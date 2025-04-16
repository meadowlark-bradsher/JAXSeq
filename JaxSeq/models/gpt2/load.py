from __future__ import annotations
from typing import Optional, Union, Callable, Tuple
from JaxSeq.bucket_manager import open_with_bucket as open
from jax.sharding import Mesh
from jax.sharding import NamedSharding
import json
import jax
import jax.numpy as jnp
from JaxSeq.models.gpt2.model import FlaxGPT2LMHeadModel
from JaxSeq.models.gpt2.config import GPT2Config
from JaxSeq.utils import match_partition_rules, inplace_float_to_dtype, file_exists
import os
import optax
from flax.training.train_state import TrainState
from JaxSeq.shard_model import shard_train_state_from_checkpoint, shard_train_state_from_params, shard_params_from_params, shard_params_from_config, shard_params_from_checkpoint, get_sharding_from_model
import math
from flax.core import unfreeze, freeze
from enum import Enum
from jaxtyping import PyTree
from transformers.tokenization_utils import PreTrainedTokenizer

class ModelLoadMode(Enum):
    HF = 'hf'
    CONFIG = 'config'
    TRAIN_STATE = 'train_state'
    TRAIN_STATE_PARAMS = 'train_state_params'
    PARAMS = 'params'

    @staticmethod
    def match_load_mode(load_mode: Union[ModelLoadMode, str], target: ModelLoadMode):
        if isinstance(load_mode, str):
            return load_mode == target.value
        return load_mode == target

def pad_embeddings(
    params: PyTree, 
    model: FlaxGPT2LMHeadModel, 
    tokenizer: PreTrainedTokenizer, 
    dtype: jnp.dtype=jnp.float32, 
) -> PyTree:
    old_size = model.config.vocab_size
    model.config.vocab_size = int(2**math.ceil(math.log2(len(tokenizer))))
    print(f'Padding embeddings from size {old_size} to size {model.config.vocab_size}. Tokenizer vocab size {len(tokenizer)}.')
    # pad embeddings
    sharding = get_sharding_from_model(model, params)
    return model.pad_embeddings(params, param_sharding=sharding, dtype=dtype)

def load_train_state_from_config(
    model_config: GPT2Config, 
    model_dtype: Union[str, jnp.dtype], 
    optim_getter: Callable[[PyTree], optax.GradientTransformation], 
    tokenizer: PreTrainedTokenizer, 
    mesh: Mesh, # should be shape (dp, fsdp, mp)
    prng_key: jax.random.PRNGKeyArray, 
    force_pad_embeddings: bool=False, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[TrainState, FlaxGPT2LMHeadModel]:
    model = FlaxGPT2LMHeadModel(model_config, dtype=model_dtype, _do_init=False)
    model.config.mesh = mesh
    # shard params
    params = freeze(shard_params_from_config(model, prng_key, params_dtype=params_dtype))
    # pad embeddings
    should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
    if should_pad:
        params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
    # shard train_state
    train_state = shard_train_state_from_params(model, params, optim_getter(params))

    return train_state, model

def load_train_state(
    model_load_mode: Union[ModelLoadMode, str], 
    model_load_path: str, 
    model_dtype: Union[str, jnp.dtype], 
    optim_getter: Callable[[PyTree], optax.GradientTransformation], 
    tokenizer: PreTrainedTokenizer, 
    mesh: Mesh, # should be shape (dp, fsdp, mp)
    prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    force_pad_embeddings: bool=False, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[TrainState, FlaxGPT2LMHeadModel]:
    
    if ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.HF):
        print("Using ModelLoadMode.HF")
        # load model
        with jax.default_device(jax.devices('cpu')[0]):
            model, params = FlaxGPT2LMHeadModel.from_pretrained(model_load_path, _do_init=False, dtype=model_dtype)
        model.config.mesh = None # None so that padding is not sharded
        # set dtype
        params = unfreeze(params)
        inplace_float_to_dtype(params, params_dtype)
        params = freeze(params)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            with jax.default_device(jax.devices('cpu')[0]):
                params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
        # shard
        model.config.mesh = mesh # back to mesh for final sharding
        params = shard_params_from_params(model, params)
        train_state = shard_train_state_from_params(model, params, optim_getter(params))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.CONFIG):
        print("Using ModelLoadMode.CONFIG")
        # load config
        assert prng_key is not None, 'Must provide prng_key when loading from config.'
        with open(model_load_path, 'r') as f:
            model_config = GPT2Config.from_dict(json.load(f))
        train_state, model = load_train_state_from_config(
            model_config=model_config, 
            model_dtype=model_dtype, 
            optim_getter=optim_getter, 
            tokenizer=tokenizer, 
            mesh=mesh, 
            prng_key=prng_key, 
            force_pad_embeddings=force_pad_embeddings, 
            params_dtype=params_dtype, 
        )
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.TRAIN_STATE):
        print("Using ModelLoadMode.TRAIN_STATE")
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = GPT2Config.from_dict(json.load(f))
        model = FlaxGPT2LMHeadModel(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh
        # shard and pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if not should_pad:
            # if no padding, just load train_state, shard as well
            train_state = shard_train_state_from_checkpoint(model, os.path.join(model_load_path, 'train_state.msgpack'), optim_getter, just_params=False, train_state_dtype=params_dtype)
        else:
            # if padding, load params, pad, shard
            params = shard_train_state_from_checkpoint(model, os.path.join(model_load_path, 'train_state.msgpack'), optim_getter, just_params=True, train_state_dtype=params_dtype)
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
            train_state = shard_train_state_from_params(model, params, optim_getter(params))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.TRAIN_STATE_PARAMS):
        print("Using ModelLoadMode.TRAIN_STATE_PARAMS")
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = GPT2Config.from_dict(json.load(f))
        model = FlaxGPT2LMHeadModel(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh
        # load params, shard params
        params = shard_train_state_from_checkpoint(model, os.path.join(model_load_path, 'train_state.msgpack'), optim_getter, just_params=True, train_state_dtype=params_dtype)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
        # shard train_state
        train_state = shard_train_state_from_params(model, params, optim_getter(params))
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.PARAMS):
        # ---------------------------------------------------------------------------------------
        # ### TODO: Finish or remove partial fallback logic
        #
        # We added a partial implementation that tries to load just `params` from a `train_state.msgpack`
        # if `params.msgpack` isn't present. However, if there is a mismatch in optimizer state keys
        # (e.g. `'opt_state' -> 'mini_step'`), we can still get a KeyError.
        #
        # Options to fix this properly:
        # 1. Patch `shard_train_state_from_checkpoint` (or `load_pytree`) so that, if we're ignoring
        #    the optimizer, we completely skip loading those keys.
        # 2. Convert the old checkpoint offline into a `params.msgpack` (params only).
        # 3. Switch to streaming partial loads (advanced).
        #
        # For now, if we run into the `KeyError`, we plan to train a new BC model or manually
        # convert the checkpoint. Eventually, we should either finalize the partial loader logic
        # or roll it back to avoid confusion.
        #
        # ---------------------------------------------------------------------------------------
        print("Using ModelLoadMode.PARAMS")

        params_path = os.path.join(model_load_path, 'params.msgpack')
        train_state_path = os.path.join(model_load_path, 'train_state.msgpack')

        config_path = os.path.join(model_load_path, 'config.json')
        if not file_exists(config_path):
            raise FileNotFoundError(f"Could not find config.json in {model_load_path}.")
        with open(config_path, 'r') as f:
            model_config = GPT2Config.from_dict(json.load(f))

        model = FlaxGPT2LMHeadModel(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh

        # Decide how to load
        if file_exists(params_path):
            # A) Load from 'params.msgpack'
            params = shard_params_from_checkpoint(
                model,
                params_path,
                params_dtype=params_dtype
            )
        elif file_exists(train_state_path):
            # B) Fallback: 'train_state.msgpack' => just load "params"
            print(f"Warning: 'params.msgpack' not found in {model_load_path}; "
                  f"falling back to 'train_state.msgpack' (params only).")
            params = shard_train_state_from_checkpoint(
                model,
                train_state_path,
                optim_getter=lambda p: optax.adam(1e-4),  # dummy or real, won't be used
                just_params=True,
                train_state_dtype=params_dtype
            )
        else:
            # C) Neither file
            raise FileNotFoundError(
                f"Could not find either 'params.msgpack' or 'train_state.msgpack' in {model_load_path}."
            )

        # Pad embeddings if needed
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))

        # Create final TrainState using the new optimizer
        train_state = shard_train_state_from_params(
            model,
            params,
            optim_getter(params)
        )
    else:
        raise ValueError(f"Invalid model_load_mode: {model_load_mode}")
    
    return train_state, model

def load_params_from_config(
    model_config: GPT2Config, 
    model_dtype: Union[str, jnp.dtype], 
    tokenizer: PreTrainedTokenizer, 
    mesh: Mesh, # should be shape (dp, fsdp, mp)
    prng_key: jax.random.PRNGKeyArray, 
    force_pad_embeddings: bool=False, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[PyTree, FlaxGPT2LMHeadModel]:
    model = FlaxGPT2LMHeadModel(model_config, dtype=model_dtype, _do_init=False)
    model.config.mesh = mesh
    # shard params
    params = freeze(shard_params_from_config(model, prng_key, params_dtype=params_dtype))
    # pad embeddings
    should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
    if should_pad:
        params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
    
    return params, model

def load_params(
    model_load_mode: Union[ModelLoadMode, str], 
    model_load_path: str, 
    model_dtype: Union[str, jnp.dtype], 
    tokenizer: PreTrainedTokenizer, 
    mesh: Mesh, # should be shape (dp, fsdp, mp)
    prng_key: Optional[jax.random.PRNGKeyArray]=None, 
    force_pad_embeddings: bool=False, 
    params_dtype: Optional[Union[str, jnp.dtype]]=jnp.float32, 
) -> Tuple[PyTree, FlaxGPT2LMHeadModel]:
    if ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.HF):
        # load model
        with jax.default_device(jax.devices('cpu')[0]):
            model, params = FlaxGPT2LMHeadModel.from_pretrained(model_load_path, _do_init=False, dtype=model_dtype)
        model.config.mesh = None # None so that padding is not sharded
        # set dtype
        params = unfreeze(params)
        inplace_float_to_dtype(params, params_dtype)
        params = freeze(params)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            with jax.default_device(jax.devices('cpu')[0]):
                params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
        # shard
        model.config.mesh = mesh # back to mesh for final sharding
        params = shard_params_from_params(model, params)
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.CONFIG):
        # load config
        assert prng_key is not None, 'Must provide prng_key when loading from config.'
        with open(model_load_path, 'r') as f:
            model_config = GPT2Config.from_dict(json.load(f))
        params, model = load_params_from_config(
            model_config=model_config, 
            model_dtype=model_dtype, 
            tokenizer=tokenizer, 
            mesh=mesh, 
            prng_key=prng_key, 
            force_pad_embeddings=force_pad_embeddings, 
            params_dtype=params_dtype, 
        )
    elif ModelLoadMode.match_load_mode(model_load_mode, ModelLoadMode.PARAMS):
        # load model
        with open(os.path.join(model_load_path, 'config.json'), 'r') as f:
            model_config = GPT2Config.from_dict(json.load(f))
        model = FlaxGPT2LMHeadModel(model_config, dtype=model_dtype, _do_init=False)
        model.config.mesh = mesh
        # load params, shard params
        params = shard_params_from_checkpoint(model, os.path.join(model_load_path, 'params.msgpack'), params_dtype=params_dtype)
        # pad embeddings
        should_pad = (force_pad_embeddings or len(tokenizer) > model.config.vocab_size)
        if should_pad:
            params = freeze(pad_embeddings(unfreeze(params), model, tokenizer, dtype=params_dtype))
    else:
        raise ValueError(f"Invalid model_load_mode: {model_load_mode}")
    
    return params, model
