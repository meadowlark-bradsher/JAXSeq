import jax
import jax.numpy as jnp

from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.checkpointing import load_pytree, save_pytree
import os

def convert_train_state_to_params(
    train_state_path: str,
    params_path: str,
    dtype: jnp.dtype = jnp.float32
):
    """
    Loads 'train_state.msgpack' from a local or GCS path, extracts the 'params',
    and saves them to 'params.msgpack' in the same or another path.

    :param train_state_path: The path or GCS URI to the train_state.msgpack file.
    :param params_path:      Where to save the new params.msgpack file.
    :param dtype:            Optional float dtype for conversion (e.g., float32).
    """
    # 1. Load the entire train_state, ignoring shape mismatch by setting target=None
    #    and sharding=None so we just read the raw dictionary.
    raw_tree = load_pytree(
        path=train_state_path,
        target=None,
        dtype=dtype,
        sharding=None  # No streaming or device placement; just read into CPU memory
    )

    # 2. Extract just the parameters
    if "params" not in raw_tree:
        raise ValueError(f"No 'params' key found in {train_state_path}!")
    params = raw_tree["params"]

    # 3. Save just the params.
    #    Note: we wrap it in a dict like {"params": params} to keep
    #    a consistent data structure. That’s typical in Flax-based code.
    #
    #    If you want truly just the raw tree, you could do: save_pytree(params, params_path, ...)
    #    Then you'd load it with something that expects a bare PyTree, not a dict.
    save_pytree(
        tree={"params": params},
        path=params_path,
        dtype=dtype,
        sharding=None
    )
    print(f"Successfully wrote only 'params' to {params_path}")

if __name__ == "__main__":
    # Example usage:
    # Suppose your train_state.msgpack is at 'gs://my-bucket/old_bc/train_state.msgpack'
    # and you want to save 'params.msgpack' at 'gs://my-bucket/old_bc/params.msgpack'.

    old_train_state_uri = "gs://my-bucket/old_bc/train_state.msgpack"
    new_params_uri = "gs://my-bucket/old_bc/params.msgpack"

    convert_train_state_to_params(old_train_state_uri, new_params_uri, dtype=jnp.float32)