#!/usr/bin/env python3

import argparse
import jax
import jax.numpy as jnp

from JaxSeq.bucket_manager import open_with_bucket as open
from JaxSeq.checkpointing import load_pytree, save_pytree


def convert_train_state_to_params(
    train_state_path: str,
    params_path: str,
    dtype: jnp.dtype
):
    """
    Loads 'train_state.msgpack' from a local or GCS path, extracts the 'params',
    and saves them to 'params.msgpack' in the same or another path.

    :param train_state_path: The path or GCS URI to the train_state.msgpack file.
    :param params_path:      Where to save the new params.msgpack file.
    :param dtype:            Float dtype for conversion (jnp.float32 or jnp.bfloat16).
    """
    print(f"Loading train_state from: {train_state_path}")
    # 1. Load the entire train_state, ignoring shape mismatch by setting target=None,
    #    so we just read the raw dictionary structure.
    raw_tree = load_pytree(
        path=train_state_path,
        target=None,   # No shape matching
        dtype=dtype,
        sharding=None  # No streaming or device placement; just read CPU memory
    )

    # 2. Extract just the parameters
    if "params" not in raw_tree:
        raise ValueError(f"No 'params' key found in {train_state_path}!")
    params = raw_tree["params"]

    # 3. Save just the params.  Usually we keep a dict structure {"params": params}.
    save_data = {"params": params}

    print(f"Saving params to: {params_path}")
    save_pytree(
        tree=save_data,
        path=params_path,
        dtype=dtype,
        sharding=None
    )
    print("Conversion complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a JAXSeq train_state.msgpack checkpoint into params-only params.msgpack."
    )
    parser.add_argument(
        "--train_state_uri",
        required=True,
        help="Local or GCS path to the train_state.msgpack file.",
    )
    parser.add_argument(
        "--params_uri",
        required=True,
        help="Local or GCS path to write the params.msgpack file.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="Data type for loading and saving params (default: float32).",
    )

    args = parser.parse_args()

    # Convert string dtype to actual jax.numpy dtype
    if args.dtype == "float32":
        dtype = jnp.float32
    else:
        dtype = jnp.bfloat16

    convert_train_state_to_params(
        train_state_path=args.train_state_uri,
        params_path=args.params_uri,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()