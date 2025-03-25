# test_util.py

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock
from jax.sharding import PartitionSpec as PS, NamedSharding, Mesh
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit

# Import the function under test:
# from your_module.util import multihost_device_put
# Adjust to match your actual import path.
from JaxSeq.utils import multihost_device_put


@pytest.mark.parametrize("assert_equal_env", [True, False])
@pytest.mark.parametrize(
    "sharding, should_call_pjit",
    [
        (None, False),
        (PS(), True),
        # If you want to test NamedSharding, you’ll need a small Mesh for demonstration:
        (NamedSharding(Mesh(jax.devices()[:1], ["x"]), PS()), True),
    ],
)
@patch("jax.experimental.multihost_utils.assert_equal", autospec=True)
@patch("jax.experimental.pjit.pjit", autospec=True)
def test_multihost_device_put(
    mock_pjit,
    mock_assert_equal,
    sharding,
    should_call_pjit,
    assert_equal_env,
):
    # Make pjit mock return an identity function
    def fake_identity_fn(x):
        return x
    mock_pjit.return_value = fake_identity_fn

    # Simple JAX array
    x = jnp.array([1, 2, 3], dtype=jnp.float32)

    # Call function
    result = multihost_device_put(
        x,
        sharding=sharding,
        assert_equal_per_host=assert_equal_env,  # pass directly
    )

    # --- Check calls ---

    # Did we call multihost_utils.assert_equal?
    if assert_equal_env:
        mock_assert_equal.assert_called_once()
    else:
        mock_assert_equal.assert_not_called()

    # Did we call pjit or not, depending on the sharding type?
    if should_call_pjit:
        mock_pjit.assert_called_once()
    else:
        mock_pjit.assert_not_called()

    # Check that result’s contents are unchanged
    np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.float32))

    # Optionally confirm it's still some JAX array type
    assert isinstance(result, jax.Array)