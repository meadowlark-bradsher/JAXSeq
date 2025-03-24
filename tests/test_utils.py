# test_util.py

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from unittest.mock import patch, MagicMock

# Import the function under test:
# from your_module.util import multihost_device_put
# Adjust to match your actual import path.
from JaxSeq.utils import multihost_device_put


@pytest.mark.parametrize("assert_equal_env", [True, False])
@patch("jax.experimental.multihost_utils.assert_equal", autospec=True)
@patch("jax.experimental.pjit.pjit", autospec=True)
def test_multihost_device_put(mock_pjit, mock_assert_equal, assert_equal_env):
    """
    Tests that `multihost_device_put` behaves correctly given:
    - The environment variable that toggles assert_equal_per_host
    - Sharding=None
    - Input is a JAX array
    """
    # Mock the environment variable controlling assert_equal
    # (or you can rely on the real environment if you prefer)
    # For demonstration, we override it directly:
    original_flag = jax.config.FLAGS.read("env:ASSERT_EQUAL_PER_HOST", None)
    # If you need to patch an environment variable, you'd do it with monkeypatch or os.environ.
    # But let's illustrate conceptually:
    #   monkeypatch.setenv("MULTIHOST_DEVICE_PUT_ASSERT_EQUAL_PER_HOST", "1" or "0")
    # For brevity, we skip that here and just rely on the test param.

    # mock_pjit will return a fake function that returns x unchanged
    def fake_identity_fn(x):
        return x
    mock_pjit.return_value = fake_identity_fn

    # Make a sample JAX array
    x = jnp.array([1, 2, 3], dtype=jnp.float32)

    # Now call the function under test
    result = multihost_device_put(
        x,
        sharding=None,
        assert_equal_per_host=assert_equal_env,
    )

    # --- Assertions / Verifications ---

    # 1) Did we call assert_equal if assert_equal_per_host == True?
    if assert_equal_env:
        mock_assert_equal.assert_called_once()
    else:
        mock_assert_equal.assert_not_called()

    # 2) Since sharding=None, pjit should still be invoked once, but with trivial specs
    mock_pjit.assert_called_once()

    # 3) Verify the “put” did not alter the data (our fake_identity_fn returns x)
    #    If you are mocking pjit entirely, you are only verifying call structure.
    #    We can still check that result is the same array values:
    np.testing.assert_array_equal(np.array(result), np.array([1, 2, 3]))

    # Optionally verify result is indeed a JAX array.
    assert isinstance(result, jax.Array)

    # Restore environment or do cleanup if necessary
    if original_flag is not None:
        jax.config.FLAGS["env:ASSERT_EQUAL_PER_HOST"] = original_flag