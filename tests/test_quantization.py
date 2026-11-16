import numpy as np
import pytest

from realm_quant.quantization import _ternarize, pack_2bit, unpack_2bit


def test_pack_unpack_round_trip_all_symbols():
    values = np.array([0, 1, 2, 3, 2, 1, 0, 3, 1], dtype=np.uint8)
    packed = pack_2bit(values)
    np.testing.assert_array_equal(unpack_2bit(packed, values.size), values)


def test_pack_rejects_out_of_range_symbols():
    with pytest.raises(ValueError):
        pack_2bit(np.array([4], dtype=np.uint8))


def test_ternary_values_are_only_zero_or_signed_scale():
    weights = np.array(
        [[1.0, -2.0, 0.0, 4.0], [-1.0, 3.0, -5.0, 2.0]],
        dtype=np.float32,
    )
    symbols, scales, restored = _ternarize(weights, 0.5)
    assert set(np.unique(symbols)).issubset({0, 1, 2})
    assert scales.shape == (2,)
    assert np.isfinite(restored).all()
    assert np.all(np.isin(restored, [-scales[0], 0.0, scales[0], -scales[1], scales[1]]))


def test_ternary_rejects_nonfinite_weights():
    with pytest.raises(ValueError):
        _ternarize(np.array([[1.0, np.nan]], dtype=np.float32), 0.7)


def test_ternary_requires_matrix_or_higher_rank():
    with pytest.raises(ValueError):
        _ternarize(np.array([1.0, 2.0], dtype=np.float32), 0.7)
