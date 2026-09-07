import math

import numpy as np
import pytest

from boundary_layer_transition.laminar_bl import (
    BLASIUS_H,
    blasius_cf,
    blasius_delta99,
    blasius_delta_star,
    blasius_theta,
    reynolds_x,
)

V = 25.0
NU = 1.4607e-5  # m^2/s, matches the default operating point to ~1e-4 rel.
X = 0.245  # m, an arbitrary interior station (s=0.35 on a 0.7 m chord)


def test_reynolds_x_hand_check():
    re_x_hand = V * X / NU
    assert reynolds_x(V, X, NU) == pytest.approx(re_x_hand, rel=1e-12)
    assert reynolds_x(V, X, NU) == pytest.approx(419524.0, rel=1e-3)


def test_reynolds_x_zero_at_leading_edge():
    assert reynolds_x(V, 0.0, NU) == 0.0


def test_blasius_delta99_hand_check():
    re_x = V * X / NU
    delta99_hand = 5.0 * X / math.sqrt(re_x)
    assert blasius_delta99(V, X, NU) == pytest.approx(delta99_hand, rel=1e-9)


def test_blasius_delta_star_hand_check():
    re_x = V * X / NU
    dstar_hand = 1.7208 * X / math.sqrt(re_x)
    assert blasius_delta_star(V, X, NU) == pytest.approx(dstar_hand, rel=1e-9)


def test_blasius_theta_hand_check():
    re_x = V * X / NU
    theta_hand = 0.664 * X / math.sqrt(re_x)
    assert blasius_theta(V, X, NU) == pytest.approx(theta_hand, rel=1e-9)


def test_blasius_cf_hand_check():
    re_x = V * X / NU
    cf_hand = 0.664 / math.sqrt(re_x)
    assert blasius_cf(V, X, NU) == pytest.approx(cf_hand, rel=1e-9)


def test_blasius_h_matches_ratio_and_literature_value():
    dstar = blasius_delta_star(V, X, NU)
    theta = blasius_theta(V, X, NU)
    assert dstar / theta == pytest.approx(BLASIUS_H, rel=1e-12)
    # Commonly cited rounded literature value is 2.59; our self-consistent
    # ratio of the published coefficients is ~2.5916 (~0.06% different).
    assert BLASIUS_H == pytest.approx(2.59, rel=2e-3)


def test_leading_edge_thickness_quantities_go_to_zero_not_nan():
    assert blasius_delta99(V, 0.0, NU) == 0.0
    assert blasius_delta_star(V, 0.0, NU) == 0.0
    assert blasius_theta(V, 0.0, NU) == 0.0


def test_leading_edge_cf_is_explicitly_singular_not_fabricated():
    cf0 = blasius_cf(V, 0.0, NU)
    assert math.isinf(cf0) and cf0 > 0


def test_no_nan_over_valid_domain_excluding_leading_edge_cf():
    x = np.linspace(0.0, 0.7, 500)
    for fn in (blasius_delta99, blasius_delta_star, blasius_theta):
        y = fn(V, x, NU)
        assert np.all(np.isfinite(y)), fn.__name__
    cf = blasius_cf(V, x, NU)
    assert np.isinf(cf[0])
    assert np.all(np.isfinite(cf[1:]))


def test_scalar_vector_consistency():
    x_vals = [0.0, 0.01, 0.1, 0.35, 0.7]
    x_arr = np.array(x_vals)
    for fn in (blasius_delta99, blasius_delta_star, blasius_theta, blasius_cf, lambda v, x, n: reynolds_x(v, x, n)):
        vec = fn(V, x_arr, NU)
        for i, xv in enumerate(x_vals):
            scalar = fn(V, xv, NU)
            if math.isinf(scalar) or math.isinf(vec[i]):
                assert math.isinf(scalar) and math.isinf(vec[i])
            else:
                assert scalar == pytest.approx(vec[i], rel=1e-12)


def test_vectorized_matches_looped_scalar_calls():
    x = np.linspace(0.001, 0.7, 37)
    vec = blasius_theta(V, x, NU)
    looped = np.array([blasius_theta(V, xi, NU) for xi in x])
    assert vec == pytest.approx(looped, rel=1e-12)


@pytest.mark.parametrize(
    "fn",
    [blasius_delta99, blasius_delta_star, blasius_theta, blasius_cf, reynolds_x],
)
def test_invalid_inputs_rejected(fn):
    with pytest.raises(ValueError):
        fn(V, -0.1, NU)
    with pytest.raises(ValueError):
        fn(0.0, 0.1, NU)
    with pytest.raises(ValueError):
        fn(V, 0.1, 0.0)
    with pytest.raises(ValueError):
        fn(V, float("nan"), NU)
