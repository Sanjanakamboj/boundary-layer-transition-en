# Verification

This document is the final, standalone verification record for Project 10
(Boundary-Layer Transition, e^N). All numbers below were freshly
recomputed in this session (via `python3 -m pytest -W error -q`,
`python3 scripts/independent_audit.py`, and the individual milestone
study scripts) and are not copied forward from earlier milestone reports
without re-running them.

## M1 laminar boundary layer

**Identities checked** (all via independent hand arithmetic or an
alternate numerical path, not by calling the same production function
twice):

- `Re_c = rho V_inf c / mu`
- `Re_x = V_inf x / nu` at a representative station
- Blasius `theta`, `delta*`, `Cf,x` closed forms
- External-flow anchor values: `U_e/V_inf` at `s=0`, `s=s_peak`, `s=1`
- Thwaites momentum-thickness integral, recomputed via an independent
  20001-point `numpy.trapezoid` quadrature on a separately sampled grid
- `x_sep,lam/c`, recomputed via an independent bisection on the sign
  change of the Thwaites `l(lambda)` correlation

**Residuals (this session):** all identity checks: `0.0` (exact, to
floating-point representation); the independent-quadrature `theta` check:
absolute `9.4e-12`, relative `6.0e-8` (both far below the documented
`~1e-3` grid-resolution expectation for that independent quadrature).

## M2 amplification

**Checked:** `Re_theta = rho U_e theta / mu` at a representative station;
`critical_re_theta(H)` at `H=3`; the local amplification-rate proxy
`dN/ds` at a representative `(Re_theta, H)` state; an independent
constant-rate `N`-factor integration control case
(`N(x) = k(x-x0)` for `dN/ds = k` over `[x0, x1]`).

**Residuals:** all `0.0` except the constant-rate integration control,
which matches to `1.6e-14` absolute (`1.3e-14` relative) -- floating-point
noise from `scipy.integrate.cumulative_trapezoid` on a smooth constant
integrand, not a modeling residual.

## M3 transition / flat-plate drag

**Checked:** the `N_crit` linear-interpolation crossing formula (hand
fraction/interpolation on a synthetic `N(s)` array); the section
skin-friction drag normalization `C_d,f = 2 * Cf * (U_e/V_inf)^2` on a
constant-`Cf`, constant-`Ue` control case; the turbulent flat-plate `Cf,x`
correlation at a representative station.

**Residuals:** all `<= 7e-18` absolute (`<=2e-16` relative) -- machine
precision.

## M4 turbulent boundary layer

**Checked:** `H1(H)` round-trip inversion at four representative `H`
values spanning both correlation branches; the Ludwieg-Tillmann `Cf`
closed form; the Green-Weeks-Brooman entrainment closure `F(H1)`; a
**genuine finite-difference residual** of the momentum-integral equation,
reconstructed from an actual solved turbulent-restart trajectory (central
difference of `theta(x)` compared against `momentum_equation_rhs()`
evaluated at the same interior station); a representative turbulent
(re-)separation location, cross-checked against the previously
established value.

**Residuals:** `H1(H)` round-trips: `<=4.0e-15` absolute. Ludwieg-Tillmann
and entrainment closures: `0.0`. Momentum-equation finite-difference
residual: absolute `2.0e-8`, relative `1.5e-6` -- this is a genuine
ODE-discretization/finite-difference residual, **not** expected to reach
machine precision, and is reported as such (per the explicit project
instruction not to treat ODE discretization differences as if they should
be machine-precision-exact). Representative turbulent separation
location: absolute `1.9e-5` chord fraction, relative `2.9e-5` -- a
grid-resolution-dependent quantity, well within the documented `~1e-2`
tolerance.

## M5 separation bubble / separated drag

**Checked:** bubble event ordering (`x_sep,lam <= x_tr,sep <= x_reattach`)
for the nominal bubble at the baseline case; the restart momentum
thickness `theta_reattach = K_theta * theta_sep` (hand multiplication
against an independently interpolated `theta_sep`); the separated-drag
closed form `C_d,sep = K_sep * L_sep^p * (U_e(x_sep)/V_inf)^2`; the exact
drag-decomposition identity `C_d,total = C_d,f + C_d,sep`.

**Residuals:** all `0.0` (event ordering is a boolean check, reported as
`0.0`/`1.0` = pass/fail, not a continuous residual).

## M6 external-flow sensitivity

**Checked:** the baseline Milestone 6 sensitivity profile reproduces the
Milestone 1 `SyntheticVelocityDistribution` defaults **bit-for-bit**
(`np.array_equal`, not merely `pytest.approx`) over a 2001-station grid;
one profile-anchor value (`te_ratio_severe`'s trailing-edge ratio); the
full-chain-orchestrated `x_sep,lam/c` (via `robustness.run_chain`)
against a direct, independent `solve_thwaites` call on the same
distribution.

**Residuals:** all `0.0`, except the profile-anchor floating-point
comparison (`5.6e-17` absolute, `1.2e-16` relative -- machine precision).

## Regression: M1-M5 headline values (this session)

Freshly reproduced via `scripts/manual_check.py`,
`scripts/laminar_boundary_layer_study.py`,
`scripts/en_amplification_study.py`, `scripts/transition_drag_study.py`,
`scripts/turbulent_boundary_layer_study.py`, and
`scripts/separation_bubble_drag_study.py`:

| Quantity | Value |
|---|---|
| `Re_c` | 1.1980e+06 |
| `U_e/V_inf` peak / location | 1.18 at `x/c=0.35` |
| `U_e/V_inf` at TE | 0.55 |
| `x_sep,lam/c` (M1 laminar separation) | 0.4314 |
| Modeled instability onset `x/c` (M2) | 0.0885 |
| `N_max` (M2, baseline) | 3.8429 |
| `N_crit` sensitivity set (M3) | `{9, 12, 14}`, all `SEPARATION_BEFORE_N_CRIT` |
| `C_d,f` fully laminar reference (M3) | 0.002451 |
| `C_d,f` fully turbulent reference (M3) | 0.008730 |
| `C_d,f` separation-triggered (M3) | 0.005044 |
| M4 separation-triggered restart `x_tr/c` | 0.4314 |
| M4 downstream turbulent separation `x/c` | 0.6704-0.7854 (rtol-dependent; `0.785` for the M3-triggered restart, `0.670` for the M5 nominal-bubble restart -- see below) |
| M5 nominal `C_d,f` | 0.002495 |
| M5 nominal `C_d,sep` | 0.019936 |
| M5 nominal `C_d,total` | 0.022430 |

**Note on the two different M4 turbulent-separation numbers above:** the
Milestone 4 study script restarts the turbulent solver *directly* at the
M1 laminar-separation station (`x_tr/c=0.4314`, `H_tr=1.286`, giving a
turbulent separation at `x/c~=0.785`), while the Milestone 5 nominal-bubble
case restarts it at the *bubble-reattachment* station (`x_tr/c~=0.461`,
`H_reattach=1.7`), giving a different (earlier) turbulent separation at
`x/c~=0.670`. Both are correct for their respective, explicitly different,
restart assumptions -- they are not a discrepancy.

## Final test suite

- **276 tests**, `pytest -W error -q`: **276 passed, 0 failed, 0
  warnings.**
- Breakdown: 200 tests from Milestones 1-4 (unchanged), 39 tests added in
  Milestone 5 (unchanged), 37 tests added in Milestone 6
  (`test_external_flow_sensitivity.py`: 17; `test_robustness.py`: 20,
  including one regression test added after a real bug was found and
  fixed during Milestone 6 development -- see DESIGN.md Section 42).

## Independent audit summary

`scripts/independent_audit.py`: **32 independent checks**, maximum
absolute residual **1.944e-05** (the grid-dependent turbulent-separation
check), maximum relative residual **2.899e-05** (same check). All
algebraic-identity checks (29 of 32) are at or below `~1e-8` absolute,
consistent with floating-point precision; the two explicitly-flagged
finite-difference/grid-dependent checks are not held to that standard,
per the project's own instruction.
