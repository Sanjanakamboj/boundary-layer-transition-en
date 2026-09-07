# Boundary-Layer Transition (e^N) -- Project 10

**Milestone 1: Laminar Boundary-Layer Foundation and Stability Inputs**

## Engineering objective

Build a rigorous, independently verified foundation for a later e^N
transition-prediction model for a sailplane wing: a representative
operating condition, an external velocity distribution, laminar
boundary-layer solutions, Reynolds-number bookkeeping, and a verification
framework. **This milestone does not predict transition.**

## Generic sailplane operating point

A single, explicitly illustrative (not manufacturer-matched) wing-section
operating point is used throughout:

| Quantity | Value |
|---|---|
| Chord, `c` | 0.70 m |
| Freestream velocity, `V_inf` | 25.0 m/s |
| Density, `rho` (ISA sea level) | 1.225 kg/m^3 |
| Dynamic viscosity, `mu` (ISA sea level) | 1.7894e-5 Pa.s |
| Kinematic viscosity, `nu` | 1.4607e-5 m^2/s |
| Mach number | 0.0735 (incompressible regime) |
| Chord Reynolds number, `Re_c` | 1.198e6 |
| Angle of attack (descriptive only) | 4.0 deg |

See [`operating_point.py`](src/boundary_layer_transition/operating_point.py)
and [DESIGN.md](DESIGN.md) for the full rationale.

## What is implemented

1. **Operating point** (`operating_point.py`) -- Re_c, Mach, incompressibility
   check, validated inputs.
2. **External velocity distribution** (`external_flow.py`) -- a transparent,
   analytic, C1-continuous synthetic `U_e(x/c)` with a favorable-gradient
   region followed by an adverse-gradient region, plus its exact analytic
   derivative `dU_e/dx`. **This is a reduced-order illustrative
   distribution, not from CFD/XFOIL/experiment/a real airfoil.**
3. **Laminar boundary layer** (`laminar_bl.py`):
   - Zero-pressure-gradient Blasius reference (`delta_99`, `delta*`, `theta`,
     `H`, `Cf,x`).
   - Thwaites' pressure-gradient-aware laminar momentum-integral method
     (`theta`, `delta*`, `H`, `Cf,x`, the pressure-gradient parameter
     `lambda`, and a laminar-separation diagnostic).
4. Full pytest suite with independent hand/reference verification, grid
   convergence testing, and invalid-input rejection.
5. Two CLI scripts and three deterministic, regenerable figures.

## Representative Milestone 1 results

(from `scripts/laminar_boundary_layer_study.py`, `n_points=4001`)

- Peak external velocity `U_e/V_inf = 1.18` at `x/c = 0.35`; trailing-edge
  `U_e/V_inf = 0.55`.
- Near the leading edge, Thwaites' `theta` matches the Blasius ZPG
  reference to ~1%, as expected for weak local pressure gradient there.
- **Predicted laminar separation** (Thwaites `l(lambda)=0` diagnostic):
  `x/c ~= 0.431`, at `lambda_sep ~= -0.0898` (literature reference `~=
  -0.09`).
- This separation prediction is a **laminar-boundary-layer diagnostic
  only** -- it is *not* a transition prediction.

## Install / run / test

```bash
python3 -m pip install -e ".[dev]"

# Tests (zero warnings expected)
python3 -m pytest -W error -q

# Manual sanity-check printout
python3 scripts/manual_check.py

# Blasius vs. Thwaites comparison + separation diagnostic
python3 scripts/laminar_boundary_layer_study.py

# Regenerate all figures (deterministic; figures/*.png)
python3 scripts/generate_figures.py
```

No installation is strictly required to run the scripts directly (they
insert `src/` onto `sys.path`), but `pip install -e ".[dev]"` is recommended
for editor/IDE support and running pytest from anywhere.

## Figures

| File | Content |
|---|---|
| `figures/external_velocity_distribution.png` | `U_e/V_inf` vs. `x/c`, favorable/adverse regions labeled |
| `figures/laminar_boundary_layer_growth.png` | `theta/c`, `delta*/c` vs. `x/c`, Thwaites vs. Blasius |
| `figures/laminar_shape_factor.png` | `H` and `lambda` vs. `x/c`, separation criterion marked |

All figures carry the caption *"Generic reduced-order model -- illustrative
only, not experimentally validated."*

## Limitations (see DESIGN.md for full detail)

- Incompressible, 2D, laminar-only; no roughness, free-stream turbulence,
  compressibility, or sweep corrections.
- The external velocity distribution is synthetic/illustrative, not derived
  from any real airfoil or CFD/experimental data.
- The leading edge is modeled as flat-plate-like (`U_e(0)` finite and
  nonzero), not as a true airfoil stagnation point.
- Thwaites' method is an approximate correlation, valid only for
  `lambda in [-0.10, 0.10]`; results are explicitly withheld (`NaN`)
  outside that domain or past the predicted separation station rather than
  extrapolated.

## What Milestone 1 does NOT yet model

- e^N amplification-factor tracking or linear stability analysis.
- Any choice of `N_crit`.
- Transition-location prediction. (A predicted laminar-separation location
  is a diagnostic, not a transition point.)
- Turbulent boundary-layer development or skin-friction/drag downstream of
  transition.
- XFOIL/CFD/real-airfoil data of any kind.
- Roughness, free-stream-turbulence, compressibility, or sweep corrections.

## Recommended Milestone 2 topic (not started)

Linear stability theory inputs and an e^N amplification-factor
calculation (Orr-Sommerfeld / Falkner-Skan-family approximate methods, or
an envelope method such as the Drela-Giles correlation) driven by the
Thwaites `theta(x)`, `H(x)`, `U_e(x)` fields established here, culminating
in an N-factor curve vs. `x/c` -- with `N_crit` selection and transition
prediction deferred to Milestone 3 unless folded in explicitly.
