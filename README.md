# Boundary-Layer Transition (e^N) -- Project 10

**Milestone 1: Laminar Boundary-Layer Foundation and Stability Inputs**
**Milestone 2: Reduced-Order Linear-Stability Proxy and e^N Amplification Tracking**

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

---

## Milestone 2: reduced-order e^N amplification tracking

### Objective

Extend the Milestone 1 Thwaites laminar boundary-layer solution into a
transparent, source-audited amplification-tracking framework: a modeled
Tollmien-Schlichting-type instability onset, a local amplification-rate
proxy, and an integrated N-factor curve vs. `x/c`. **This milestone tracks
amplification only -- it does not select `N_crit` and does not predict a
transition location.**

### Why e^N tracks amplification rather than directly predicting transition

The e^N method does not compute a transition location from first
principles. It tracks how much small disturbances have been amplified,
`N(x) = ln(A(x)/A0)`, as they travel downstream through the (locally)
unstable region of the boundary layer. Transition is then *empirically*
associated with the station where `N(x)` reaches a calibrated threshold,
`N_crit`, whose correct value depends on the actual disturbance
environment (free-stream turbulence level, surface roughness, acoustics)
of the case at hand -- it is not a universal constant computable from the
boundary-layer solution alone. Milestone 2 computes the `N(x)` curve;
choosing `N_crit` and thereby predicting a transition location is
explicitly deferred.

### Exact distinction between the five terms

| Term | What it means here | What it is NOT |
|---|---|---|
| **Instability onset** | First `x/c` where the local, modeled criterion (`Re_theta >= Re_theta,crit(H)`) is satisfied; `N=0` there by construction | Not a transition point |
| **N-factor growth** | The accumulated amplification `N(x) = integral(dN/ds) ds` from onset onward, using a reduced-order proxy `dN/ds`, not a solved eigenvalue | Not a measure of actual disturbance amplitude in any specific real environment |
| **N_crit** | An empirical, environment-dependent threshold value of `N` at which real disturbances are typically observed to trigger transition | **Not selected in Milestone 2** |
| **Transition** | The real physical laminar-to-turbulent breakdown, conventionally associated with `N(x)` reaching `N_crit` | **Not predicted anywhere in Milestone 1 or 2** |
| **Laminar separation** (M1 diagnostic) | The Thwaites `l(lambda)=0` station -- a laminar-boundary-layer diagnostic | Not equivalent to transition; a separated laminar shear layer often transitions and may reattach turbulently, neither of which is modeled here |

### Selected reduced-order amplification method

A source audit (see [DESIGN.md](DESIGN.md) Section 8) found that the exact
published polynomial coefficients used inside XFOIL's Drela/Gleyzes-
Cousteix-Bonnet envelope method could not be verified against a primary
source within this session. Rather than reproduce unverified numbers,
Milestone 2 implements an explicitly labeled **reduced-order engineering
proxy** (`stability.py`):

```
Re_theta,crit(H) = 200 * exp(-1.0 * (H - H_blasius))      [sourced anchor: 200 at H=H_blasius]
dN/d(x/c)         = 10.0 * (H - 1) * max(0, Re_theta - Re_theta,crit(H)) / 200
N(x/c)            = integral (dN/d(x/c)) d(x/c)            [from x/c=0; dN/d(x/c)=0 pre-onset]
```

`Re_theta,crit,ZPG = 200` is the one sourced number (the classical
Blasius/Tollmien-Schlichting ZPG neutral-stability point). The `H`-shape of
the neutral curve and the amplification-rate law are project-chosen
proxies calibrated only for legibility, **not** fit to any published
amplification-rate table. Full derivation, sign convention, and the
sourced-vs-project-assumption breakdown are in DESIGN.md Sections 8-11.

### Baseline Milestone 2 results

(from `scripts/en_amplification_study.py`, `n_points=4001`, same baseline
operating point as Milestone 1)

- Modeled instability onset: `x/c ~= 0.0885`
- M1 laminar-separation diagnostic: `x/c ~= 0.4314` (unchanged from M1)
- Available amplification length: `Delta(x/c) ~= 0.343`
- Maximum accumulated N before separation: `N_max ~= 3.84`
- No N_crit is selected; therefore no transition location is predicted.

### Sensitivity results (`V_inf` / `Re_c`, geometry and `U_e/V_inf` shape fixed)

| `V_inf` [m/s] | `Re_c` | onset `x/c` | separation `x/c` | `N_max` |
|---|---|---|---|---|
| 20 | 9.58e5 | 0.126 | 0.4314 | 2.89 |
| 25 | 1.198e6 | 0.089 | 0.4314 | 3.84 |
| 30 | 1.438e6 | 0.069 | 0.4314 | 4.75 |

The M1 laminar-separation `x/c` is numerically invariant across all three
cases -- a genuine Thwaites/Falkner-Skan similarity result for this
geometrically self-similar family (`lambda` is independent of `Re_c` at
fixed `x/c` when `U_e/V_inf(x/c)` is held fixed), not an artifact, and is
explained explicitly in DESIGN.md Section 14. The modeled onset and `N_max`
both vary meaningfully with `Re_c`, as expected since `Re_theta` itself is
`Re_c`-dependent at fixed `x/c`.

### Reproduce Milestone 2

```bash
# Amplification-tracking study + sensitivity table
python3 scripts/en_amplification_study.py

# Regenerate the 3 Milestone 2 figures (deterministic; does not touch the M1 figures)
python3 scripts/generate_m2_figures.py
```

### Milestone 2 figures

| File | Content |
|---|---|
| `figures/stability_inputs.png` | `Re_theta`, local critical `Re_theta(H)`, `H`, and `lambda` vs. `x/c`; onset and separation marked |
| `figures/n_factor_growth.png` | `N` vs. `x/c`; onset, amplification-active region, and separation marked; "No N_crit selected" caption |
| `figures/n_factor_reynolds_sensitivity.png` | `N(x/c)` for the three `V_inf`/`Re_c` cases; no `N_crit` line drawn |

### Milestone 2 limitations

- `stability.py` does **not** solve the Orr-Sommerfeld eigenvalue problem
  and does **not** reproduce any specific published envelope-amplification
  correlation (e.g. XFOIL's) -- see the source-audit note above.
- The onset criterion and amplification-rate law are project-calibrated
  proxies, anchored at only one sourced number (the ZPG neutral point);
  their quantitative accuracy away from that anchor is not established.
- No receptivity, free-stream-turbulence, or surface-roughness modeling
  (these determine the real `N_crit` in practice and are entirely outside
  this milestone's scope).
- Stability outputs are computed only over the Milestone 1 valid,
  attached-laminar domain; nothing is modeled for, or beyond, the M1
  laminar-separation diagnostic.

## What Milestone 2 does NOT yet model

- `N_crit` selection of any kind.
- A predicted transition location.
- Turbulent skin friction, drag, or any post-transition boundary-layer
  development.
- A true Orr-Sommerfeld/PSE eigenvalue solution or a validated envelope
  amplification correlation.
- Receptivity / disturbance-environment modeling.

## Recommended Milestone 3 topic (not started)

Given a chosen, documented `N_crit` (with its own source audit -- e.g.
typical wind-tunnel `N_crit ~= 8-9`, typical free-flight/low-turbulence
`N_crit ~= 11-12`, per the e^N literature), locate the transition station
as the first `x/c` where the Milestone 2 `N(x/c)` curve reaches `N_crit`
(honestly reporting if the Milestone 2 baseline case reaches laminar
separation before any plausible `N_crit`, in which case transition would be
associated with separation rather than the smooth `N`-envelope crossing),
and begin turbulent boundary-layer / skin-friction modeling downstream of
that station.
