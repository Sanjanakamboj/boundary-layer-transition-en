# Boundary-Layer Transition (e^N) -- Project 10

**Milestone 1: Laminar Boundary-Layer Foundation and Stability Inputs**
**Milestone 2: Reduced-Order Linear-Stability Proxy and e^N Amplification Tracking**
**Milestone 3: N_crit Transition Criterion, Turbulent Skin Friction, and Drag Impact**
**Milestone 4: Pressure-Gradient Turbulent Boundary Layer and Transition-to-TE Drag**

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

---

## Milestone 3: N_crit transition criterion, turbulent skin friction, and drag impact

### Objective

Extend Milestone 2 into an actual (sensitivity-range) transition-location
estimate and its first-order skin-friction drag consequence: select a
defensible `N_crit` sensitivity range from primary literature, locate
where (if anywhere) the Milestone 2 `N(x)` curve reaches it, introduce a
separate "assumed transition at separation" bookkeeping scenario for cases
where no crossing exists, add sourced laminar/turbulent flat-plate
skin-friction correlations, and estimate the resulting section
skin-friction drag coefficient.

### What N_crit means

`N_crit` is **not a universal material constant** -- it is an
environment/receptivity-dependent threshold: the amplification level at
which small disturbances typically become large enough, given the actual
disturbance environment (free-stream turbulence, acoustics, surface
roughness), to trigger transition. A lower `N_crit` corresponds to a
noisier environment; a higher `N_crit` to a quieter one. This project does
not model receptivity explicitly and does not invent a turbulence-
intensity-to-`N_crit` mapping.

### Sourced N_crit sensitivity range

Fetched and quoted verbatim from Mark Drela's primary XFOIL documentation
(`web.mit.edu/drela/Public/web/xfoil/xfoil_doc.txt`) in this session:

| Situation | `N_crit` |
|---|---|
| Sailplane | 12-14 |
| Motorglider | 11-13 |
| Clean wind tunnel | 10-12 |
| Average wind tunnel | 9 (standard "e^9 method") |
| Dirty wind tunnel | 4-8 |

This project uses **`N_crit` in {9, 12, 14}** as its sensitivity range:
9 (the historical "e^9" average-wind-tunnel reference) and 12/14 (the
documented sailplane range -- directly relevant to this project's vehicle
type). This is a sensitivity range, not a selected or recommended design
value.

### Exact transition-event/status logic

`transition.locate_transition_n_crit(asol, n_crit)` returns exactly one of:

- **`N_CRIT_CROSSING`**: `N(x)` reaches `N_crit` within the Milestone 1
  valid (attached-laminar) domain; the crossing `x_tr/c` is linearly
  interpolated between grid stations.
- **`SEPARATION_BEFORE_N_CRIT`**: the M1 laminar-separation diagnostic
  occurs first.
- **`NO_EVENT_IN_DOMAIN`**: neither occurs (only arises for hypothetical
  cases with no predicted separation and insufficient amplification).

Separately, `transition.separation_triggered_transition(asol)` builds an
explicitly labeled **"separation-triggered" bookkeeping scenario**
(`x_tr = x_sep`) -- used for drag estimation only when no N_crit crossing
exists. **This is never an e^N result and is never presented as one.**

### Baseline outcome

At the Milestone 1/2 baseline operating point (`Re_c ~= 1.198e6`),
`N_max ~= 3.84`, well below every sourced `N_crit` value. The correct,
honestly reported result for all three (`N_crit = 9, 12, 14`) is:

**`SEPARATION_BEFORE_N_CRIT`** -- the modeled laminar boundary layer
reaches its Thwaites separation diagnostic (`x/c ~= 0.4314`) before enough
amplification has accumulated to cross any plausible `N_crit`. No
crossing is invented to force a transition-location result.

### Reynolds-number sensitivity

The existing physical sensitivity space (`V_inf`, hence `Re_c`) was
searched -- **not the M2 amplification-rate proxy constants** -- for a
naturally occurring `N_crit` crossing, extended up to `V_inf = 90` m/s
(`M ~= 0.26`, still comfortably within the M1 incompressible assumption,
`M < 0.3`):

| `V_inf` [m/s] | `Re_c` | `N_max` | `N_crit=9` | `N_crit=12` | `N_crit=14` |
|---|---|---|---|---|---|
| 20 | 9.58e5 | 2.89 | separation first | separation first | separation first |
| 25 | 1.198e6 | 3.84 | separation first | separation first | separation first |
| 30 | 1.438e6 | 4.75 | separation first | separation first | separation first |
| 40 | 1.917e6 | 6.40 | separation first | separation first | separation first |
| **60** | 2.875e6 | 9.25 | **crossing at x/c=0.428** | separation first | separation first |
| **90** | 4.313e6 | 12.78 | **crossing at x/c=0.378** | **crossing at x/c=0.423** | separation first |

Two genuine `N_crit` crossings were found this way (`V_inf = 60` m/s
crosses `N_crit=9`; `V_inf = 90` m/s crosses both `N_crit=9` and `12`),
without altering any Milestone 2 amplification-model constant.

### Separation-triggered assumption

Because the transition location cannot be determined by e^N crossing for
the baseline case, Milestone 3's drag bookkeeping falls back to the
explicitly labeled "assumed immediate transition at the M1
laminar-separation diagnostic" scenario (`x_tr/c = x_sep/c ~= 0.4314`).
This is a modeling convenience -- not a claim that separation equals
transition physically.

### Skin-friction correlations

- Laminar, local (reused from Milestone 1): `Cf,x = 0.664 / sqrt(Re_x)`
  (Blasius).
- Laminar, average: `Cf_bar = 1.328 / sqrt(Re_L)`.
- Turbulent, local: `Cf,x = 0.0592 * Re_x^(-1/5)` (smooth-wall,
  incompressible, zero-pressure-gradient 1/7-power-law correlation;
  White, *Viscous Fluid Flow*; validity `5e5 <= Re_x <= 1e7`).
- Turbulent, average: `Cf_bar = 0.074 * Re_L^(-1/5)`.

The turbulent correlation is evaluated at the physical `x` from the real
leading edge (not a virtual-origin correction) -- an explicitly labeled
simplification; see DESIGN.md for why the more accurate mixed-boundary-
layer correction was not implemented.

### C_d,f convention

```
C_d,f = 2 * integral_0^1 Cf(x/c) * [U_e(x/c)/V_inf]^2 d(x/c)
```

Reference length = chord; both surfaces represented by the same M1
synthetic `U_e(x/c)` shape (an explicit simplification, since the M1
model is not upper/lower-surface-resolved); `Cf` from freestream-Reynolds-
number flat-plate correlations; dynamic-pressure weighting from the actual
M1 pressure-gradient distribution. Wetted-length/surface-slope corrections
are neglected.

### Baseline drag-impact results

| Case | `C_d,f` |
|---|---|
| Fully laminar reference | 0.002451 |
| Fully turbulent reference | 0.008730 |
| Separation-triggered (`x_tr/c = 0.4314`) | 0.005044 |

Separation-triggered drag is **+105.8%** relative to fully laminar and
**-42.2%** relative to fully turbulent -- i.e. it falls, as physically
expected, strictly between the two references.

### Prescribed transition-location sensitivity

| `x_tr/c` | `C_d,f` | % vs. laminar | % vs. turbulent |
|---|---|---|---|
| 0.1 | 0.008048 | +228.3% | -7.8% |
| 0.3 | 0.006273 | +155.9% | -28.1% |
| 0.5 | 0.004464 | +82.1% | -48.9% |
| 0.7 | 0.003255 | +32.8% | -62.7% |
| 0.9 | 0.002643 | +7.8% | -69.7% |

Later transition consistently and substantially reduces drag toward the
laminar reference -- directly demonstrating why maintaining laminar flow
farther aft matters. The lowest-drag case is **not** called an "optimum"
(no structural, stall, or off-design constraints are modeled).

### Reproduce Milestone 3

```bash
python3 scripts/transition_drag_study.py
python3 scripts/generate_m3_figures.py
```

### Milestone 3 figures

| File | Content |
|---|---|
| `figures/ncrit_transition_sensitivity.png` | `N(x/c)` vs. the sourced `N_crit` sensitivity lines; baseline shows no crossing |
| `figures/transition_reynolds_map.png` | Transition-event outcome (crossing / separation-first / no-event) vs. `V_inf`/`Re_c` and `N_crit` |
| `figures/skin_friction_distributions.png` | `Cf(x/c)` for laminar/turbulent/separation-triggered/prescribed cases |
| `figures/transition_drag_impact.png` | `C_d,f` vs. prescribed `x_tr/c`, with laminar/turbulent references and the separation-triggered case marked |

### Limitations

- `N_crit` is a sensitivity range, not a selected design value; no
  transition location is asserted as "the" transition point for this
  project.
- The skin-friction drag model uses zero-pressure-gradient flat-plate
  correlations as a bookkeeping approximation over the M1 pressure-
  gradient velocity distribution -- not a full turbulent pressure-gradient
  boundary-layer solution.
- The turbulent correlation's physical-x evaluation neglects the
  virtual-origin/mixed-boundary-layer history effect (documented, not
  implemented).
- The separation-triggered scenario assumes *immediate* transition at
  separation; no separation-bubble reattachment model is implemented.
- No form/pressure drag, viscous-inviscid interaction, or 3D effects are
  modeled -- only 2D section skin friction.
- Not validated against wind-tunnel or flight-test data.

## What Milestone 3 does NOT model

- A single, selected, "correct" `N_crit` for this project.
- A turbulent pressure-gradient integral boundary-layer solution.
- Separation-bubble transition/reattachment physics.
- Form drag, pressure drag, or 3D/finite-span effects.
- Any claim of experimental validation.

---

## Milestone 4: pressure-gradient turbulent boundary layer and transition-to-TE drag

### Why beyond flat-plate bookkeeping

Milestone 3's turbulent skin friction used a zero-pressure-gradient flat-
plate correlation evaluated at the physical distance from the leading
edge -- convenient, but blind to (a) the actual momentum thickness the
flow has already accumulated by the assumed transition station and (b)
the real pressure gradient acting on the turbulent layer downstream.
Milestone 4 replaces that with an actual two-equation turbulent integral
boundary-layer method (Head's entrainment method) propagated along the
real Milestone 1 `U_e(x)`, connecting the full pipeline: M1 external flow
+ laminar Thwaites -> M2 e^N amplification -> M3 transition-event logic ->
M4 turbulent propagation -> improved skin-friction drag estimate.

### Chosen method and sources

**Head's entrainment method** (Head, M. R., 1958, ARC R&M 3152), using the
Green-Weeks-Brooman (RAE TR 72231, 1973) curve fit for the entrainment
closure and `H1(H)` correlation, and the Ludwieg-Tillmann (1950, NACA TM
1285) skin-friction correlation. The exact equation set and constants were
fetched and verified from a peer-reviewed source (Cambridge Core, *Flow*
journal) in this session; a small correction (a missing `+3.3` offset on
one branch of a secondary source's quoted `H1(H)` formula) was identified
and applied via a direct continuity cross-check -- see DESIGN.md Section
24 for the full audit.

```
Momentum integral:  dtheta/dx + (2+H)(theta/Ue)(dUe/dx) = Cf/2
Entrainment:         (1/Ue) d(Ue theta H1)/dx = 0.0299 (H1-3.0)^-0.6169
H1(H):                0.8234(H-1.1)^-1.287 + 3.3   (H<=1.6)
                      1.55(H-0.6778)^-3.064 + 3.3   (H>1.6)
Skin friction:       Cf = 0.246 * 10^(-0.678 H) * Re_theta^-0.268
```

### State variables

`theta(x)`, entrainment shape factor `H1(x)` (the two integrated ODE
states), with `H(x)` (via numerical inversion of `H1(H)`),
`delta*(x) = H*theta`, and `Cf(x)` (Ludwieg-Tillmann) derived at each
station.

### Transition initialization

`theta_tr` is taken by continuous interpolation of the M1 laminar
`theta(x)` at the assumed transition station (rejected if that station
lies beyond the M1 laminar-separation diagnostic -- the M1 laminar state
is never used where M1 itself considers it invalid). The initial shape
factor uses `H_tr = 72/56 ~= 1.286`, the exact 1/7-power-law turbulent
profile value (the same idealized profile underlying Milestone 3's
turbulent flat-plate correlations) -- a documented nominal choice, with
its sensitivity explicitly quantified (see below), not fit to produce any
particular drag result.

### Turbulent separation criterion

The `H1(H)` correlation's own asymptote (`H1 -> 3.3` as `H -> infinity`)
is used directly: this project declares turbulent separation when the
integrated `H1` state crosses down through `3.32` (a small, documented
numerical margin above that asymptote), via a terminal ODE event -- not
inferred from "Cf got small."

### Baseline result

Propagating the turbulent boundary layer from the Milestone 3
separation-triggered station (`x_tr/c = 0.4314`, `theta_tr = 0.240 mm`,
`H_tr = 1.286`):

- **The turbulent boundary layer itself separates** at `x/c ~= 0.785`
  (`H` has risen to `~4.4` there) -- **it does not reach the trailing
  edge**. This is an honest, unforced finding: this project's synthetic
  adverse-gradient `U_e(x/c)` (a 53% deceleration from its peak to the
  trailing edge) is severe enough that essentially every transition
  scenario examined leads to turbulent separation before the trailing
  edge, regardless of where transition is assumed to occur.

### M3 vs. M4 skin-friction drag comparison

| Scenario | `x_tr/c` | M3 `C_d,f` | M4 `C_d,f` | % difference | M4 status |
|---|---|---|---|---|---|
| Fully turbulent | 0.00 | 0.008730 | 0.006611 | -24.3% | turbulent separation |
| Early prescribed | 0.10 | 0.008048 | 0.006441 | -20.0% | turbulent separation |
| Late prescribed | 0.40 | 0.005330 | 0.004350 | -18.4% | turbulent separation |
| Separation-triggered | 0.4314 | 0.005044 | 0.004144 | -17.8% | turbulent separation |
| N_crit=9 crossing (V=60 m/s) | 0.4282 | 0.003904 | 0.003265 | -16.4% | turbulent separation |

**M4 consistently gives *less* drag than M3 in every scenario examined**
(16-24% lower). This is reported and explained, not assumed: M3's
flat-plate correlation, referenced to the freestream velocity and the
physical distance from the leading edge, has no memory of the actual
(already-thickened) momentum deficit at transition and always predicts a
"fresh" turbulent layer; M4's integral solution correctly inherits that
thickened state and additionally reflects the real, sourced collapse of
`Cf` toward zero as the turbulent layer itself approaches separation. This
direction is not asserted as a general rule -- a milder pressure gradient
or an earlier, longer attached turbulent run could reverse it.

### Reynolds sensitivity

Every `V_inf` case examined (20-90 m/s, `Re_c` from 9.6e5 to 4.3e6) also
reaches turbulent separation from the separation-triggered station, at a
gently increasing `x/c` (~0.78 to ~0.83) as `Re_c` rises; `C_d,f` decreases
monotonically with `Re_c` (0.00443 at 20 m/s to 0.00293 at 90 m/s),
consistent with the usual Reynolds-number scaling of skin friction.

### Initialization (`H_tr`) sensitivity

| `H_tr` | Status | `C_d,f` |
|---|---|---|
| 1.20 | turbulent separation | 0.004705 |
| 1.286 (nominal) | turbulent separation | 0.004144 |
| 1.40 | turbulent separation | 0.003862 |
| 1.60 | turbulent separation | 0.003687 |

A modest (~1.28x) but non-negligible sensitivity -- quantified explicitly,
not hidden, and consistent with the finding (Section 27, DESIGN.md) that
Head's system relaxes toward a common downstream equilibrium largely
independent of the exact starting `H_tr`, over a longer development length
than this project's short chord provides.

### Reproduce Milestone 4

```bash
python3 scripts/turbulent_boundary_layer_study.py
python3 scripts/generate_m4_figures.py
```

### Milestone 4 figures

| File | Content |
|---|---|
| `figures/turbulent_boundary_layer_history.png` | `theta/c`, `H`, `Cf` vs. `x/c` for the separation-triggered case, with transition and turbulent separation marked |
| `figures/m3_vs_m4_skin_friction.png` | Local `Cf(x)` comparing M3 flat-plate vs. M4 pressure-gradient turbulent treatment (laminar portions coincide) |
| `figures/m3_vs_m4_drag_impact.png` | `C_d,f` vs. assumed `x_tr/c` for both models, with laminar/turbulent references and the separation-triggered case marked |
| `figures/turbulent_reynolds_sensitivity.png` | `C_d,f` and turbulent-separation `x/c` vs. `Re_c` |

### Limitations

- Head's method is a classical reduced-order integral method, not a
  field/RANS/CFD solution, and is not validated against experimental data
  in this project.
- Beyond a Milestone-4-predicted turbulent separation, `Cf` is held
  constant (frozen) as a bookkeeping convention -- this is not a resolved
  separated-flow solution and excludes any associated pressure-drag rise.
- The "instantaneous laminar-to-turbulent restart" transition
  initialization does not resolve an actual laminar-separation bubble.
- No compressibility, 3D, or form/pressure-drag effects are modeled.
- The turbulent-origin treatment for the "fully turbulent from the
  leading edge" seed uses a self-derived (not independently sourced)
  flat-plate `theta` scaling consistent with Milestone 3's own turbulent
  correlation.

## What Milestone 4 does NOT model

- A resolved separated-flow (post-turbulent-separation) solution.
- A laminar-separation-bubble transition/reattachment model.
- Compressible, 3D, or form/pressure-drag effects.
- Any claim of CFD or experimental validation.

## Recommended Milestone 5 topic (not started)

Milestone 4 found that this project's synthetic adverse gradient drives
the turbulent boundary layer to separation before the trailing edge in
every scenario examined. A natural next step is a sourced, reduced-order
treatment of the post-separation region -- e.g. a documented pressure-
drag/form-drag estimate for the separated region (rather than the current
frozen-`Cf` bookkeeping placeholder), and/or a simple, sourced
laminar-separation-bubble transition model to replace the "instantaneous
restart" assumption -- to move from a skin-friction-only `C_d,f` toward a
more complete section drag polar, with any new closure constants audited
the same way `N_crit` and Head's-method constants were audited here.
