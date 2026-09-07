# Boundary-Layer Transition (e^N) — Reduced-Order Sailplane Section Study

A transparent, source-audited, reduced-order pipeline connecting external
flow, laminar boundary-layer development, e^N amplification, N_crit
transition-event logic, a pressure-gradient turbulent boundary layer, a
laminar-separation-bubble closure, and section skin-friction/separated-
drag bookkeeping — for one generic, illustrative sailplane wing section.

**This is a portfolio/technical-interview engineering study, not a design
tool.** It does not use CFD, XFOIL, or real-airfoil data; the external
velocity distribution is a synthetic, explicitly illustrative shape; the
e^N amplification model is a reduced-order proxy, not an Orr-Sommerfeld/
PSE solution; the separated-flow drag term is a parametric bookkeeping
closure, not a validated pressure/form-drag correlation; and no result
here is, or substitutes for, a CFD/RANS solution, an experimentally
validated transition or drag prediction, a stall angle, a "safe"
operating envelope, or a complete/certified airfoil drag polar.

## Engineering question

Starting from a representative sailplane wing-section operating point,
how far does laminar flow persist, when (if ever) does amplification
reach a plausible transition threshold, what happens to the boundary
layer through separation and turbulent reattachment, and what does that
imply for section skin-friction drag — and **how much of that story is a
consequence of the specific illustrative pressure distribution chosen,
versus more general Reynolds-number/boundary-layer behavior?**

## Representative operating point

Generic, non-manufacturer-matched mid-span wing section: chord
`c = 0.70 m`, `V_inf = 25.0 m/s`, ISA sea level (`rho = 1.225 kg/m^3`,
`mu = 1.7894e-5 Pa.s`), giving `Re_c = 1.198e6` and `Mach = 0.0735`
(comfortably incompressible, `M < 0.3`). The synthetic external velocity
distribution accelerates from `U_e/V_inf = 1.0` at the leading edge to a
peak of `1.18` at `x/c = 0.35`, then decelerates to `0.55` at the
trailing edge — loosely motivated by the qualitative shape of a
laminar-flow airfoil's suction-side distribution, but **not derived from
any real airfoil, CFD, or wind-tunnel data**.

## Main findings

1. **Laminar separation precedes any plausible transition threshold.**
   Thwaites' method predicts laminar separation at `x/c = 0.4314`; the
   e^N amplification proxy reaches only `N_max = 3.84` there, below every
   sourced `N_crit` sensitivity value (`{9, 12, 14}`). Status:
   `SEPARATION_BEFORE_N_CRIT` — not an invented crossing.
2. **A restarted turbulent boundary layer separates again.** Whether
   restarted instantaneously at the laminar-separation station or via the
   Milestone 5 bubble-reattachment closure, the turbulent layer
   consistently re-separates under the continued adverse gradient
   (`x/c ~= 0.67-0.79` depending on the restart assumption) — it never
   reaches the trailing edge in the baseline family.
3. **Separated-flow drag dominates the modeled total.** For the nominal
   bubble, `C_d,sep = 0.019936` vs. `C_d,f = 0.002495` — about 89% of
   `C_d,total = 0.022430`.
4. **These two qualitative findings are robust across a 7-case synthetic
   pressure-distribution sensitivity family** (separation-before-N_crit
   and downstream re-separation both hold in 7/7 cases), but the
   *quantitative* separation location, `N_max`, and drag are clearly
   profile-sensitive, and the **unsourced separated-drag coefficient
   `K_sep` is the single most influential modeled assumption in the
   entire project** (up to 267% swing in `C_d,total`) — more influential
   than any external-flow shape parameter tested.

See [RESULTS.md](RESULTS.md) for the full quantitative summary and
[VERIFICATION.md](VERIFICATION.md) for the independent verification
record.

## Model architecture

### External flow

A C1-continuous Hermite-smoothstep blend, `U_e(x/c)`, parameterized by a
peak velocity ratio, peak location, and trailing-edge ratio
(`src/boundary_layer_transition/external_flow.py`). Exact analytic
derivative provided and cross-checked against finite differences.

### Laminar boundary layer

Zero-pressure-gradient Blasius reference plus Thwaites' pressure-
gradient-aware momentum-integral method (`laminar_bl.py`), sourced from
Schlichting & Gersten and White's *Viscous Fluid Flow* (the Thwaites
`l(lambda)`/`H(lambda)` correlation is the White/Cebeci-Bradshaw curve fit
to Thwaites' original 1949 data, not a direct digitization of it — see
DESIGN.md for the full audit).

### e^N amplification

A reduced-order linear-stability **proxy** (`stability.py`) — not a
solved Orr-Sommerfeld eigenvalue problem — anchored at the one sourced
number available (the classical Blasius Tollmien-Schlichting neutral
point, `Re_theta,crit,ZPG = 200`). The onset criterion and amplification-
rate law beyond that anchor are explicitly labeled project assumptions.

### N_crit transition logic

`transition.py` locates (by linear interpolation) where the amplification
history first reaches an externally supplied `N_crit`, restricted to the
region where the M1 laminar solution remains valid. `N_crit` itself is
sourced from Drela's primary XFOIL documentation (fetched and quoted
verbatim in this session): sailplane `12-14`, average wind tunnel `9`.
**Never selected as a single "correct" design value.**

### Turbulent boundary layer

Head's (1958) entrainment method with the Green-Weeks-Brooman closure and
Ludwieg-Tillmann (1950) skin friction (`turbulent_bl.py`), propagated
along the actual M1 `U_e(x)` from an assumed transition/restart station
to the trailing edge or turbulent separation (detected via the closure's
own `H1 -> 3.3` asymptote, not "Cf got small").

### Separation bubble

A transparent parametric sensitivity closure (`separation_bubble.py`) for
the laminar-separation -> separated-shear-layer-transition -> turbulent-
reattachment (or open-separation) sequence. Horton (1969) and Gaster
(1967) were checked directly (web search) but yielded no compact,
independently verifiable correlation usable with this project's state
variables — Gaster's own criterion is documented in the literature as not
generally valid — so short/nominal/long bubble and open-separation
scenarios are evaluated instead, every constant explicitly labeled as a
project assumption.

### Drag bookkeeping

`skin_friction.py` (M3 flat-plate bookkeeping and the shared section-drag
integral) and `separated_drag.py` (the M5 separated pressure/form-drag
closure, `C_d,sep = K_sep L_sep^p (U_e(x_sep)/V_inf)^2`, `K_sep` and `p`
unsourced). `C_d,total = C_d,f + C_d,sep` is enforced as an exact identity
in code.

## External-flow sensitivity

`external_flow_sensitivity.py` and `robustness.py` construct a fixed,
7-case deterministic family (peak ratio `1.10/1.18/1.26`; peak location
`0.25/0.35/0.45`; TE ratio `0.45/0.55/0.70`, one-factor-at-a-time,
defined *before* any downstream result was inspected) and re-propagate
each through the **unmodified** M1-M5 chain. Result: `x_sep,lam/c` in
`[0.368, 0.503]` (most sensitive to **peak location**, not the TE-ratio/
adverse-gradient severity one might have assumed); `N_max` in `[3.24,
4.83]`; `C_d,total` in `[0.0198, 0.0253]` (most M6-sensitive to peak
ratio). See RESULTS.md for the full table and the computed (not
hard-coded) cross-model sensitivity ranking.

## Verification

- **276 tests**, `pytest -W error -q`: all passing, zero warnings.
- **32 independent checks** (`scripts/independent_audit.py`): hand
  arithmetic, alternate numerical paths, and reconstructed ODE/finite-
  difference residuals — never the same production function called twice.
  Maximum residual: `1.9e-5` (a grid-dependent quantity, explicitly not
  held to machine-precision); all algebraic identities: `<=1e-8`.
- Full detail in [VERIFICATION.md](VERIFICATION.md).

## Featured figures

| File | Content |
|---|---|
| `figures/final_boundary_layer_summary.png` | Flagship 4-panel summary: external flow, laminar/instability/separation timeline, bubble/reattachment timeline, drag decomposition |
| `figures/final_external_flow_sensitivity.png` | Mild/baseline/severe adverse-gradient `U_e(x/c)` and their laminar-separation/`N_max` consequences |
| `figures/final_transition_mechanism_map.png` | Categorical mechanism outcome across the external-flow x Reynolds sensitivity grid |
| `figures/final_drag_sensitivity.png` | `C_d,total` (friction + separated) across the external-flow family |
| `figures/final_model_sensitivity_ranking.png` | Deterministic one-factor sensitivity tornado chart |

24 figures total across all milestones; see `figures/` for the complete
set (M1 laminar foundation, M2 amplification, M3 N_crit/flat-plate drag,
M4 pressure-gradient turbulent BL, M5 bubble/separated drag, M6 final
synthesis).

## Reproducing the analysis

```bash
python3 -m pip install -e ".[dev]"

# Full test suite (zero warnings expected)
python3 -m pytest -W error -q
ruff check .

# Study scripts, in pipeline order
python3 scripts/manual_check.py
python3 scripts/laminar_boundary_layer_study.py
python3 scripts/en_amplification_study.py
python3 scripts/transition_drag_study.py
python3 scripts/turbulent_boundary_layer_study.py
python3 scripts/separation_bubble_drag_study.py
python3 scripts/final_robustness_study.py
python3 scripts/independent_audit.py

# Regenerate all figures (deterministic; figures/*.png)
python3 scripts/generate_figures.py
python3 scripts/generate_m2_figures.py
python3 scripts/generate_m3_figures.py
python3 scripts/generate_m4_figures.py
python3 scripts/generate_m5_figures.py
python3 scripts/generate_m6_figures.py
```

## Limitations

- Generic, non-manufacturer-matched section; synthetic `U_e(x/c)`, not
  CFD/XFOIL/experimental data.
- Reduced-order e^N proxy, not an Orr-Sommerfeld/PSE solution.
- `N_crit` is a sourced *sensitivity range*, never a selected design value.
- Head's integral method is a classical reduced-order closure, not
  CFD/RANS.
- The laminar-separation-bubble closure and the separated-drag closure
  are both explicitly labeled parametric sensitivity models; `K_sep`
  dominates the uncertainty in the final drag numbers.
- No roughness, receptivity, compressibility correction, or 3D effect is
  modeled.
- Not a complete or certified airfoil drag polar; not validated against
  CFD, XFOIL, or experimental data.

## Development history

Six milestones, each building additively on a frozen, independently
verified baseline (full detail in `DESIGN.md`):

1. **Laminar foundation** — operating point, synthetic `U_e(x/c)`, Blasius
   + Thwaites laminar boundary layer.
2. **e^N amplification** — reduced-order linear-stability proxy and
   N-factor tracking.
3. **N_crit transition + flat-plate drag** — sourced `N_crit` sensitivity
   range, transition-event logic, M3 skin-friction bookkeeping.
4. **Pressure-gradient turbulent boundary layer** — Head's entrainment
   method replacing the M3 flat-plate turbulent assumption downstream of
   transition.
5. **Separation bubble + post-separation drag** — a parametric
   laminar-separation-bubble closure and a distinct separated pressure/
   form-drag term.
6. **Final robustness audit and portfolio synthesis** (this milestone) —
   external-flow shape sensitivity, a full independent end-to-end audit,
   final portfolio figures, and this documentation set. Development
   stops here.
