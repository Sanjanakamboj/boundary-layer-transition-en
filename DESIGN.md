# DESIGN.md -- Milestone 1: Laminar Boundary-Layer Foundation and Stability Inputs

This document records the coordinate/sign conventions, source audit,
derivations, numerical methods, leading-edge treatment, verification
strategy, and validity limitations for Milestone 1 of the Project 10
(Boundary-Layer Transition, e^N) sailplane wing study.

**Scope reminder:** Milestone 1 implements only an operating point, an
external velocity distribution, and laminar boundary-layer solutions
(Blasius + Thwaites). It does **not** implement e^N amplification,
N_crit selection, transition-location prediction, turbulent boundary-layer
modeling, or drag estimation. A predicted laminar-separation location is a
laminar-boundary-layer diagnostic only, not a transition prediction.

---

## 1. Coordinate and sign conventions

- `x` is the dimensional streamwise (chordwise) coordinate, m, measured
  from the modeled leading edge (`x = 0`) to the modeled trailing edge
  (`x = c`).
- `s = x / c` is the nondimensional chordwise coordinate, the primary
  independent variable used throughout `external_flow.py` and reported in
  all tables/figures, `s in [0, 1]`.
- `U_e(x)` (or `U_e(s)`) is the external (boundary-layer-edge) velocity,
  m/s, always positive over the modeled domain (no reversed flow is
  represented -- this milestone does not model separated-flow reverse
  velocities).
- `y` (implicit, not gridded in this milestone) is the wall-normal
  coordinate; boundary-layer thickness quantities (`delta_99`, `delta*`,
  `theta`) are integral properties over `y` at a fixed `x`, consistent with
  standard boundary-layer theory.
- All boundary-layer thickness quantities are non-negative by definition.
  `H = delta*/theta >= 1` (with `H` significantly above 1 as separation is
  approached).
- Positive `dU_e/dx` = favorable (accelerating) pressure gradient;
  negative `dU_e/dx` = adverse (decelerating) pressure gradient. This
  follows directly from the inviscid Bernoulli relation `dp/dx = -rho U_e
  dU_e/dx` at the boundary-layer edge (not implemented explicitly here,
  only its qualitative sign consequence is used).
- Thwaites' pressure-gradient parameter is defined as
  `lambda = (theta^2 / nu) * dU_e/dx`, i.e. `lambda > 0` favorable,
  `lambda < 0` adverse -- consistent with the sign convention above.

---

## 2. Source audit

Sources actually consulted (from training-time technical knowledge of
these standard, widely reproduced classical references; equations were
cross-checked internally for self-consistency, e.g. `BLASIUS_H` is
computed as the ratio of the two Blasius coefficients used rather than
taken as an independently rounded constant):

1. **H. Schlichting & K. Gersten, *Boundary-Layer Theory*, 9th ed.**
   Used for: the qualitative structure of the Blasius flat-plate solution
   (Chapter 7) and the general form of the laminar momentum-integral
   equation. Valid range: incompressible, steady, 2D laminar boundary
   layer.

2. **F. M. White, *Viscous Fluid Flow*, 3rd ed.**
   Used for: the specific numerical coefficients implemented here --
   Blasius `delta_99 = 5.0 x/sqrt(Re_x)`, `delta* = 1.7208 x/sqrt(Re_x)`,
   `theta = 0.664 x/sqrt(Re_x)`, `Cf,x = 0.664/sqrt(Re_x)` (Table 4.1-type
   summary of the Blasius solution); and the two-branch polynomial
   correlation for Thwaites' method, `l(lambda)` and `H(lambda)`, attributed
   in White to Cebeci & Bradshaw (Table 4.1 / Eqs. 4.75-4.76 region).
   These are the exact constants implemented in `laminar_bl.py`.

3. **B. Thwaites, "Approximate calculation of the laminar boundary layer,"
   *Aeronautical Quarterly*, 1(3), 1949, pp. 245-280.**
   Original source of the momentum-integral method and its correlation
   concept (`l(lambda)`, `H(lambda)` as universal functions of a single
   pressure-gradient parameter `lambda`). Thwaites' own tabulated data is
   *not* reproduced digit-for-digit here; the polynomial curve fit to that
   data (per White/Cebeci-Bradshaw, item 2) is used instead. This
   substitution is a **documented, deliberate project choice** (see
   "Ambiguity" below), not a claim that Thwaites' original table was
   consulted directly.

4. **T. Cebeci & P. Bradshaw, *Momentum Transfer in Boundary Layers*,
   Hemisphere/McGraw-Hill, 1977.**
   Attributed source (via White) of the specific two-branch polynomial fit
   used for `l(lambda)` and `H(lambda)`.

**Ambiguity documented:** Several textbooks quote slightly different
polynomial fits to Thwaites' original tabulated `l(lambda)`/`H(lambda)`
data (Thwaites' own approximate formulas, White's fit, Cebeci-Bradshaw's
fit, and Curle's fit all differ at the few-percent level, particularly for
`lambda < 0`). This project uses the specific two-branch fit reproduced in
White's *Viscous Fluid Flow* (attributed to Cebeci-Bradshaw), consistently,
everywhere in the code. The separation criterion is **not** hard-coded from
a secondary citation; instead `thwaites_separation_lambda()` numerically
solves `l(lambda) = 0` for the *actual implemented* correlation and returns
that root (`lambda_sep ~= -0.0898` for this implementation), which is then
compared to the commonly cited literature value of `lambda_sep ~= -0.09`
as an independent consistency check (see `test_separation_predicted_and_
lambda_root_matches_literature`).

**What is a project choice vs. sourced physics:**
- Sourced physics: the Blasius coefficients; the Thwaites momentum-integral
  formulation `theta^2(x) = 0.45 nu/U_e^6 integral_0^x U_e^5 dx'`; the
  `l(lambda)`, `H(lambda)` polynomial correlations; the ZPG limit
  `lambda -> 0`.
- Project choices: the specific generic operating point (chord, speed,
  altitude); the synthetic `U_e(x/c)` shape (smoothstep blend) and its
  numeric parameters (`s_peak=0.35`, `u_peak_ratio=1.18`, `u_te_ratio=0.55`);
  the flat-plate-like leading-edge boundary condition `theta(0)=0` with
  `U_e(0)` finite and nonzero (rather than a true stagnation-point airfoil
  leading edge); the correlation validity domain `lambda in [-0.10, 0.10]`;
  the decision to report `H`/`delta*`/`Cf` as `NaN` (not extrapolated) once
  `lambda` exits that domain or once separation has been predicted.

---

## 3. Derivations / equations implemented

### 3.1 Operating point (`operating_point.py`)

- `nu = mu / rho`
- `a = sqrt(gamma R T)` (ideal-gas speed of sound), `gamma=1.4`,
  `R=287.05 J/(kg K)`
- `M = V_inf / a`
- `Re_c = rho V_inf c / mu`
- Incompressibility check: `M < 0.3` (a conventional, project-chosen
  threshold, not a hard physical boundary) -- verified true for the
  Milestone 1 baseline (`M ~= 0.073`).

### 3.2 External velocity distribution (`external_flow.py`)

A C1-continuous, two-segment Hermite smoothstep blend between
`(s=0, U/V_inf=1)`, `(s=s_peak, U/V_inf=u_peak_ratio)`, and
`(s=1, U/V_inf=u_te_ratio)`. Smoothstep: `f(t) = 3t^2 - 2t^3`, `t in [0,1]`,
with `f'(0)=f'(1)=0`, guaranteeing the assembled `U_e(s)` is continuous and
continuously differentiable (C1) everywhere on `[0,1]`, including at the
segment boundary `s=s_peak` (both one-sided derivatives vanish there).

`dU_e/dx` is provided both in closed form (exact chain-rule derivative of
the smoothstep blend) and via a central-difference numerical estimate for
verification; the two agree to the stencil's expected `O(h^2)` truncation
error (tested to `abs=1e-3` at `h = 1e-6 c`).

**This distribution is explicitly synthetic/illustrative** -- it is not
derived from XFOIL, CFD, wind-tunnel data, or a specific airfoil geometry.
It is only loosely, qualitatively motivated by the shape of a laminar-flow
airfoil's suction-side velocity distribution (accelerate to a suction peak,
then decelerate toward the trailing edge).

### 3.3 Blasius ZPG reference (`laminar_bl.py`, part A)

```
Re_x     = U_inf x / nu
delta_99 = 5.0    x / sqrt(Re_x)
delta*   = 1.7208 x / sqrt(Re_x)
theta    = 0.664  x / sqrt(Re_x)
H        = delta*/theta  (computed self-consistently = 1.7208/0.664 ~= 2.5916;
                           commonly rounded to 2.59 in textbooks)
Cf,x     = 0.664 / sqrt(Re_x)
```

### 3.4 Thwaites pressure-gradient-aware method (`laminar_bl.py`, part B)

```
theta(x)^2 = 0.45 nu / U_e(x)^6 * integral_0^x U_e(x')^5 dx'      [theta(0)=0]
lambda(x)  = theta(x)^2 / nu * dU_e/dx
l(lambda)  = 0.22 + 1.57 lambda - 1.8 lambda^2            (0 <= lambda <= 0.1)
           = 0.22 + 1.402 lambda + 0.018 lambda/(lambda+0.107)  (-0.1 <= lambda < 0)
H(lambda)  = 2.61 - 3.75 lambda + 5.24 lambda^2            (0 <= lambda <= 0.1)
           = 2.088 + 0.0731/(lambda+0.14)                  (-0.1 <= lambda < 0)
delta*(x)  = H(lambda) theta(x)
Cf,x(x)    = 2 nu l(lambda) / (U_e(x) theta(x))
```

Separation is predicted at the first `x` (interpolated) where `l(lambda)`
crosses from positive to non-positive.

---

## 4. Numerical method

- The Thwaites integral `integral_0^x U_e^5 dx'` is evaluated by
  cumulative trapezoidal quadrature (`scipy.integrate.cumulative_trapezoid`)
  on a uniform grid in `x` (equivalently `s`), default `n_points=2001`
  (`4001` used in the scripts for tighter convergence).
- Convergence is demonstrated by refining `n_points` (251 -> 4001) and
  checking that `theta` at a fixed station and the predicted separation
  location `s_sep` both converge monotonically with shrinking
  station-to-station differences (see
  `test_grid_convergence_theta_and_separation_location`).
- `H`, `delta*`, and `Cf,x` are reported only where `lambda` is within the
  correlation's documented validity domain `[-0.10, 0.10]` **and** the
  station has not yet passed the predicted separation location. Past
  separation, `lambda` can numerically drift back into the nominal range
  as `dU_e/dx` flattens toward the trailing edge, even though the
  attached-flow correlation is no longer physically meaningful there; this
  region is explicitly masked to `NaN` rather than silently reporting a
  plausible-looking but invalid number.

---

## 5. Leading-edge treatment (x = 0)

Two distinct situations are treated differently and explicitly, per the
Milestone 1 numerical-robustness requirement:

1. **Removable singularity (delta_99, delta*, theta, Blasius and
   Thwaites):** the raw formula `x / sqrt(Re_x)` is a `0/0` indeterminate
   form at `x=0` in floating point (`Re_x=0` there), even though its
   analytic limit is exactly zero (`x/sqrt(Re_x) = sqrt(x nu/V) -> 0` as
   `x -> 0`). This module evaluates the algebraically equivalent,
   non-singular closed form `sqrt(x nu / V)` directly, so these quantities
   return exactly `0.0` at `x=0` -- the true limit, not a fabricated
   placeholder.

2. **True (non-removable) singularity (Blasius Cf,x):** `Cf,x = 0.664 /
   sqrt(Re_x)` genuinely diverges to `+infinity` as `x -> 0` (infinite wall
   shear at the idealized flat-plate leading edge). This module returns
   `inf` at `x=0`, explicitly, and this is verified by a dedicated test
   (`test_leading_edge_cf_is_explicitly_singular_not_fabricated`). It is
   never masked or replaced with a finite number.

3. **Thwaites theta(0)=0 boundary condition:** the momentum-integral
   equation is solved starting from `theta(0)=0` with `U_e(0)` finite and
   nonzero. This is standard practice for a flat-plate-like leading edge
   with a prescribed nonzero edge velocity there (as in this synthetic
   distribution), and is distinct from a true stagnation-point airfoil
   leading edge, where `U_e(0)=0` and a different (regular, non-singular
   in a different sense) starting procedure is required. Modeling a true
   stagnation-point leading edge is out of scope for Milestone 1; see
   Section 7.

4. **Thwaites Cf,x at x=0:** `theta(0)=0` makes `Cf,x = 2 nu l(lambda)/
   (U_e theta)` a `.../0` form at the leading edge. This is reported as
   `NaN` (formally undefined, `0/0` in the physical limit since `l(0)` is
   also effectively 0 there), not a fabricated finite value.

---

## 6. Verification strategy

Independent verification (not merely re-calling the production function) is
implemented for every item required by the milestone:

- `Re_c`, `Re_x`: hand-computed via two independent arithmetic paths
  (`rho V c / mu` vs. `V c / nu` with `nu` computed separately).
- Blasius `delta_99`, `delta*`, `theta`, `Cf,x`: hand-computed with
  `math.sqrt`/plain arithmetic against the closed-form coefficients,
  independent of the vectorized numpy implementation.
- `H`: verified as the exact ratio `delta*/theta` and compared to the
  commonly cited literature value 2.59 (within ~0.06%).
- External-flow endpoints/reference stations: exact values at `s=0`,
  `s=s_peak`, `s=1` checked against the distribution's defining control
  points.
- `dU_e/dx`: closed-form vs. central-difference numerical derivative,
  cross-checked to `O(h^2)` tolerance; plus hand-derived zero-derivative
  points at the leading edge and at the peak (both are stationary points of
  the smoothstep construction, verifiable by inspection of the formula).
- Dimensional/nondimensional consistency: `U_e(x) == V_inf * ratio(x/c)`
  and `dU_e/dx == (V_inf/c) * d(ratio)/ds` checked at 51 stations.
- Thwaites integral evaluation: (a) a hand-rolled `numpy.trapezoid`
  quadrature on an independently sampled grid, compared against the
  production solver's `scipy.integrate.cumulative_trapezoid`-based result
  at an interior station; (b) a from-scratch manual cumulative
  trapezoidal-sum reimplementation compared against `scipy`'s function on
  the same grid.
- Zero-pressure-gradient recovery: `l(0)=0.22`, `H(0)=2.61` checked exactly
  against the correlation's own definition; the *implied* ZPG skin-friction
  coefficient derived by hand from `l(0)` and the Blasius `theta` scaling
  (`0.6627`) is checked against the Blasius `Cf` coefficient (`0.664`, within
  ~0.2%); `H(0)` is checked against `BLASIUS_H` (within ~1%). At the
  near-leading-edge station of the actual synthetic distribution (weak but
  nonzero `lambda`), the study script reports the resulting theta deviation
  from the Blasius reference (~1%), documented as expected given the small
  residual pressure gradient there.
- Scalar/vector consistency: every public function is checked with a scalar
  argument against the corresponding element of a vectorized call, across
  `operating_point`, `external_flow`, and `laminar_bl`.
- Invalid-input rejection: nonpositive/nonfinite `chord`, `V_inf`, `rho`,
  `mu`, `nu`, `T`; negative `x`; out-of-range `s`; invalid distribution
  shape parameters (`s_peak` outside `(0,1)`, `u_peak_ratio<=1`,
  `u_te_ratio` outside `(0, u_peak_ratio)`); `n_points < 3`.
- Grid convergence: `theta` at a fixed station and the predicted separation
  location both converge (differences shrink monotonically, and the
  finest-grid-pair relative change is `< 1e-4` for `theta` and
  `< 1e-3` chord-fraction for `s_sep`) as `n_points` is refined from 251 to
  4001.

---

## 7. Validity limitations

- Incompressible flow only; valid because `M ~= 0.073 << 0.3` for the
  chosen operating point. Not re-derived or re-checked for other operating
  points a user might substitute.
- 2D (no sweep, no spanwise flow) laminar boundary layer only.
- No surface roughness, free-stream turbulence intensity, or leading-edge
  contamination effects.
- The external velocity distribution is synthetic/illustrative, not tied to
  any real airfoil's actual pressure distribution; quantitative results
  (e.g. the specific separation location `x/c ~= 0.43`) are illustrative
  of the *method*, not a prediction for any real wing.
- The leading edge is modeled as flat-plate-like (`U_e(0)` finite and
  nonzero, `theta(0)=0`), not as a true stagnation point. A real airfoil
  leading edge requires a stagnation-point starting solution for Thwaites'
  method (often handled via a small-`x` series expansion); this is
  explicitly not implemented in Milestone 1.
- Thwaites' method is itself an approximate, single-parameter correlation
  method; it is known from the literature to carry a few-percent-level
  systematic error (visible here in the small ZPG-limit mismatches
  documented above) and to become unreliable near and beyond its
  documented `lambda` validity domain `[-0.10, 0.10]` -- which is why this
  implementation refuses to report `H`/`delta*`/`Cf,x` outside that domain
  or beyond the predicted separation station, rather than extrapolating.
- A Thwaites-predicted laminar separation is **not** a transition
  prediction. Real laminar boundary layers commonly transition upstream of
  where they would otherwise laminar-separate, and separated laminar shear
  layers can reattach turbulently; neither phenomenon is modeled here.
  e^N amplification-factor tracking against a chosen `N_crit` (Milestone 2+)
  is required before any transition-location claim can be made.
