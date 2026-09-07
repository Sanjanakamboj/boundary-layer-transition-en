# Results

A concise, quantitative summary of Project 10 (Boundary-Layer Transition,
e^N): a reduced-order, source-audited laminar-to-turbulent transition and
skin-friction/separated-drag study for a generic sailplane wing section.

## Engineering objective

Build a transparent, independently verified, reduced-order pipeline
connecting external flow -> laminar boundary layer -> e^N amplification
-> transition-event logic -> turbulent boundary layer -> laminar-
separation-bubble closure -> section drag bookkeeping, for one
illustrative operating point, and quantify how much of the resulting
story depends on the specific synthetic pressure distribution chosen.

## Baseline operating point

Generic (non-manufacturer-matched) sailplane wing section: chord
`c = 0.70 m`, `V_inf = 25.0 m/s`, ISA sea level (`rho = 1.225 kg/m^3`,
`mu = 1.7894e-5 Pa.s`), giving `Re_c = 1.198e6` and `Mach = 0.0735`
(comfortably incompressible). The external velocity distribution is a
synthetic, illustrative `U_e(x/c)`: accelerates from `U_e/V_inf = 1.0` at
the leading edge to a peak of `1.18` at `x/c = 0.35`, then decelerates to
`0.55` at the trailing edge. This is **not** derived from CFD, XFOIL, or
any real airfoil.

## Laminar separation result

Thwaites' pressure-gradient-aware laminar integral method predicts
laminar separation at **`x/c = 0.4314`** (`lambda_sep ~= -0.0898`, matching
the literature reference `~-0.09`).

## e^N amplification result

A reduced-order linear-stability proxy (anchored at the classical Blasius
Tollmien-Schlichting neutral point, `Re_theta,crit,ZPG = 200`) predicts a
modeled instability onset at **`x/c = 0.0885`**, accumulating to
**`N_max = 3.843`** by the laminar-separation station. This is well below
every sourced `N_crit` sensitivity value.

## N_crit transition interpretation

Sourced from Mark Drela's primary XFOIL documentation (fetched and quoted
verbatim): `N_crit` sensitivity range **`{9, 12, 14}`** (9 = "average wind
tunnel" / the classical e^9 method; 12/14 = the documented "sailplane"
range). For the baseline case, **laminar separation occurs before N
reaches any of these values** -- the correct, honestly reported status is
`SEPARATION_BEFORE_N_CRIT` for all three, not an invented crossing.
Extending the Reynolds sensitivity to `V_inf = 60` and `90` m/s (both
`Mach < 0.3`) finds **genuine** attached-flow `N_crit=9` crossings at
`x/c = 0.428` and `0.378` respectively -- found by searching the existing
physical sensitivity space, never by retuning the amplification model.

## Turbulent boundary-layer result

Head's (1958) entrainment method (Green-Weeks-Brooman closure,
Ludwieg-Tillmann skin friction) propagates the turbulent boundary layer
from an assumed transition station. Restarted directly at the M1 laminar-
separation station (`x_tr/c=0.4314`, `H_tr=1.286`), the turbulent layer
itself separates again at **`x/c = 0.785`** -- it does not reach the
trailing edge. This finding (a downstream turbulent re-separation) proved
robust across every profile in the Milestone 6 external-flow sensitivity
family (7/7 cases) and across V_inf up to 90 m/s wherever the mechanism
stayed separation-induced.

## Separation-bubble / drag result

A transparent, source-audited (Horton 1969; Gaster 1967, both confirmed
not to yield a compact verifiable correlation) parametric bubble closure
replaces the instantaneous-restart assumption. For the **nominal bubble**
(`dx_tr,sep/c=0.008`, `dx_reattach/c=0.022`, `K_theta=5`, `H_reattach=1.7`):
reattachment at `x/c=0.461`, followed by turbulent re-separation at
`x/c=0.670`.

**Final drag decomposition (nominal bubble):**

| Component | Value |
|---|---|
| `C_d,f` (friction, attached + reattached regions only) | 0.002495 |
| `C_d,sep` (separated pressure/form-drag bookkeeping) | 0.019936 |
| **`C_d,total`** | **0.022430** |

The separated-flow penalty dominates the modeled total (~89% of
`C_d,total`) for the nominal bubble. Short/long bubbles and an explicit
open-separation case were also evaluated (`C_d,total` = 0.00975 / 0.0552 /
0.380 respectively) -- see DESIGN.md and README.md for the full table.

## External-flow sensitivity

A fixed, deterministic 7-case family (peak ratio 1.10/1.18/1.26; peak
location 0.25/0.35/0.45; TE ratio 0.45/0.55/0.70, one-factor-at-a-time)
was propagated through the unmodified M1-M5 chain:

| Quantity | Range across the family |
|---|---|
| `x_sep,lam/c` | [0.3677, 0.5027] |
| `N_max` | [3.241, 4.827] |
| `C_d,total` (nominal bubble) | [0.01976, 0.02530] |

**Most influential external-flow parameter for `x_sep,lam/c`: peak
location** (spread 0.135, vs. 0.030 for TE ratio and 0.004 for peak
ratio) -- **not** the TE-ratio/adverse-gradient severity that might have
been assumed; this was determined by computation, not assumed in advance.
For `C_d,total`, peak ratio has the largest Milestone-6 effect (12.8%
swing) among the three external-flow parameters.

## Robust findings

(Robust *within this reduced-order sensitivity family* -- not claimed
universal.)

- Laminar separation occurring before any sourced `N_crit` is reached: **7/7**
  external-flow cases at the baseline `V_inf`.
- A downstream turbulent re-separation after bubble reattachment: **7/7**
  cases.
- The exact separation-before-N_crit / re-separation *mechanism* is not
  an artifact of the one arbitrary baseline pressure-distribution choice.

## Model-sensitive findings

- The *quantitative* `x_sep,lam/c`, `N_max`, and `C_d,total` all vary
  measurably across the external-flow family (see ranges above).
- **Cross-model sensitivity ranking** (max `|Delta% C_d,total|`,
  one-factor-at-a-time, computed fresh in this session): `M5 K_sep`
  (266.6%) and `M5 bubble/reattachment length` (126.7%) dominate; `M6 peak
  ratio` (12.8%) and `M4 H_tr` (9.8%) are intermediate; `M6 peak location`
  (2.3%), `M5 H_reattach` (2.3%), `M5 K_theta` (2.2%), and `M6 TE ratio`
  (0.4%) are minor. **The unsourced separated-drag coefficient `K_sep` is
  the single most influential modeled assumption in the entire project** --
  more influential than any external-flow shape parameter tested.

## Final drag decomposition

See "Separation-bubble / drag result" above for the nominal-bubble table.
For context: `C_d,f` alone ranges `[0.002451, 0.008730]` across the M3
fully-laminar and fully-turbulent flat-plate references. Once the M5
separated-flow penalty is included, **every** M5 bubble/open-separation
scenario's `C_d,total` exceeds the M3 fully-turbulent friction-only
reference (0.008730) -- even the shortest bubble (`C_d,total=0.009749`).
This is a direct, quantitative illustration of why representing separated-
flow drag as a distinct term (rather than omitting it, as the M3/M4
friction-only bookkeeping did) matters for this baseline case.

## Important limitations

- Generic, non-manufacturer-matched sailplane section; synthetic
  `U_e(x/c)`, not CFD/XFOIL/experimental data.
- Reduced-order e^N proxy anchored at one sourced number (Blasius ZPG
  neutral point); not an Orr-Sommerfeld/PSE solution.
- `N_crit` is a sourced *sensitivity range*, never selected as "the"
  design value.
- Head's integral method is a classical reduced-order closure, not
  CFD/RANS.
- The laminar-separation-bubble closure and the separated-drag closure
  are both explicitly labeled parametric sensitivity models -- neither is
  independently validated, and `K_sep` in particular is the dominant
  source of uncertainty in the final drag numbers.
- No roughness, receptivity, compressibility correction, or 3D effect is
  modeled. Not a complete or certified airfoil drag polar.
