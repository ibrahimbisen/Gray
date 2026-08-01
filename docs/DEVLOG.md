# Development log

A record of how Phases 1–2 were built, including the reasoning behind decisions and
the things that turned out to be wrong. Kept because several findings here were
expensive to reach and cheap to lose, and because a plain list of what the code does
would not explain *why* it does it that way.

Session 1 — 1 Aug 2026.

---

## Starting point

The repo was 129 files: 39 SolidWorks parts, 30 STLs, 25 build photos, one URDF, and
exactly one Python file — a 14-line PyBullet stub that loaded the model, dropped it on
a plane, and spun `stepSimulation()` without ever commanding a motor. The README
advertised reinforcement learning; no training code, environment, reward function or
policy had ever been committed.

Last push: Nov 2022. The mechanical work was complete and the robot had been built
and photographed.

---

## Choosing an approach

The first substantive decision was whether to train a policy from scratch or layer it
on a hand-written gait.

**Decided: classical gait first, then RL as a residual on top.** Not as a fallback —
as a load-bearing component. RL from scratch on a 12-DOF quadruped takes days and
frequently fails to converge; a residual on a working gait converges in hours,
degrades gracefully toward that gait if the policy is bad, and proves the calibration
before a network ever sends a command.

This was initially presented as independent reasoning. **It is not** — it is a
published architecture (Policies Modulating Trajectory Generators), and specifically
D²-GMBC, which validated *Bézier gait + RL modulation + domain randomization*
sim-to-real on the SpotMicro class of hobby-servo quadruped. The prior art should have
been named up front. Corrected after the user pushed back and asked whether any
research had actually been done.

That pushback also changed the plan in two ways:

1. **Stop writing from scratch.** `OpenQuadruped/spot_mini_mini` (MIT) exists for
   almost exactly this robot. Read it as reference; reimplement on a current stack
   because its code is 2020-era PyBullet.
2. **Use the GPU from day one.** Initial advice was to develop on the Mac and escalate
   to the RTX 4070 Ti later. That was too timid — NVIDIA demonstrated zero-shot
   sim-to-real on Spot in Isaac Lab at ~90k FPS. The desktop should be the training
   machine from the start.

**Python over MATLAB**, decided on ecosystem rather than capability. MATLAB does have
an RL Toolbox with documented walking-robot examples, so this was not a capability
argument; but Simscape does not integrate cleanly with RL agent training, none of the
quadruped sim-to-real ecosystem lives there, and deployment to a Raspberry Pi is
Python-native.

---

## Phase 1 — the digital twin

### The mass model

The URDF claimed **1.254 kg**. The twelve DS3218MG servos alone are ~720 g, so this
was clearly incomplete. The first instinct — "the CAD numbers are junk, rebuild from
scratch" — turned out to be wrong, and finding out why produced a much better model.

A least-squares fit over all 13 links against mesh volumes recovered SolidWorks'
assumptions to an **RMS residual of 0.017 g**:

```
printed parts : SOLID plastic at 1.0227 g/cm3
servos        : 64.3 g each, 8 of 12 modelled
```

An exact fit. The URDF is a *coherent* model of the printed structure plus the servos
that were in the CAD — it simply omits the battery, Pi, electronics, fasteners, and
the four hip-pitch servos. 687 g of real robot that was never modelled.

Two consequences followed:

- Its **COM and inertia are trustworthy** and are now reused rather than recomputed.
  Validation: mesh centroids match SolidWorks' inertial origins to **0.1 mm** on every
  servo-free link, and differ by a consistent **12.5 mm** on exactly the four thigh
  links — the fingerprint of SolidWorks correctly treating the servo as denser than
  the plastic. A uniform-density recomputation would have been *worse*.
- Only completeness needed fixing, not correctness.

Corrected total: **1.625 kg** (hollowed parts) to **2.00 kg** (solid), against Stanford
Pupper's 2.1 kg in the same size class.

### Three bugs that do not announce themselves

**Servo geometry baked into link meshes.** Splitting `fl_top` into connected bodies
gave a largest body of 73.92 cm³ — exactly `Left_Thigh.STL` — plus sub-bodies
measuring 40.0 × 20.3 mm and 40.2 × 20.2 mm, which is the DS3218MG's 40 × 20 footprint.
Applying resin density to the whole mesh would have counted every servo twice, once as
plastic and once as datasheet mass. Baked-in counts: `base_link` 4, each `*_top` 1,
hips and shanks 0.

A related scare resolved harmlessly: five meshes report "not watertight." They are
multi-body STLs, not broken geometry — the exporter merged each link's sub-assembly
into one file. Hole-filling changes their volume by 0.00%.

**The export is Y-up.** This was the expensive one. Loaded raw, the robot lies on its
side: two legs pointing vertically up, two down, feet spanning 19 mm instead of 216 mm.
It was caught by noticing that the two *front* hip mounts differed along world Z, then
confirmed geometrically — `base_link`'s thin axis (25 mm of a 197 × 25 × 158 mm plate)
runs along Y, and for a Z-up robot the thin axis must be Z. Both rotation signs were
tested; +90° about X puts all four feet below the trunk. Fixed in the URDF rather than
patched downstream, so any consumer gets a correct model.

This is the class of bug that produces a policy that never learns to walk with no
visible reason why.

**MuJoCo silently discards the root link's mass.** With no parent joint, `base_link`
gets welded to the world: its geoms are emitted loose in `<worldbody>`, the four hips
are promoted to top-level bodies, and its 723 g vanish. The model reported 901 g
instead of 1625 g. Fixed by rebuilding the trunk as a real floating body.

### Result

Drop test: settles upright at 182.5 mm, uprightness 0.999, residual speed 0.6 mm/s.
Visual meshes decimated 21 MB → 3.6 MB and deliberately excluded from LFS so an
LFS-skipped clone still loads.

---

## Phase 2 — walking

### Why the IK is closed-form

Both pitch joints share an axis. That means the foot's offset *along* that axis is
constant regardless of joint angle, which splits the problem into one rotation of a
rigid plane plus a planar 2-link reach — both exactly solvable. No iteration, no
Jacobian, no convergence checks, which is what makes it viable at 50 Hz on a Pi.

Validation: **FK matches MuJoCo to 1 nanometre** across 400 random poses × 4 legs. IK
round-trips to 0.3 nm over 3000 poses, never violates joint limits, and correctly
rejects unreachable targets. Nine tests, including a guard against the knee flipping
branch mid-stride.

Leg geometry recovered: **thigh 141.2 mm, shank 170.0 mm, 311.5 mm max reach**, with
per-leg spread under 0.6 mm.

### Gait tuning

The swing curve is a 7-point Bézier whose control points bracket each endpoint
horizontally, giving near-vertical lift-off and descent. That shape was chosen for
**low touchdown speed rather than minimum swing time**, because the parts are SLA
resin — more brittle than printed thermoplastic under repeated impact, and walking is
nothing but repeated impact. The same concern becomes a reward term in Phase 3.

Sweeping step length and period produced two non-obvious limits:

- **Above ~2 strides/s the robot travels backward.** The feet skid; stance loses grip
  before it can push. A hard constraint imposed by servo speed.
- **Speed trades against straightness.** 160 mm/0.7 s reaches 71.6 mm/s but veers
  65 mm; 120 mm/0.6 s runs nearly straight at 52.8 mm/s.

Chose straight. Drift originates in the few-mm per-leg asymmetry of the CAD assembly,
and is exactly the sort of thing the RL layer absorbs for free — spending a baseline's
quality fighting it would be wasted.

Final: **675 mm in 12 s at 56 mm/s**, three feet down at all times, uprightness 0.987,
no falls.

Trot walks but wanders ~140 mm. Expected — with only two feet down it needs active
balance, which is Phase 3's job rather than a tuning problem.

---

## What the build photos settled

The user pointed out that `Overview/` contains build photographs. Two things came out
of reading them that the CAD alone did not show:

**The knee is a pushrod linkage.** A thigh-mounted DS3218 drives the shank through a
ball-jointed metal rod. Harmless in simulation — a four-bar driving a hinge is still a
hinge, so the kinematic tree is right — but **servo angle ≠ joint angle on the real
robot**, and the linkage changes effective torque and speed. Recorded as a Phase 4
blocker; a policy deployed without that mapping will not reach commanded angles.

**Servo placement confirmed.** Abduction on the body, hip-pitch at the shoulder, knee
on the thigh, nothing on the shank. This had been an assumption in the mass model; the
photos validated it.

Worth noting as a general lesson: the photos answered in one minute a question the
geometry could not answer at all.

---

## Corrections made during the session

Recorded because the reasoning matters more than the conclusions:

1. **"Your URDF is junk"** → it is incomplete but internally coherent, and its
   COM/inertia are good enough to reuse. Reversed after the least-squares fit.
2. **"We'll write this from scratch"** → prior art exists for this exact robot class;
   port the method instead. Reversed after the user asked whether any research had
   been done.
3. **"Train on the Mac, escalate later"** → the 4070 Ti should be the training machine
   from the start.
4. **"The PC will pull ~2 MB"** → 133 MB for a normal clone, because the working tree
   still carries `Overview/` and non-LFS assemblies. Sparse checkout gets it to 6.3 MB;
   both were tested by cloning fresh and running the gait.

---

## Open

- Are the SLA parts hollowed or solid? ±15% on total mass. Randomization covers it;
  a scale settles it at reassembly.
- Armature is an *estimate* (0.003 kg·m², from a ~6e-8 kg·m² rotor behind ~245:1).
  It dominates link inertia at ~8e-5, so it matters — randomize it hard.
- The pushrod linkage ratio is unmeasured and unmeasurable until reassembly.
