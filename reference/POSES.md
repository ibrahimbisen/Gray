# Reference poses and joint limits

Given by the owner (Ibrahim) on 2 August 2026, from knowledge of the real machine,
which is currently in pieces and cannot be measured by hand. Recorded as stated, then
checked against the CAD.

Renders are generated FROM the saved numbers by `python tools/render_pose.py`, so a
picture can never drift out of step with the data it claims to show. They live in
`reference/poses/`. The angles are also in `progress/poses.json`.

Angles are DEGREES in the physical convention the pose editor uses:

    hip     +  leg swings OUT, away from the body
    top     +  leg swings FORWARD, towards the nose
    bottom  +  foot lifts UP

That convention exists because the legs are mirrored and the raw joint angles disagree
with each other: +10 degrees of knee lifts the front-right foot 26 mm and drops the
front-left one 22 mm. The per-leg sign is measured at startup, not written down.

---

## The three poses

| pose | hips | thighs | knees | trunk | clipping |
|---|---|---|---|---|---|
| resting, limp | +55 | -21 | +39 | 42 mm | none |
| sitting, ready | 0 | -22 | +39 | 104 mm | none |
| standing, max height | 0 | +24 | -43 | 281 mm | none |

All three are clear on all 66 part pairs, checked triangle-by-triangle against the
real meshes. 240 mm of range from lying to fully extended.

### resting, limp
Powered off, body limp, legs pulled in. Intended as the pose every training episode
starts from, and the state the real robot is in before it is asked to stand.

OWNER'S DESCRIPTION: eight points of contact - four hips and four toes on the floor.

MEASURED, and it is six points, not eight:

    fr_top      0.0 mm      <- front thighs are the lowest parts
    fl_top      0.9 mm
    fr_hip      2.7 mm      <- all four hips are down, as described
    fl_hip      2.8 mm
    bl_hip      3.2 mm
    br_hip      3.2 mm
    fr toe      9.7 mm      <- toes are in the air
    fl toe     10.7 mm
    br toe     20.6 mm
    bl toe     27.6 mm

The hips are exactly as described. The toes are not down, and no knee angle brings the
back ones down: sweeping the knee from -20 to +30 with these hips and thighs leaves the
back toes at 10.9 mm and 13.6-17.6 mm throughout. With the hips 55 degrees out the back
legs are held clear of the floor by geometry, not by a setting.

RESOLVED, PROVISIONALLY: the owner's own words - "i think the model is slightly wrong
but if you measured that it's fine". So the measurement is what everything downstream
uses: six contact points, four hips and two front thighs, toes clear of the floor.

FLAGGED FOR RE-CHECK ON ASSEMBLY. The owner believes the real robot rests differently
and that belief is not being discarded, only set aside in favour of the number that
can actually be checked today. The most likely cause of a genuine difference is the
foot geometry: the four legs are not identical in the CAD (see the last section) and
the toes are modelled as 12 mm spheres, so a real rubber foot that squashes or sits
lower than the sphere would bring the toes down without changing anything else.

If the assembled robot does rest on its toes, the fix is in the model, not in these
angles - the pose itself is collision-free and 42 mm is the lowest trunk height found
by hand or by search.

### sitting, ready
Powered on, all limbs active, ready to stand.

It does NOT hold this pose: released under gravity with the servos holding, it sags
from 104 mm to 110 mm and ends up resting on nine parts including both back thighs,
level 0.964. Worth knowing before it is used as a target.

### standing, max height
Legs fully extended, the highest the robot goes.

Holds well: 281 mm commanded, settles at 269 mm, dead level at 0.998, resting on all
four feet and shanks.

---

## Joint travel, all three

OWNER, from the assembled machine:

    hip     -82 (in)   to  +92 (out)     174 deg
    thigh   -35 (back) to  +65 (forward) 100 deg
    knee    -40 (down) to  +39 (up)       79 deg

Against the servo's own 270 degrees that is 64%, 37% and 29%. These are HARDWARE
stops - where the assembled joint runs out of travel - and cannot be derived from the
CAD, which is why they are recorded here.

THE CAD AGREES BY NOT DISAGREEING for the thigh and the knee: swinging either across
its whole stated range, and well past it, produces no self-collision at all. Thigh
checked -60 to +90, knee -60 to +65, both at hips 0; and both stated end stops
re-checked against four knee positions each. Everything clear. So for those two the
owner's number IS the limit - there is nothing in the geometry to find.

The hip is the exception and is dealt with below.

WHAT THE KNEE BUYS, measured at hips 0 and thighs 0 - this is the ride-height range:

    knee -40   trunk 264 mm      the tallest the knee alone reaches
    knee   0   trunk 199 mm
    knee +35   trunk 102 mm      the lowest
    knee +40   trunk 101 mm      and past here it stops changing: the leg has folded
                                 far enough that something else is carrying the robot

DISCREPANCY, unresolved: the owner's own "standing, max height" pose uses knee -43,
which is 3 degrees past the -40 stated here. Both are clear of collisions. At -43 the
trunk is 281 mm, at -40 it is 276 mm - so honouring the stated limit costs 5 mm of
height. Recorded rather than silently corrected.

## Hip travel in detail

OWNER: inward to about -82 degrees, outward to about +92 degrees.

These are MECHANICAL limits of the hip joint on the assembled robot - where the
hardware stops - and they cannot be derived from the CAD, which is why they are
recorded here rather than measured. Total 174 degrees of the servo's 270.

The CAD imposes a SECOND limit, of a different kind, and the binding one at any moment
is whichever is tighter:

    INWARD, legs hanging straight: the two front feet meet each other at -28 degrees.
    Long before -82. Measured by swinging the hips in with thigh and knee at zero;
    first clash is fl_bottom into fr_bottom.

    INWARD at the full -82: only reachable with the legs folded up out of each other's
    way. Of 20 thigh/knee combinations tried at -82, only two were clear, both with the
    knee at +80 and the thigh back at -20 or -40. Everything else has the left and
    right legs passing through each other.

    OUTWARD: no self-collision found anywhere. Clear at +92, and still clear at +120,
    past the stated limit. So +92 is the hardware stopping it, not the geometry.

So the hip limit is not one number. It is -82 to +92 from the hardware, further
restricted by what the knees and thighs are doing at the time.

---

## Note on the legs not matching each other

The four legs are NOT identical in the CAD. In the same pose the back-left toe sits
27.6 mm higher than the front-right, and with every joint at zero the four feet sit at
0.0, 2.5, 8.2 and 18.5 mm off the floor. Foot positions differ by about 10 mm between
legs.

This is why one set of angles cannot put all four toes down, and why the pose editor's
"Reset" solves a small per-leg knee angle (0.0, -1.0, -3.3, +7.5 degrees) to stand the
robot square instead of simply zeroing everything.

---

## The learning plan, as the owner set it

Staged, each stage starting where the last one ended:

    1  resting limp  42 mm  ->  sitting ready  104 mm     get the legs underneath
    2  sitting ready 104 mm ->  standing       187 mm     push up and hold
    3  standing             ->  walking                   later

STANDING HEIGHT IS 187 mm, being 60% of the 312 mm leg (thigh 141 + shank 170), which
is the owner's "about 60 percent". The other readings of that phrase were 166 mm (60%
of the 276 mm maximum) and 183 mm (60% of the way up from resting); 187 mm was chosen
because expressing stance as a fraction of LEG EXTENSION is the usual convention and
it leaves the knee genuinely bent, with travel left to absorb a footfall and to push
off from. Reached at thigh 0, knee +4.

EVERY COMMANDED HEIGHT SAGS. Measured by commanding a pose and letting the servos
hold it against the robot's 1.625 kg:

    commanded 187 mm  ->  holds 172 mm   (sag 17 mm)
    commanded 166 mm  ->  holds 146 mm   (sag 19 mm)
    commanded 199 mm  ->  holds 183 mm   (sag 16 mm)
    commanded 276 mm  ->  holds 262 mm   (sag 14 mm)

All four hold dead level (0.998-0.999) and stop moving. So a target height must be
commanded about 17 mm above where it is wanted, or the reward has to be written
against the settled height rather than the commanded one.

KNOWN PROBLEM WITH STAGE 1: "sitting ready" does not hold itself up. Commanded at
104 mm it sags to 110 mm and ends resting on nine parts including both back thighs,
level 0.964. As written, stage 1 finishes at a pose the robot falls out of. Either
re-pose it into something the servos can hold, or treat it as a pose to pass THROUGH
and make stage 1 run all the way into standing.
