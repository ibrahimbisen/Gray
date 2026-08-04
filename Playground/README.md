# Playground

Somewhere to poke at a trained policy by hand. Nothing in here trains anything,
scores anything, or is read by the dashboard.

> **Not while something is training.** This opens a second process on the same
> graphics card, and there is no room for one — runs are sized to fill the card on
> purpose. Doing it once took the whole machine down and lost the run with it.
> Check the dashboard first: if anything is running, pause the queue and wait.
> RULES.md rule 4.

## Drive it yourself

```
Playground\pilot.bat                 the newest walk run
Playground\pilot.bat --run 25        a run by number, name, or folder
Playground\pilot.bat --run walk_v3 --checkpoint model_1500.pt
```

A window opens with one robot in it, and you steer it — with a **gamepad** if one
is plugged in, otherwise from the **numpad**.

## The gamepad

| control | does |
| --- | --- |
| left stick | walk: away from you is forward, sideways is a crab step |
| right stick | turn on the spot |
| `A` | stop |
| `B` | put it back on its feet |
| `ZL` held | drive it past the range it was trained on |

Full stick is the fastest speed it was ever trained on, so you cannot
accidentally ask for something meaningless. Push half way, walk half speed.

The pad is read on the physics thread, not through the window, so it keeps
working when the MuJoCo window is not the one in front.

If a stick does nothing or does the wrong thing:

```
Playground\pad.bat        prints every axis and button live, as you move them
```

The map it should match is at the top of `PadPilot` in
[pad.py](pad.py) — three axis names and three button numbers, measured on the
PowerA Switch pad. `Playground\pilot.bat --keys` ignores the pad entirely.

## The numpad

```
    7  crab left       8  faster         9  crab right
    4  turn left       5  STOP           6  turn right
                       2  slower
    0  straight ahead at the usual speed
```

| key | does |
| --- | --- |
| numpad `8` / `2` | forward, ±0.05 m/s per press |
| numpad `4` / `6` | turn on the spot, ±0.10 rad/s per press |
| numpad `7` / `9` | crab sideways, ±0.05 m/s per press |
| numpad `5` | stop — tell it to stand still |
| numpad `0` | straight ahead at the usual speed |
| mouse drag | move the camera |
| **ctrl + drag on the robot** | **shove it, and watch it catch itself** |
| space | pause |
| enter | put it back on its feet |
| `-` / `=` | slow motion / fast forward |
| `P` | show a live plot of every reward term |

**Why not WASD.** MuJoCo binds every letter from A to Z to one of its own view
toggles — `W` is wireframe, `S` is shadow, `A` is auto-connect, `D` is static
bodies, and so on for the whole alphabet. It handles them inside its C++ window
before Python sees the key, so they cannot be turned off: WASD steered the robot
and scrambled the picture at the same time. The numpad is the one block MuJoCo
leaves alone.

Speeds are **sticky**: a press changes the speed and it stays there until you
change it again, like a throttle. The window only reports keys going down, never
coming back up, so hold-to-walk is not available.

## What the readout means

Bottom left, while you drive:

```
told       +0.25 fwd   +0.00 side   +0.00 turn
doing      +0.21 fwd   +0.02 side   -0.04 turn
off line   +38 mm      heading -2 deg
falls      0           upright 0.98
```

- **told** — the three numbers you are sending. This is everything the policy
  knows about what you want.
- **doing** — what the trunk is really doing, smoothed over about half a stride.
  Raw, it is unreadable: every footfall makes the trunk surge at about four times
  its mean speed. This is smoothed exactly the way the walk reward smooths it, so
  it is the same number the policy was scored on.
- **off line** — how far sideways it has ended up from the line it was sent
  along, measured from where it was and which way it pointed when the straight
  command started. Resets when you ask for a turn: a turn is not drift.
- **heading** — how far it has rotated off that line. Different failure: a robot
  can hold a perfectly straight line while slowly turning to face sideways.
- **falls** — how many times it has gone over since the window opened.
- **upright** — 1.00 is level, 0.00 is on its side. Below 0.50 counts as fallen.

If **OUTSIDE WHAT IT TRAINED ON** appears, the command is outside the range the
policy ever saw. The walk policy trained on 0.15–0.35 m/s forward, ±0.10
sideways, ±0.5 rad/s turn. Ask for 0.9 m/s and it will fall over, and that tells
you nothing at all about the policy.

## What this is not

It shows how a policy **feels**. It does not show whether it **passes** — the eye
is easily fooled by a robot that looks busy. For that:

```
python scripts/verify.py Gray-Walk
python scripts/drive.py --run 25 --cases "0.25,0,0; 0.25,0.10,0"
```

If the *doing* line here and `drive.py` disagree for the same command, the
readout is wrong, not the policy.

## Files

| file | what it is |
| --- | --- |
| `pilot.py` | picks the run, builds the world, opens the window |
| `control.py` | the keyboard. Three floats — forward, sideways, turn |
| `pad.py` | the gamepad, read straight from Windows. Same three floats |
| `hud.py` | writes those floats into the policy's command, draws the readout |
| `pilot.bat` | launcher, because `.venv\Scripts\python.exe` is blocked here |
| `pad.bat` | the live pad probe |

`control.py` and `pad.py` know nothing about simulation on purpose — they hand
over three numbers and nothing else. The real robot takes the same three, so
whatever drives it later plugs in here without the rest changing.
