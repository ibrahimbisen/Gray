# Gray

Gray is a 12-DOF quadruped robot. Two college students designed the robot and made the
parts. The software teaches the robot to walk with reinforcement learning in Python.

This project is in progress.

## Where the project is

The mechanical robot is complete. The work now is in simulation: teach one policy to
walk, teach a second policy to stand up, then join them.

- [PLAN.md](PLAN.md) — the stages, and the number each stage must reach to pass.
- [RULES.md](RULES.md) — the decisions that hold across the project, and why.
- [CLAUDE.md](CLAUDE.md) — the facts about the hardware that do not change.

The robot has 12 servos, three IMUs on the trunk, and a potentiometer on each joint.
The knee is a pushrod linkage, so the servo angle is not the joint angle.

## How to run it

Start two processes in two terminals. Each process continues if you close the other one.

```
python run.py                 the training centre, http://127.0.0.1:8000
python run.py --runner        works through the queue, unattended
```

The dashboard is where you put runs in the queue. The runner is what does them.
The two are separate because a closed dashboard must not stop a six-hour job.
If nothing starts after you queue a job, the top of the page tells you: the runner
is not running.

Each job in the queue is `train`, then `film`, then `verify`. The runner does one job
at a time. Two training processes on one graphics card stop each other and make no
progress, so a queue of one is the rule (RULES.md rule 4). Put as many jobs in the queue
as you want, and come back in the morning.

> **Caution: do not open a simulator window while a run trains.** A second process on
> the same card once stopped the machine and lost the run. This includes
> `Playground\pilot.bat`, `scripts/drive.py`, and `find_stance.py --render`.
> The dashboard is safe, because it uses the CPU only.

Other entry points:

```
python run.py --check                     what the dashboard can see, then exit
python scripts/train.py Gray-Walk --iterations 3000
python scripts/verify.py Gray-Walk        score a policy against its bar
python tools/prepare_model.py "sim/URDF and Meshes V2"   rebuild the model
python tools/api_contract_check.py        the pages against the API they read
node tools/render_check.js                every panel shows (the server must be up)
```

The three tasks are `Gray-Stand`, `Gray-Push` and `Gray-Walk`. To drive a trained policy
by hand, read [Playground/README.md](Playground/README.md). For what each reward term
pays for, read [docs/REWARDS.md](docs/REWARDS.md).

## Latest tests

<img src="Overview/Test.gif" width="250" height="250"/><br />

## The CAD model

This image shows the URDF file.<br />
<img src="Overview/Screenshot 2022-05-28 114828.png" width="250" height="150"/><br />
This image shows the body with the computer parts, and without the legs.<br />
<img src="Overview/Screenshot 2022-05-28 114849.png" width="250" height="150"/><br />

## Manufacture

This image shows the controller for the robot.<br />
<img src="Overview/IMG_5251.jpg" width="250" height="150"/><br />
This image shows the legs.<br />
<img src="Overview/photo_2022-09-18_12-46-38.jpg" width="250" height="150"/><br />
This image shows the body.<br />
<img src="Overview/photo_2022-09-18_12-47-05.jpg" width="250" height="150"/><br />
This image shows the body with the computer parts, ready for a test.<br />
<img src="Overview/RObot_from_angle.JPG" width="250" height="150"/><br />

## Simulation

These animations show what the reinforcement learning made, at three times.

<img src="Overview/Gifs/1.gif" width="150" height="250"/><img src="Overview/Gifs/2.gif" width="150" height="250"/><img src="Overview/Gifs/3.gif" width="150" height="250"/>

This work is in progress.

## Contributors

 - Ibrahim Eren Bisen
 - Emin Alp Arslan
