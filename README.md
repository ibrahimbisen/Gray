# Gray
A quadruped robot designed and manufactrued by college students from scratch and trained with pyhton reinforcement learning.

This project is WIP.

## Running it

Two processes, two terminals. Both survive the other being closed.

```
python run.py                 # the training centre, http://127.0.0.1:8000
python run.py --runner        # works through the queue, unattended
```

The dashboard is where you queue runs; the runner is what actually runs them.
Closing the dashboard must not kill a six-hour training job, which is why they
are separate. If nothing starts after you queue it, the page says so at the top —
that means the runner is not running.

Each queued job is `train → film → verify`, one at a time, because two training
processes on one card deadlock rather than fail (RULES.md rule 4). Queue as many
as you like and come back in the morning.

Other entry points:

```
python run.py --check         # what the dashboard can see, then exit
python scripts/train.py Gray-Walk --iterations 3000
python scripts/verify.py Gray-Walk        # score a policy against its bar
python tools/prepare_model.py "sim/URDF and Meshes V2"   # rebuild the model
python tools/api_contract_check.py        # pages vs the API they read
node tools/render_check.js                # every panel renders (server must be up)
```
## Latest Tests:
<img src="Overview/Test.gif" width="250" height="250"/><br />

## CAD Modelling:
This is a rendering of the URDF file<br />
<img src="Overview/Screenshot 2022-05-28 114828.png" width="250" height="150"/><br />
This is a rendering of the main body with computer systems - legs<br />
<img src="Overview/Screenshot 2022-05-28 114849.png" width="250" height="150"/><br />

## Manufacturing:

This is the picture of the Controller we are desiging for the robot<br />
<img src="Overview/IMG_5251.jpg" width="250" height="150"/><br />
This is a picture of the legs that we have designed<br />
<img src="Overview/photo_2022-09-18_12-46-38.jpg" width="250" height="150"/><br />
This is a picture of the main body<br />
<img src="Overview/photo_2022-09-18_12-47-05.jpg" width="250" height="150"/><br />
This is a picture of the main body combined with the computer systems for testing<br />
<img src="Overview/RObot_from_angle.JPG" width="250" height="150"/><br />


## Simulation:
These are couple of Gifs showing the progress that has been done by the machine learning algorithm we are using throughout time.

<img src="Overview/Gifs/1.gif" width="150" height="250"/><img src="Overview/Gifs/2.gif" width="150" height="250"/><img src="Overview/Gifs/3.gif" width="150" height="250"/>

Currently WIP.

## Contributors:
 - Ibrahim Eren Bisen
 - Emin Alp Arslan
