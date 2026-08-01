"""Gray as an mjlab entity.

The MJCF in sim/models/gray.xml is written to be run standalone by scripts/walk.py,
so it ships with its own floor, light and twelve <position> actuators. mjlab supplies
all three itself - the scene owns the terrain and lighting, and the actuator configs
below create their own <position> actuators so that gains, armature and command delay
are randomisable. Handing the file over unedited would give a duplicate ground plane
and twenty-four actuators for twelve joints, so `get_spec` strips them first.

Everything physical here traces back to gray/config/robot.yaml, which is generated -
if a number looks wrong, fix the tool that emits it, not this file.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from gray.gait import GaitGenerator, GaitParams
from gray.kinematics import LEGS, load_legs

GRAY_XML: Path = Path(__file__).resolve().parents[1] / "sim" / "models" / "gray.xml"
assert GRAY_XML.exists(), f"missing {GRAY_XML} - run tools/make_mjcf.py"

SEGMENTS = ("hip", "top", "bottom")          # matches (q0, q1, q2) in gray.kinematics
JOINT_ORDER = tuple(f"{leg}_{seg}" for leg in LEGS for seg in SEGMENTS)
FOOT_GEOM_REGEX = "^(fl|fr|br|bl)_bottom_collision$"

# DS3218MG, from robot.yaml. Position-commanded, no feedback, 50 Hz PWM ceiling.
SERVO_STIFFNESS = 20.0       # kp, as tuned in the standalone MJCF
SERVO_DAMPING = 0.5          # kv
SERVO_EFFORT_NM = 1.96       # 20 kg.cm at 6.8 V
SERVO_ARMATURE = 0.003       # ESTIMATE: ~6e-8 kg.m^2 rotor behind ~245:1. Dominates
                             # the ~8e-5 link inertia, so randomise it hard.

# Command latency, in physics timesteps. robot.yaml puts real servo lag at 10-40 ms;
# at the 0.005 s timestep configured in gray_env that is 2-8 steps.
DELAY_MIN_STEPS = 2
DELAY_MAX_STEPS = 8


def get_spec() -> mujoco.MjSpec:
  """Load Gray's MJCF and hand mjlab a bare robot."""
  spec = mujoco.MjSpec.from_file(str(GRAY_XML))

  # mjlab's scene owns the ground and the lighting.
  for geom in list(spec.worldbody.geoms):
    if geom.name == "floor":
      spec.delete(geom)
  for light in list(spec.worldbody.lights):
    spec.delete(light)

  # The actuator configs above re-create these with randomisable gains; leaving the
  # XML's twelve in place would give twenty-four actuators driving twelve joints.
  for actuator in list(spec.actuators):
    spec.delete(actuator)

  return spec


def _standing_pose() -> dict[str, float]:
  """The Phase 2 neutral stance, as a per-joint default.

  Reusing GaitGenerator.stand() rather than hardcoding numbers keeps the RL default
  pose identical to the one the classical gait starts from, so a zero residual on
  step one is exactly the Phase 2 controller.
  """
  stand = GaitGenerator(load_legs(), GaitParams()).stand()
  return {f"{leg}_{seg}": float(stand[leg][i])
          for leg in LEGS for i, seg in enumerate(SEGMENTS)}


# walk.py spawns at stance_height + 15 mm and lets the robot settle; same here.
INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, GaitParams.stance_height + 0.015),
  joint_pos=_standing_pose(),
  joint_vel={".*": 0.0},
)

# Only the feet are meant to touch the ground. Everything else keeps a collider so a
# fall is detected, but frictionless (condim=1) so a scraping knee cannot accidentally
# push the robot along and be rewarded for it.
COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  solref=(0.01, 1),
  condim={FOOT_GEOM_REGEX: 3, ".*_collision": 1},
  priority={FOOT_GEOM_REGEX: 1},
  friction={FOOT_GEOM_REGEX: (0.8,)},   # midpoint of the 0.4-1.2 randomisation
)

SERVO_ACTUATOR = BuiltinPositionActuatorCfg(
  # Anchored to the twelve joint names rather than ".*", which also matched the `imu`
  # site and made mjlab warn about ambiguous transmission targets.
  target_names_expr=("^(fl|fr|br|bl)_(hip|top|bottom)$",),
  stiffness=SERVO_STIFFNESS,
  damping=SERVO_DAMPING,
  effort_limit=SERVO_EFFORT_NM,
  armature=SERVO_ARMATURE,
  delay_min_lag=DELAY_MIN_STEPS,
  delay_max_lag=DELAY_MAX_STEPS,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(SERVO_ACTUATOR,),
  soft_joint_pos_limit_factor=0.95,
)


def get_gray_robot_cfg() -> EntityCfg:
  """A fresh Gray config. Fresh each call so callers cannot mutate a shared one."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer

  from mjlab.entity.entity import Entity

  mujoco.viewer.launch(Entity(get_gray_robot_cfg()).spec.compile())
