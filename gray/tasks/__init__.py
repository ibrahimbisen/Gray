"""Gray's training tasks, registered with mjlab.

Found automatically through the "mjlab.tasks" entry point declared in
pyproject.toml, so `train Gray-Stand` works from anywhere once the project is
installed with `pip install -e .`.
"""

from mjlab.tasks.registry import register_mjlab_task

from gray.tasks.stand_env_cfg import stand_env_cfg, stand_ppo_cfg

register_mjlab_task(
    task_id="Gray-Stand",
    env_cfg=stand_env_cfg(),
    play_env_cfg=stand_env_cfg(play=True),
    rl_cfg=stand_ppo_cfg(),
)
