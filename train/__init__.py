"""Phase 3 - residual reinforcement learning on top of Gray's classical gait.

Everything in this package may import MuJoCo, torch and mjlab. That is exactly why
it is NOT part of `gray/`: `gray/` is imported unchanged by the Raspberry Pi, which
will never have any of those installed. Keep the boundary.

The policy learns *corrections* to the Phase 2 Bezier gait, never a gait from
scratch - see docs/PROJECT_NOTES.md and arXiv 2010.12070 (D2-GMBC).
"""
