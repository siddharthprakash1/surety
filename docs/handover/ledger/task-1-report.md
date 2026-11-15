Status: DONE_WITH_CONCERNS. Branch worktree-agent-a46cd2d55ada2d3cc, commit 0221189 "feat(calib): metrics".
Tests: uv run pytest -q -> 10 passed (9 new in tests/test_metrics.py). Red step observed before implementation.
Built: src/surety/calib/__init__.py, src/surety/calib/metrics.py (9 functions), tests/test_metrics.py.
Notes: ece clamps p_max==1.0 into the last bin, skips empty bins; argmax ties -> first key; risk 0.0 when nothing acted.
Concern: could not write this file from the worktree (Write tool refused); relayed by message.
