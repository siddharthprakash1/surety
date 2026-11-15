# Task 4 report
- Status: DONE. Branch task-4.
- Files: src/surety/supervise/{__init__,client,prompts}.py, tests/test_supervisor.py
- Tests: 8 passed (7 new + existing). TDD red (collection error) then green.
- Notes: ids matched as whole tokens (not adjacent to \w or '-'); candidate_ids restrict both prompt and parsing; ids not in state.options are skipped in prompt; HTTP errors raise via raise_for_status.
