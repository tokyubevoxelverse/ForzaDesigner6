"""When the runner emits an engine_run error from a CUDA OOM, the
message should include the 'restart EXE' suggestion so users know
how to recover (CUDA cache survives the failed process and the
EXE process itself can hold an old context)."""
from __future__ import annotations

import re

from forza_abyss_painter.runtime import torch_runner


def test_engine_run_oom_message_suggests_restart():
    """Find the runner code that emits the engine_run error event
    and confirm it includes 'restart' guidance."""
    import inspect
    src = inspect.getsource(torch_runner)
    # The OOM-class message should mention restart. We don't pin the
    # exact phrasing, just that the word appears in proximity to
    # 'engine_run'.
    engine_run_block = re.search(
        r'"stage":\s*"engine_run".*?\n(?:.*?\n){0,15}',
        src,
        re.DOTALL,
    )
    assert engine_run_block, "Could not find engine_run stage block in run()"
    block_text = engine_run_block.group(0).lower()
    assert "restart" in block_text, (
        "engine_run error message should suggest restarting the EXE "
        "to release the CUDA cache. Got block:\n" + engine_run_block.group(0)
    )
