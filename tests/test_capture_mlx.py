from collections.abc import Callable
from pathlib import Path
from runpy import run_path
from typing import cast

_visible_text = cast(
    Callable[[str], str],
    run_path(str(Path(__file__).parents[1] / "scripts/capture_mlx.py"))[
        "_visible_text"
    ],
)


def test_visible_text_extracts_final_channel_and_rejects_truncated_thought() -> None:
    assert _visible_text("<|channel>thought\nplan<channel|>answer<audio|>") == "answer"
    assert _visible_text("<|channel>thought\nunfinished plan") == ""
    assert _visible_text("<|channel>thought\nplan<|channel>final\nanswer") == ("answer")
