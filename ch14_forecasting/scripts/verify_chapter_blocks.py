"""Verify printed Chapter 14 code-block output against execution."""

from __future__ import annotations

import contextlib
import io
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "ch14_forecasting" / "ch14_forecasting.md"
FENCE = re.compile(r"```(python|bash|text)?\n(.*?)\n```", re.DOTALL)


def _normalize(text: str) -> str:
    lines = text.strip().replace("\r\n", "\n").splitlines()
    return "\n".join(line.rstrip() for line in lines)


def _run_python(source: str, namespace: dict[str, object]) -> str:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, str(MANUSCRIPT), "exec"), namespace)
    return output.getvalue()


def main() -> None:
    os.chdir(ROOT)
    blocks = FENCE.findall(MANUSCRIPT.read_text())
    namespace: dict[str, object] = {"__name__": "__main__"}
    checked = 0
    index = 0
    while index < len(blocks):
        language, source = blocks[index]
        if language != "python":
            index += 1
            continue
        if index + 1 >= len(blocks) or blocks[index + 1][0] not in {"", "text"}:
            index += 1
            continue
        expected = blocks[index + 1][1]
        actual = _run_python(source, namespace)
        if _normalize(actual) != _normalize(expected):
            raise AssertionError(
                f"Output mismatch after python block {checked + 1}\n"
                f"Expected:\n{expected}\n\nActual:\n{actual}"
            )
        checked += 1
        index += 2
    print(f"Verified {checked} Chapter 14 printed code blocks.")


if __name__ == "__main__":
    main()
