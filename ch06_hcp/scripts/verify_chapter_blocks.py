"""Execute every printed Chapter 6 block and verify its quoted output."""

from __future__ import annotations

import contextlib
import io
import re
import subprocess
from pathlib import Path


CHAPTER = Path(__file__).resolve().parents[1] / "ch06_hcp_account_targeting.md"
REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_PATTERN = re.compile(
    r"```(bash|python)\n(.*?)\n```\n\n```text\n(.*?)\n```",
    re.DOTALL,
)


def main() -> None:
    blocks = BLOCK_PATTERN.findall(CHAPTER.read_text(encoding="utf-8"))
    namespace = {"__name__": "__chapter_block_verifier__"}
    failures = []

    for index, (language, source, expected) in enumerate(blocks, start=1):
        if language == "bash":
            completed = subprocess.run(
                source,
                shell=True,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            actual = completed.stdout.rstrip()
        else:
            stream = io.StringIO()
            with contextlib.chdir(REPO_ROOT), contextlib.redirect_stdout(stream):
                exec(
                    compile(source, f"<chapter-block-{index}>", "exec"),
                    namespace,
                )
            actual = stream.getvalue().rstrip()

        if actual != expected.rstrip():
            failures.append(
                {
                    "block": index,
                    "language": language,
                    "expected": expected.rstrip(),
                    "actual": actual,
                }
            )

    if failures:
        for failure in failures:
            print(
                f"\nBlock {failure['block']} ({failure['language']}) mismatch"
                f"\nExpected:\n{failure['expected']}"
                f"\nActual:\n{failure['actual']}"
            )
        raise SystemExit(1)

    print(f"Verified {len(blocks)} Chapter 6 blocks and quoted outputs.")


if __name__ == "__main__":
    main()
