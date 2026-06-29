# Chapter 7: Competitive Intelligence and Market Access

Generate the Chapter 7 supplemental data:

```bash
uv run python ch07_competitive/generation_modules/generate_ch07_data.py
```

Run the analysis and figures:

```bash
uv run python ch07_competitive/scripts/run_analysis.py
uv run python ch07_competitive/scripts/build_figures.py
```

Build and execute the 2 companion notebooks:

```bash
uv run python ch07_competitive/scripts/build_notebooks.py
uv run jupyter nbconvert --to notebook --execute --inplace ch07_competitive/ch07_walkthrough.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace ch07_competitive/ch07_exercise_solutions.ipynb
```

Run the publication checks:

```bash
uv run pytest tests/test_ch07_competitive.py -q
uv run ruff check ch07_competitive tests/test_ch07_competitive.py
uv run python ch07_competitive/scripts/verify_chapter_blocks.py
```
