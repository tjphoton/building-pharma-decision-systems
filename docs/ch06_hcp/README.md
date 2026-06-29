# Chapter 6: HCP Targeting

Run the complete workflow from the repository root:

```bash
uv run python ch06_hcp/scripts/generate_ch06_data.py
uv run python ch06_hcp/scripts/run_analysis.py
MPLCONFIGDIR=/tmp/matplotlib uv run python ch06_hcp/scripts/build_figures.py
uv run python ch06_hcp/scripts/build_notebooks.py
uv run python ch06_hcp/scripts/verify_chapter_blocks.py
uv run ruff check ch06_hcp/generation_modules ch06_hcp/scripts tests/test_ch06_targeting.py tests/test_ch06_generation_isolation.py
uv run pytest tests/test_ch06_targeting.py tests/test_ch06_generation_isolation.py -q
```

Chapter 6 reads stable identifiers and source tables from Chapters 3 through 5. Supplemental data is written only under `ch06_hcp/data/generated/`. Analysis CSVs and `manifest.json` are written to `assets/generated_outputs/` and remain untracked.

The generator manifest records source hashes. `tests/test_ch06_generation_isolation.py` verifies that regeneration leaves every upstream source unchanged.

Figures are rebuilt as SVG and PNG. The figure builder removes stale Chapter 6 exports before writing the current 8-figure set.
