# Autonomous Training Optimization Program

This document defines how an autonomous coding agent should iteratively improve the training pipeline in this repository.

## Goal

Improve the training setup to achieve the best possible validation outcome under the repository's existing constraints.

Primary metric to optimize:
- `val_bpb`: `lower` is better

Secondary metrics / constraints:
- `training_seconds`: should stay close to the fixed budget (`300s` measured training time)
- `peak_vram_mb`: should stay within available single-GPU memory (no OOM)
- `mfu_percent`: higher is better when `val_bpb` is not regressing
- run must complete successfully and print the final summary block

Per-run budget:
- wall clock budget: fixed `TIME_BUDGET = 300` seconds of measured training time from `prepare.py`
- note: `train.py` excludes startup/compile overhead by only counting steady-state steps after step 10
- timeout rule for automation: if total wall-clock runtime exceeds `900s`, terminate and mark `timeout`

Success criteria:
- improve `val_bpb` versus baseline/best-known commit
- do not crash (`FAIL`, exception, OOM, NaN/Inf behavior)
- keep evaluation protocol unchanged (`evaluate_bpb` path)
- prefer simpler, attributable changes when performance is similar

## Repository scope

Read these files first for context:
- `README.md`
- `train.py`
- `prepare.py`
- `pyproject.toml`

Primary training entrypoint:
- `uv run train.py`

Primary evaluation path:
- final metric comes from `prepare.evaluate_bpb(...)`, called at end of `train.py`
- summary is printed after a `---` separator and includes `val_bpb`, runtime, memory, and model stats

Artifacts and logs to inspect:
- stdout/stderr from training run (`run.log`)
- final printed summary block in `run.log`
- optional local cache artifacts in `~/.cache/autoresearch/` (data/tokenizer), but these are not experiment outputs

## What can be modified

The agent may modify:
- `train.py` only
- model architecture knobs in `train.py` (e.g. `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN`)
- optimization and schedule knobs in `train.py` (LRs, betas, warmup/warmdown, weight decay, batch sizing)
- training-loop internals in `train.py` that do not alter evaluation integrity

## What cannot be modified

The agent must NOT:
- modify validation metric definition (`evaluate_bpb`) or how it is computed
- modify dataset contents, shard pinning, tokenizer training logic, or byte accounting in `prepare.py`
- modify held-out validation shard behavior (`VAL_SHARD`/`VAL_FILENAME` path)
- add new dependencies or external services
- bypass final evaluation or hardcode reported metrics
- rewrite unrelated project files/infrastructure

Do NOT modify:
- `prepare.py`
- `pyproject.toml`
- dataset/cache contents under `~/.cache/autoresearch/`
- official metric reporting format in the final summary block

## Baseline run

The first experiment must always be the baseline:
1. run the pipeline exactly as-is from a clean commit:
   - `uv run train.py > run.log 2>&1`
2. confirm run completed and printed the final summary block
3. extract `val_bpb` and runtime stats from the block
4. record in `autoexp_results.tsv` as baseline (`status=keep` for initial reference)
5. compare all subsequent experiments against this baseline/best-known result

## Output format

At the end of each run, extract these fields from the final summary block in `run.log`:

```text
val_bpb:          <float>
training_seconds: <float>
total_seconds:    <float>
peak_vram_mb:     <float>
mfu_percent:      <float>
total_tokens_M:   <float>
num_steps:        <int>
num_params_M:     <float>
depth:            <int>
```

The block is printed exactly after a line containing `---`.
If a run exits early (e.g. prints `FAIL` and exits), mark missing fields as `NA`.

## Logging results

Log every experiment to:
- `autoexp_results.tsv`

Use tab-separated format with this header:

```text
commit	primary_metric	status	description	runtime_sec	peak_vram_mb	notes
```

Field mapping:
- `commit`: short git commit hash of the experiment
- `primary_metric`: value of `val_bpb`
- `status`: `keep`, `discard`, `crash`, `timeout`, `oom`, or `nan`
- `description`: one-line description of change tested
- `runtime_sec`: `total_seconds` from summary (or `NA`)
- `peak_vram_mb`: from summary (or `NA`)
- `notes`: why kept/discarded or failure diagnosis

Do not commit `autoexp_results.tsv` unless explicitly requested.

## Search space hints

High-priority knobs:
- learning rates: `EMBEDDING_LR`, `UNEMBEDDING_LR`, `MATRIX_LR`, `SCALAR_LR`
- optimization schedule: `WARMUP_RATIO`, `WARMDOWN_RATIO`, `FINAL_LR_FRAC`, `ADAM_BETAS`, `WEIGHT_DECAY`
- throughput/memory tradeoff: `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE` (with implied `grad_accum_steps`)
- model capacity: `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, `WINDOW_PATTERN`

Low-priority or risky knobs:
- major rewrites to attention/optimizer kernels
- dataloader packing semantics
- changing numerical behavior that risks instability without clear hypothesis

Known repo-specific traps:
- OOM from increasing `DEVICE_BATCH_SIZE`, `DEPTH`, or model width too aggressively
- invalid batch math if `TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) != 0` (assert crash)
- instability fast-fail in `train.py`: run exits if `train_loss` is NaN or `> 100`
- kernel/device compatibility issues around Flash Attention kernel loading on non-supported GPUs

## Experimentation principles

- Always make one coherent, attributable experiment per commit
- Keep edits focused to `train.py`
- Preserve reproducibility (`torch.manual_seed(42)` path should remain deterministic)
- Prefer changes that improve `val_bpb` without inflating runtime/memory excessively
- Revert non-improving or unstable experiments
- Continue from best-known commit only

Reasonable experiments:
- LR/schedule tuning
- batch size vs grad accumulation tradeoffs
- depth/width/window-pattern tradeoffs under fixed 5-minute training budget
- optimizer hyperparameter adjustments already supported in current code

Unreasonable experiments:
- modifying validation logic or data split definitions
- skipping final eval
- hardcoding metric outputs
- broad speculative rewrites that confound attribution

## The experiment loop

LOOP FOREVER:

1. Check current branch and identify best-known `val_bpb` from `autoexp_results.tsv`.
2. Choose one promising `train.py` change.
3. Edit only allowed file(s) (`train.py`).
4. Commit with a short message describing the hypothesis.
5. Run experiment:
   - `uv run train.py > run.log 2>&1`
6. Parse `run.log` summary block and extract metrics.
7. On failure, inspect `tail -n 80 run.log` and classify (`crash`/`oom`/`nan`/`timeout`).
8. Append one row to `autoexp_results.tsv`.
9. If `val_bpb` improved meaningfully and run is stable, mark `keep` and continue from this commit.
10. Otherwise mark `discard` (or failure status) and return to previous best commit.
11. Repeat.

## Failure handling

Treat these as failures unless trivial to fix:
- Python exception / import error / assertion failure
- CUDA OOM
- `FAIL` printed by training loop (NaN or exploding loss)
- missing final summary block
- timeout beyond `900s` wall clock

Crash handling policy:
- small, obvious mistakes can be fixed and retried once
- if idea appears fundamentally unstable or too costly, log and discard

## Branching and reproducibility

Recommended branch naming:
- `autoexp/<date>-<tag>`

For each run:
- keep experiment changes committed before running
- avoid mixing unrelated modifications
- keep logs (`run.log`) and `autoexp_results.tsv` local unless asked to commit
- preserve the best-known runnable state

## Final instruction to the agent

You are acting as an autonomous training researcher for this repo.
Run disciplined, incremental experiments on `train.py` to reduce `val_bpb` under the fixed 5-minute training budget.
Keep evaluation integrity intact, log every run to `autoexp_results.tsv`, and prefer robust improvements over risky rewrites.
