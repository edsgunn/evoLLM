# 2026-08-26 — `baserate` 6143004: no data at all

**There is no run directory here, because the job never created one.** This
README is the whole record, and it exists so the failure is findable rather
than only being a gap between job ids.

## What happened

Submitted at 11:19 to measure the mate-handshake base rate under **zero
genomes** — the frozen base model, no adapters. Sequence:

| time | event |
|---|---|
| 11:19:28 | job starts, `configs/node_4room_7b_chr001_evict.yaml` |
| 11:19:56 | room `gpu0`'s engine builds fine — pool 83,745 blocks |
| 11:19:59 | room `gpu1`'s engine calls `nvmlDeviceGetHandleByIndex(1)` → `NVMLError_InvalidArgument`; engine core dies |
| 11:20–15:19 | **the job hangs** |
| 15:19:23 | cancelled by Slurm at its 4-hour time limit |

Thirty-one seconds of work, four GPU-hours consumed, zero events written.

## Why

The four-room reference config was submitted to `slurm/precheck_handshake.sh`,
which requested `--gpus=1`. The config declares one room per GPU on devices
0–3. Room `gpu0` worked; room `gpu1` pinned `CUDA_VISIBLE_DEVICES=1` and asked
NVML for a physical device that had not been allocated to the job.

A submission error, not a bug in the world.

## What was changed

`slurm/precheck_handshake.sh` now requests `--gpus=4`, matching
`slurm/run_experiment.sh`. The base rate is only interpretable against a main
run, so it has to be measurable under the same world — which means the same
number of rooms. Retried as job **6146533**.

## What was NOT changed, and remains a live risk

**An engine-start failure does not terminate the job.** It hung for 3h59m after
the run was already dead. Every future job is exposed to the same thing: any
failure during engine construction burns the whole allocation silently. This is
unfixed and is the more expensive of the two problems — the submission error
cost one mistake, this cost four hours.

## Can anything here be cited?

No. The job produced no events, no genomes and no traces. The base-rate
question it was meant to answer — what the frozen base model does with no
genome — **remains open**, and until it is answered every "takeoff" claim in
this project means "better than a randomly perturbed base model" rather than
better than the base model itself.
