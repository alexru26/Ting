# TODO

All items from the previous TODO are implemented on `rework`:

## Training data - DONE
- [src/botzone_ingest.py](src/botzone_ingest.py) replays `data/data.txt` (98,209
  rounds) into the shared trajectory schema: draw decisions, claim decisions
  (including ignored HU/GANG declarations; ignored PENG/CHI lack the
  follow-up discard and are skipped and counted), rob-the-kong decisions,
  final-score rewards, and `steps_from_end`. On the 16-round sample it
  produces zero label/legality disagreements with the judge.
- Botzone logs, local self-play, and any future corpora mix via repeated
  `--dataset path:weight` flags; caches are per-source and merged at train.
- Train/validation splits are by match (`split_by_game`), never by turn.
- Filtering/weighting: forced turns carry zero policy loss by construction;
  fan-backward credit decay (`--credit-gamma`) up-weights decisive turns
  near wins; `--outcome-scale` up-weights high-scoring trajectories;
  `--draw-weight` down-weights drawn games; the auxiliary win head provides
  the good-vs-merely-legal signal.
- The 27 finalist checkpoints are frozen league/evaluation opponents only
  ([src/finalist_opponents.py](src/finalist_opponents.py)); the adapter was
  validated at 55-58% exact-move agreement (~10% chance) on replayed logs.

## Supervised learning - DONE
- Weighted masked-action loss (outcome-sensitive), value loss on decayed
  returns, auxiliary win head (`--win-weight`).
- Data-mix controls via dataset specs; per-state-type metrics: decision vs
  forced counts, top-k masked accuracy, value MSE, win accuracy, and ECE
  calibration in `eval-cnn`.

## Reinforcement learning - DONE
- Opponent league samples rule baseline, finalists (`--finalist-dir/prob`),
  historical checkpoints (`--opponents` + `--league-dir`), and candidate
  mirrors (`--self-play-prob`); promoted candidates snapshot into the league
  dir and persist for later runs.
- Promotion requires the baseline gate AND the paired duplicate-wall gate.
- Reward shaping stays explicit (`--score-delta-weight --fan-weight
  --placement-weight`).

## Research notes - DONE
- Tjong fan-backward: `backfill_decayed_returns` / `--credit-gamma` give
  earlier decisions nonzero, decayed credit (also applied to supervised
  value targets).
- Suphx global reward baseline: PPO advantages subtract the value head's
  prediction and are normalized per update.
- Suphx oracle guiding: 3 oracle planes (opponent hands) exist only in the
  simulator and anneal `--oracle-start` -> `--oracle-end` across training;
  they are structurally zero at inference and in supervised data.

## Next steps
- Run the full-scale pipeline on RunPod:
  1. `python src/botzone_ingest.py --input data/data.txt --output data/botzone.jsonl --workers 32`
  2. `python src/imitation.py train-cnn --dataset data/botzone.jsonl --out src/model.h5 --epochs 20 --channels 64 --blocks 6 --hidden-size 512 --batch-size 1024 --device auto --verbose`
  3. PPO league loop with `--finalist-dir data/models --league-dir checkpoints/league --oracle-start 0.7 --oracle-end 0.0`
- Consider sweeping `--credit-gamma`, `--outcome-scale`, and the oracle
  schedule; duplicate-wall + SPRT is the comparison harness.
- If Botzone runtime latency ever matters: cached shanten makes feature
  extraction ~8ms; the full-size net adds ~1ms on CPU.
