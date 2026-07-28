# Ting Mahjong Agent

A neural Chinese Standard Mahjong bot.

## Botzone Deployment

- Zip only the Python sources (`__main__.py`, `bot.py`, `policy.py`,
  `state.py`, `tiles.py`, `scoring.py`, `features.py`, `model.py`) and
  upload that as the bot; keep it under the 4 MB zip limit.
- Upload `model.h5` through Botzone's **Manage Storage** (~268 MB quota);
  the bot finds it at the relative `data/model.h5` path first, then falls
  back to a copy next to the sources (see `policy.candidate_model_paths`).
- The bot answers in long-running mode (`>>>BOTZONE_REQUEST_KEEP_RUNNING<<<`),
  so the torch import and checkpoint load happen once per game instead of
  once per turn; it degrades transparently to the classic
  restart-every-turn mode if the platform ignores the marker.

## Runtime Decision Flow

1. [src/bot.py](src/bot.py) reads the JSON payload and rebuilds
   [src/state.py](src/state.py) `GameState` from the full request/response history.
2. `GameState.enumerate_legal_actions()` enumerates exactly the moves the judge
   will accept: HU only when the hand actually wins (>= 8 fan), meld discards
   that account for tiles consumed by the meld, claims only on other players'
   discards, BUGANG only with the fourth tile in hand, and no PASS on draw
   turns. Legality is guaranteed by construction, so the bot can never emit an
   INVALID move.
3. Turns with a single legal action are answered directly without model
   inference. For real decisions, [src/features.py](src/features.py) converts the
   state into the schema v4 features and [src/model.py](src/model.py) scores the
   legal actions; [src/policy.py](src/policy.py) picks the argmax deterministically.
4. Failures are explicit: a missing or incompatible checkpoint, malformed
   input, or an illegal model output raises instead of silently degrading.

In short: state reconstruction -> legal action enumeration -> feature
extraction -> masked model scoring -> action string.

## Model Input Contract (feature schema v5)

Inputs are split into tile planes and a meta vector, defined in
[src/features.py](src/features.py) and enforced at checkpoint load time:

- Tile planes (21 x 34, supervised values on a lossless 0.25 grid):
  - 4 hand-count threshold planes (>=1, >=2, >=3, >=4 copies).
  - 1 own-meld plane and 3 opponent-meld planes (relative seat order),
    with melds expanded to per-tile counts (PENG=3, GANG=4, CHI=1 each).
  - 1 own-discard plane and 3 opponent-discard planes (relative seat order).
  - 4 unseen-count threshold planes (remaining copies from our perspective).
  - 1 one-hot plane for the current event tile (drawn or claimed).
  - 1 plane marking tiles that reduce shanten (acceptance set).
  - 3 oracle planes (opponent hands by relative seat): populated only during
    oracle-guided RL in the simulator and annealed to zero (Suphx-style
    curriculum); always zero in supervised data and at inference.
- Meta vector (31 values): seat and prevalent-wind one-hots, decision phase
  (draw / discard-response / bugang-response), relative last actor, last
  request action, plus normalized scalars for flowers, meld count, game
  progress, shanten, tenpai flag, acceptance count, and the can-win / win-fan
  signal.

The expensive fan-conditioned lookahead features from schema v3 (per-action
wait-fan profiles) were removed: they cost hundreds of fan-calculator calls
per decision and duplicated what the value head should learn.

## Model Architecture

[src/model.py](src/model.py) defines a residual policy-value network:

- The 18x34 planes are reshaped onto a 4x9 grid (suit rows + honor row) and
  processed by a Conv2d stem plus residual blocks, then fused with the meta
  vector into a shared hidden state.
- Heads: action-family logits (PASS/HU/GANG/PLAY/BUGANG/PENG/CHI),
  family-conditioned argument logits (2 x 35), a scalar value head, and an
  auxiliary win head (predicts a positive-score outcome) that teaches the
  trunk to separate strong actions from merely legal ones.
- An action is scored as `family + arg1 + arg2` logits, and the policy is a
  softmax over the legal actions only. Training uses this same masked scoring,
  so the training objective matches the runtime decision rule exactly.
- One shared trunk with per-family conditioned heads gives per-action-family
  specialisation while keeping a single checkpoint.

Checkpoints are HDF5 files with the architecture and feature contract stored
as attributes; loading verifies both strictly and raises on any mismatch.
Large weight tensors are stored as per-channel symmetric int8 (small ones as
float16) and dequantized to float32 on load, keeping the file well under
Botzone's 4 MB zip limit; `save()` raises if the file would exceed the
`MODEL_FILE_BYTE_LIMIT` budget.

## Training

### 1. Training data

The primary supervised source is the Botzone match export in `data/data.txt`
(98,209 real rounds; `data/sample.txt` is a 16-round preview).
[src/botzone_ingest.py](src/botzone_ingest.py) replays every round, reconstructs
each player's information state, and emits trajectory JSONL in the same
schema as local self-play - including claim decisions, ignored HU/GANG
declarations, per-player final-score rewards, and `steps_from_end` for
credit decay:

```bash
python src/botzone_ingest.py --input data/data.txt --output data/botzone.jsonl --workers 12 --verbose
```

Local self-play data from the rule-based teacher remains available as a
secondary source:

```bash
python src/local_game.py --games 1000 --seed 1 --export-dataset data/local_data.jsonl
```

### 2. Supervised training

[src/imitation.py](src/imitation.py) trains from compact pre-encoded caches
(auto-built, quantized uint8 planes). `--dataset` may be repeated with
per-source weights to control the data mix; splits are by match so
near-duplicate states never leak across train/validation:

```bash
python src/imitation.py train-cnn \
    --dataset data/botzone.jsonl:1.0 --dataset data/local_data.jsonl:0.3 \
    --out src/model.h5 --epochs 16 --channels 128 --blocks 12 --hidden-size 512 \
    --batch-size 1024 --learning-rate 0.0005 --device auto --verbose
```

The objective is outcome-weighted masked legal-action cross-entropy
(fan-backward credit via `--credit-gamma`, winning trajectories up-weighted
via `--outcome-scale`, drawn games down-weighted via `--draw-weight`), plus
a value loss on decayed returns and an auxiliary win-prediction loss
(`--win-weight`). Forced turns contribute zero policy loss automatically.
`eval-cnn` reports top-k masked accuracy, masked CE, value MSE, win-head
accuracy, and ECE calibration, split by decision vs forced states:

```bash
python src/imitation.py eval-cnn --dataset data/botzone.jsonl --model src/model.h5
```

### 3. Reinforcement learning and self-play

[src/rl_self_play.py](src/rl_self_play.py) fine-tunes a checkpoint with batched
PPO updates against a sampled opponent league: the rule-based baseline, the
27 frozen IJCAI finalist imitation checkpoints in `data/models/`
(driven through [src/finalist_opponents.py](src/finalist_opponents.py); they are
evaluation/league opponents, never training labels), historical h5
checkpoints, and mirror copies of the current candidate:

```bash
python src/rl_self_play.py ppo-train --model src/model.h5 --games 512 \
    --eval-games 64 --update-every 8 --device auto \
    --finalist-dir data/models --finalist-prob 0.35 \
    --league-dir checkpoints/league --self-play-prob 0.2 \
    --oracle-start 0.7 --oracle-end 0.0 --credit-gamma 0.97 \
    --learning-rate 0.00003 --target-kl 0.02 --snapshot-every 200
```

PPO fine-tuning should use a much smaller learning rate than supervised
training (`--learning-rate`, applied on top of the loaded checkpoint) and a
KL guard (`--target-kl`) that stops update epochs early when the policy
drifts too far from the data-collection policy; `--snapshot-every` writes a
rolling `.snapshot.h5` so long runs are recoverable. `finalist-eval` plays
the candidate against tables of three finalists with the candidate seat
rotating, which is the primary strength metric:

```bash
python src/rl_self_play.py finalist-eval --model src/model.h5 \
    --finalist-dir data/models --games 256 --verbose
```

Returns use Tjong-style fan-backward decay, advantages subtract the value
head's prediction (Suphx-style learned baseline) and are normalized, and the
oracle curriculum anneals perfect-information planes to zero over training.
Promotion requires both the baseline gate and a paired duplicate-wall gate
(SPRT + score-aware, [src/model_governance.py](src/model_governance.py));
promoted candidates are snapshotted into `--league-dir` so later runs face
them as opponents. `ppo-eval` and `duplicate-wall` remain available for
standalone evaluation.

## Local Testing

Run a single game and print the final state:

```bash
python src/local_game.py --games 1 --seed 42
```

Pipe a JSON payload through the bot:

```bash
echo '{"requests": ["0 0 0", "1 0 0 0 0 W1 W2 W3 B4 B5 B6 T7 T8 T9 J1 J1 F1 F2", "2 W4"], "responses": ["PASS", "PASS"]}' | python src/bot.py
```

Run the full test suite:

```bash
python -m unittest -q
```
