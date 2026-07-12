# Ting Mahjong Agent

Ting is a Chinese Standard Mahjong bot with a neural-first runtime and a deterministic rule-based fallback. The codebase is organized so the Botzone entry point stays safe and lightweight, while training, evaluation, and governance live in separate modules.

## Architecture

The runtime path starts in [src/__main__.py](src/__main__.py), which hands control to [src/bot.py](src/bot.py). The bot reconstructs the full game state from the Botzone request/response history using [src/state.py](src/state.py), then selects a policy through [src/policy.py](src/policy.py).

The AI has three layers:

- State and legality layer: [src/state.py](src/state.py) and [src/bot.py](src/bot.py) rebuild the position, enumerate legal actions, and enforce a final legality firewall before any response is emitted.
- Decision layer: [src/policy.py](src/policy.py) contains the default goal-based Mahjong policy, the neural policy wrapper, the PPO rollout policy used by self-play, and the bounded search planner hook.
- Model and governance layer: [src/imitation.py](src/imitation.py), [src/rl_self_play.py](src/rl_self_play.py), and [src/model_governance.py](src/model_governance.py) handle training, evaluation, promotion gating, registry metadata, and duplicate-wall comparisons.

### Runtime decision flow

At inference time the bot follows this sequence:

1. Reconstruct the current state from the full history.
2. Use the neural policy by default and auto-discover the bundled model artifact under `src/`.
3. Ask the selected policy for an action.
4. Verify the action against the current request context.
5. Fall back to a safe rule-based move if the model output is invalid or unavailable.

Neural inference is the default path. The legacy rule policy remains as a strict safety fallback whenever model loading fails, inference fails, uncertainty handling requests fallback, or the emitted action is not legal. Search-time augmentation is optional and budgeted so the Botzone runtime remains safe.

### Policy components

- [src/policy.py](src/policy.py): the main rule-based Mahjong policy, including discard scoring, kong/peng/chi decisions, neural fallback, uncertainty handling, and search-time hooks.
- [src/model.py](src/model.py): the PyTorch CNN policy-value model used for learned inference.
- [src/search_planner.py](src/search_planner.py): bounded rollout planning over candidate actions with a hard runtime budget.
- [src/model_governance.py](src/model_governance.py): duplicate-wall paired evaluation, Elo ladder helpers, SPRT-style promotion gating, and model registry records.

## Training And Evaluation

Training is split into stages so the runtime bot never depends on training-only code.

### 1. Data generation

[src/local_game.py](src/local_game.py) runs local Mahjong games and can export JSONL trajectories for supervised or RL training. Each record contains the request context, legal actions, chosen action, reward placeholder, and deterministic feature vector.

Example:

```bash
python src/local_game.py --games 200 --seed 42 --export-dataset data/train.jsonl
```

### 2. Supervised pretraining

[src/imitation.py](src/imitation.py) is now neural-only and supports the CNN checkpoint family backed by PyTorch.

A checkpoint family is the storage/behavior contract for a model artifact: how it is serialized, what heads/statistics it contains, and which inference code path can load it. The repository now keeps only one active family for deployment and training: CNN policy-value checkpoints.

Example commands:

```bash
python src/imitation.py train-cnn --dataset data/train.jsonl
```

The CNN is the deployable deep model path. Relative output paths are normalized under `src/` so the model is easy to package with the bot.

### 3. Reinforcement learning and self-play

[src/rl_self_play.py](src/rl_self_play.py) implements the self-play worker, reward shaping, PPO fine-tuning, baseline evaluation, and the duplicate-wall evaluation mode.

Example commands:

```bash
python src/rl_self_play.py self-play --games 100 --seed 42
python src/rl_self_play.py ppo-train --model src/model.h5 --games 32 --eval-games 16
python src/rl_self_play.py ppo-eval --model src/model.h5 --games 16
python src/rl_self_play.py duplicate-wall --model src/model.h5 --games 16
```

### 4. Governance and promotion

The current evaluation layer focuses on reproducibility and low-variance comparisons:

- duplicate-wall paired evaluation to reduce randomness
- Elo ladder helpers for relative checkpoint tracking
- SPRT-like promotion gate logic for candidate acceptance
- model registry entries with version, checksum, training corpus, and metrics
- regression coverage for legality, deterministic replay, and fallback consistency

In this repository, governance means model lifecycle controls that make evaluation and deployment reproducible and auditable (registry metadata, checksums, deterministic paired evaluation, and regression safeguards). Promotion means deciding whether a candidate checkpoint replaces the current baseline, based on measured performance gates instead of manual intuition.

## Model Files

The repository keeps deployable model artifacts inside `src/` so Botzone packaging stays simple. The default CNN artifact path is [src/model.h5](src/model.h5).

The model loader supports CNN checkpoints (HDF5) for neural policy runtime.

## Environment And Runtime Flags

The bot does not require environment variables on Botzone. By default it runs neural-first and attempts to load the bundled model artifact from `src/model.h5`.

Environment variables are optional local overrides:

- `TING_POLICY_MODE`: `rule`, `neural`, or another supported policy mode
- `TING_POLICY_MODEL_PATH`: path to the checkpoint to load
- `TING_POLICY_RISK_MODE`: adaptation mode used by the neural policy wrapper
- `TING_POLICY_TEMPERATURE`: sampling temperature for neural action selection
- `TING_POLICY_ENABLE_SEARCH`: enables bounded search-time planning
- `TING_POLICY_SEARCH_DISABLE`: hard disables search planning
- `TING_POLICY_BELIEF_WEIGHT`: weight for the belief-related features in the neural stack

The bot always keeps the legality firewall in [src/bot.py](src/bot.py), so a bad model output does not break protocol correctness.

## Repository Layout

- [src/state.py](src/state.py) - request replay and game-state reconstruction
- [src/bot.py](src/bot.py) - Botzone I/O adapter and legality firewall
- [src/policy.py](src/policy.py) - rule-based policy, neural wrapper, and search hooks
- [src/features.py](src/features.py) - deterministic feature extraction
- [src/action_codec.py](src/action_codec.py) - action vocabulary and encode/decode helpers
- [src/dataset.py](src/dataset.py) - JSONL trajectory schema, reader, and writer
- [src/local_game.py](src/local_game.py) - local simulator and dataset export
- [src/imitation.py](src/imitation.py) - supervised training and offline evaluation
- [src/rl_self_play.py](src/rl_self_play.py) - PPO/self-play and candidate evaluation
- [src/model_governance.py](src/model_governance.py) - registry metadata, Elo, and promotion gating
- [tests/](tests/) - unit and integration coverage for the runtime, training, and governance layers

## Local Testing

Run a single game and print the final state:

```bash
python src/local_game.py --games 1 --seed 42
```

Run the full test suite:

```bash
python -m unittest -q
```
