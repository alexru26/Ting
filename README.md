# Ting Mahjong Agent

## Model Architecture

The live policy is a hybrid stack: [src/state.py](src/state.py) reconstructs the full game state from request/response history, [src/features.py](src/features.py) converts that state into a deterministic feature dict, [src/model.py](src/model.py) turns the features into action scores and value estimates, and [src/runtime_model.py](src/runtime_model.py) wraps checkpoint loading plus inference. [src/policy.py](src/policy.py) chooses between the neural model, a rule-based fallback, and optional search.

Model inputs are split into a tile tensor and a meta vector:

- Tile tensor: 11 channels over the 34 tile types.
- Channel 1: normalized hand counts.
- Channel 2: normalized seen-tile counts.
- Channel 3: normalized self-discard counts.
- Channel 4: normalized pack counts.
- Channels 5-7: normalized discard counts for up to three opponents.
- Channel 8: raw hand counts.
- Channel 9: raw seen-tile counts.
- Channel 10: hand shanten, broadcast across all tile slots.
- Channel 11: acceptancy, broadcast across all tile slots.

The meta vector is 70 values and carries the rest of the context:

- 8 legacy state scalars from the reconstructed game state.
- Request context: request type, seat, target-player flag.
- One-hot encodings for the event action and raw request action.
- State quality signals: schema version, shanten, acceptancy, efficiency deltas, immediate hu availability, fan estimates, and tenpai profile.
- Action-conditioned fan deltas for PASS, HU, GANG, PLAY, BUGANG, PENG, and CHI.
- Opponent temporal summaries for up to three opponents: history length, pack count, honor ratios, and suit ratios.

The network produces these outputs:

- Family logits over PASS, HU, GANG, PLAY, BUGANG, PENG, and CHI.
- Two conditioned argument heads for tile selection and discard selection.
- Belief logits and belief probabilities over the remaining-tile distribution.
- A main value head, an auxiliary value head, and an efficiency-bonus head.

At runtime, [src/runtime_model.py](src/runtime_model.py) converts those outputs into either a chosen action string or a probability distribution over legal actions, and [src/bot.py](src/bot.py) validates the final action before returning it.

## Runtime Decision Flow

1. [src/bot.py](src/bot.py) reads the input payload, rebuilds [src/state.py](src/state.py) `GameState` from the full history, and rejects malformed turns with a safe `PASS`.
2. [src/policy.py](src/policy.py) selects the active policy. The default is the neural policy, but rule-based fallback remains available and the model path can be overridden with environment variables.
3. The policy calls [src/features.py](src/features.py) to extract the tile channels and meta features for the current seat.
4. [src/runtime_model.py](src/runtime_model.py) loads [src/model.py](src/model.py), scores the legal actions, and applies the configured temperature, belief weight, and efficiency weight.
5. If search is enabled, [src/search_planner.py](src/search_planner.py) can replace the raw model choice with a bounded rollout action.
6. [src/bot.py](src/bot.py) validates legality one last time and falls back to `PASS` or a simple discard if the model output is unusable.

In short, the runtime path is state reconstruction -> feature extraction -> model scoring -> legality check -> action string.

## Training And Evaluation

### 1. Data generation

Training data is generated locally by [src/local_game.py](src/local_game.py), which simulates full 4-player rounds and can export trajectory records through [src/dataset.py](src/dataset.py). Each JSONL record stores the game id, turn index, player id, request type, request action, chosen action, legal actions, reward, extracted features, and metadata.

To locally generate dataset of opponent models:

```bash
python src/local_game.py --games 100 --opponent-registry data/OPPONENTS.json --random-opponents --export-dataset data/DATA.jsonl
```

### 2. Supervised pretraining

The supervised pipeline has three stages:

- Pre-encode the JSONL dataset with `preencode-cnn` to cache tensors in `.npz` form for faster training.
- Train the CNN with `train-cnn`, which learns from the action family plus argument targets, the value targets, and the belief target derived from seen tiles.
- Evaluate the checkpoint with `eval-cnn`, which reports masked cross-entropy and top-k metrics on the same trajectory format.

To pre-encode the dataset for faster training:

```bash
python src/imitation.py preencode-cnn --dataset data/DATA.jsonl --output data/DATA.preencoded.npz --device cpu
```

To train on the pre-encoded dataset:

```bash
python src/imitation.py train-cnn --dataset data/DATA.jsonl --cache data/DATA.preencoded.npz --out src/MODEL.h5 --epochs 10 --hidden-size 64 --batch-size 2048 --device auto --verbose
```

### 3. Reinforcement learning and self-play

Reinforcement learning and self-play live in [src/rl_self_play.py](src/rl_self_play.py): `ppo-train` fine-tunes a checkpoint against rule-based and registry-backed opponents, while `ppo-eval` measures the candidate against baseline opponents. The reward signal combines score delta, winner fan bonus, and a placement proxy.

```bash
python src/rl_self_play.py ppo-train --model src/MODEL.h5 --games 100 --eval-games 10 --device cpu --opponent-registry data/OPPONENTS.json
```

## Local Testing

Run a single game and print the final state:

```bash
python src/local_game.py --games 1 --seed 42
```

Run the full test suite:

```bash
python -m unittest -q
```
