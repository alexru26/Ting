# Python Mahjong Agent

This folder contains the Python agent, the local simulator, and the Botzone entry point for Chinese Standard Mahjong.

## Structure

- `state.py` - game state tracking
- `tiles.py` - tile utilities
- `scoring.py` - hand evaluation and fan calculator bridge
- `policy.py` - action selection heuristics used by the simulator and bot
- `bot.py` - Botzone-compatible interface entry point
- `local_game.py` - local round simulator and terminal board view
- `main.py` - minimal runner that starts the bot entry point
- `tests/` - unit tests for state, policy, and local game output

## How the AI works

This agent is a hybrid Mahjong policy with a deterministic, rule-based baseline and optional learned inference.
The default mode remains deterministic from the current game state, while neural mode can load offline-trained checkpoints.

### 1. Input and state reconstruction

The entrypoint is `__main__.py` -> `bot.py`.

For each turn, Botzone provides:

- `requests`: full request history
- `responses`: this bot's past responses

`GameState.from_history(...)` in `state.py` replays the whole sequence into a structured state:

- my hand, melds, discards, flowers
- opponent discards and exposed melds
- seen tile counts
- current request context (`type`, actor, action, relevant tile)

This lets the policy reason from a full reconstructed position instead of only the latest request.

### 2. Goal selection (offense model)

The policy evaluates several hand goals from `scoring.py`:

- `STANDARD`
- `SEVEN_PAIRS`
- `PURE_FLUSH`
- `MIXED_FLUSH`
- `ALL_TRIPLETS`

Each goal has:

- an estimated base fan value
- a goal-specific shanten estimate
- utility:

`utility = base_fan / (shanten + 1)^1.5`

The highest utility goal is selected for the current hand.

### 3. Turn decision flow

The policy in `policy.py` routes decisions by request type.

#### On draw (`type 2`)

1. Check self-draw win (`HU`) via fan calculator wrapper (`can_win`)
2. Check concealed kong (`GANG tile`)
3. Check supplement kong (`BUGANG tile`)
4. Otherwise choose discard based on goal + defense

#### On opponent discard (`type 3 PLAY`)

1. Check win on discard (`HU`)
2. Check open kong (`GANG`)
3. Check `PENG discard_tile` followed by best discard if beneficial
4. Check `CHI mid discard` only if discard came from left player and improves hand
5. Else `PASS`

#### On gang event (`type 3 GANG/BUGANG`)

- Only attempt robbing the kong (`HU`) on `BUGANG`
- Otherwise `PASS`

### 4. Discard scoring (offense + defense)

When discarding, the policy computes:

- offense term: lower shanten after discard is better
- defense term: danger score for the tile against opponent exposed information

Combined score:

`score = offense - DEFENSE_WEIGHT * danger`

where `DEFENSE_WEIGHT` is currently `0.4`.

Danger increases for tiles that look live and potentially useful to opponents, and decreases if an opponent already discarded that tile (safer signal).

### 5. Botzone action safety guard

`bot.py` includes an action legality guard before sending output:

- validates action format against current request type
- checks tile possession for `PLAY`, `PENG`, `CHI`, `GANG`, `BUGANG`
- verifies left-player restriction for `CHI`

If policy output is invalid for the current reconstructed state, the bot falls back to a safe action:

- `PLAY <first tile>` on draw
- `PASS` otherwise

This is designed to prevent Botzone `invalid action` failures.

### 6. Fan evaluation dependency

Winning checks use `MahjongFanCalculator` when available. If unavailable locally, wrapper functions return conservative defaults.

On Botzone, the runtime provides the fan calculator library.

### 7. Current limitations

- No probabilistic hidden-hand inference yet
- No long-horizon search (MCTS/expectimax) yet
- Current learned model is count-based imitation/policy-value, not a deep network
- Defensive model is lightweight and based on exposed public information only

## Local testing

Run a single game and print the final state:

```bash
python local_game.py --games 1 --seed 42
```

Run a single game with the simple terminal board view:

```bash
python local_game.py --games 1 --seed 42 --tui --tui-delay 0 --no-clear
```

Run multiple games for a quick batch summary:

```bash
python local_game.py --games 100
```

## Dataset export for ML

The local simulator can export per-decision trajectories to JSONL for training and analysis.

Export one game:

```bash
python local_game.py --games 1 --seed 42 --export-dataset data/train.jsonl
```

Export a larger deterministic batch:

```bash
python local_game.py --games 200 --seed 42 --export-dataset data/train.jsonl
```

Each line is one decision record with this top-level shape:

```json
{
	"game_id": "game-0",
	"turn_index": 0,
	"player_id": 0,
	"request_type": 2,
	"request_action": "DRAW",
	"action": "PLAY W1",
	"legal_actions": ["PASS", "HU", "PLAY W1"],
	"reward": 0.0,
	"done": false,
	"features": {
		"hand_counts": [0, 0],
		"seen_counts": [0, 0],
		"self_discard_counts": [0, 0],
		"pack_counts": [0, 0],
		"opponent_discard_counts": [[0, 0], [0, 0], [0, 0]],
		"meta": [0, 0, 0, 0, 2, 0, 1, 0]
	},
	"metadata": {}
}
```

Notes:

- `legal_actions` is generated from the same legality model used by runtime checks.
- `features` are deterministic and tile-order stable, suitable for reproducible training runs.
- `meta` encodes context fields in fixed order: my_id, quan, flowers, n_packs, request_type, last_actor, last_action_id, last_tile_idx.
- `reward` and `done` are currently placeholders for later RL phases.

### Supervised imitation training

Train a frequency-based imitation checkpoint from exported trajectories:

```bash
python src/imitation.py train --dataset data/train.jsonl --out data/model.json
```

Train a count-based policy-value checkpoint with multi-head action statistics and value estimation:

```bash
python src/imitation.py train-pv --dataset data/train.jsonl --out data/model_pv.json
```

Evaluate offline masked top-k metrics and calibration (ECE):

```bash
python src/imitation.py eval --dataset data/train.jsonl --model data/model.json --topk 1,3,5
```

Evaluate policy-value checkpoints with masked cross-entropy and value MSE:

```bash
python src/imitation.py eval-pv --dataset data/train.jsonl --model data/model_pv.json --topk 1,3,5
```

Use the trained checkpoint at inference time:

```bash
set TING_POLICY_MODE=neural
set TING_POLICY_MODEL_PATH=data/model.json
```

`TING_POLICY_MODEL_PATH` may point to either checkpoint format (`frequency_lookup_v1` or `count_policy_value_v1`).

`NeuralPolicy` still keeps strict legality masking and falls back to rule policy on any load/inference issue.

## Botzone submission

Use the root-level `__main__.py` as the Botzone entry module. Botzone is launching the upload with `python -m`, so it must find `__main__.py` at the archive root.

1. Make sure `bot.py` runs without local-only files such as `local_game.py`.
2. Keep the Python modules that `bot.py` imports in the same upload bundle, including `state.py`, `tiles.py`, `policy.py`, and `scoring.py`.
3. Do not include the compiled local test extension in the submission bundle; Botzone already provides the Mahjong fan calculator runtime.
4. Zip the folder contents, not the parent folder. The archive root must contain `__main__.py`, `bot.py`, `policy.py`, `scoring.py`, `state.py`, and `tiles.py` directly.
5. If Botzone says it cannot find `__main__`, the zip layout is wrong. Recreate the zip and inspect it before uploading.
6. Test with a few local games first so you know the bot prints valid JSON responses when driven through the Botzone protocol.

## Useful entry points

- `python local_game.py --games 1 --tui` for a visual CLI replay
- `python -m unittest` for the local tests
- `python __main__.py` or `python -m python_agent` for the Botzone-style entry path
- `python bot.py` for the underlying stdin/stdout entry point

## Next steps

1. Improve the discard policy with stronger goal selection and opponent modeling.
