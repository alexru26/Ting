# Ting Mahjong Agent

## Architecture

### Runtime decision flow

## Training And Evaluation

### 1. Data generation

```bash
python src/local_game.py --games 100 --export-dataset data/DATALOCATION.jsonl
```

```bash
python src/local_game.py --games 100 --opponent-registry data/OPPONENTS.json --random-opponents --export-dataset data/DATA.jsonl
```

### 2. Supervised pretraining

```bash
python src/imitation.py preencode-cnn --dataset data/DATA.jsonl --output data/DATA.preencoded.npz --decision-only --device cpu
```

```bash
python src/imitation.py train-cnn --dataset data/DATA.jsonl --cache data/DATA.preencoded.npz --out src/MODEL.h5 --epochs 10 --hidden-size 64 --batch-size 2048 --device auto --verbose
```

### 3. Reinforcement learning and self-play

```bash
python src/rl_self_play.py ppo-train --model src/MODEL.h5 --games 100 --eval-games 10 --device auto --opponent-registry data/OPPONENTS.json
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
