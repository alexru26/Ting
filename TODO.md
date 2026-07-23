# TODO

## Training data
- Ingest the Botzone-exported records in `data/` as a supervised training source, alongside the existing local self-play trajectories.
- Normalize every record into the same trajectory schema used by the current imitation pipeline so Botzone logs, local games, and finalist logs can be mixed safely.
- Split training and validation by match or replay seed, not by individual turn, to avoid leaking near-duplicate states across splits.
- Add record filtering and weighting so the model does not learn all actions equally:
	- down-weight forced or low-information turns,
	- up-weight decisive turns near wins,
	- up-weight trajectories with stronger final score or fan outcomes,
	- keep a separate signal for good actions versus merely legal actions.
- Treat the 27 fused Torch imitation checkpoints from the IJCAI 2026 Mahjong Competition as frozen evaluation opponents, not as training labels.

## Supervised learning
- Extend imitation training to support weighted loss, so winning or high-scoring trajectories contribute more than low-value or noisy ones.
- Add an auxiliary quality/value target so the model learns to distinguish strong actions from weak but legal actions.
- Keep the current masked-action objective, but make the mask-aware weighting outcome-sensitive.
- Add data-mix controls for Botzone logs, local self-play, and finalist corpora so the training recipe can be tuned instead of hard-coded.
- Track separate metrics for decision states, forced states, top-k masked accuracy, and calibration.

## Reinforcement learning
- Expand the opponent league to sample from:
	- the rule-based baseline,
	- the 27 external imitation-policy networks,
	- previous RL checkpoints,
	- the current candidate.
- Keep previous checkpoints in the league even when a newer checkpoint becomes the active candidate.
- Use historical checkpoints as training opponents so the policy does not overfit a single baseline style.
- Make RL promotion depend on duplicate-wall evaluation and score-aware gates, not just raw win rate.
- Continue shaping rewards with score delta, fan, and placement proxy, but keep the shaping explicit and tunable.

## Research notes to incorporate
- Tjong paper: the "fan backward" idea propagates credit backward from a winning hand to the earlier actions that built it, instead of only rewarding the terminal win. This is useful for sparse Mahjong rewards because it gives earlier decisions nonzero learning signal.
- Suphx: global reward prediction trains a baseline model to predict the final round or match outcome, then uses that prediction as a variance-reducing baseline in policy-gradient updates. This is standard actor-critic logic, but it is especially important in Mahjong because draw luck makes raw outcomes very noisy.
- Suphx: oracle guiding trains with perfect-information features early in RL, then gradually masks those oracle features away so the policy learns from information it will actually have at inference time.
- Use oracle-guided training as a curriculum: strong early gradient signal, then anneal toward deployable partial-information inputs.

## Next implementation steps
- Add a Botzone-log loader and a unified trajectory merger.
- Add weighted imitation loss and a value-or-quality auxiliary head.
- Add opponent sampling over finalist models and historical checkpoints.
- Add a reward-backpropagation pass for fan-based credit assignment.
- Add a global reward predictor baseline for PPO-style updates.
- Add an oracle-feature masking curriculum for RL warm-starting.

