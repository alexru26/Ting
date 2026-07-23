# TODO

All items from the previous TODO were addressed on the `rework` branch:

1. **Remove the old rule-based fallback path** - DONE. The neural model is
   the only runtime decision source ([src/policy.py](src/policy.py)); the rule
   policy now lives in [src/rule_policy.py](src/rule_policy.py) and is used only
   as the dataset-generation teacher and evaluation baseline. Set
   `TING_LOG_DECISIONS=1` to log each decision's source (`model` vs `forced`
   single-legal-action turns).
2. **Rework the feature set** - DONE. Feature schema v4 (18 tile planes + 31
   meta values, see README) replaces the fan-conditioned lookahead features;
   the model input contract is enforced at checkpoint load.
3. **Debug the invalid action at runtime** - DONE. Root causes were in
   `enumerate_legal_actions`: PASS offered on draw turns, HU offered without a
   winning hand, PENG/CHI discards that were consumed by the meld itself,
   BUGANG without the fourth tile, and claims offered on our own discards.
   All fixed with regression tests in [tests/test_state.py](tests/test_state.py).
4. **Remove redundant and legacy code** - DONE. Deleted external ingestion,
   package-profile layer, search planner, action codec, split manifests, and
   the unusable `.pkl` opponent-registry path (those checkpoints came from a
   different codebase and silently fell back to rule-based play).
5. **Remove ml_packages.py** - DONE. Direct imports; missing dependencies
   fail at import time.
6. **Reduce fallback behavior** - DONE. Strict checkpoint loading (schema +
   shape verified), no exception swallowing at runtime, deterministic greedy
   action selection.
7. **Optimize training** - DONE. Single-pass pre-encoding into a quantized
   uint8 cache (auto-built/auto-refreshed), one batched training path with
   masked legal-action cross-entropy, batched evaluation, batched PPO updates,
   and dataset rewards backfilled from final scores (they were always 0
   before, so the value head had no signal).
8. **Consider multiple models per action** - DECIDED. One shared trunk with
   family-conditioned argument heads scores each action family with its own
   conditioned head while keeping the single checkpoint Botzone needs; see
   README "Model Architecture".

## Next steps

- Regenerate a large dataset (`local_game.py --games 5000+`) and train the
  full-size model on RunPod (`--channels 64 --blocks 6 --hidden-size 512`).
- PPO league: keep promoted checkpoints and pass them via `--opponents` for
  self-play diversity.
- `data/models/*.pkl` and `data/opponents_registry.json` are legacy artifacts
  from another codebase and are no longer referenced; delete them when
  convenient.
