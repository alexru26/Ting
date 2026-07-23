## TODO

1. Remove the old rule-based fallback path.
- Make the neural model the only runtime decision source.
- Expose clear logging/metrics so it is obvious how often each decision comes from the model versus any non-neural path before the fallback is removed.

2. Rework the feature set fed into the model.
- Review which state, history, opponent, and action-context features are actually useful.
- Remove noisy, redundant, or hard-to-maintain features.
- Update the model input contract and training pipeline together so they stay aligned.

3. Debug the occasional invalid action produced at runtime.
- Reproduce the Botzone-side INVALID move case.
- Trace the full path from feature extraction to action decoding and legality checks.
- Fix any mismatch between encoded actions, legal-action filtering, and the final emitted move.

4. Remove redundant and legacy code paths.
- Delete dataset-generation code that is no longer needed for external game ingestion.
- Simplify the training/runtime surface area where duplicate logic exists.

5. Remove the `ml_packages.py` dependency layer.
- Require the needed packages directly.
- Fail fast if dependencies are missing instead of masking the problem with a package-profile abstraction.

6. Reduce fallback behavior and tighten execution constraints.
- Prefer explicit failures over silent recovery when inputs, models, or dependencies are invalid.
- Keep runtime behavior deterministic and constrained.

7. Optimize training for learning efficiency.
- Improve data encoding, batching, caching, and evaluation flow.
- Remove unnecessary passes, re-reads, and duplicated tensor work.
- Preserve the strongest learning signal while keeping training fast and memory-efficient.

8. Consider multiple models.
- Consider the possibility of using different models for different actions (DISCARD, CHII, PENG, GANG, etc.)

## Additional Context
- The relevant src files will be bundled into a zip for Botzone. That is the ultimate point of the project.
- This TODO are all the things that are worth thinking a lot about. There may be more things that come up that are worth addressing. If so, do so. This TODO only serves as a rough outline for what I am thinking about right now.
- To ensure rigor in this project, use plenty of tests. Also manually verify that things look right in terminal in addition to the unit tests. 
- The final model will be trained on a pod via RunPod, so there are not any major resource constraints. 