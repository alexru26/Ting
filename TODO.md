# TODO
## Task 1
The current runtime model file is about 7 MB (`src/model.h5`), which already exceeds Botzone's 4 MB zip limit before counting bot code. The other files packed into the zip only take around 30 KB, and the main issue is the model file. Address this issue. It might help to force the model file to be less than a certain limit or save the model in another file type. Do not have the final model fallback to the old rule policy at all.

## Task 2
Fully train a new model by following these steps. First, ssh into the runpod:

```bash
ssh hb4mp6onn4r2fc-64411f15@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Upon ssh-ing into the machine, you will automatically be in the Ting directory. There is no need to run git pull or git lfs pull. Furthermore, the botzone txt data has already been ingested and pre-encoded at /workspace/persistent. You can immediately start supervised training and PPO training. Evaluate and train the model until it performs well. Make sure to keep checkpoints and try different parameters if necessary. Don't train the model off of mainly the rule policy, since it's not very good. The final model should consistently do well against the finalists. If the model is unable to do well, find places for improvement across the entire codebase. You can edit the repo locally, push the changes onto a new branch, then use that branch in the runpod. Overall, keep on iterating and improving on the project until the trained model is satisfactory. Once you are done training the model, run this in the runpod:

```bash
runpodctl send PATH_TO_MODEL
```

It should output a code. Then accept the file locally with:

```bash
runpodctl receive GENERATED_RUNPOD_CODE
```

Make sure the model is stored as src/model.h5 and push it to a new branch.