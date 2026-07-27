# TODO
## Task 1
I zip up the following files: \_\_main\_\_.py, bot.py, policy.py, state.py, tiles.py, scoring.py, features.py, model.py, model.h5. But when I run this on Botzone, the following error occurs:

{"debug":"botzone inception log:\n1 line read from stdin\n1 line read from stdin\n...done\nstdin sent to child, begin reading output\nrequested force close stdin...\nset alarm: 11 second(s).\n1 line read from child\nnormal exited (raw = 256) with status = 1\nverdict = RE\n-- RE occurred --\nworking dir = /var/sandbox/box0/\nhostname:\n","error":"错误：程序崩溃 / ERROR: Program raised a runtime error","output":"stderr:\nstart to close fds...\nTraceback (most recent call last):\n  File \"/usr/lib/python3.6/runpy.py\", line 193, in _run_module_as_main\n    \"\_\_main\_\_\", mod_spec)\n  File \"/usr/lib/python3.6/runpy.py\", line 85, in _run_code\n    exec(code, run_globals)\n  File \"6a67a9dc27e7bf01db05d8ea.py36/\_\_main\_\_.py\", line 6, in &lt;module&gt;\n  File \"6a67a9dc27e7bf01db05d8ea.py36/bot.py\", line 42, in run\n  File \"6a67a9dc27e7bf01db05d8ea.py36/bot.py\", line 37, in handle_input\n  File \"6a67a9dc27e7bf01db05d8ea.py36/policy.py\", line 73, in create_policy\n  File \"6a67a9dc27e7bf01db05d8ea.py36/policy.py\", line 43, in \_\_init\_\_\n  File \"6a67a9dc27e7bf01db05d8ea.py36/policy.py\", line 36, in load_model\n  File \"6a67a9dc27e7bf01db05d8ea.py36/model.py\", line 785, in load\nFileNotFoundError: Model checkpoint not found: /6a67a9dc27e7bf01db05d8ea.py36/model.h5\n\n\nstdout:\n/6a67a9dc27e7bf01db05d8ea.py36/model.h5\n"}

It seems that it cannot parse the path even though the zip contains model.h5. 

## Task 2
Read through the official documentation about Bots on Botzone: https://wiki.botzone.org.cn/index.php?title=Bot/en. Note that it specifies that you can upload data to a data path via "Manage Storage." You get around 268 MB of storage. In other words, it might be better to store the model into the data path. Look through the documentation for any other relevant information.

## Task 3
Double check the codebase to make sure it matches the game on Botzone: https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong/en.

Then, thoroughly research other prominent Mahjong ML models and consider how to improve the model. There is still a lot of room for improvement. Note that the current model.h5 in the repository is still considerably under the 4 mb limit. After the improvements, make sure everything works as expected. All changes should be in a new branch. Then, fully train a model from start to end through runpod. The command to ssh into the runpod is as follows:

```bash
ssh 8o5n5v7yas57ks-64411fe1@ssh.runpod.io -i ~/.ssh/id_ed25519
```

When on the runpod, be wary of the fact that you might suddenly get disconnected from it. Do not stop to ask the user at any moment. Just continue to work through everything until you are done and you push all the changes onto github.