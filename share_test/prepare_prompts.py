# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

import json, random

with open("ShareGPT_V3_unfiltered_cleaned_split.json") as f:
    data = json.load(f)

prompts = []
for x in data[:1000]:
    conv = x["conversations"]
    user_msgs = [c["value"] for c in conv if c["from"] == "human"]
    if user_msgs:
        prompts.append(user_msgs[0])

with open("prompts.txt", "w") as f:
    for p in prompts:
        f.write(p.replace("\n", " ") + "\n")