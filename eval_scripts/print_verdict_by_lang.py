import jsonlines
from tqdm import tqdm
from collections import Counter, defaultdict

stat = defaultdict(Counter)

with jsonlines.open("api_aux_test_submission_java-evaluated.jsonl") as jrp:
    for sample in tqdm(jrp):
        verdict = "PASSED"
        for ut in sample["unittests"]:
            if ut["exec_outcome"] != "PASSED":
                verdict = ut["exec_outcome"]
                break
        stat[sample["lang_cluster"]][f"{sample['exec_outcome']}-{verdict}"] += 1

import json
print(json.dumps(stat, indent=4))
    