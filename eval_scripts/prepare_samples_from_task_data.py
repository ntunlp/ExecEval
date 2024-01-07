import datasets 
from tqdm import tqdm
import jsonlines
from copy import deepcopy

cfg = "apr"

dataset = datasets.load_dataset("NTU-NLP-sg/xCodeEval", cfg)

with jsonlines.open(f"{cfg}_code_samples_heldout.jsonl", "w") as jwp:
    def append_samples(dts):
        for dt in tqdm(dts):
            sample = deepcopy(dt)
            sample["source_code"] = sample["bug_source_code"]
            sample["exec_outcome"] = sample["bug_exec_outcome"]
            sample["task_id"] = f'{sample["apr_id"]}-bug'
            jwp.write(sample)
            if sample["fix_source_code"]:
                sample = deepcopy(dt)
                sample["source_code"] = sample["fix_source_code"]
                sample["exec_outcome"] = sample["fix_exec_outcome"]
                sample["task_id"] = f'{sample["apr_id"]}-fix'
                jwp.write(sample)

    # append_samples(dataset["train"])
    append_samples(dataset["validation"])
    append_samples(dataset["test"])