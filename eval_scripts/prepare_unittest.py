import json
import jsonlines
from tqdm import tqdm

uts = {}

with jsonlines.open("api_aux_test_submission_java.jsonl") as jrp:
    for sample in tqdm(jrp):
        s = sample["hidden_unit_tests"].replace("'", "\"")
        _uts = json.loads(s)
        if uts.get(sample['src_uid']) is not None:
            assert len(uts[sample['src_uid']]) == len(_uts), f"{len(uts[sample['src_uid']])}, {len(_uts)}"
        uts[sample['src_uid']] = _uts

with open("test_unittest_db.json", "w") as wp:
    json.dump(uts, wp)
    