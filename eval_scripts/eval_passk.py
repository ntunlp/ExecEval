import itertools
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
import jsonlines
import numpy as np
import tqdm

sys.path.extend(
    [Path(__file__).parent.parent, Path(__file__).parent.parent / "execution_engine"]
)
# exit(0)
# sys.path.extend([
from api_comm import APICommunication
from exec_outcome import ExecOutcome
from yaml import safe_load


def estimate_pass_at_k(
    num_samples: int | list[int] | np.ndarray,
    num_correct: list[int] | np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def evaluate_functional_correctness(
    sample_file: str,
    k: list[int] = [1, 10, 100],
    n_workers: int = 4,
    limits_by_lang: dict = {},
    compile_n_execute_args_by_lang: dict = {},
    eval_result_file: str | None = None,
    unittest_file: str = "unittest_db.json",
    execeval_url: str = "http://localhost:5000",
    block_network: bool = True,
    stop_on_first_fail: bool = True,
    use_sanitizer: bool = False,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    if eval_result_file is None:
        eval_result_file = f"{sample_file.split('.')[0]}-evaluated.jsonl"

    with open(unittest_file) as ut_rp:
        unittest_db = json.load(ut_rp)
    # Check the generated samples against test suites.

    with APICommunication(execeval_url) as execeval:
        execute_code = execeval.execute_code
        supported_langs = {r["runtime_name"] for r in execeval.get_runtimes()}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            results = defaultdict(list)
            with jsonlines.open(sample_file) as sample_rp:
                for idx, sample in tqdm.tqdm(
                    enumerate(sample_rp), desc="Reading samples"
                ):
                    src_uid = sample["src_uid"]
                    source_code = sample["source_code"]
                    task_id = sample["task_id"]
                    lang = sample["lang"]
                    if src_uid not in unittest_db:
                        continue
                    unittests = unittest_db[src_uid]
                    if len(unittests) == 0:
                        continue
                    if lang not in supported_langs:
                        continue

                    args = (
                        lang,
                        source_code,
                        unittests,
                        limits_by_lang[lang],
                        block_network,
                        stop_on_first_fail,
                        use_sanitizer,
                        compile_n_execute_args_by_lang.get(lang, {}).get("compile_cmd"),
                        compile_n_execute_args_by_lang.get(lang, {}).get(
                            "compile_flags"
                        ),
                        compile_n_execute_args_by_lang.get(lang, {}).get("execute_cmd"),
                        compile_n_execute_args_by_lang.get(lang, {}).get(
                            "execute_flags"
                        ),
                        idx,
                        task_id,
                    )

                    future = executor.submit(execute_code, *args)
                    futures.append(future)
                    completion_id[task_id] += 1
                    n_samples += 1

            print("Running test suites...")
            for idx, future in tqdm.tqdm(
                enumerate(as_completed(futures)),
                desc="Test running",
                total=len(futures),
            ):
                result = future.result()
                unittests, sample_idx, task_id = result
                if not isinstance(unittests, list) and "error" in unittests:
                    """
                    [TODO] log it
                    """
                    print("ERROR: ", unittests["error"])
                    continue
                results[task_id].append((sample_idx, unittests))
    print("Calculate pass@k.")
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [
            all(x["exec_outcome"] == ExecOutcome.PASSED.value for x in r[1])
            for r in result
        ]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }

    # Finally, save the results in one file:
    def combine_results():
        with jsonlines.open(sample_file) as sample_rp:
            cnt = 0
            for idx, sample in enumerate(sample_rp):
                cnt += 1
                if sample["lang"] not in supported_langs:
                    continue
                task_id = sample["task_id"]
                if len(results[task_id]) == 0:
                    continue
                if results[task_id][0][0] > idx:
                    continue
                result = results[task_id].pop(0)

                sample["unittests"] = result[1]
                _exec_outcomes = [
                    r["exec_outcome"]
                    for r in result[1]
                    if r["exec_outcome"] != ExecOutcome.PASSED.value
                ] + [ExecOutcome.PASSED.value]

                sample["exec_outcome"] = _exec_outcomes[0]
                yield sample

    print(f"Writing results to {eval_result_file}...")
    with jsonlines.open(eval_result_file, "w") as result_wp:
        for result in tqdm.tqdm(combine_results(), total=n_samples):
            result_wp.write(result)

    return pass_at_k


def entry_point(
    sample_file: str,
    k: str | list | tuple = "1,2,5,10",
    n_workers: int = 4,
    compile_n_execute_args_by_lang_cfg_file: str | None = None,
    limits_by_lang_cfg_file: str | None = None,
    unittest_file: str = "unittest_db.json",
    execeval_url: str = "http://localhost:5000",
    block_network: bool = True,
    stop_on_first_fail: bool = True,
    use_sanitizer: bool = False,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    """
    [TODO]
    compile_n_execute_args_by_lang_cfg_file: str | None = None,
    limits_by_lang_cfg_file: str | None = None,

    assume yaml files and consider config.yaml for compile..args,
    and resource_limits.py for limits_by_lang
    """
    limits_by_lang, compile_n_execute_args_by_lang = None, {}
    if limits_by_lang_cfg_file is None:
        limits_by_lang_cfg_file = "limits_by_lang.yaml"
    if not os.path.exists(limits_by_lang_cfg_file):
        print(
            "Need resource limit defaults for all runtimes, provide the path to default 'limits_by_lang.yaml' or to the modified one."
        )
        exit(-1)
    with open(limits_by_lang_cfg_file) as limit_cfg_rp:
        limits_by_lang = safe_load(limit_cfg_rp)

    if compile_n_execute_args_by_lang_cfg_file is not None and os.path.exists(
        compile_n_execute_args_by_lang_cfg_file
    ):
        with open(
            compile_n_execute_args_by_lang_cfg_file
        ) as compile_n_execute_args_by_lang_rp:
            compile_n_execute_args_by_lang = safe_load(
                compile_n_execute_args_by_lang_rp
            )

    ks = list(map(int, k.split(","))) if isinstance(k, str) else list(k)
    results = evaluate_functional_correctness(
        sample_file,
        ks,
        n_workers,
        block_network=block_network,
        limits_by_lang=limits_by_lang,
        compile_n_execute_args_by_lang=compile_n_execute_args_by_lang,
        unittest_file=unittest_file,
        execeval_url=execeval_url,
        stop_on_first_fail=stop_on_first_fail,
        use_sanitizer=use_sanitizer,
    )

    print(results)


def main():
    fire.Fire(entry_point)


sys.exit(main())
