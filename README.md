# ExecEval

A distributed, extensible, secure solution for evaluating machine generated code with unit tests in multiple programming languages.

This repository is a part of our ongoing effort to build large scale execution based evaluation benchmark published as [xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval](https://arxiv.org/abs/2303.03004). If you are using this tool, plesae consider citing the paper.

```
@misc{khan2023xcodeeval,
      title={xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation and Retrieval}, 
      author={Mohammad Abdullah Matin Khan and M Saiful Bari and Xuan Long Do and Weishi Wang and Md Rizwan Parvez and Shafiq Joty},
      year={2023},
      eprint={2303.03004},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
Part of this work was submitted as a requirement for the Master of Science degree in Computer Science and Applications at the Islamic University of Technology by Muhammad Abdullah Matin Khan. (The thesis or project report will be added upon publication).

```
@misc{khan2024xcodeeval,
      title={Development of a Code Search Engine Using Natural Language Processing Techniques}, 
      author={Mohammad Abdullah Matin Khan},
      year={2024},
      publication={Journal of Engineering and Technology (JET)}
      url=TBA
}
```

## Dependencies:

-   [docker-ce](https://docs.docker.com/engine/install/)

## Steps (Assuming dependencies satisfied):

1. Clone this [ExecEval repository](https://github.com/ntunlp/ExecEval).
2. `cd ExecEval`
3. `docker build . -t exec-eval:1.0`
4. `docker run -it -p x:y -e NUM_WORKERS=67 exec-eval:1.0`. This will expose port `y` (default `5000`) as `http://localhost:y` on the local machine whereas port `x` is used within the docker container which can be set by environment variable `GUNICORN_PORT`. The `NUM_WORKERS` is an environment variable representing the number of parallel execution engine workers. It is recommended to not use all cpus, as if cpu goes into 100% load it might affect execution speed of the codes uncontrollably, and keeping some cpus free for evaluation script. A valid example assuming less cpus available: `docker run -it -p 5000:5000 -e NUM_WORKERS=5 exec-eval:1.0`

### Expected outcome:

A http server should be running on `$PORT=y` (default `5000`) which can parallely execute codes and return their output.

## Some helpful definitions:

### Definition of ExtendedUnittest:

```py
# dataclass
class ExtendedUnittest:
    input: str
    output: list[str] = field(default_factory=list)
    result: str | None = None
    exec_outcome: ExecOutcome | None = None
```

### Definition of ExecOutcome:

```py
class ExecOutcome(Enum):
    PASSED = "PASSED"   # code executes and output matches expected output
    WRONG_ANSWER = "WRONG_ANSWER" # code executes and output does NOT matches expected output
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED" # code executes and didn't exit in time, output is ignored in this case
    RUNTIME_ERROR = "RUNTIME_ERROR" # code failed to execute (crashed)
    COMPILATION_ERROR = "COMPILATION_ERROR" # code failed to compile
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED" # code exceeded memory limit during execution
```

### Definition of ResourceLimits:

For detailed description of each attributes go to [man page of getrlimit](https://man7.org/linux/man-pages/man2/getrlimit.2.html).

```py
class ResourceLimits:
    core: int = 0  # RLIMIT_CORE
    data: int = -1  # RLIMIT_DATA
    #    nice: int = 20  # RLIMIT_NICE
    fsize: int = 0  # RLIMIT_FSIZE
    sigpending: int = 0  # RLIMIT_SIGPENDING
    #    memlock: int = -1  # RLIMIT_MEMLOCK
    rss: int = -1  # RLIMIT_RSS
    nofile: int = 4  # RLIMIT_NOFILE
    msgqueue: int = 0  # RLIMIT_MSGQUEUE
    rtprio: int = 0  # RLIMIT_RTPRIO
    stack: int = -1  # RLIMIT_STACK
    cpu: int = 2  # RLIMIT_CPU, CPU time, in seconds.
    nproc: int = 1  # RLIMIT_NPROC
    _as: int = 2 * 1024 ** 3  # RLIMIT_AS set to 2GB by default
    locks: int = 0  # RLIMIT_LOCKS
    # rttime: int = 2  # RLIMIT_RTTIME, Timeout for real-time tasks.
```

## API endpoints:

### API to execute code:

-   End point: /api/execute_code
-   Method: POST
-   Content-type: application/json
-   Post request json format:

```py
# json of dict of this dataclass
class JobData:
    language: str # language of the code to be executed, usually found in sample["lang"] field
    source_code: str #source_code, usually found in sample["source_code"] field
    unittests: list[ExtendedUnittest] # unittests, usually found in unittest_db[sample["src_uid"]] field which do contain more key value pairs than input, output; so skip them
    compile_cmd: str | None = None # compiler program e.g. gcc, g++, clang++, go, rustc, javac
    compile_flags: str | None = None # flags passed during compilation e.g. "-std=c++11 -lm -static ...
    execute_cmd: str | None = None # executor program (mainly interpreter for interpreted languages) e.g. python2, pypy2, ruby, php
    execute_flags: str | None = None # flags to executor program e.g. "-o -nologo", "-W ignore
    limits: ResourceLimits = field(default_factory=ResourceLimits) # Resource limits
    block_network: bool = True # block network access for codes executed by ExecEval (True is safer)
    stop_on_first_fail: bool = True # stops executing a code if a unit test fails (True for faster execution)
    use_sanitizer: bool = False # This kept to allow some codes of xCodeEval (e.g. MS C++) to execute on linux during testing ExecEval with xCodeEval test data. (False should be ok)

```

-   Response json format: ExtendedUnittest

### API to get list of runtimes available:

-   End point: /api/all_runtimes
-   Method: GET
-   Content-type: application/json
-   Response format:

```json
[
	{
		"compile_cmd": "gcc", // program to compile with
		"compile_flags": "-fno-optimize-sibling-calls -w -fno-strict-aliasing -DONLINE_JUDGE -include limits.h -fno-asm -s -O2 -DONLINE_JUDGE -include math.h -static -lm", // default compiler flags
		"execute_cmd": "",
		"execute_flags": "",
		"has_sanitizer": true,
		"is_compiled": true,
		"runtime_name": "GNU C",
		"timelimit_factor": 1
	},
	{
		"compile_cmd": "python3",
		"compile_flags": "-W ignore -m py_compile",
		"execute_cmd": "python3", // program to execute with
		"execute_flags": "-W ignore -OO -s -S", // flags to execute with
		"has_sanitizer": false, // is a sanitizer implemented in execution_engine/settings.py
		"is_compiled": true, // true if there is a compile cmd
		"runtime_name": "Python 3", // name which needs to match with the language passed in api for execute code
		"timelimit_factor": 3 // a multiplier for time allowed to execute as some languages are slower than others
	}
	// etc.
]
```

## Evaluation

### pass@k

Check the `eval_scripts` directory. The dependencies are mentioned in `requirements.txt`. Run `pip install -r eval_scripts/requirements.txt`. The entry point is through `eval_passk.py`. Run `python eval_scripts/eval_passk.py --help` for description of arguments.

#### Example of most typical usage:

```sh
python eval_scripts/eval_passk.py $path_to_samples_to_evaluate --k "1,2,5,10" --n_workers 129 --limits_by_lang_cfg_file eval_scripts/limits_by_lang.yaml --unittest_file $path_to_unittest_db_file --execeval_url "http://localhost:5000" --use_sanitizer 0

```

## **IMPORTANT**

-   pip dependencies to run evaluation script is listed in `eval_scripts/requirements.txt`.
-   Sanitize functions are available in `execution_engine/settings.py`.
-   Default compiler or execution flags are available in `execution_engine/config.yaml`.
-   Default resource limits for all supported languages are available in `eval_scripts/limits_by_lang.yaml`.
-   The machine generated codes to be executed should be a list of json with following key value pairs present to work properly:
    -   source_code: the code to be executed.
    -   lang: the language/runtime to use to execute in `ExecEval`.
    -   src_uid: the unique id to retrieve unittests from unittest_db.
    -   task_id: an unique id assigned by machine/model trainer to represent the task they are solving. For example, **program synthesis** should have `task_id` same as `src_uid` whereas **Code translation** can have `task_id` same as the index of the test sample for which the code is generated.
-   Be extra careful with the files used to run the scripts, for most parts following the files i.e. `unittest_db` by **xCodeEval** and other files by **ExecEval** should be okay except for the file with machine generated codes.

## Security measures:

-   Use seperate unpreviledged user for each worker to limit access to different resources.
-   Use `prlimit` to limit resources allowed for the execution.
-   Use `seccomp` to limit socket syscalls (can be easily extended to arbitrary syscall blocker with the caveat that some syscalls are required by some languages to execute code).
-   Thus arbitrary resource usage is restricted.
-   Compilation is not so secure as execution with the assumption that the code needs to find vulnerability in the compiler to exploit this point. (This part not tested)
