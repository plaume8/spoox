import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
from datetime import datetime


@dataclass
class Trial:
    trial_id: str
    task_name: str
    agent_id: str
    model_id: str
    # from spoox logs
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    agent_system_stopped: Optional[bool] = None
    agent_system_exec_time_sec: Optional[float] = None
    agent_usage_stats: Any = None
    # from harbor/TB logs
    verifier_tests: Optional[int] = None
    verifier_tests_passed: Optional[int] = None
    verifier_tests_failed: Optional[int] = None
    verifier_tests_passed_relative: Optional[float] = None
    exception_type: Optional[str] = None


@dataclass
class TBRun:
    run_id: str
    model: str
    agent: str
    started_at: datetime
    finished_at: datetime
    n_total_trials: int
    n_attempts: int
    i_of_n_attempts: int

    n_errors: int
    accuracy_mean: float
    accuracy_mean_rel: float
    solved_tasks: list[str]
    unsolved_tasks: list[str]
    exception_stats: dict[str, list[str]]

    trials: dict[str, Trial]
    total_prompt_tokens: int
    total_completion_tokens: int


def split_trail_list(ts):
    split_ts = []
    ts_cur = ts[:1]
    for t in ts[1:]:
        if t[0] < ts_cur[-1][0]:
            # new list
            split_ts.append(ts_cur.copy())
            ts_cur = [t]
        else:
            # same list
            ts_cur.append(t)
    split_ts.append(ts_cur.copy())
    return split_ts


def parse_trial(t_path: Path, agent_id: str, model_id: str) -> Trial:
    # read in spoox meta data + usage stats pkl
    try:
        t_spoox_path = \
        [d for d in (t_path / 'agent' / 'spoox').iterdir() if d.is_dir and d.name.startswith("spoox_logs_")][0]
        with (t_spoox_path / 'meta_data.json').open("r") as f:
            spoox_meta_data = json.load(f)
    except FileNotFoundError:
        spoox_meta_data = None
    try:
        t_spoox_path = \
        [d for d in (t_path / 'agent' / 'spoox').iterdir() if d.is_dir and d.name.startswith("spoox_logs_")][0]
        with (t_spoox_path / 'usage_stats.pkl').open("rb") as f:
            spoox_usage_stats = pickle.load(f)
    except FileNotFoundError:
        spoox_usage_stats = None
    except EOFError:
        spoox_usage_stats = None
    # read in harbor verifier json
    try:
        with (t_path / 'verifier' / 'ctrf.json').open("r") as f:
            verifier_data = json.load(f)
    except FileNotFoundError:
        verifier_data = None
    # read in result json (no exception cause it should be part of it)
    with (t_path / 'result.json').open("r") as f:
        result_data = json.load(f)

    # craft Trial data object
    trial = Trial(t_path.name, t_path.name.split('__')[0], agent_id, model_id)
    # parse spoox meta data
    if spoox_meta_data is not None:
        trial.total_prompt_tokens = spoox_meta_data.get('model-client-total-usage-prompt-tokens')
        trial.total_completion_tokens = spoox_meta_data.get('model-client-total-usage-completion-tokens')
        trial.agent_system_stopped = spoox_meta_data.get('agent-system-stopped')
        trial.agent_system_exec_time_sec = spoox_meta_data.get('agent-system-exec-time-sec')
        trial.agent_usage_stats = spoox_usage_stats
    # parse verifier data
    if verifier_data is not None:
        verifier_summary = verifier_data['results']['summary']
        trial.verifier_tests = verifier_summary.get('tests')
        trial.verifier_tests_passed = verifier_summary.get('passed')
        trial.verifier_tests_failed = verifier_summary.get('failed')
        if trial.verifier_tests > 0:
            trial.verifier_tests_passed_relative = trial.verifier_tests_passed / trial.verifier_tests
        if result_data['exception_info'] is not None:
            trial.exception_type = result_data['exception_info'].get('exception_type')
        assert trial.verifier_tests == trial.verifier_tests_failed + trial.verifier_tests_passed
    return trial

