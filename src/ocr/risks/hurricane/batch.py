"""Coiled batch jobs for the hurricane pipeline, launched like the stage runner and fire.

Each job runs `python -m ocr.risks.hurricane.<module> <args>` on Coiled VMs. Coiled package-syncs
the local environment, including the editable `ocr` package, so the code travels with it; the VMs
read and write S3 with the Coiled workspace's own AWS access. Launching needs only a Coiled login.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from ocr.risks.hurricane import store

POLL_SECONDS = 30


def run(
    module: str,
    args: Sequence[str],
    *,
    name: str,
    over: Sequence[int | str] | None = None,
    vm_type: str = 'm8g.xlarge',
    max_workers: int = 1,
    spot: bool = True,
    secret_env: dict[str, str] | None = None,
    max_retries: int = 2,
    wait: bool = False,
) -> dict:
    """Submit one batch job; with `over`, one task per value (as $COILED_BATCH_TASK_INPUT). Tasks
    retry `max_retries` times, which covers reclaimed spot VMs."""
    import coiled

    job = coiled.batch.run(
        command=['python', '-m', f'ocr.risks.hurricane.{module}', *args],
        name=re.sub(r'[^A-Za-z0-9_-]', '-', name),
        region='us-west-2',
        arm=True,
        vm_type=[vm_type],
        scheduler_vm_type=[vm_type],  # Coiled refuses clusters that mix ARM and x86
        spot_policy='spot_with_fallback' if spot else 'on-demand',
        max_workers=max_workers,
        map_over_values=[str(v) for v in over] if over else None,
        forward_aws_credentials=False,
        secret_env=secret_env,
        # the tasks write where this process would (store.S3_ROOT), whatever the VM's default
        env={'PYTHONUNBUFFERED': '1', 'CHAZ_S3_ROOT': store.S3_ROOT},
        max_retries=max_retries,
        tag={'Project': 'OCR'},
    )
    print(
        f'{name}: job {job["job_id"]}, cluster {job["cluster_id"]}; '
        f'logs: coiled batch logs {job["cluster_id"]}',
        flush=True,
    )
    if wait:
        tasks = wait_for(job['cluster_id'])
        print(summary(tasks), flush=True)
        if any(failed(t) for t in tasks) or not tasks:
            raise RuntimeError(f'{name}: tasks failed; logs: coiled batch logs {job["cluster_id"]}')
    return job


def wait_for(cluster_id: int) -> list[dict]:
    """Every task of the job once all have finished, or the cluster has stopped."""
    import coiled

    while True:
        jobs = coiled.batch.status(cluster_id)
        tasks = [t for j in jobs for t in j['tasks']]
        finished = tasks and all(t['state'] in ('done', 'error') for t in tasks)
        if finished or (jobs and jobs[0]['cluster_state'] in ('error', 'stopped')):
            return tasks
        time.sleep(POLL_SECONDS)


def failed(task: dict) -> bool:
    """A task that errored, exited non-zero, or never finished (its cluster stopped first)."""
    return task['state'] != 'done' or bool(task['exit_code'])


def summary(tasks: list[dict]) -> str:
    return f'{sum(not failed(t) for t in tasks)} of {len(tasks)} tasks succeeded'


def parse_years(spec: str) -> list[int]:
    """'1981-2016,2019' -> [1981, ..., 2016, 2019]."""
    years = []
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        years.extend(range(int(lo), int(hi or lo) + 1))
    return years
