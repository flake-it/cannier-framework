import os
import sys
import json
import shlex
import sqlite3
import subprocess as sp

from cannier_framework.utils import (
    get_counters_max, load_subjects, fetch_counters, manage_pool
)

from cannier_framework.globals import (
    VOLUME_DIR, CONT_HOME_DIR, VENV_DIR, CONT_SUBJECTS_DIR, CONT_VOLUME_DIR,
    STDOUT_DIR, ARGS
)

HOST_VOLUME_DIR = os.path.abspath(VOLUME_DIR)
EXECUTABLE = os.path.join(CONT_HOME_DIR, VENV_DIR, "bin", "cannier")


class Container:
    def __init__(
        self, container_id, proj, mode, victim_nodeid, *commands_exec
    ):
        self.container_id = container_id
        self.proj = proj
        self.mode = mode
        self.victim_nodeid = victim_nodeid
        self.commands_exec = commands_exec

    def manage(self):
        work_dir = os.path.join(CONT_SUBJECTS_DIR, self.proj, self.proj)
        env = os.environ.copy()
        venv_dir = os.path.join(CONT_SUBJECTS_DIR, self.proj, VENV_DIR)
        env["PATH"] = os.path.join(venv_dir, "bin") + ":" + env["PATH"]
        db_file = os.path.join(CONT_VOLUME_DIR, f"{self.proj}.sqlite3")

        for command in self.commands_exec[:-1]:
            sp.run(shlex.split(command), check=True, cwd=work_dir, env=env)

        sp.run(
            [
                *shlex.split(self.commands_exec[-1]), "-v", "-p", "no:cov", 
                "-p", "no:flaky", "-p", "no:xdist", "-p", "no:sugar", "-p", 
                "no:replay", "-p", "no:forked", "-p", "no:ordering", "-p", 
                "no:randomly", "-p", "no:flakefinder", "-p", "no:random_order", 
                "-p", "no:rerunfailures", f"--mode={self.mode}", 
                f"--db-file={db_file}", f"--victim-nodeid={self.victim_nodeid}"
            ],
            check=True, cwd=work_dir, env=env
        )

    def stop(self):
        sp.run(
            ["docker", "stop", self.container_id], stdout=sp.DEVNULL, 
            stderr=sp.DEVNULL
        )

    def start(self):
        stdout_file = os.path.join(STDOUT_DIR, self.container_id)

        try:
            with open(stdout_file, "a") as f:
                proc = sp.run(
                    [
                        "docker", "run", 
                        f"-v={HOST_VOLUME_DIR}:{CONT_VOLUME_DIR}:rw", "--rm", 
                        "--init", "--cpus=1", f"--name={self.container_id}", 
                        "cannier-experiment", EXECUTABLE, "manage", 
                        self.container_id, self.proj, self.mode, 
                        self.victim_nodeid, *self.commands_exec
                    ],
                    stdout=f, stderr=f, timeout=ARGS.timeout
                )
        except sp.TimeoutExpired:
            success = False
            fate = "expired"
            self.stop()
        else:
            success = proc.returncode == 0
            fate = "succeeded" if success else "failed"

        return f"{fate}: {self.container_id}", success

    
def manage_container(proj, mode, run_id, victim_nodeid, *commands_exec):
    Container(proj, mode, run_id, victim_nodeid, *commands_exec).manage()


def fetch_victims(cur):
    cur.execute(
        "select nodeid "
        "from item "
        "where n_runs_features = ? and "
        "n_runs_baseline = ? and "
        "n_runs_shuffle = ? and "
        "(n_fail_baseline = 0 or n_fail_baseline = n_runs_baseline) and "
        "n_fail_baseline != n_fail_shuffle and "
        "n_runs_victim = 0",
        (ARGS.n_repeats, ARGS.n_reruns, ARGS.n_reruns)
    )

    return [victim_nodeid for victim_nodeid, in cur.fetchall()]


def iter_containers_proj(
    plugin_modes, db_file, proj, commands_exec, counters_max
):
    with sqlite3.connect(db_file) as con:
        counters = fetch_counters(con.cursor())

    for mode in plugin_modes:
        if mode == "victim":
            with sqlite3.connect(db_file) as con:
                victims = fetch_victims(con.cursor())

            for i, victim_nodeid in enumerate(victims):
                yield Container(
                    f"{proj}_{mode}_{i}", proj, mode, victim_nodeid, 
                    *commands_exec
                )
        else:
            n_remaining = max(0, counters_max[mode] - counters[mode])

            for i in range(n_remaining):
                yield Container(
                    f"{proj}_{mode}_{i}", proj, mode, "", *commands_exec
                )


def iter_containers(plugin_modes):
    with open(os.path.join("schema.sql"), "r") as f:
        schema = f.read()

    counters_max = get_counters_max()

    for repo, (_, _, commands_exec) in load_subjects().items():
        proj = repo.split("/", 1)[1]
        db_file = os.path.join(VOLUME_DIR, f"{proj}.sqlite3")

        if not os.path.exists(db_file):
            with sqlite3.connect(db_file) as con:
                con.executescript(schema)

        yield from iter_containers_proj(
            set(plugin_modes), db_file, proj, commands_exec, counters_max
        )


def run_containers(*plugin_modes):
    os.makedirs(STDOUT_DIR, exist_ok=True)
    os.makedirs(VOLUME_DIR, exist_ok=True)
    
    args = iter_containers(plugin_modes)
    all_success = manage_pool(Container.start, args)

    if not all_success:
        print(f"cannier-framework: one or more containers failed.")
        sys.exit(1)
