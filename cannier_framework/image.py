import os
import json
import shlex
import random
import subprocess as sp

from multiprocessing import Pool

from cannier_framework.utils import load_subjects

from cannier_framework.globals import (
    CONT_HOME_DIR, CONT_SUBJECTS_DIR, VENV_DIR, CONT_VOLUME_DIR, ARGS
)


EXECUTABLE = os.path.join("/usr", "bin","python3.8")
PIP_INSTALL = ("pip", "install", "-I", "--no-deps")
PLUGIN_DIR = os.path.join(CONT_HOME_DIR, "pytest-cannier")


def setup_project(repo, sha, commands_setup):
    proj = repo.split("/", 1)[1]
    url = f"https://github.com/{repo}"
    work_dir = os.path.join(CONT_SUBJECTS_DIR, proj, proj)
    venv_dir = os.path.join(CONT_SUBJECTS_DIR, proj, VENV_DIR)
    env = os.environ.copy()
    env["PATH"] = os.path.join(venv_dir, "bin") + ":" + env["PATH"]
    req_file = os.path.join(CONT_SUBJECTS_DIR, proj, "requirements.txt")
    
    sp.run(["git", "clone", url, work_dir], check=True)
    sp.run(["git", "reset", "--hard", sha], cwd=work_dir, check=True)
    sp.run(["virtualenv", f"--python={EXECUTABLE}", venv_dir], check=True)
    sp.run([*PIP_INSTALL, "pip==20.2.4"], check=True, env=env)
    sp.run([*PIP_INSTALL, "-r", req_file], check=True, env=env)
    sp.run([*PIP_INSTALL, PLUGIN_DIR], check=True, env=env)

    for command in commands_setup:
        sp.run(shlex.split(command), check=True, cwd=work_dir, env=env)


def setup_image():
    os.makedirs(CONT_VOLUME_DIR, exist_ok=True)

    args = [
        (repo, sha, commands_setup) 
        for repo, (sha, commands_setup, _) in load_subjects().items()
    ]

    random.shuffle(args)

    with Pool(processes=min(ARGS.processes, len(args))) as pool:
        pool.starmap(setup_project, args)