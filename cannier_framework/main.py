import gc
import os
import sys
import inspect
import numpy as np

from argparse import ArgumentParser

from cannier_framework.globals import ARGS
from cannier_framework.image import setup_image
from cannier_framework.collate import collate_data
from cannier_framework.figures import make_figures
from cannier_framework.technique import make_points
from cannier_framework.model import make_shap, make_preds
from cannier_framework.container import manage_container, run_containers


MENU = {
    "setup": setup_image,
    "manage": manage_container,
    "run": run_containers,
    "collate": collate_data,
    "shap": make_shap,
    "preds": make_preds,
    "points": make_points,
    "figures": make_figures
}


def main():
    gc.freeze()
    np.seterr(all="ignore")
    parser = ArgumentParser("cannier-framework")
    parser.add_argument("command")
    parser.add_argument("args", nargs="*")
    parser.add_argument("--processes", type=int, default=os.cpu_count())
    parser.add_argument("--timeout", type=int, default=28800)
    parser.add_argument("--n-repeats", type=int, default=30)
    parser.add_argument("--n-reruns", type=int, default=2500)
    parser.parse_args(namespace=ARGS)

    try:
        fn = MENU[ARGS.command]
    except KeyError:
        print(f"cannier-framework: '{ARGS.command}' is not a valid command.")
        sys.exit(1)

    try:
        inspect.signature(fn).bind(*ARGS.args)
    except TypeError:
        print(f"cannier-framework: incorrect arguments for '{ARGS.command}'.")
        sys.exit(1)

    fn(*ARGS.args)
