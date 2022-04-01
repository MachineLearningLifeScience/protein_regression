"""
This is a little helper script to make use of array jobs.
"""
import pickle
import argparse
import os
from os.path import sep

from util.execution.run_local import run_local


def main(**args):
    f = open(os.getcwd() + sep + "cluster_run_scripts" + sep + args["timestamp"] + ".pkl", "rb")
    run_command_list = pickle.load(f)
    run_command = run_command_list[args["job"]]
    f.close()
    run_local(run_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-j", "--job", type=int)
    parser.add_argument("-t", "--timestamp", type=str)
    args = parser.parse_args()
    main(**vars(args))
