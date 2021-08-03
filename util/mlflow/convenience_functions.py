import mlflow
import os
from os.path import join


mlflow.set_tracking_uri('file:'+join(os.getcwd(), join("results", "mlruns")))


def make_experiment_name_from_tags(tags: dict):
    return "".join([t + "_" + tags[t] + "__" for t in tags.keys()])


def find_experiments_by_tags(tags: dict):
    exps = mlflow.tracking.MlflowClient().list_experiments()
    def all_tags_match(e):
        for tag in tags.keys():
            if tag not in e.tags:
                return False
            if e.tags[tag] != tags[tag]:
                return False
        return True
    return [e for e in exps if all_tags_match(e)]
