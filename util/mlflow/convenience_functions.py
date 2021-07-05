import mlflow
import os


mlflow.set_tracking_uri(os.path.join("file:"+"/Users/Bruger/Documents/PhD/Speciale_paper/gitlab/protein_regression"+"/results/mlruns"))


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
