"""
Utility function to merge mlflow results from different sources.
Merges by meta.yaml accounts for input results directories.
"""
import os
import shutil
import yaml


def align_meta_files(mlflow_results_dir: str):
    if not os.path.exists(mlflow_results_dir):
        raise ValueError(f"Specified directory: {mlflow_results_dir} does not exist!")
    for exp_dir in os.listdir(mlflow_results_dir):
        change_cnt = 0.
        if exp_dir.startswith("."):
            continue
        # parse experiment meta file
        exp_meta_file = mlflow_results_dir + os.sep + exp_dir + os.sep + "meta.yaml"
        with open(exp_meta_file) as infile:
            meta_file = yaml.safe_load(infile)
            meta_exp_id = meta_file.get("experiment_id")
            meta_exp_name = meta_file.get("name")
            meta_artifacts = meta_file.get("artifact_location")
        # parse and check experiment run meta file
        for run_dir in os.listdir(mlflow_results_dir + os.sep + exp_dir):
            # check if artifact path exists
            artifact_location_exists = os.path.isdir(meta_artifacts.replace("file://", ""))
            # check we're looking at a run directory for changes
            results_subdir_is_not_a_directory = not os.path.isdir(mlflow_results_dir + os.sep + exp_dir + os.sep + run_dir)
            if artifact_location_exists and results_subdir_is_not_a_directory:
                continue
            # TODO check against stored meta info
            # if diff correct
            run_meta_file_name =  mlflow_results_dir + os.sep + exp_dir + os.sep + run_dir + os.sep + "meta.yaml"
            if not os.path.exists(run_meta_file_name):
                continue
            with open(run_meta_file_name) as infile:
                run_meta_file = yaml.safe_load(infile)
            try:
                run_exp_id = run_meta_file.get("experiment_id")
            except AttributeError: # NOTE: case meta.yaml empty
                print(f"Error at {run_exp_id}/{run_dir} -> deleting...")
                shutil.rmtree(mlflow_results_dir + os.sep + exp_dir + os.sep + run_dir)
                continue
            if not run_exp_id == meta_exp_id or not artifact_location_exists: # misaligned output files or wrong artifact location
                change_cnt += 1
                run_meta_file["experiment_id"] = meta_exp_id
                # adjust for mlflow results artifact directory
                run_meta_file["artifact_uri"] = mlflow_results_dir + os.sep + meta_exp_id + os.sep + run_dir + os.sep + "artifacts"
            assert run_meta_file.get("experiment_id") == meta_exp_id
            assert os.path.isdir(run_meta_file.get("artifact_uri").replace("file://", ""))
            with open(run_meta_file_name, "w") as outfile:
                yaml.safe_dump(run_meta_file, outfile)
        print(f"Experiment {meta_exp_id}: {change_cnt} runs adjusted.")


if __name__ == "__main__":
    # NOTE: THIS SCRIPT PRESUPPOSES THAT ALL FILES ARE IN ONE MLRUNS OUTPUT DIRECTORY
    # NOTE: IT IS REQUIRED THAT THE EXPERIMENT DIR DO NOT CONTAIN meta.yaml FILES, ONLY THE INDIVIDUAL OUTPUT DIRECTORIES
    # NOTE: if results have been copied from another source into this mlruns directory, check that all meta information aligns if the names are correct
    MLFLOW_DIRECTORY = "/Users/rcml/protein_regression/results/mlruns/"
    align_meta_files(MLFLOW_DIRECTORY)