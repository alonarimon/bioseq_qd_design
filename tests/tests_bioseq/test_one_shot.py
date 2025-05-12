import json
import os.path
import subprocess
import sys

from pydantic.schema import datetime

test_file_dir = os.path.dirname(os.path.abspath(__file__))  # tests/tests_bioseq/
project_root = os.path.dirname(os.path.dirname(test_file_dir))  # project root

def run_elm():

    run_elm_path = os.path.join(project_root, "run_elm.py")
    if not os.path.exists(run_elm_path):
        raise FileNotFoundError(f"run_elm.py not found at {run_elm_path}")

    configs_to_run = ["oneshot_bio_elmconfig", "oneshot_similarity_bd_elmconfig"]
    run_dirs = {}

    for config_name in configs_to_run:
        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M")
        output_dir = os.path.join(
            project_root, "logs", "elm", f"test_{config_name}", timestamp
        )
        command = [
            sys.executable, run_elm_path,
            "--config-name", config_name,

            # overrides
            "env.size_of_refs_collection=10",
            "qd.init_steps=1",
            "qd.total_steps=100",
            "output_dir=test_logs",
            f"run_name=test_{config_name}",
            f"wandb_group=test_suite",
            f"hydra.job.override_dirname=",
            f"hydra.run.dir=logs/elm/test_{config_name}/{timestamp}",
        ]

        result = subprocess.run(command,
                                cwd=project_root,
                                capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Test for {config_name} failed")
            print(result.stderr)

        # Save directory path for later comparison
        run_dirs[config_name] = os.path.join(output_dir, "oracle_evaluations", "oracle_evaluation.json")

    return run_dirs


def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def compare_results(new_results, reference_results, tolerance=0.0000001):
    """
    Compare new results with reference results.
    :param new_results: New results to compare.
    :param reference_results: Reference results to compare against.
    :return: True if results match, False otherwise.
    """
    # Check if both are dictionaries
    if not isinstance(new_results, dict) or not isinstance(reference_results, dict):
        raise ValueError("Both new_results and reference_results should be dictionaries.")
    # Check if they have the same keys
    if new_results.keys() != reference_results.keys():
        print("⚠ Keys do not match between new results and reference!")
        return False
    # Compare values
    for key in new_results:
        if isinstance(new_results[key], (int, float)) and isinstance(reference_results[key], (int, float)):
            if abs(new_results[key] - reference_results[key]) > tolerance:
                print(f"⚠ Key '{key}' differs: new={new_results[key]}, ref={reference_results[key]}")
                return False
        elif isinstance(new_results[key], dict) and isinstance(reference_results[key], dict):
            if not compare_results(new_results[key], reference_results[key]):
                return False
        else:
            if new_results[key] != reference_results[key]:
                print(f"⚠ Key '{key}' differs: new={new_results[key]}, ref={reference_results[key]}")
                return False
    return True



def test_oneshot():

    # Run the ELM configurations
    run_dirs = run_elm()

    # Set fixed reference file paths (adjust if needed)
    reference_paths = {
        "oneshot_bio_elmconfig": os.path.join(project_root, "logs", "elm", f"test_oneshot_bio_elmconfig", '25-05-06_17-27', "oracle_evaluations", "oracle_evaluation.json"),
        "oneshot_similarity_bd_elmconfig": os.path.join(project_root, "logs", "elm", f"test_oneshot_similarity_bd_elmconfig", '25-05-06_17-29', "oracle_evaluations", "oracle_evaluation.json"),
    }

    for config_name in ["oneshot_bio_elmconfig", "oneshot_similarity_bd_elmconfig"]:
        new_results = load_json(run_dirs[config_name])
        reference_results = load_json(reference_paths[config_name])

        print(f"Comparing results for {config_name}...")
        match = compare_results(new_results, reference_results)
        if match:
            print(f"✅ Test for {config_name} passed.")
        else:
            print(f"❌ Test for {config_name} failed.")
            print("See differences above.")
            raise AssertionError(f"Test for {config_name} failed. Results do not match.")

if __name__ == "__main__":
    test_oneshot()
