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
        else:
            print(f"✅ Test for {config_name} passed")

        # Save directory path for later comparison
        run_dirs[config_name] = os.path.join(output_dir, "oracle_evaluations", "oracle_evaluation.json")

    return run_dirs


def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def compare_results(new_results, reference_results):
    if new_results == reference_results:
        print("✅ New results match the reference exactly.")
    else:
        print("⚠ Differences found between new results and reference!")
        # Print differences (simple key-wise comparison)
        for key in new_results:
            if key in reference_results:
                if new_results[key] != reference_results[key]:
                    print(f"  Key '{key}' differs: new={new_results[key]}, ref={reference_results[key]}")
            else:
                print(f"  Key '{key}' is missing in reference.")
        for key in reference_results:
            if key not in new_results:
                print(f"  Key '{key}' is missing in new results.")



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
        compare_results(new_results, reference_results)

if __name__ == "__main__":
    test_oneshot()
