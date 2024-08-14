import argparse
import os
import json

def merge_output(output_dir):
    outputs = os.listdir(output_dir)
    output_file = os.path.join(output_dir, "all_subjects_summary.json")

    all_data = {"total": {"corr": 0, "wrong": 0, "acc": 0}}
    for file in outputs:
        if "summary" in file:
            chunk_file = os.path.join(output_dir, file)
            with open(chunk_file, 'r') as rf:
                data = json.load(rf)
                all_data["total"]["corr"] += data["total"]["corr"]
                all_data["total"]["wrong"] += data["total"]["wrong"]
    all_data["total"]["acc"] = all_data["total"]["corr"]/(all_data["total"]["corr"] + all_data["total"]["wrong"])
    with open(output_file, 'w') as wf:
        json.dump(all_data, wf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Directory containing the result pickle files.')
    args = parser.parse_args()
    
    merge_output(args.output_dir)
