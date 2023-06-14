import pandas as pd
from pathlib import Path
import os
from skm_pyutils.path import get_all_files_in_dir


def count(df, split="Split"):
    if len(df) == 0:
        return
    entries = df[split].unique()
    counts = {}
    for entry in entries:
        counts[entry] = len(df[df[split] == entry])
    return counts


def write_out_count(counts_list, input_paths, output_path):
    print(f"Writing counts to {output_path}")
    print(counts_list, input_paths)
    if os.path.exists(output_path):
        os.remove(output_path)
    for counts, path in zip(counts_list, input_paths):
        splits = path.split("_")
        name1 = splits[-3]
        name2 = splits[-2]
        name_to_write = f"{name1}_{name2}"
        with open(output_path, "a") as f:
            f.write(f"{name_to_write}:\n")
            for key, val in counts.items():
                f.write(f"\t{key} - {val}\n")
            f.write("\n")

    with open(output_path, "r") as f:
        print(f.read())


def main(path_with_dfs):
    csv_files = get_all_files_in_dir(str(path_with_dfs), ".csv", return_absolute=False)
    paths = []
    counts_list = []
    for path in csv_files:
        if not (
            path.startswith("concat_cca_correlation") and path.endswith("converted.csv")
        ):
            continue
        df = pd.read_csv(path_with_dfs / path)
        counts = count(df)
        if counts is None:
            continue
        paths.append(path)
        counts_list.append(counts)
    write_out_count(counts_list, paths, str(path_with_dfs / "counts.txt"))


if __name__ == "__main__":
    main(Path(snakemake.input[0]).parent)
