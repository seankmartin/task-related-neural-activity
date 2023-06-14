import pandas as pd
from pathlib import Path
from skm_pyutils.path import get_all_files_in_dir


def create_delay_splits(df, split="Trial result"):
    columns = list(df.columns)
    columns.remove(split)
    columns.remove("Delay")
    new_df = pd.DataFrame()
    for delay in df["Delay"].unique():
        matching_rows = df["Delay"] == delay
        for val in columns:
            new_df["Split"] = df[matching_rows][split].values
            new_df[f"Delay_{delay}_{val}"] = df[matching_rows][val].values
    return new_df


def main(path_with_dfs):
    csv_files = get_all_files_in_dir(str(path_with_dfs), ".csv")
    for path in csv_files:
        if path.endswith("converted.csv"):
            continue
        df = pd.read_csv(path)
        df_converted = create_delay_splits(df)
        df_converted.to_csv(path[:-4] + "_converted.csv")


if __name__ == "__main__":
    main(Path(snakemake.input[0]).parent)
