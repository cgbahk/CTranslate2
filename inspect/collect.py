# TODO Move into one script, just like playground code

from pathlib import Path
import pandas as pd
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--csv_path")
    parser.add_argument("--md_path")

    args = parser.parse_args()

    ws_dir = Path(args.workspace)
    assert ws_dir.is_dir()

    df = pd.DataFrame(sorted(ws_dir.glob("**/manifest.yml")), columns=["mani_path"])

    columns = [
        "memory_in_MiB_base",
        "memory_in_MiB_model_loaded",
        "memory_in_MiB_peak",
        "unix_time_in_second_begin",
        "unix_time_in_second_end",
        "status",
    ]

    def func_factory_for_key(key):

        def get_val_by_key_for(mani_path):
            with open(mani_path) as mani_yml_file:
                mani = yaml.safe_load(mani_yml_file)
                return mani[key]

        return get_val_by_key_for

    for col_name in columns:
        df[col_name] = df["mani_path"].apply(func_factory_for_key(col_name))

    if args.csv_path:
        print(f"Generating csv '{args.csv_path}'")
        df.to_csv(args.csv_path, index=False)

    if args.md_path:
        with open(args.md_path, "w") as md_file:
            md_file.write(df.to_markdown(index=False))
            md_file.write("\n")


if __name__ == "__main__":
    main()
