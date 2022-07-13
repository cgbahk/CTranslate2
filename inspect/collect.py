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
        "longest_hyp_token_count",
    ]

    def func_factory_for_key(key):

        def get_val_by_key_for(mani_path):
            with open(mani_path) as mani_yml_file:
                mani = yaml.safe_load(mani_yml_file)
                return mani[key]

        return get_val_by_key_for

    for col_name in columns:
        df[col_name] = df["mani_path"].apply(func_factory_for_key(col_name))

    ct2_options = [
        "beam_size",
    ]

    def func_factory_for_ct2_option(key):

        def get_val_by_key_for(mani_path):
            with open(mani_path) as mani_yml_file:
                mani = yaml.safe_load(mani_yml_file)
                return mani["ct2_translate_option"][key]

        return get_val_by_key_for

    for opt_name in ct2_options:
        df[opt_name] = df["mani_path"].apply(func_factory_for_ct2_option(opt_name))

    df["translation_time_in_second"] = \
        df["unix_time_in_second_end"] - df["unix_time_in_second_begin"]

    def get_batches_info(row):
        batches_info_path = row.mani_path.parent / "batches_info.yml"
        assert batches_info_path.is_file()
        row["batches_info_path"] = batches_info_path

        with open(batches_info_path) as batches_info_file:
            batches_info = yaml.safe_load(batches_info_file)

        row["num_batch"] = batches_info["num_batch"]

        first_batch = batches_info["batches"][0]
        L = first_batch["examples"][0]
        N = first_batch["num_example"]

        row["first_batch_longest_sentence_length"] = L
        row["first_batch_num_sentence"] = N
        row["first_batch_num_token"] = sum(first_batch["examples"])

        max_NL = 0
        max_NL2 = 0

        for batch_info in batches_info["batches"]:
            batch_N = batch_info["num_example"]
            batch_L = batch_info["examples"][0]

            if max_NL < batch_N * batch_L:
                max_NL = batch_N * batch_L

            if max_NL2 < batch_N * batch_L**2:
                max_NL2 = batch_N * batch_L**2

        row["max_NL_over_batch"] = max_NL
        row["max_NL2_over_batch"] = max_NL2

        return row

    df = df.apply(get_batches_info, axis=1)

    if args.csv_path:
        print(f"Generating csv '{args.csv_path}'")
        df.to_csv(args.csv_path, index=False)

    if args.md_path:
        with open(args.md_path, "w") as md_file:
            md_file.write(df.to_markdown(index=False))
            md_file.write("\n")


if __name__ == "__main__":
    main()
