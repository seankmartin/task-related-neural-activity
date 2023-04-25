import pandas as pd
from trna.allen import load_allen
from trna.ibl import load_ibl
from trna.common import load_config


def save_brain_regions(ibl_recording_container, out_dir):
    result_list = []
    for recording in ibl_recording_container.load_iter():
        session_regions = []
        for k, v in recording.data.items():
            if not k.startswith("probe"):
                continue
            unit_table = v[1]
            try:
                regions = unit_table["acronym"]
            except KeyError:
                print(unit_table)
                continue
            session_regions.extend(regions)
        result_list.append((recording.attrs["session"], list(set(session_regions))))
    df = pd.DataFrame(result_list, columns=["session", "regions"])
    out_loc = out_dir / "tables" / "ibl_brain_regions.csv"
    df.to_csv(out_loc, index=False)


def main(config_path=None, main_config=None):
    if main_config is None:
        config = load_config(config_path)
    else:
        config = load_config(config=main_config)
    allen_recording_container, allen_loader = load_allen(
        config["allen_data_dir"], config["allen_manifest"]
    )
    allen_table = allen_loader.get_sessions_table()
    out_loc = config["output_dir"] / "tables" / "allen_sessions.csv"
    out_loc.parent.mkdir(parents=True, exist_ok=True)
    allen_table.to_csv(out_loc, index=False)
    ibl_recording_container, ibl_loader = load_ibl(config["ibl_data_dir"])
    out_loc = config["output_dir"] / "tables" / "ibl_sessions.csv"
    ibl_table = ibl_loader.get_sessions_table()
    ibl_table.to_csv(out_loc, index=False)
    save_brain_regions(ibl_recording_container, config["output_dir"])


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        main(main_config=snakemake.config)
    else:
        main()
