from trna.allen import load_allen
from trna.ibl import load_ibl
from trna.common import load_config

def get_suitable_sessions(allen_table, ibl_table, out_dir, config): 
    out_loc = config["output_dir"] / "tables" / "suitable_allen.csv"
    out_loc = config["output_dir"] / "tables" / "suitable_ibl.csv"


def main(config_path=None):
    config = load_config(config_path)
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
    get_suitable_sessions(allen_table, ibl_table, config["output_dir"], config)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        main(snakemake.config["config_path"])
    else:
        main()
