import pandas as pd
from pathlib import Path

from trna.allen import load_allen
from trna.ibl import load_ibl
from trna.common import load_config

from simuran.plot.unit import plot_unit_properties
from simuran.loaders.allen_loader import BaseAllenLoader
from simuran.bridges.allen_vbn_bridge import AllenVBNBridge
from simuran.bridges.ibl_wide_bridge import IBLWideBridge
from simuran import set_only_log_to_file


def plot_cells(recording_container, brain_regions, output_dir):
    is_allen = isinstance(recording_container[0].loader, BaseAllenLoader)
    bridge = AllenVBNBridge() if is_allen else IBLWideBridge()

    unit_tables = []
    for recording in recording_container:
        unit_table, spike_train = bridge.spike_train(
            recording, brain_regions=brain_regions
        )
        unit_tables.append(unit_table)

    unit_table = pd.concat(unit_tables)

    if is_allen:
        unit_properties = [""]
        structure_name = "structure_acronym"
    else:
        structure_name = "acronym"
        unit_properties = ["firing_rate"]

    log_scales = [True]
    plot_unit_properties(
        unit_table,
        unit_properties,
        log_scales,
        output_dir,
        structure_name=structure_name,
    )


def region_to_string(brain_regions):
    s = ""
    for r in brain_regions:
        if isinstance(r, str):
            s += r + "_"
        else:
            s += "_".join(r) + "_"
    return s[:-1].replace("/", "-")


def main(config_location, brain_table_location, output_dir):
    config = load_config(config=config_location)
    brain_table = pd.read_csv(brain_table_location)
    for brain_region_pair in config["allen_brain_regions"]:
        allen_recording_container, allen_loader = load_allen(
            config["allen_data_dir"], config["allen_manifest"], brain_region_pair
        )
        print(
            f"Loaded {len(allen_recording_container)} recordings from Allen with brain regions {brain_region_pair}"
        )
        br_str = region_to_string(brain_region_pair)
        n = "allen"
        plot_cells(
            allen_recording_container, brain_region_pair, output_dir / n / br_str
        )
    for brain_region_pair in config["ibl_brain_regions"]:
        ibl_recording_container, ibl_loader = load_ibl(
            config["ibl_data_dir"], brain_table, config["ibl_brain_regions"]
        )
        print(
            f"Loaded {len(ibl_recording_container)} recordings from IBL with brain regions {config['ibl_brain_regions']}"
        )
        n = "ibl"
        br_str = region_to_string(brain_region_pair)
        plot_cells(ibl_recording_container, ["VISp"], output_dir / n / br_str)


if __name__ == "__main__":
    set_only_log_to_file(snakemake.log[0])
    main(snakemake.config, snakemake.input[0], Path(snakemake.output[0]))
