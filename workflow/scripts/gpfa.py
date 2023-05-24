import pandas as pd
from trna.allen import load_allen
from trna.ibl import load_ibl
from trna.plot import simple_trajectory_plot, plot_gpfa_distance
from trna.common import (
    regions_to_string,
    name_from_recording,
    load_data,
    load_config,
    write_config,
)
from trna.gpfa import analyse_single_recording

from simuran.plot.figure import SimuranFigure
from simuran.loaders.allen_loader import BaseAllenLoader
from simuran import set_only_log_to_file


def plot_data(recording, info, out_dir, brain_regions, rel_dir=None):
    regions_as_str = regions_to_string(brain_regions)
    for key, value in info.items():
        correct = value["correct"]
        incorrect = value["incorrect"]
        fig = simple_trajectory_plot(correct, incorrect)
        out_name = name_from_recording(
            recording, f"gpfa_{key}_{regions_as_str}.png", rel_dir=rel_dir
        )
        fig = SimuranFigure(fig, str(out_dir / out_name))
        fig.save()


def analyse_container(overwrite, config, recording_container, brain_regions):
    is_allen = isinstance(recording_container[0].loader, BaseAllenLoader)
    regions = []
    for region in brain_regions:
        if isinstance(region, str):
            regions.append(region)
        else:
            for sub_region in region:
                regions.append(sub_region)
    if is_allen:
        rel_dir_path = "allen_data_dir"
        n = "allen"
    else:
        rel_dir_path = "ibl_data_dir"
        n = "ibl"
    for i, recording in enumerate(recording_container):
        output_dir = (
            config["output_dir"]
            / "gpfa"
            / recording.get_name_for_save(rel_dir=config[rel_dir_path])
        )
        info = load_data(
            recording, output_dir, regions, rel_dir=config[rel_dir_path], bit="gpfa"
        )
        if (info == "No pickle data found") or overwrite:
            recording = recording_container.load(i)
            info = analyse_single_recording(
                recording,
                config["gpfa_params"],
                output_dir,
                config[rel_dir_path],
                regions,
                filter_prop=config[f"{n}_filter_properties"],
            )
        if info is not None:
            plot_data(
                recording, info, output_dir, regions, rel_dir=config[rel_dir_path]
            )
    all_info = []
    for i, recording in enumerate(recording_container):
        output_dir = (
            config["output_dir"]
            / "gpfa"
            / recording.get_name_for_save(rel_dir=config[rel_dir_path])
        )
        info = load_data(recording, output_dir, regions, rel_dir=config[rel_dir_path])
        if info == "No pickle data found":
            print(
                f"No pickle data found for {recording.get_name_for_save(rel_dir=config[rel_dir_path])}"
            )
        elif info is not None:
            info["name"] = recording.get_name_for_save(rel_dir=config[rel_dir_path])
            all_info.append(info)
    print(f"Analysed {len(all_info)} recordings with sufficient units")
    output_dir = config["output_dir"] / "gpfa"
    regions_str = regions_to_string(regions)
    plot_gpfa_distance(all_info, output_dir, regions_str, n)

    return all_info


def main(main_config, brain_table_location, overwrite=False):
    config = load_config(config=main_config)
    brain_table = pd.read_csv(brain_table_location)
    for brain_region_pair in config["allen_brain_regions"]:
        allen_recording_container, allen_loader = load_allen(
            config["allen_data_dir"], config["allen_manifest"], brain_region_pair
        )
        print(
            f"Loaded {len(allen_recording_container)} recordings from Allen with brain regions {brain_region_pair}"
        )
        full_info = analyse_container(
            overwrite, config, allen_recording_container, brain_region_pair
        )
        for brain_region in brain_region_pair:
            rows = []
            for i, recording in enumerate(allen_recording_container):
                if recording.get_name_for_save(rel_dir=config["allen_data_dir"]) in [
                    info["name"] for info in full_info
                ]:
                    rows.append(i)
            sub_recording_container = allen_recording_container.filter_table(
                rows, inplace=False
            )
            analyse_container(
                overwrite=overwrite,
                config=config,
                recording_container=sub_recording_container,
                brain_regions=[brain_region],
            )
    for brain_region_pair in config["ibl_brain_regions"]:
        ibl_recording_container, ibl_loader = load_ibl(
            config["ibl_data_dir"], brain_table, brain_region_pair
        )
        print(
            f"Loaded {len(ibl_recording_container)} recordings from IBL with brain regions {brain_region_pair}"
        )
        if len(ibl_recording_container) == 0:
            print("No recordings found for this brain region pair")
            continue
        full_info = analyse_container(
            overwrite, config, ibl_recording_container, brain_region_pair
        )
        for brain_region in brain_region_pair:
            rows = []
            for i, recording in enumerate(ibl_recording_container):
                if recording.get_name_for_save(rel_dir=config["ibl_data_dir"]) in [
                    info["name"] for info in full_info
                ]:
                    rows.append(i)
            sub_recording_container = ibl_recording_container.filter_table(
                rows, inplace=False
            )
            if len(sub_recording_container) == 0:
                print("No recordings found for this brain region")
                continue
            analyse_container(
                overwrite=overwrite,
                config=config,
                recording_container=sub_recording_container,
                brain_regions=[brain_region],
            )

    write_config(main_config, config, config["output_dir"] / "gpfa" / "config.yaml")


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        set_only_log_to_file(snakemake.log[0])
        main(
            snakemake.config,
            snakemake.input[0],
            snakemake.params.overwrite,
        )
    else:
        main(None, r"G:/OpenData/OpenDataResults/tables/ibl_brain_regions.csv", False)
