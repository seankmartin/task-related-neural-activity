import pickle

import pandas as pd
from trna.common import load_config, split_spikes_into_trials, split_trajectories
from trna.allen import load_allen
from trna.ibl import load_ibl
from trna.dimension_reduction import elephant_gpfa
from trna.plot import simple_trajectory_plot

from simuran.bridges.ibl_wide_bridge import IBLWideBridge
from simuran.bridges.allen_vbn_bridge import AllenVBNBridge
from simuran.plot.figure import SimuranFigure
from simuran.loaders.allen_loader import BaseAllenLoader


def name_from_recording(recording, filename, rel_dir=None):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    name = name + "--" + filename
    return name


def save_info_to_file(info, recording, out_dir, rel_dir=None):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    save_name = out_dir / "pickles" / (name + "_gpfa" + ".pkl")
    save_name.parent.mkdir(parents=True, exist_ok=True)
    with open(save_name, "wb") as f:
        pickle.dump(info, f)


def analyse_single_recording(recording, gpfa_window, out_dir, base_dir, brain_regions):
    print("Analysing recording: " + recording.get_name_for_save())
    rel_dir = base_dir
    is_allen = isinstance(recording.loader, BaseAllenLoader)
    bridge = AllenVBNBridge() if is_allen else IBLWideBridge()
    unit_table, spike_train = bridge.spike_train(recording, brain_regions=brain_regions)
    trial_info = bridge.trial_info(recording)

    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    unit_table.to_csv(
        out_dir
        / "tables"
        / name_from_recording(recording, "unit_table.csv", rel_dir=rel_dir)
    )
    per_trial_spikes = split_spikes_into_trials(
        spike_train, trial_info["trial_times"], end_time=gpfa_window
    )
    try:
        gpfa_result, trajectories = elephant_gpfa(
            per_trial_spikes, gpfa_window, num_dim=3
        )
    except ValueError:
        print("Not enough spikes for GPFA")
        return None
    correct, incorrect = split_trajectories(trajectories, trial_info["trial_correct"])
    info = {"correct": correct, "incorrect": incorrect}
    save_info_to_file(info, recording, out_dir, rel_dir)
    print(
        "Finished analysing: "
        + recording.get_name_for_save()
        + f" with {len(correct)} correct and {len(incorrect)} incorrect trials and {len(unit_table)} units"
    )
    return info


def load_data(recording, out_dir, rel_dir=None):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    save_name = out_dir / "pickles" / (name + "_gpfa" + ".pkl")
    if save_name.exists():
        print(
            "Loading pickle data for: " + recording.get_name_for_save(rel_dir=rel_dir)
        )
        with open(save_name, "rb") as f:
            info = pickle.load(f)
        return info
    else:
        return None


def plot_data(recording, info, out_dir, rel_dir=None):
    correct = info["correct"]
    incorrect = info["incorrect"]
    fig = simple_trajectory_plot(correct, incorrect)
    out_name = name_from_recording(recording, "gpfa.png", rel_dir=rel_dir)
    fig = SimuranFigure(fig, str(out_dir / "gpfa" / out_name))
    fig.save()


def analyse_container(overwrite, config, recording_container):
    is_allen = isinstance(recording_container[0].loader, BaseAllenLoader)
    if is_allen:
        rel_dir_path = "allen_data_dir"
        brain_regions = config["allen_brain_regions"]
    else:
        rel_dir_path = "ibl_data_dir"
        brain_regions = config["ibl_brain_regions"]
    for i, recording in enumerate(recording_container):
        info = load_data(recording, config["output_dir"], rel_dir=config[rel_dir_path])
        if info is None or overwrite:
            recording = recording_container.load(i)
            info = analyse_single_recording(
                recording,
                config["gpfa_window"],
                config["output_dir"],
                config[rel_dir_path],
                brain_regions,
            )
        if info is not None:
            plot_data(
                recording, info, config["output_dir"], rel_dir=config[rel_dir_path]
            )


def main(main_config, brain_table_location, overwrite=False):
    config = load_config(config=main_config)
    brain_table = pd.read_csv(brain_table_location)
    allen_recording_container, allen_loader = load_allen(
        config["allen_data_dir"],
        config["allen_manifest"],
        config["allen_brain_regions"],
    )
    print(
        f"Loaded {len(allen_recording_container)} recordings from Allen with brain regions {config['allen_brain_regions']}"
    )
    ibl_recording_container, ibl_loader = load_ibl(
        config["ibl_data_dir"], brain_table, config["ibl_brain_regions"]
    )
    print(
        f"Loaded {len(ibl_recording_container)} recordings from IBL with brain regions {config['ibl_brain_regions']}"
    )

    analyse_container(overwrite, config, ibl_recording_container)
    analyse_container(overwrite, config, allen_recording_container)


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        main(snakemake.config, snakemake.input[0], snakemake.params.overwrite)
    else:
        main(None, r"G:/OpenData/OpenDataResults/tables/ibl_brain_regions.csv", False)
