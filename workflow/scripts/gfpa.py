import pickle
from trna.common import load_config, split_spikes_into_trials, split_trajectories
from trna.allen import load_allen
from trna.ibl import load_ibl
from trna.dimension_reduction import elephant_gpfa
from trna.plot import simple_trajectory_plot

from simuran.bridges.ibl_wide_bridge import (
    one_spike_train,
    one_trial_info,
    one_recorded_regions,
)
from simuran.bridges.allen_vbn_bridge import (
    allen_spike_train,
    allen_trial_info,
    allen_recorded_regions,
)
from simuran.plot.figure import SimuranFigure


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


def analyse_single_recording(recording, gpfa_window, out_dir, base_dir, is_allen=False):
    print("Analysing recording: " + recording.get_name_for_save())
    rel_dir = base_dir
    if is_allen:
        unit_table, spike_train = allen_spike_train(
            recording, brain_regions=["CA1", "TH", "VISp"]
        )
        trial_info = allen_trial_info(recording)
    else:
        unit_table, spike_train = one_spike_train(recording)
        trial_info = one_trial_info(recording)

    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    unit_table.to_csv(
        out_dir
        / "tables"
        / name_from_recording(recording, "unit_table.csv", rel_dir=rel_dir)
    )
    per_trial_spikes = split_spikes_into_trials(
        spike_train, trial_info["trial_times"], end_time=gpfa_window
    )
    gpfa_result, trajectories = elephant_gpfa(per_trial_spikes, gpfa_window, num_dim=3)
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
        print("Loading data for: " + recording.get_name_for_save(rel_dir=rel_dir))
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


def analyse_container(overwrite, config, ibl_recording_container, is_allen=False):
    if is_allen:
        rel_dir_path = "allen_data_dir"
    else:
        rel_dir_path = "ibl_data_dir"
    for i, ibl_recording in enumerate(ibl_recording_container):
        # TODO this is a hack to only do a few
        if i > 3:
            break
        info = load_data(
            ibl_recording, config["output_dir"], rel_dir=config[rel_dir_path]
        )
        if info is None and not overwrite:
            ibl_recording = ibl_recording_container.load(i)
            info = analyse_single_recording(
                ibl_recording,
                config["gpfa_window"],
                config["output_dir"],
                config[rel_dir_path],
                is_allen=is_allen,
            )
        plot_data(
            ibl_recording, info, config["output_dir"], rel_dir=config[rel_dir_path]
        )


def main(overwrite=False):
    config = load_config()
    allen_recording_container, allen_loader = load_allen(
        config["allen_data_dir"], config["allen_manifest"]
    )
    out_loc = config["output_dir"] / "tables" / "ibl_sessions.csv"
    out_loc.parent.mkdir(parents=True, exist_ok=True)
    allen_table
    ibl_recording_container, ibl_loader = load_ibl(config["ibl_data_dir"])
    out_loc = config["output_dir"] / "tables" / "ibl_sessions.csv"
    ibl_loader.get_sessions_table().to_csv(out_loc, index=False)

    analyse_container(overwrite, config, ibl_recording_container)
    analyse_container(overwrite, config, allen_recording_container, is_allen=True)


if __name__ == "__main__":
    main()
