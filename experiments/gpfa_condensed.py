from trna.common import load_config, split_spikes_into_trials, split_trajectories
from trna.allen import load_allen
from trna.ibl import load_ibl
from trna.dimension_reduction import elephant_gpfa
from trna.plot import simple_trajectory_plot
from simuran.bridges.ibl_wide_bridge import one_spike_train, one_trial_info
from simuran.bridges.allen_vbn_bridge import allen_spike_train, allen_trial_info
from simuran.plot.figure import SimuranFigure


def name_from_recording(recording, filename, rel_dir=None):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    name += "--" + filename
    return name


def analyse_single_recording(recording, gpfa_window, out_dir, is_allen=False):
    if is_allen:
        print(
            "Analysing allen recording: "
            + recording.get_name_for_save(rel_dir=out_dir.parent)
        )
        unit_table, spike_train = allen_spike_train(recording)
        trial_info = allen_trial_info(recording)
        out_name = name_from_recording(recording, "gpfa", rel_dir=out_dir.parent)
    else:
        print(
            "Analysing IBL recording: "
            + recording.get_name_for_save()
        )
        unit_table, spike_train = one_spike_train(recording)
        trial_info = one_trial_info(recording)
        out_name = name_from_recording(recording, "gpfa")
    per_trial_spikes = split_spikes_into_trials(
        spike_train, trial_info["trial_times"], end_time=gpfa_window
    )
    gpfa_result, trajectories = elephant_gpfa(per_trial_spikes, gpfa_window, num_dim=3)
    correct, incorrect = split_trajectories(trajectories, trial_info["trial_correct"])
    fig = simple_trajectory_plot(correct, incorrect)
    fig = SimuranFigure(fig, out_dir / out_name)
    fig.save()


def main():
    config = load_config()
    allen_recording_container, allen_loader = load_allen(
        config["allen_data_dir"], config["allen_manifest"]
    )
    example_allen = allen_recording_container.load(0)
    trial_info = allen_trial_info(example_allen)
    ibl_recording_container, ibl_loader = load_ibl(config["ibl_data_dir"])

    # TODO replace by for loop
    example_one = ibl_recording_container.load(0)
    example_allen = allen_recording_container.load(0)

    analyse_single_recording(example_one, config["gpfa_window"], config["output_dir"])
    analyse_single_recording(
        example_allen, config["gpfa_window"], config["output_dir"], is_allen=True
    )


if __name__ == "__main__":
    main()
