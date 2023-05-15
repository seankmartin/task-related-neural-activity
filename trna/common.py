import pickle
import numpy as np
from pathlib import Path
from simuran import ParamHandler
from simuran.core.utils import convert_filter


def save_info_to_file(info, recording, out_dir, regions, rel_dir=None, bit="gpfa"):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    regions_as_str = regions_to_string(regions)
    save_name = out_dir / "pickles" / (name + regions_as_str + f"_{bit}" + ".pkl")
    save_name.parent.mkdir(parents=True, exist_ok=True)
    with open(save_name, "wb") as f:
        pickle.dump(info, f)


def load_data(recording, out_dir, regions, rel_dir=None, bit="gpfa"):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    regions_as_str = regions_to_string(regions)
    save_name = out_dir / "pickles" / (name + regions_as_str + f"_{bit}" + ".pkl")
    if save_name.exists():
        print(
            "Loading pickle data for: "
            + recording.get_name_for_save(rel_dir=rel_dir)
            + regions_as_str
        )
        with open(save_name, "rb") as f:
            info = pickle.load(f)
        return info
    else:
        return "No pickle data found"


def name_from_recording(recording, filename, rel_dir=None):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    name = name + "--" + filename
    return name


def regions_to_string(brain_regions):
    s = ""
    for r in brain_regions:
        if isinstance(r, str):
            s += r + "_"
        else:
            s += "_".join(r) + "_"
    return s[:-1].replace("/", "-")


def ensure_enough_units(unit_table, min_num_units, brain_region_str):
    has_enough = True
    if len(unit_table) < min_num_units:
        has_enough = False
    brain_regions = unit_table[brain_region_str].unique()
    for region in brain_regions:
        if isinstance(region, str):
            if len(unit_table[unit_table[brain_region_str] == region]) < min_num_units:
                has_enough = False
                break
        else:
            num_units = 0
            for sub_region in region:
                num_units += len(unit_table[unit_table[brain_region_str] == sub_region])
            if num_units < min_num_units:
                has_enough = False
                break
    return has_enough


def decimate_train_to_min(unit_table, spike_train, threshold):
    brain_regions = unit_table["brain_region"].unique()
    new_spike_train = {}
    new_unit_table = unit_table.copy()
    for region in brain_regions:
        region_units = unit_table[unit_table["brain_region"] == region]
        if len(region_units) > threshold:
            random_units = np.random.choice(range(len(region_units)), threshold)
            for i in random_units:
                new_spike_train[region_units.iloc[i].index] = spike_train[
                    region_units.iloc[i].index
                ]
        else:
            for i in range(len(region_units)):
                new_spike_train[region_units.iloc[i].index] = spike_train[
                    region_units.iloc[i].index
                ]
    new_unit_table = new_unit_table.loc[new_spike_train.keys()]
    return new_spike_train


def split_spikes_into_trials(
    spike_train, trial_start_ends, end_time=None, num_trials=None, delay=0
):
    """
    Split the spike train into trials.

    Parameters:
    -----------
    spike_train: dict
        Dictionary of spike trains for each neuron.
    trial_start_ends: List of tuples
        Trial start and end information.
    end_time: float
        End time of each trial.
        If None, uses 1.2 * median of the trial lengths.
    num_trials: int
        Number of trials to use. If None, uses all trials.
    delay: int
        The delay in milliseconds to apply to the spike train.

    Returns:
    --------
    new_spike_train: list
        List of lists of spike trains for each neuron in each trial.

    """
    delay = delay / 1000
    end_time = (
        end_time
        if end_time is not None
        else np.median([t[1] - t[0] for t in trial_start_ends]) * 1.2
    )
    new_spike_train = []
    for i, (start, end) in enumerate(trial_start_ends):
        if (num_trials is not None) and (i == num_trials):
            break
        start_time = start + delay
        end_time_final = start + end_time + delay
        if start_time < 0:
            start_time = start
        trial_spike_train = []
        for k, v in spike_train.items():
            to_use = v[(v >= start_time) & (v <= end_time_final)]
            trial_spike_train.append(to_use - start)
        new_spike_train.append(trial_spike_train)
    return new_spike_train


def split_trajectories(trajectories, trial_correct):
    """
    Split the trajectories into correct and incorrect trials.

    Parameters:
    -----------
    trajectories: np.ndarray
        The trajectories of the neurons with GPFA applied.
    trial_correct: np.ndarray
        Whether each trial was correct or not.

    Returns:
    --------
    correct: np.ndarray
        The trajectories of the neurons with GPFA applied for correct trials.
    incorrect: np.ndarray
        The trajectories of the neurons with GPFA applied for incorrect trials.

    """
    correct = trajectories[np.array(trial_correct).astype(bool)]
    incorrect = trajectories[~np.array(trial_correct).astype(bool)]
    return correct, incorrect


def load_config(config_path=None, config=None):
    """
    Establish the config for the project.

    Parameters:
    -----------
    config_path: str
        The path to the config file. If None, uses the default.
        The default is the config file config.yml in the project.

    Returns:
    --------
    parameters: ParamHandler
        The parameters for the project. Like a dict.

    """
    if config is None:
        config_path = (
            config_path or Path(__file__).parent.parent / "config" / "config.yaml"
        )
        parameters = ParamHandler(source_file=config_path)
    else:
        parameters = ParamHandler(attrs=config)

    try:
        from google.colab import drive

        drive.mount("/content/drive")
        data_directory = Path(parameters["drive_dir"])
    except ModuleNotFoundError:
        data_directory = Path(parameters["local_dir"])

    parameters["allen_data_dir"] = data_directory / parameters["allen_name"]
    parameters["ibl_data_dir"] = data_directory / parameters["ibl_name"]
    parameters["output_dir"] = data_directory / parameters["output_name"]

    return parameters
