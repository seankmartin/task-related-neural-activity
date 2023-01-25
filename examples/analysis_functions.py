import numpy as np
from matplotlib import pyplot as plt
import simuran as smr

# Convenience function to compute the PSTH
def makePSTH(spikes, start_times, window_duration, binSize=0.001):
    bins = np.arange(0, window_duration + binSize, binSize)
    counts = np.zeros(bins.size - 1)
    for i, start in enumerate(start_times):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start + window_duration)
        counts = counts + np.histogram(spikes[startInd:endInd] - start, bins)[0]

    counts = counts / start_times.size
    return counts / binSize, bins


def example_plot(
    good_units,
    spike_times,
    stimulus_change_times,
    area_of_interest="VISp",
    time_before_change=1,
    duration=2.5,
):
    # Here's where we loop through the units in our area of interest and compute their PSTHs
    area_change_responses = []
    area_units = good_units[good_units["structure_acronym"] == area_of_interest]
    for iu, unit in area_units.iterrows():
        unit_spike_times = spike_times[iu]
        unit_change_response, bins = makePSTH(
            unit_spike_times,
            stimulus_change_times - time_before_change,
            duration,
            binSize=0.01,
        )
        area_change_responses.append(unit_change_response)
    area_change_responses = np.array(area_change_responses)

    # Plot the results
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches([12, 4])

    clims = [np.percentile(area_change_responses, p) for p in (0.1, 99.9)]
    im = ax[0].imshow(area_change_responses, clim=clims)
    ax[0].set_title("Active Change Responses for {}".format(area_of_interest))
    ax[0].set_ylabel("Unit number, sorted by depth")
    ax[0].set_xlabel("Time from change (s)")
    ax[0].set_xticks(np.arange(0, bins.size - 1, 20))
    _ = ax[0].set_xticklabels(np.round(bins[:-1:20] - time_before_change, 2))

    ax[1].plot(
        bins[:-1] - time_before_change, np.mean(area_change_responses, axis=0), "k"
    )
    ax[1].set_title(
        "{} population active change response (n={})".format(
            area_of_interest, area_change_responses.shape[0]
        )
    )
    ax[1].set_xlabel("Time from change (s)")
    ax[1].set_ylabel("Firing Rate")

    return fig


def ccf_unit_plot(session_units_channels):
    from matplotlib import pyplot as plt

    fig = plt.figure()
    fig.set_size_inches([14, 8])
    ax = fig.add_subplot(111, projection="3d")

    def plot_probe_coords(probe_group):
        ax.scatter(
            probe_group["left_right_ccf_coordinate"],
            probe_group["anterior_posterior_ccf_coordinate"],
            -probe_group[
                "dorsal_ventral_ccf_coordinate"
            ],  # reverse the z coord so that down is into the brain
        )
        return probe_group["probe_id"].values[0]

    probe_ids = session_units_channels.groupby("probe_id").apply(plot_probe_coords)

    ax.set_zlabel("D/V")
    ax.set_xlabel("Left/Right")
    ax.set_ylabel("A/P")
    ax.legend(probe_ids)
    ax.view_init(elev=55, azim=70)


def filter_good_units(unit_channels, sort_=True):
    # Very NB https://allensdk.readthedocs.io/en/latest/_static/examples/nb/visual_behavior_neuropixels_quality_metrics.html
    # This would depend on the analysis, but some ideas here
    if sort_:
        unit_channels = unit_channels.sort_values(
            "probe_vertical_position", ascending=False
        )
    good_unit_filter = (
        (unit_channels["isi_violations"] < 0.4) # Well isolated units
        & (unit_channels["nn_hit_rate"] > 0.9) # Well isolated units
        & (unit_channels["amplitude_cutoff"] < 0.1) # Units that have most of their activations
        & (unit_channels["presence_ratio"] > 0.9) # Tracked for 90% of the recording
    )

    return unit_channels[good_unit_filter]


def verify_filtering(units):
    from scipy.ndimage import gaussian_filter1d

    smr.set_plot_style()
    ax = smr.setup_ax(xlabel="log$_{10}$ firing rate (spikes / s)")
    for region, value in region_dict.items():
        data = np.log10(units[units.structure_acronym.isin(value)]['firing_rate'])
        bins = np.linspace(-3, 2, 100)
        histogram, bins = np.histogram(data, bins, density=True)
        ax.plot(bins[:-1], gaussian_filter1d(histogram, 1), c=color_dict[region])
    smr.despine()
    plt.legend(region_dict.keys())
    return ax


def get_brain_regions_units(units, n=20):
    return units["structure_acronym"].value_counts()[:n].index.tolist()


region_dict = {
    "cortex": [
        "VISp",
        "VISl",
        "VISrl",
        "VISam",
        "VISpm",
        "VIS",
        "VISal",
        "VISmma",
        "VISmmp",
        "VISli",
    ],
    "thalamus": [
        "LGd",
        "LD",
        "LP",
        "VPM",
        "TH",
        "MGm",
        "MGv",
        "MGd",
        "PO",
        "LGv",
        "VL",
        "VPL",
        "POL",
        "Eth",
        "PoT",
        "PP",
        "PIL",
        "IntG",
        "IGL",
        "SGN",
        "VPL",
        "PF",
        "RT",
    ],
    "hippocampus": ["CA1", "CA2", "CA3", "DG", "SUB", "POST", "PRE", "ProS", "HPF"],
    "midbrain": [
        "MB",
        "SCig",
        "SCiw",
        "SCsg",
        "SCzo",
        "PPT",
        "APN",
        "NOT",
        "MRN",
        "OP",
        "LT",
        "RPF",
        "CP",
    ],
}

color_dict = {'cortex' : '#08858C',
              'thalamus' : '#FC6B6F',
              'hippocampus' : '#7ED04B',
              'midbrain' : '#FC9DFE'}