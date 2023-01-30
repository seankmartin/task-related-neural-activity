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
