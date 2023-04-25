import simuran as smr
from simuran.loaders.one_loader import OneAlyxLoader

from trna.filter_to_regions import filter_ibl_table


def load_ibl(ibl_cache_dir, brain_region_table=None, brain_regions=None):
    """
    Load the IBL data.

    Parameters
    ----------
    ibl_cache_dir : str
        The directory to cache the data in.

    Returns
    -------
    one_recording_container : RecordingContainer
        The recording container for the IBL data.
    one_loader : OneAlyxLoader
        The loader for the IBL data.

    """
    one_loader = OneAlyxLoader(cache_directory=ibl_cache_dir)
    one_sessions = one_loader.get_sessions_table()
    if brain_regions is not None and len(brain_regions) > 0:
        one_sessions = filter_ibl_table(one_sessions, brain_region_table, brain_regions)
    one_recording_container = smr.RecordingContainer.from_table(
        one_sessions, one_loader
    )

    return one_recording_container, one_loader
