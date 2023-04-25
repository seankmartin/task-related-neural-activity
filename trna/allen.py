import simuran as smr
from simuran.loaders.allen_loader import AllenVisualBehaviorLoader

from trna.filter_to_regions import filter_allen_table


def load_allen(allen_cache_dir, allen_manifest, brain_regions=None):
    """
    Load the Allen data.

    Parameters
    ----------
    allen_cache_dir : str
        The directory to cache the data in.
    allen_manifest : str
        The path to the manifest file.

    Returns
    -------
    allen_recording_container : RecordingContainer
        The recording container for the Allen data.
    allen_loader : AllenVisualBehaviorLoader
        The loader for the Allen data.

    """
    allen_loader = AllenVisualBehaviorLoader(
        cache_directory=allen_cache_dir, manifest=allen_manifest
    )
    allen_sessions = allen_loader.get_sessions_table()
    if brain_regions is not None and len(brain_regions) > 0:
        allen_sessions = filter_allen_table(allen_sessions, brain_regions)
    allen_recording_container = smr.RecordingContainer.from_table(
        allen_sessions, allen_loader
    )

    return allen_recording_container, allen_loader
