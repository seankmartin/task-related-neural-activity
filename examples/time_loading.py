from time import perf_counter
from pathlib import Path

from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)
from pynwb import NWBHDF5IO

ALLEN_DIR = Path(r"D:\example-data\allen-data")
ALLEN_MANIFEST = "visual-behavior-neuropixels_project_manifest_v0.4.0.json"
EXAMPLE_SESSION = 1044385384


def timeit(name):
    def inner(func):
        def wrapper(*args, **kwargs):
            t1 = perf_counter()
            func(*args, **kwargs)
            t2 = perf_counter()

            print(f"{name} took {t2 - t1:.2f} seconds")

        return wrapper

    return inner


@timeit("Allen SDK loader")
def allen_example(cache):
    return cache.get_ecephys_session(EXAMPLE_SESSION)


@timeit("NWB loader")
def nwb_example(nwb_path):
    nwb_io = NWBHDF5IO(nwb_path, "r", load_namespaces=True)
    return nwb_io.read()


def main(cache_is_s3=True):
    if cache_is_s3:
        cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=ALLEN_DIR)
    else:
        cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(
            cache_dir=ALLEN_DIR
        )
    cache.load_manifest(ALLEN_MANIFEST)

    nwb_path = (
        ALLEN_DIR
        / "visual-behavior-neuropixels-0.4.0"
        / "behavior_ecephys_sessions"
        / str(EXAMPLE_SESSION)
        / f"ecephys_session_{EXAMPLE_SESSION}.nwb"
    )

    nwb_example(nwb_path)
    allen_example(cache)

    allen_example(cache)
    nwb_example(nwb_path)


main(cache_is_s3=True)
main(cache_is_s3=False)
