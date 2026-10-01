import json
import os


DATASET_SHORT_NAMES = {
    "abalone": "abalone",
    "ada": "ada_prior",
    "avila": "avila",
    "connect": "connect-4",
    "eeg": "eeg",
    "google": "google",
    "house": "house_prices",
    "jungle": "jungle_chess_2pcs_raw_endgame_complete",
    "micro": "microaggregation2",
    "mozilla4": "mozilla4",
    "obesity": "obesity",
    "page": "page-blocks",
    "pbcseq": "pbcseq",
    "pol": "pol",
    "run_or_walk": "Run_or_walk_information",
    "shuttle": "shuttle",
    "uscensus": "USCensus",
    "wall": "wall-robot-navigation",
}

# Keep backward-compatible aliases while making DATASET_SHORT_NAMES the
# canonical set shown by command-line tools.
DATASET_ALIASES = {
    **DATASET_SHORT_NAMES,
    "google_play_store": "google",
    "google_play_store_apps": "google",
    "house_prices": "house_prices",
    "jungle_chess": "jungle_chess_2pcs_raw_endgame_complete",
    "jungle_chess_2pcs_raw_endgame_complete": "jungle_chess_2pcs_raw_endgame_complete",
    "microaggregation2": "microaggregation2",
    "page_blocks": "page-blocks",
    "page-blocks": "page-blocks",
    "runorwalk": "Run_or_walk_information",
    "run_or_walk_information": "Run_or_walk_information",
    "us_census": "USCensus",
    "wall_robot_navigation": "wall-robot-navigation",
    "wall-robot-navigation": "wall-robot-navigation",
}


def find_project_root(start_path):
    cur = os.path.abspath(start_path)
    while True:
        data_dir = os.path.join(cur, "data")
        if os.path.isdir(data_dir):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def resolve_dataset_name(dataset_name):
    name = str(dataset_name or "").strip()
    if not name:
        return "google"
    return DATASET_ALIASES.get(name.lower(), name)


def list_dataset_short_names():
    """Return the command-line dataset aliases and their data-directory names."""
    return dict(DATASET_SHORT_NAMES)


def resolve_dataset_paths(project_root, dataset_name):
    ds_name = resolve_dataset_name(dataset_name)
    ds_dir = os.path.join(project_root, "data", ds_name)
    data_path = os.path.join(ds_dir, "data.csv")
    info_path = os.path.join(ds_dir, "info.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"dataset data file not found: {data_path}")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"dataset info file not found: {info_path}")
    return ds_name, data_path, info_path


def load_dataset_label(project_root, dataset_name):
    ds_name, _, info_path = resolve_dataset_paths(project_root, dataset_name)
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    label = info.get("label")
    if not label:
        raise ValueError(f"Missing 'label' in {info_path}")
    return ds_name, label
