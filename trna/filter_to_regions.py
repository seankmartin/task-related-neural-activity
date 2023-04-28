import ast


def filter_allen_table(allen_table, brain_regions):
    def filter_regions(x):
        x = ast.literal_eval(x)
        return all([br in x for br in brain_regions])

    desired_sessions = allen_table[
        allen_table["structure_acronyms"].apply(filter_regions)
    ]
    filtered_table = allen_table.loc[
        allen_table["behavior_session_id"].isin(desired_sessions["behavior_session_id"])
    ]
    return filtered_table


def filter_ibl_table(ibl_table, region_table, brain_regions):
    def check_brain_regions(x):
        x = ast.literal_eval(x)
        matches = []
        for br in brain_regions:
            if isinstance(br, str):
                matches.append(br in x)
            else:
                matches.append(any([b in x for b in br]))

    desired_sessions = region_table[
        region_table["regions"].apply(lambda x: all([br in x for br in brain_regions]))
    ]
    filtered_table = ibl_table.loc[
        ibl_table["session"].isin(desired_sessions["session"])
    ]
    return filtered_table
