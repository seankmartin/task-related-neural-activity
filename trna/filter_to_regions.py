import ast


def filter_allen_table(allen_table, brain_regions):
    def check_brain_regions(x):
        x = ast.literal_eval(x)
        all_matches = []
        for brain_region_set in brain_regions:
            matches = []
            for br in brain_region_set:
                if isinstance(br, str):
                    matches.append(br in x)
                else:
                    matches.append(any([b in x for b in br]))
            all_matches.append(all(matches))
        return any(all_matches)

    desired_sessions = allen_table[
        allen_table["structure_acronyms"].apply(check_brain_regions)
    ]
    filtered_table = allen_table.loc[
        allen_table["behavior_session_id"].isin(desired_sessions["behavior_session_id"])
    ]
    return filtered_table


def filter_ibl_table(ibl_table, region_table, brain_regions):
    def check_brain_regions(x):
        x = ast.literal_eval(x)
        all_matches = []
        for brain_region_set in brain_regions:
            matches = []
            for br in brain_region_set:
                if isinstance(br, str):
                    matches.append(br in x)
                else:
                    matches.append(any([b in x for b in br]))
            all_matches.append(all(matches))
        return any(all_matches)

    desired_sessions = region_table[region_table["regions"].apply(check_brain_regions)]
    filtered_table = ibl_table.loc[
        ibl_table["session"].isin(desired_sessions["session"])
    ]
    return filtered_table
