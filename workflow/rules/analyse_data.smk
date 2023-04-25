rule analyse_gpfa:
    input:
        f"{config['local_dir']}/OpenDataResults/tables/ibl_brain_regions.csv"
    output:
        f"{config['local_dir']}/OpenDataResults/tables/allen_gpfa.csv",
        f"{config['local_dir']}/OpenDataResults/tables/ibl_gpfa.csv"
    params:
        overwrite = True
    script:
        "../scripts/gpfa.py"