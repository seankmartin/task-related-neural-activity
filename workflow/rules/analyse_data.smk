rule analyse_gpfa:
    input:
        f"{config['local_dir']}/OpenDataResults/tables/ibl_brain_regions.csv"
    output:
        directory(f"{config['local_dir']}/OpenDataResults/gpfa")
    params:
        overwrite = False
    log:
        f"{config['local_dir']}/OpenDataResults/logs/gpfa.log"
    script:
        "../scripts/gpfa.py"