rule analyse_gpfa:
    input:
        f"{config['local_dir']}/OpenDataResults/tables/ibl_brain_regions.csv"
    output:
        directory(f"{config['local_dir']}/OpenDataResults/gpfa/TODO")
    params:
        overwrite = False
    log:
        f"{config['local_dir']}/OpenDataResults/logs/gpfa.log"
    script:
        "../scripts/gpfa.py"
        
rule plot_units:
    input:
        f"{config['local_dir']}/OpenDataResults/tables/ibl_brain_regions.csv"
    output:
        directory(f"{config['local_dir']}/OpenDataResults/unit_properties")
    log:
        f"{config['local_dir']}/OpenDataResults/logs/unit_plot.log"
    script:
        "../scripts/plot_cells.py"