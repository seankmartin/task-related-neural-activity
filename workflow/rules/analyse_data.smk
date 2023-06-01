rule analyse_gpfa:
    input:
        f"{config['local_dir']}/OpenDataResults/tables/ibl_brain_regions.csv"
    output:
        f"{config['local_dir']}/OpenDataResults/gpfa/png/gpfa_variance_CA1_TH_SUB_allen.png"
    params:
        overwrite = False
    log:
        f"{config['local_dir']}/OpenDataResults/logs/gpfa.log"
    script:
        "../scripts/gpfa.py"

rule analyse_cca:
    input:
        f"{config['local_dir']}/OpenDataResults/tables/ibl_brain_regions.csv"
    output:
        f"{config['local_dir']}/OpenDataResults/cca/png/allen_CA1_SUB_cca_correlation.png"
    params:
        overwrite = False
    log:
        f"{config['local_dir']}/OpenDataResults/logs/cca.log"
    script:
        "../scripts/cca.py"       

rule plot_units:
    input:
        f"{config['local_dir']}/OpenDataResults/tables/ibl_brain_regions.csv"
    output:
        directory(f"{config['local_dir']}/OpenDataResults/unit_properties")
    log:
        f"{config['local_dir']}/OpenDataResults/logs/unit_plot.log"
    script:
        "../scripts/plot_cells.py"

rule split_tables:
    input:
        f"{config['local_dir']}/OpenDataResults/cca/rate_correlation_allen_CA1_SUB.csv"
    output:
        f"{config['local_dir']}/OpenDataResults/cca/rate_correlation_allen_CA1_SUB_converted.csv"