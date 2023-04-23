rule create_tables:
    output:
        f"{config['local_dir']}/OpenDataResults/tables/allen_sessions.csv",
        f"{config['local_dir']}/OpenDataResults/tables/ibl_sessions.csv",
    script:
        "../scripts/create_tables.py"