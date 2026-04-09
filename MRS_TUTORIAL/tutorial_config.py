"""Editable inputs for the workshop notebooks."""

TG_CONFIG = {
    "sample_name": "Tutorial Polymer",
    "density_file": "TG/tg_density_data.dat",
    "rough_tg_guess_k": 350.0,
    "n_eq": 5000,
    "n_prod": 5000,
    "fit_sizes": (5, 6, 7, 8),
}

DC_CONFIG = {
    "sample_name": "Tutorial Polymer",
    "dipole_file": "DC/dc_dipole_fluctuation_data.dat",
    "temperature_k": 300.0,
}

TC_CONFIG = {
    "sample_name": "Tutorial Polymer",
    "thermo_file": "TC/tc_thermo_data.dat",
    "profile_file": "TC/tc_temperature_profile.dat",
    "num_points": 2000,
    "num_sets": 8,
}
