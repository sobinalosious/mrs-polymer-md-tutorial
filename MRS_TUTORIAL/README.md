# Polymer MD Property Tutorial

This folder is an analysis-only tutorial for three polymer properties derived from precomputed molecular dynamics outputs:

- glass transition temperature (`Tg`)
- dielectric constant (`DC`)
- thermal conductivity (`TC`)
- specific heat capacity (`Cp`)

The tutorial is designed for conference or workshop delivery with a small, self-contained analysis environment that does not require LAMMPS, HPC access, or the full polymer-generation workflow.

## How To Use

Open the notebooks in `notebooks/` and run them top-to-bottom.

Each notebook:

- introduces the property and the corresponding MD observable
- loads the prepared output file(s)
- makes clear plots automatically
- prints the final result in a readable format
- saves a summary figure to `outputs/`

## Folder Layout

```text
MRS_TUTORIAL/
  TC/                         raw thermal-conductivity MD outputs
  TG/                         raw glass-transition MD outputs
  DC/                         raw dielectric MD outputs
  notebooks/                  tutorial notebooks
  outputs/                    figures saved during notebook runs
  tutorial_config.py          editable tutorial inputs
  tutorial_utils.py           shared analysis + plotting helpers
  environment.yml             Binder / conda environment
  requirements.txt            pip fallback
```

## Recommended Delivery Mode

Use Binder as the primary workshop link.

This repository now also includes a repo-root Binder environment in `binder/environment.yml`, so you can launch the whole repository directly into the tutorial notebooks.

After pushing this folder to GitHub, a Binder launch URL can look like:

```text
https://mybinder.org/v2/gh/<github-user>/<repo>/<branch>?urlpath=lab/tree/MRS_TUTORIAL/notebooks/00_Start_Here.ipynb
```

You can also use local Jupyter:

```bash
cd MRS_TUTORIAL
conda env create -f environment.yml
conda activate polymer-md-tutorial
jupyter lab
```

## Notebook Order

1. `00_Start_Here.ipynb`
2. `01_Glass_Transition_Temperature.ipynb`
3. `02_Dielectric_Constant.ipynb`
4. `03_Thermal_Conductivity.ipynb`
5. `04_Specific_Heat_Capacity.ipynb`

## Inputs You Should Review Before Sharing

Open `tutorial_config.py` and check the editable values.

All three tutorial cases are self-contained with the files currently present.

For dielectric constant, this tutorial intentionally focuses only on the dipole component from:

- `DC/dc_dipole_fluctuation_data.dat`

That keeps the exercise aligned with the workshop goal of teaching how dielectric response is extracted from MD fluctuations.

For specific heat capacity, the current folder includes a precomputed Cp result file:

- `CP/cp_enthalpy_temperature_data.dat`

The Cp notebook uses this file to generate an enthalpy-vs-temperature plot and compute Cp from the fitted slope.

## Teaching Notes

This setup is intentionally analysis-first:

- no installation of LAMMPS or PySIMM
- no job submission
- no HPC queue waits

That keeps the tutorial aligned with the goal of teaching how polymer properties are calculated from MD outputs, rather than teaching cluster operations.
