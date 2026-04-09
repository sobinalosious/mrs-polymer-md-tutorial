from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


EPSILON_0 = 8.854187817e-12
KB = 1.380649e-23
CONVERSION_DIPOLE = 1.602176634e-29
CONVERSION_VOLUME = 1e-30


def tutorial_root() -> Path:
    return Path(__file__).resolve().parent


def output_dir(root: Path | None = None) -> Path:
    root = tutorial_root() if root is None else Path(root)
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    return out


def use_workshop_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfa",
            "axes.edgecolor": "#2f3b52",
            "axes.labelcolor": "#1f2937",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.linewidth": 1.0,
            "xtick.color": "#1f2937",
            "ytick.color": "#1f2937",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "grid.color": "#d6dce5",
            "grid.alpha": 0.6,
            "grid.linewidth": 0.8,
            "font.size": 11.5,
            "font.family": "DejaVu Serif",
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d6dce5",
            "legend.framealpha": 0.95,
            "lines.linewidth": 2.3,
            "lines.markersize": 6,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.tick_params(direction="out", length=4.5, width=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis == "y":
        ax.grid(axis="y")
    elif grid_axis == "x":
        ax.grid(axis="x")
    else:
        ax.grid(True)


def save_figure_bundle(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"))
    fig.savefig(stem.with_suffix(".svg"))


def print_section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def check_tutorial_setup(root: Path | str | None = None) -> dict[str, Any]:
    root = tutorial_root() if root is None else Path(root)
    required = [
        root / "TG" / "tg_density_data.dat",
        root / "TC" / "tc_thermo_data.dat",
        root / "TC" / "tc_temperature_profile.dat",
        root / "DC" / "dc_dipole_fluctuation_data.dat",
        root / "CP" / "cp_enthalpy_temperature_data.dat",
    ]
    missing_required = [str(path.relative_to(root)) for path in required if not path.exists()]
    print_section("Tutorial File Check")
    if missing_required:
        print("Missing required files:")
        for item in missing_required:
            print(f"  - {item}")
    else:
        print("All required TG / TC / DC raw output files are present.")

    return {
        "missing_required": missing_required,
    }


def run_tg_analysis(
    root: Path | str | None = None,
    sample_name: str = "Tutorial Polymer",
    density_file: str = "TG/tg_density_data.dat",
    rough_tg_guess_k: float = 350.0,
    n_eq: int = 5000,
    n_prod: int = 5000,
    fit_sizes: tuple[int, ...] = (5, 6, 7, 8),
    save_figure: bool = True,
) -> dict[str, Any]:
    root = tutorial_root() if root is None else Path(root)
    use_workshop_style()

    density_path = root / density_file
    data = np.loadtxt(density_path, comments="#")
    temps_all = data[:, 1]
    dens_all = data[:, 2]

    nstep = n_eq + n_prod
    n_total = len(temps_all)
    n_tpoints = n_total // nstep
    if n_tpoints < 2:
        raise ValueError("Not enough temperature points found for Tg fitting.")

    avg_t = []
    avg_rho = []
    for i in range(n_tpoints):
        start = i * nstep + n_eq
        stop = start + n_prod
        if stop > n_total:
            break
        avg_t.append(np.mean(temps_all[start:stop]))
        avg_rho.append(np.mean(dens_all[start:stop]))

    avg_t = np.array(avg_t)
    avg_rho = np.array(avg_rho)

    order = np.argsort(avg_t)
    t_use = avg_t[order]
    rho_use = avg_rho[order]

    valid_fit_sizes = [n for n in fit_sizes if 1 <= n <= len(t_use) // 2]
    if not valid_fit_sizes:
        raise ValueError("No valid fit sizes remain for the Tg analysis.")

    fit_records: list[dict[str, float]] = []
    for n in valid_fit_sizes:
        low_idx = np.arange(n)
        high_idx = np.arange(len(t_use) - n, len(t_use))

        sl_low, ic_low, *_ = stats.linregress(t_use[low_idx], rho_use[low_idx])
        sl_high, ic_high, *_ = stats.linregress(t_use[high_idx], rho_use[high_idx])
        tg = (ic_high - ic_low) / (sl_low - sl_high)
        fit_records.append(
            {
                "n_fit": float(n),
                "tg_k": float(tg),
                "sl_low": float(sl_low),
                "ic_low": float(ic_low),
                "sl_high": float(sl_high),
                "ic_high": float(ic_high),
            }
        )

    tg_vals = np.array([record["tg_k"] for record in fit_records])
    tg_mean = float(np.mean(tg_vals))
    tg_std = float(np.std(tg_vals, ddof=0))

    lower_bound = rough_tg_guess_k - 150.0
    upper_bound = rough_tg_guess_k + 150.0
    within_window = bool(
        np.all((tg_vals >= lower_bound) & (tg_vals <= upper_bound))
        and lower_bound <= tg_mean <= upper_bound
    )

    representative = fit_records[len(fit_records) // 2]
    n = int(representative["n_fit"])
    low_idx = np.arange(n)
    high_idx = np.arange(len(t_use) - n, len(t_use))
    rho_tg = representative["sl_low"] * tg_mean + representative["ic_low"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    axes[0].plot(
        t_use,
        rho_use,
        marker="o",
        linewidth=2.4,
        color="#0f766e",
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Averaged density",
    )
    axes[0].scatter(
        t_use[low_idx],
        rho_use[low_idx],
        s=70,
        color="#c2410c",
        edgecolors="white",
        linewidths=0.9,
        label="Low-T fit region",
        zorder=3,
    )
    axes[0].scatter(
        t_use[high_idx],
        rho_use[high_idx],
        s=70,
        color="#6d28d9",
        edgecolors="white",
        linewidths=0.9,
        label="High-T fit region",
        zorder=3,
    )
    axes[0].plot(
        t_use,
        representative["sl_low"] * t_use + representative["ic_low"],
        linestyle="--",
        linewidth=2.2,
        color="#c2410c",
        label=f"Low-T fit (n={n})",
    )
    axes[0].plot(
        t_use,
        representative["sl_high"] * t_use + representative["ic_high"],
        linestyle="--",
        linewidth=2.2,
        color="#6d28d9",
        label=f"High-T fit (n={n})",
    )
    axes[0].scatter(
        [tg_mean],
        [rho_tg],
        color="#dc2626",
        edgecolors="white",
        linewidths=1.0,
        s=120,
        label=f"Tg = {tg_mean:.1f} K",
        zorder=4,
    )
    axes[0].axvline(tg_mean, color="#dc2626", linestyle=":", linewidth=1.8)
    axes[0].set_title("Density-Temperature Crossover")
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Density (g/cm$^3$)")
    axes[0].legend(loc="best")
    style_axis(axes[0])

    axes[1].plot(
        valid_fit_sizes,
        tg_vals,
        marker="o",
        linewidth=2.4,
        color="#1d4ed8",
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    axes[1].axhspan(tg_mean - tg_std, tg_mean + tg_std, color="#bfdbfe", alpha=0.45, label="Mean ± std")
    axes[1].axhline(tg_mean, color="#1d4ed8", linestyle="--", linewidth=2, label=f"Mean Tg = {tg_mean:.1f} K")
    axes[1].axhline(lower_bound, color="#9ca3af", linestyle=":", linewidth=1.5, label="Acceptance window")
    axes[1].axhline(upper_bound, color="#9ca3af", linestyle=":", linewidth=1.5)
    axes[1].set_title("Sensitivity to Fit Window")
    axes[1].set_xlabel("Number of points in each fit")
    axes[1].set_ylabel("Estimated Tg (K)")
    axes[1].legend(loc="best")
    style_axis(axes[1], grid_axis="y")

    fig.suptitle(f"Glass Transition Tutorial: {sample_name}", fontsize=16, fontweight="bold")

    if save_figure:
        save_figure_bundle(fig, output_dir(root) / "glass_transition_summary")

    print_section("Glass Transition Temperature")
    print(f"Sample: {sample_name}")
    print(f"Input file: {density_path.name}")
    print(f"Cooling points used: {len(t_use)}")
    print(f"Rough Tg guide: {rough_tg_guess_k:.1f} K")
    print("\nFit-by-fit estimates:")
    for record in fit_records:
        print(f"  n_fit = {int(record['n_fit'])}: Tg = {record['tg_k']:.2f} K")
    print(f"\nFinal Tg = {tg_mean:.2f} +/- {tg_std:.2f} K")
    if within_window:
        print("Result status: accepted within the expected Tg window.")
    else:
        print("Result status: outside the expected Tg window. Review the fit regions.")

    return {
        "sample_name": sample_name,
        "tg_mean_k": tg_mean,
        "tg_std_k": tg_std,
        "accepted": within_window,
        "fit_records": fit_records,
        "figure": fig,
    }


def calculate_dielectric_constant_component(
    dipole_moments: np.ndarray,
    avg_volume_angstrom3: float,
    temperature_k: float,
) -> float:
    m_sq_avg = np.mean(np.sum(dipole_moments**2, axis=1))
    m_avg_sq = np.sum(np.mean(dipole_moments, axis=0) ** 2)
    avg_volume_m3 = avg_volume_angstrom3 * CONVERSION_VOLUME
    return 1.0 + ((m_sq_avg - m_avg_sq) * (CONVERSION_DIPOLE**2)) / (
        EPSILON_0 * KB * temperature_k * avg_volume_m3 * 3.0
    )


def run_dc_analysis(
    root: Path | str | None = None,
    sample_name: str = "Tutorial Polymer",
    dipole_file: str = "DC/dc_dipole_fluctuation_data.dat",
    temperature_k: float = 300.0,
    save_figure: bool = True,
) -> dict[str, Any]:
    root = tutorial_root() if root is None else Path(root)
    use_workshop_style()

    dipole_path = root / dipole_file
    raw = np.loadtxt(dipole_path)
    dipole = raw[:, :3]
    magnitude = raw[:, 3]
    volume = raw[:, 4]
    avg_volume_angstrom3 = float(np.mean(volume))

    epsilon_dipole = float(
        calculate_dielectric_constant_component(
            dipole_moments=dipole,
            avg_volume_angstrom3=avg_volume_angstrom3,
            temperature_k=temperature_k,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    sample_index = np.arange(len(dipole))

    stride = max(1, len(dipole) // 1000)
    axes[0].plot(sample_index[::stride], dipole[::stride, 0], color="#dc2626", linewidth=1.8, alpha=0.95, label="Mx")
    axes[0].plot(sample_index[::stride], dipole[::stride, 1], color="#0284c7", linewidth=1.8, alpha=0.95, label="My")
    axes[0].plot(sample_index[::stride], dipole[::stride, 2], color="#059669", linewidth=1.8, alpha=0.95, label="Mz")
    axes[0].axhline(0.0, color="#94a3b8", linestyle=":", linewidth=1.2)
    axes[0].set_title("Dipole-Moment Components")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Dipole moment (e*A)")
    axes[0].legend(loc="upper right", ncol=3)
    style_axis(axes[0])

    axes[1].plot(
        sample_index[::stride],
        magnitude[::stride],
        color="#7c3aed",
        linewidth=2.1,
        label="|M|",
    )
    axes[1].fill_between(sample_index[::stride], magnitude[::stride], color="#ddd6fe", alpha=0.55)
    axes[1].axhline(np.mean(magnitude), color="#4c1d95", linestyle="--", linewidth=1.8, label=f"Mean |M| = {np.mean(magnitude):.2f}")
    axes[1].set_title("Total Dipole Magnitude")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("|M| (e*A)")
    axes[1].legend(loc="upper right")
    axes[1].text(
        0.03,
        0.95,
        f"$\\epsilon_{{dipole}}$ = {epsilon_dipole:.3f}\n"
        f"T = {temperature_k:.0f} K\n"
        f"Samples = {len(dipole)}\n"
        f"Average volume = {avg_volume_angstrom3:.1f} A$^3$",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=11.2,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.45"},
    )
    style_axis(axes[1])

    fig.suptitle(f"Dielectric Constant Tutorial: {sample_name}", fontsize=16, fontweight="bold")

    if save_figure:
        save_figure_bundle(fig, output_dir(root) / "dielectric_constant_summary")

    print_section("Dielectric Constant")
    print(f"Sample: {sample_name}")
    print(f"Input file: {dipole_path.name}")
    print(f"Samples used: {len(dipole)}")
    print(f"Average volume: {avg_volume_angstrom3:.2f} A^3")
    print(f"Dipolar dielectric constant = {epsilon_dipole:.4f}")
    print("Reported quantity in this tutorial: dipole component only")

    return {
        "sample_name": sample_name,
        "epsilon_dipole": epsilon_dipole,
        "figure": fig,
    }


def _split_thermo_windows(thermo_df: pd.DataFrame, num_points: int) -> list[pd.DataFrame]:
    if len(thermo_df) < num_points:
        raise ValueError(f"Thermo file has only {len(thermo_df)} rows; expected at least {num_points}.")
    data = thermo_df.tail(num_points).reset_index(drop=True)
    batch = len(data) // 10
    return [data.iloc[i * batch : (i + 1) * batch].copy() for i in range(10)]


def _split_temp_windows(profile_file: Path) -> list[pd.DataFrame]:
    with open(profile_file, "r", encoding="utf-8") as handle:
        contents = handle.readlines()

    contents_for_use = contents[3:]
    batch = len(contents_for_use) // 10
    windows = []
    for i in range(10):
        subset = contents_for_use[i * batch : (i + 1) * batch]
        keep_positions = []
        cleaned_rows = []
        for j, line in enumerate(subset):
            parts = line.split()
            if len(parts) != 4:
                continue
            temperature = float(parts[3])
            if temperature >= 250.0:
                keep_positions.append(j)
        if len(keep_positions) < 2:
            continue
        head = keep_positions[1]
        tail = keep_positions[-1]
        for line in subset[head:tail]:
            parts = line.split()
            if len(parts) != 4:
                continue
            cleaned_rows.append((float(parts[1]), float(parts[3])))
        windows.append(pd.DataFrame(cleaned_rows, columns=["distance", "temperature"]))
    return windows


def _heat_flux_regression(window_df: pd.DataFrame) -> dict[str, Any]:
    x_length = float(window_df["Lx"].iloc[0]) * 1e-10
    y_length = float(window_df["Ly"].iloc[0]) * 1e-10
    z_length = float(window_df["Lz"].iloc[0]) * 1e-10
    time = window_df["Step"].to_numpy(dtype=float) * 0.25 * 1e-15
    heat_source = window_df["f_4"].to_numpy(dtype=float)
    heat_sink = window_df["f_5"].to_numpy(dtype=float)
    energy = ((heat_sink - heat_source) / (2.0 * 23.06)) * 1.6022e-19
    slope, intercept, r_value, p_value, std_err = stats.linregress(time, energy)
    return {
        "time": time,
        "energy": energy,
        "heat_source": heat_source,
        "heat_sink": heat_sink,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
        "x_length": x_length,
        "y_length": y_length,
        "z_length": z_length,
    }


def _temperature_gradient_regression(temp_df: pd.DataFrame) -> dict[str, Any]:
    distance = temp_df["distance"].to_numpy(dtype=float)
    temperature = temp_df["temperature"].to_numpy(dtype=float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(distance, temperature)
    return {
        "distance": distance,
        "temperature": temperature,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_value": float(r_value),
    }


def _thermal_conductivity(
    slope_heat: float,
    slope_temp: float,
    x_length: float,
    y_length: float,
    z_length: float,
) -> float:
    return float(slope_heat / (y_length * z_length * abs(slope_temp / x_length)))


def run_tc_analysis(
    root: Path | str | None = None,
    sample_name: str = "Tutorial Polymer",
    thermo_file: str = "TC/tc_thermo_data.dat",
    profile_file: str = "TC/tc_temperature_profile.dat",
    num_points: int = 2000,
    num_sets: int = 8,
    save_figure: bool = True,
) -> dict[str, Any]:
    root = tutorial_root() if root is None else Path(root)
    use_workshop_style()

    thermo_path = root / thermo_file
    profile_path = root / profile_file

    thermo_df = pd.read_csv(
        thermo_path,
        sep=r"\s+",
        header=None,
        names=[
            "Step",
            "Temp",
            "E_pair",
            "E_mol",
            "TotEng",
            "Press",
            "Density",
            "Volume",
            "Lx",
            "Ly",
            "Lz",
            "f_4",
            "f_5",
        ],
    )

    hf_windows = _split_thermo_windows(thermo_df, num_points=num_points)
    temp_windows = _split_temp_windows(profile_path)
    if len(temp_windows) < len(hf_windows):
        raise ValueError("Temperature profile does not contain enough windowed segments.")

    start_index = max(0, len(hf_windows) - num_sets)
    selected_hf = hf_windows[start_index:]
    selected_temp = temp_windows[start_index : start_index + len(selected_hf)]

    records = []
    for offset, (hf_df, temp_df) in enumerate(zip(selected_hf, selected_temp), start=start_index + 1):
        heat = _heat_flux_regression(hf_df)
        gradient = _temperature_gradient_regression(temp_df)
        tc_value = _thermal_conductivity(
            slope_heat=heat["slope"],
            slope_temp=gradient["slope"],
            x_length=heat["x_length"],
            y_length=heat["y_length"],
            z_length=heat["z_length"],
        )
        records.append(
            {
                "window": offset,
                "tc": tc_value,
                "heat": heat,
                "gradient": gradient,
            }
        )

    tc_values = np.array([record["tc"] for record in records])
    tc_mean = float(np.mean(tc_values))
    tc_std = float(np.std(tc_values, ddof=0))

    representative = min(records, key=lambda record: abs(record["tc"] - tc_mean))
    rep_heat = representative["heat"]
    rep_gradient = representative["gradient"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)

    axes[0].plot(rep_heat["time"] * 1e12, rep_heat["energy"], color="#1d4ed8", linewidth=2.4, label="Heat transferred")
    axes[0].plot(
        rep_heat["time"] * 1e12,
        rep_heat["slope"] * rep_heat["time"] + rep_heat["intercept"],
        color="#dc2626",
        linestyle="--",
        linewidth=2.2,
        label=f"Linear fit (R = {rep_heat['r_value']:.3f})",
    )
    axes[0].set_title(f"Representative Heat-Flux Fit (window {representative['window']})")
    axes[0].set_xlabel("Time (ps)")
    axes[0].set_ylabel("Accumulated energy (J)")
    axes[0].legend(loc="best")
    style_axis(axes[0])

    axes[1].scatter(
        rep_gradient["distance"],
        rep_gradient["temperature"],
        s=34,
        color="#0f766e",
        edgecolors="white",
        linewidths=0.5,
        label="Temperature profile",
    )
    axes[1].plot(
        rep_gradient["distance"],
        rep_gradient["slope"] * rep_gradient["distance"] + rep_gradient["intercept"],
        color="#f97316",
        linestyle="--",
        linewidth=2.2,
        label=f"Linear fit (R = {rep_gradient['r_value']:.3f})",
    )
    axes[1].set_title(f"Representative Temperature Gradient (window {representative['window']})")
    axes[1].set_xlabel("Reduced position")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].legend(loc="best")
    style_axis(axes[1])

    window_ids = [record["window"] for record in records]
    axes[2].plot(
        window_ids,
        tc_values,
        marker="o",
        linewidth=2.4,
        color="#7c3aed",
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    axes[2].axhspan(tc_mean - tc_std, tc_mean + tc_std, color="#ddd6fe", alpha=0.5, label="Mean ± std")
    axes[2].axhline(tc_mean, color="#6d28d9", linestyle="--", linewidth=1.8, label=f"Mean = {tc_mean:.4e}")
    axes[2].set_title("Window-by-Window Thermal Conductivity")
    axes[2].set_xlabel("Window")
    axes[2].set_ylabel("Thermal conductivity")
    axes[2].legend(loc="best")
    style_axis(axes[2], grid_axis="y")

    fig.suptitle(f"Thermal Conductivity Tutorial: {sample_name}", fontsize=16, fontweight="bold")

    if save_figure:
        save_figure_bundle(fig, output_dir(root) / "thermal_conductivity_summary")

    print_section("Thermal Conductivity")
    print(f"Sample: {sample_name}")
    print(f"Input files: {thermo_path.name}, {profile_path.name}")
    print(f"Thermo points used: {num_points}")
    print(f"Windows used in final average: {len(records)}")
    print("\nWindow-by-window conductivity:")
    for record in records:
        print(f"  window {record['window']:>2}: k = {record['tc']:.6e}")
    print(f"\nMean thermal conductivity = {tc_mean:.6e}")
    print(f"Standard deviation = {tc_std:.6e}")

    return {
        "sample_name": sample_name,
        "tc_mean": tc_mean,
        "tc_std": tc_std,
        "records": records,
        "figure": fig,
    }


def run_cp_analysis(
    root: Path | str | None = None,
    sample_name: str = "Tutorial Polymer",
    cp_data_file: str = "CP/cp_enthalpy_temperature_data.dat",
    num_blocks: int = 100,
    save_figure: bool = True,
) -> dict[str, Any]:
    root = tutorial_root() if root is None else Path(root)
    use_workshop_style()

    cp_path = root / cp_data_file
    data = np.loadtxt(cp_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(-1, 5)

    temperature = data[:, 0]
    total_energy_kcal = data[:, 1]
    volume_ang3 = data[:, 3]
    density_gcc = data[:, 4]

    total_steps = len(data)
    block_size = total_steps // num_blocks
    if block_size < 1:
        raise ValueError("num_blocks is too large for the Cp dataset.")

    ang3_to_m3 = 1e-30
    kcal_to_j = 4184.0
    atm_to_pa = 101325.0
    avogadro_number = 6.02214076e23

    volume_m3 = volume_ang3 * ang3_to_m3
    density_kg_m3 = density_gcc * 1000.0
    energy_j = total_energy_kcal * kcal_to_j / avogadro_number
    enthalpy_j = energy_j + atm_to_pa * volume_m3
    mass_kg = volume_m3 * density_kg_m3

    avg_temperatures = []
    avg_enthalpies = []
    avg_masses = []
    for i in range(num_blocks):
        block = data[i * block_size : (i + 1) * block_size]
        if len(block) == 0:
            continue
        t_block = block[:, 0]
        e_block = block[:, 1]
        v_block = block[:, 3] * ang3_to_m3
        rho_block = block[:, 4] * 1000.0
        e_j_block = e_block * kcal_to_j / avogadro_number
        h_j_block = e_j_block + atm_to_pa * v_block
        m_block = v_block * rho_block
        avg_temperatures.append(np.mean(t_block))
        avg_enthalpies.append(np.mean(h_j_block))
        avg_masses.append(np.mean(m_block))

    avg_temperatures = np.array(avg_temperatures)
    avg_enthalpies = np.array(avg_enthalpies)
    avg_masses = np.array(avg_masses)

    slope_raw, intercept_raw = np.polyfit(temperature, enthalpy_j, 1)
    fit_raw = slope_raw * temperature + intercept_raw
    slope_avg, intercept_avg = np.polyfit(avg_temperatures, avg_enthalpies, 1)
    fit_avg = slope_avg * avg_temperatures + intercept_avg
    cp_value = float(slope_avg / np.mean(avg_masses))

    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    ax.scatter(
        temperature[:: max(1, len(temperature) // 2000)],
        enthalpy_j[:: max(1, len(enthalpy_j) // 2000)],
        s=10,
        color="#cbd5e1",
        alpha=0.55,
        label="Raw data",
    )
    ax.plot(
        np.sort(temperature),
        fit_raw[np.argsort(temperature)],
        color="#94a3b8",
        linewidth=1.6,
        linestyle="--",
        label="Raw-data fit",
    )
    ax.scatter(
        avg_temperatures,
        avg_enthalpies,
        s=42,
        color="#b91c1c",
        edgecolors="white",
        linewidths=0.8,
        label="Block averages",
        zorder=3,
    )
    ax.plot(
        avg_temperatures,
        fit_avg,
        color="#7f1d1d",
        linewidth=2.5,
        label="Block-averaged fit",
    )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Enthalpy (J)")
    ax.set_title(f"Enthalpy vs Temperature: {sample_name}")
    ax.legend(loc="best")
    style_axis(ax)
    ax.text(
        0.04,
        0.96,
        f"$C_p$ = {cp_value:.2f} J kg$^{{-1}}$ K$^{{-1}}$\n"
        f"Blocks = {len(avg_temperatures)}\n"
        f"dH/dT = {slope_avg:.3e} J K$^{{-1}}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11.0,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.45"},
    )

    if save_figure:
        save_figure_bundle(fig, output_dir(root) / "specific_heat_capacity_summary")

    print_section("Specific Heat Capacity")
    print(f"Sample: {sample_name}")
    print(f"Input file: {cp_path.name}")
    print(f"Raw-data slope (dH/dT) = {slope_raw:.4e} J/K")
    print(f"Block-fit slope (dH/dT) = {slope_avg:.4e} J/K")
    print(f"Mean mass per block = {np.mean(avg_masses):.4e} kg")
    print(f"Cp = {cp_value:.4f} J/(kg·K)")

    return {
        "sample_name": sample_name,
        "cp_value": cp_value,
        "figure": fig,
    }
