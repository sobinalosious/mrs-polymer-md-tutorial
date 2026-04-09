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
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbf8",
            "axes.edgecolor": "#2c2c2c",
            "axes.labelcolor": "#1f1f1f",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.color": "#1f1f1f",
            "ytick.color": "#1f1f1f",
            "grid.color": "#d7d7d2",
            "grid.alpha": 0.7,
            "font.size": 11,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d8d8d8",
            "savefig.bbox": "tight",
        }
    )


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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(t_use, rho_use, marker="o", linewidth=2.2, color="#0f766e", label="Averaged density")
    axes[0].scatter(t_use[low_idx], rho_use[low_idx], s=65, color="#b45309", label="Low-T fit region", zorder=3)
    axes[0].scatter(
        t_use[high_idx],
        rho_use[high_idx],
        s=65,
        color="#7c3aed",
        label="High-T fit region",
        zorder=3,
    )
    axes[0].plot(
        t_use,
        representative["sl_low"] * t_use + representative["ic_low"],
        linestyle="--",
        linewidth=2,
        color="#b45309",
        label=f"Low-T fit (n={n})",
    )
    axes[0].plot(
        t_use,
        representative["sl_high"] * t_use + representative["ic_high"],
        linestyle="--",
        linewidth=2,
        color="#7c3aed",
        label=f"High-T fit (n={n})",
    )
    axes[0].scatter([tg_mean], [rho_tg], color="#dc2626", s=110, label=f"Tg = {tg_mean:.1f} K", zorder=4)
    axes[0].axvline(tg_mean, color="#dc2626", linestyle=":", linewidth=1.8)
    axes[0].set_title("Density-Temperature Crossover")
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Density (g/cm$^3$)")
    axes[0].legend(loc="best")

    axes[1].plot(valid_fit_sizes, tg_vals, marker="o", linewidth=2.2, color="#2563eb")
    axes[1].axhspan(tg_mean - tg_std, tg_mean + tg_std, color="#93c5fd", alpha=0.35, label="Mean ± std")
    axes[1].axhline(tg_mean, color="#1d4ed8", linestyle="--", linewidth=2, label=f"Mean Tg = {tg_mean:.1f} K")
    axes[1].axhline(lower_bound, color="#9ca3af", linestyle=":", linewidth=1.5, label="Acceptance window")
    axes[1].axhline(upper_bound, color="#9ca3af", linestyle=":", linewidth=1.5)
    axes[1].set_title("Sensitivity to Fit Window")
    axes[1].set_xlabel("Number of points in each fit")
    axes[1].set_ylabel("Estimated Tg (K)")
    axes[1].legend(loc="best")

    fig.suptitle(f"Glass Transition Tutorial: {sample_name}", fontsize=16, fontweight="bold")

    if save_figure:
        fig.savefig(output_dir(root) / "glass_transition_summary.png", dpi=220)

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

    sample_step = max(20, len(dipole) // 120)
    running_x = []
    running_eps = []
    for end in range(200, len(dipole) + 1, sample_step):
        running_x.append(end)
        running_eps.append(
            calculate_dielectric_constant_component(
                dipole_moments=dipole[:end],
                avg_volume_angstrom3=float(np.mean(volume[:end])),
                temperature_k=temperature_k,
            )
        )

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sample_index = np.arange(len(dipole))

    stride = max(1, len(dipole) // 1000)
    axes[0, 0].plot(sample_index[::stride], dipole[::stride, 0], linewidth=1.5, color="#ef4444", label="Mx")
    axes[0, 0].plot(sample_index[::stride], dipole[::stride, 1], linewidth=1.5, color="#0ea5e9", label="My")
    axes[0, 0].plot(sample_index[::stride], dipole[::stride, 2], linewidth=1.5, color="#10b981", label="Mz")
    axes[0, 0].set_title("Dipole Components vs Sample")
    axes[0, 0].set_xlabel("Sample index")
    axes[0, 0].set_ylabel("Dipole moment (e*A)")
    axes[0, 0].legend(loc="best")

    axes[0, 1].hist(magnitude, bins=35, color="#8b5cf6", edgecolor="white")
    axes[0, 1].set_title("Distribution of Total Dipole Magnitude")
    axes[0, 1].set_xlabel("|M| (e*A)")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].plot(running_x, running_eps, color="#1d4ed8", linewidth=2.3)
    axes[1, 0].axhline(epsilon_dipole, color="#dc2626", linestyle="--", linewidth=1.8, label=f"Final = {epsilon_dipole:.3f}")
    axes[1, 0].set_title("Convergence of Dipolar Dielectric Constant")
    axes[1, 0].set_xlabel("Number of samples included")
    axes[1, 0].set_ylabel("$\\epsilon_{dipole}$")
    axes[1, 0].legend(loc="best")

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.0,
        0.9,
        "This tutorial focuses only on the dipole-fluctuation contribution.\n\n"
        "Key takeaway:\n"
        "- the total dipole moment fluctuates during MD\n"
        "- the variance of that dipole is linked to dielectric response\n"
        "- better sampling gives a more stable dielectric estimate",
        fontsize=12,
        va="top",
        ha="left",
        bbox={"facecolor": "#eff6ff", "edgecolor": "#93c5fd", "boxstyle": "round,pad=0.6"},
    )

    fig.suptitle(f"Dielectric Constant Tutorial: {sample_name}", fontsize=16, fontweight="bold")

    if save_figure:
        fig.savefig(output_dir(root) / "dielectric_constant_summary.png", dpi=220)

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

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(rep_heat["time"] * 1e12, rep_heat["energy"], color="#2563eb", linewidth=2.1, label="Heat transferred")
    axes[0, 0].plot(
        rep_heat["time"] * 1e12,
        rep_heat["slope"] * rep_heat["time"] + rep_heat["intercept"],
        color="#dc2626",
        linestyle="--",
        linewidth=2.0,
        label=f"Linear fit (R = {rep_heat['r_value']:.3f})",
    )
    axes[0, 0].set_title(f"Representative Heat-Flux Fit (window {representative['window']})")
    axes[0, 0].set_xlabel("Time (ps)")
    axes[0, 0].set_ylabel("Accumulated energy (J)")
    axes[0, 0].legend(loc="best")

    axes[0, 1].scatter(rep_gradient["distance"], rep_gradient["temperature"], s=30, color="#0f766e", label="Temperature profile")
    axes[0, 1].plot(
        rep_gradient["distance"],
        rep_gradient["slope"] * rep_gradient["distance"] + rep_gradient["intercept"],
        color="#f97316",
        linestyle="--",
        linewidth=2.0,
        label=f"Linear fit (R = {rep_gradient['r_value']:.3f})",
    )
    axes[0, 1].set_title(f"Representative Temperature Gradient (window {representative['window']})")
    axes[0, 1].set_xlabel("Reduced position")
    axes[0, 1].set_ylabel("Temperature (K)")
    axes[0, 1].legend(loc="best")

    window_ids = [record["window"] for record in records]
    axes[1, 0].plot(window_ids, tc_values, marker="o", linewidth=2.2, color="#7c3aed")
    axes[1, 0].axhspan(tc_mean - tc_std, tc_mean + tc_std, color="#ddd6fe", alpha=0.45, label="Mean ± std")
    axes[1, 0].axhline(tc_mean, color="#6d28d9", linestyle="--", linewidth=1.8, label=f"Mean = {tc_mean:.4e}")
    axes[1, 0].set_title("Window-by-Window Thermal Conductivity")
    axes[1, 0].set_xlabel("Window")
    axes[1, 0].set_ylabel("Thermal conductivity")
    axes[1, 0].legend(loc="best")

    cumulative = np.cumsum(tc_values) / np.arange(1, len(tc_values) + 1)
    axes[1, 1].plot(np.arange(1, len(tc_values) + 1), cumulative, marker="o", linewidth=2.2, color="#0891b2")
    axes[1, 1].axhline(tc_mean, color="#0f766e", linestyle="--", linewidth=1.8, label="Final mean")
    axes[1, 1].set_title("Convergence of the Running Mean")
    axes[1, 1].set_xlabel("Number of windows included")
    axes[1, 1].set_ylabel("Running mean thermal conductivity")
    axes[1, 1].legend(loc="best")

    fig.suptitle(f"Thermal Conductivity Tutorial: {sample_name}", fontsize=16, fontweight="bold")

    if save_figure:
        fig.savefig(output_dir(root) / "thermal_conductivity_summary.png", dpi=220)

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
