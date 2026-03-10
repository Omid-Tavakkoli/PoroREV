#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_config_file(path: str) -> Dict[str, str]:
    """Parse key=value pairs from config text file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    config: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            # Allow multiple assignments on one line, e.g. "pore = 1, solid = 0".
            parts = [p.strip() for p in line.split(",") if p.strip()]
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if key:
                    config[key] = value

    return config


def parse_numeric(value: str) -> Union[int, float]:
    """Parse numeric text as int if possible, otherwise float."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def get_required(config: Dict[str, str], key: str) -> str:
    """Read required key from config, raising a clear error if missing."""
    if key not in config or config[key] == "":
        raise ValueError(f"Missing required key '{key}' in config file.")
    return config[key]


def load_tif(path: str) -> np.ndarray:
    """Load a tif/tiff image stack as 3D numpy array."""
    try:
        import tifffile

        arr = tifffile.imread(path)
    except Exception:
        # Fallback to imageio if tifffile is unavailable
        try:
            import imageio.v3 as iio

            arr = iio.imread(path)
        except Exception as exc:
            raise RuntimeError(
                "Could not read TIFF. Install tifffile or imageio:\n"
                "  pip install tifffile imageio"
            ) from exc

    if arr.ndim != 3:
        raise ValueError(f"TIFF must be 3D. Loaded shape: {arr.shape}")
    return arr


def load_raw(path: str, dims_zyx: Tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
    """Load raw binary volume with given dimensions."""
    expected_size = int(np.prod(dims_zyx))
    data = np.fromfile(path, dtype=dtype)
    if data.size != expected_size:
        raise ValueError(
            f"RAW size mismatch. Expected {expected_size} voxels, got {data.size}. "
            "Check --dims and --dtype."
        )
    return data.reshape(dims_zyx)


def to_binary(volume: np.ndarray, pore_value: Union[int, float], solid_value: Union[int, float]) -> np.ndarray:
    """Convert to uint8 binary: solid stays 0, all non-solid values become pore (1)."""
    if pore_value == solid_value:
        raise ValueError("Config values 'pore' and 'solid' must be different.")
    _ = pore_value  # Kept for config completeness/documentation.
    return (volume != solid_value).astype(np.uint8)


def get_sides(binary_vol: np.ndarray, min_side: int, step: int) -> List[int]:
    """Create valid cube sides from min_side to min(volume shape)."""
    min_dim = min(binary_vol.shape)
    sides = list(range(min_side, min_dim + 1, step))
    if not sides:
        raise ValueError(
            f"No valid cube sizes. min_side={min_side}, min(volume shape)={min_dim}"
        )
    return sides


def compute_random_stats_for_side(
    binary_vol: np.ndarray, side: int, samples: int, rng: np.random.Generator
) -> Tuple[float, float, float, float]:
    """Compute mean/std/95% CI half-width/CV for random cubes of one side."""
    z, y, x = binary_vol.shape
    max_z = z - side
    max_y = y - side
    max_x = x - side
    if min(max_z, max_y, max_x) < 0:
        raise ValueError(f"Cube side {side} does not fit in volume shape {binary_vol.shape}.")

    vals = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        z0 = int(rng.integers(0, max_z + 1))
        y0 = int(rng.integers(0, max_y + 1))
        x0 = int(rng.integers(0, max_x + 1))
        cube = binary_vol[z0 : z0 + side, y0 : y0 + side, x0 : x0 + side]
        vals[i] = float(cube.mean())

    mean_p = float(np.mean(vals))
    std_p = float(np.std(vals, ddof=1)) if samples > 1 else 0.0
    ci95_half = 1.96 * std_p / np.sqrt(samples) if samples > 1 else 0.0
    cv = std_p / mean_p if mean_p > 0 else float("nan")
    return mean_p, std_p, float(ci95_half), float(cv)


def compute_rev_curve_random(
    binary_vol: np.ndarray,
    min_side: int = 10,
    step: int = 10,
    samples: int = 100,
    seed: int = 42,
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Compute random-sampling porosity statistics for increasing cube sizes."""
    if binary_vol.ndim != 3:
        raise ValueError("Input volume must be 3D.")
    if samples <= 0:
        raise ValueError("--samples must be a positive integer.")

    sides = get_sides(binary_vol, min_side=min_side, step=step)
    rng = np.random.default_rng(seed)

    means: List[float] = []
    stds: List[float] = []
    ci95_halfs: List[float] = []
    cvs: List[float] = []
    for s in sides:
        mean_p, std_p, ci95_half, cv = compute_random_stats_for_side(
            binary_vol, side=s, samples=samples, rng=rng
        )
        means.append(mean_p)
        stds.append(std_p)
        ci95_halfs.append(ci95_half)
        cvs.append(cv)

    return sides, means, stds, ci95_halfs, cvs


def detect_rev_by_ci(
    sides: List[int],
    ci95_halfs: List[float],
    threshold: float,
    consecutive: int = 3,
) -> Optional[int]:
    """Return first REV side where CI half-width stays below threshold."""
    if consecutive <= 0:
        raise ValueError("--rev-consecutive must be positive.")

    count = 0
    start_idx = 0
    for i, ci_half in enumerate(ci95_halfs):
        if ci_half <= threshold:
            if count == 0:
                start_idx = i
            count += 1
            if count >= consecutive:
                return sides[start_idx]
        else:
            count = 0
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate porosity REV from a segmented 3D rock image."
    )
    parser.add_argument(
        "--config",
        default="input.txt",
        help="Path to config text file (default: input.txt)",
    )
    args = parser.parse_args()

    cfg = parse_config_file(args.config)

    in_path = get_required(cfg, "filename")
    if not os.path.isabs(in_path):
        in_path = os.path.join(os.path.dirname(os.path.abspath(args.config)), in_path)

    ext = os.path.splitext(in_path)[1].lower()
    dtype_name = cfg.get("dtype", "uint8")

    if ext in [".tif", ".tiff"]:
        vol = load_tif(in_path)
    elif ext == ".raw":
        size_x = int(get_required(cfg, "filesize_x"))
        size_y = int(get_required(cfg, "filesize_y"))
        size_z = int(get_required(cfg, "filesize_z"))
        if min(size_x, size_y, size_z) <= 0:
            raise ValueError("filesize_x/filesize_y/filesize_z must be positive integers.")
        dims_zyx = (size_z, size_y, size_x)
        vol = load_raw(in_path, dims_zyx, np.dtype(dtype_name))
    else:
        raise ValueError("Unsupported format. Use .raw or .tif/.tiff")

    pore_value = parse_numeric(get_required(cfg, "pore"))
    solid_value = parse_numeric(get_required(cfg, "solid"))
    binary_vol = to_binary(vol, pore_value=pore_value, solid_value=solid_value)

    min_side = int(cfg.get("min_side", "10"))
    step = int(cfg.get("step", "10"))
    samples = int(cfg.get("samples", "100"))
    seed = int(cfg.get("seed", "42"))
    save_fig = cfg.get("save_fig", "porosity_rev.png")
    save_csv = cfg.get("save_csv", "porosity_rev.csv")
    rev_ci_threshold = cfg.get("rev_ci_threshold", None)
    rev_consecutive = int(cfg.get("rev_consecutive", "3"))

    if min_side <= 0 or step <= 0:
        raise ValueError("--min-side and --step must be positive integers.")

    sns.set(style="white")
    palette = sns.color_palette(
        ["#000000", "#CC79A7", "#56B4E9", "#E69F00", "#009E73", "#F0E442", "#0072B2", "#D55E00"]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sides, means, stds, ci95_halfs, cvs = compute_rev_curve_random(
        binary_vol,
        min_side=min_side,
        step=step,
        samples=samples,
        seed=seed,
    )

    with open(save_csv, "w", encoding="utf-8") as f:
        f.write("cube_size_voxel,mean_porosity,std_porosity,ci95_halfwidth,cv\n")
        for s, m, sd, ci, cv in zip(sides, means, stds, ci95_halfs, cvs):
            f.write(f"{s},{m:.8f},{sd:.8f},{ci:.8f},{cv:.8f}\n")

    low = np.array(means) - np.array(ci95_halfs)
    high = np.array(means) + np.array(ci95_halfs)
    ax.plot(
        sides,
        means,
        marker="o",
        linewidth=1.5,
        color=palette[2],
        markerfacecolor=palette[2],
        markeredgecolor=palette[2],
        alpha=0.9,
        label=f"Mean (samples/size={samples})",
    )
    ax.fill_between(sides, low, high, color=palette[2], alpha=0.22, label="95% CI")
    ax.set_title("Porosity REV Analysis (Random Sampling)")
    print("Mode: random")
    print(f"Samples per size: {samples}, seed: {seed}")
    print(f"Computed {len(sides)} points from side={sides[0]} to side={sides[-1]}")
    print(f"Pore value: {pore_value}, solid value: {solid_value}")

    if rev_ci_threshold is not None and rev_ci_threshold != "":
        rev_ci_threshold_val = float(rev_ci_threshold)
        if rev_ci_threshold_val <= 0:
            raise ValueError("--rev-ci-threshold must be positive.")
        rev_size = detect_rev_by_ci(
            sides,
            ci95_halfs,
            threshold=rev_ci_threshold_val,
            consecutive=rev_consecutive,
        )
        if rev_size is None:
            print(
                "No REV size found with current CI criterion "
                f"(threshold={rev_ci_threshold_val}, consecutive={rev_consecutive})."
            )
        else:
            print(
                f"Estimated REV size by CI criterion: {rev_size} voxels "
                f"(threshold={rev_ci_threshold_val}, consecutive={rev_consecutive})"
            )

    ax.set_xlabel("Cube size [voxel]")
    ax.set_ylabel("Porosity")

    # Ensure full frame (all spines visible), same style as ua_us.py
    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    plt.show()

    print(f"Volume shape (Z,Y,X): {binary_vol.shape}")
    print(f"Saved figure: {save_fig}")
    print(f"Saved CSV:    {save_csv}")


if __name__ == "__main__":
    main()
