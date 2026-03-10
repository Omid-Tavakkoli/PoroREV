## PoroREV

This repository estimate porosity REV (Representative Elementary Volume) from a segmented 3D image using random cube sampling.

The script supports:
- `.raw` volumes (with dimensions from `input.txt`)
- `.tif` / `.tiff` 3D stacks (dimensions read from file)

---

### Installation
1) Python 3.9+ recommended.
2) Install dependencies:
- **pip**:

```bash
pip install numpy matplotlib seaborn tifffile imageio
```

- **conda**:

```bash
conda create -n pororev python=3.10 -y
conda activate pororev
conda install -c conda-forge numpy matplotlib seaborn tifffile imageio -y
```
---

### Inputs

- `PoroREV.py`: main script
- `input.txt`: user-editable configuration file

---

### How to Run

- **Default config file**:

```bash
python3 PoroREV.py
```

-**Custom config file path**:

```bash
python3 PoroREV.py --config input.txt
```

### Configuration (`input.txt`)

**1) Phase labels**

- `pore`: pore label value in the segmented image
- `solid`: solid label value in the segmented image

Important behavior:
- The current code maps voxels equal to `solid` to `0` (solid).
- All other voxel values are mapped to `1` (pore).
- `pore` and `solid` must be different values.

**2) Input file**

- `filename`: input image path (`.raw`, `.tif`, `.tiff`)
  - Relative paths are resolved relative to the config file location.

**3) RAW file settings (used only for `.raw`)**

- `filesize_x`, `filesize_y`, `filesize_z`: raw dimensions in voxels
- `dtype`: raw voxel type (examples: `uint8`, `uint16`, `int16`, `float32`)

Notes:
- For `.tif/.tiff`, dimensions are read from the image stack.
- For `.tif/.tiff`, raw size fields and `dtype` are ignored.

**4) REV sampling settings**

- `min_side`: smallest sampled cube side length (voxel)
- `step`: cube side increment between samples
- `samples`: random cubes per side length
- `seed`: random seed for reproducibility

**5) Output settings**

- `save_fig`: output figure path (`.png`, `.pdf`, etc.)
- `save_csv`: output table path

**6) Optional REV detection by CI**

- `rev_ci_threshold` (optional): CI half-width threshold
- `rev_consecutive`: required consecutive side lengths below threshold

If `rev_ci_threshold` is commented out or missing, automatic REV-size detection is skipped.

### Outputs

**1. Figure (`save_fig`)**:
   - Mean porosity vs. cube size
   - Shaded 95% confidence interval
**2. CSV (`save_csv`) with columns**:
   - `cube_size_voxel`
   - `mean_porosity`
   - `std_porosity`
   - `ci95_halfwidth`
   - `cv`
