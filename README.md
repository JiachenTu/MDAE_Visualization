# MDAE Dual Corruption Visualization

Publication-quality visualizations of the MDAE (Masked Diffusion Autoencoder) dual corruption strategy for CVPR submission using real brain MRI data.

## Overview

This package generates high-resolution, publication-ready figures demonstrating the MDAE pretraining strategy using **real brain MRI data** from multiple datasets. The visualizations clearly show the dual corruption approach that combines:

1. **Spatial Masking**: Blocky patch-based masking with variable ratios (16×16 or 16³ patches)
2. **Diffusion Noise**: DDPM-style variance-preserving schedule with noise corruption

## Supported Datasets

- **M4Raw**: Multi-acquisition MRI dataset (T1, T2, FLAIR) - **Recommended for 2D grid visualization**
- **BraTS18/BraTSPED**: Brain tumor segmentation datasets - For 3D volume visualization

## Generated Figures

### 1. Dual Corruption Grid (`dual_corruption_grid.pdf`)
**NEW**: 2D grid visualization using M4Raw T1 MRI data showing the complete dual corruption space:
- **Columns**: Diffusion timesteps from clean (t=0) to noisy (T_max)
- **Rows**: Masking ratios from 0% to 90%
- **Highlighted cell**: Typical training sample with medium corruption (75% mask, t=0.5)
- Demonstrates how MDAE explores the full corruption spectrum during training

### 2. **NEW**: 3D Volume Visualizations (PyVista)
Interactive 3D volume renderings of the dual corruption pipeline:
- **Clean Volume** (`volume_3d_clean.png`): Transparent 3D rendering of brain MRI
- **Blocky Mask Pattern** (`volume_3d_mask_pattern.png`): 3D visualization of 16³ patch masking
- **4-Panel Comparison** (`volume_3d_comparison_4panel.png`): Side-by-side corruption stages
- **Multi-angle Views** (`volume_3d_multiview.png`): Axial, coronal, sagittal, and isometric views
- **Mask Overlay** (`volume_3d_mask_overlay.png`): Volume with mask overlay showing masked regions

These 3D visualizations provide deeper insight into the spatial structure of the corruption process.

### 3. Dual Corruption Overview (`dual_corruption_overview.pdf`)
Multi-row figure showing the complete corruption pipeline:
- Clean volume $x_0$
- Spatial mask $m$ (16³ blocky patches)
- Noisy volume $x_t = \alpha(t)x_0 + \sigma(t)\varepsilon$
- Doubly corrupted input $\tilde{x} = m \odot x_t$
- Reconstruction target (masked regions only)

Demonstrates 3 different masking ratios (25%, 50%, 90%) with different noise levels.

### 2. Masking Ratio Comparison (`masking_ratio_comparison.pdf`)
Comparison of 6 different masking ratios: 10%, 25%, 50%, 75%, 90%, 95%

Shows how the stochastic masking strategy enables learning across the full corruption spectrum.

### 3. Step-by-Step Process (`step_by_step_process.pdf`)
Detailed 9-step visualization of the complete MDAE training procedure:
1. Load clean volume
2. Sample $p_{\text{mask}} \sim \mathcal{U}(0.01, 0.99)$
3. Generate blocky mask (16³ patches)
4. Sample timestep $t$ and noise $\varepsilon$
5. Apply noise corruption: $x_t = \alpha(t)x_0 + \sigma(t)\varepsilon$
6. Apply spatial masking: $\tilde{x} = m \odot x_t$
7. Network reconstruction: $\hat{x} = g_\theta(\tilde{x}, t)$
8. Compute loss on masked regions only

## Installation

### Using Conda (Recommended)

```bash
# Activate nnseg environment
conda activate nnseg

# Navigate to visualization directory
cd /home/jiachen/projects/MDAE/Visualizations

# Install dependencies
pip install -r requirements.txt
```

### Using pip

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start: 3D Volume Visualization (PyVista)

Generate publication-quality 3D volume renderings using BraTS brain MRI:

```bash
# Generate all 3D visualizations
python visualize_dual_corruption_3d.py

# Generate specific visualizations only
python visualize_dual_corruption_3d.py --visualizations volume mask comparison

# High-resolution output (600 DPI)
python visualize_dual_corruption_3d.py --dpi 600

# Custom corruption parameters
python visualize_dual_corruption_3d.py --mask-percentage 0.9 --timestep 0.7
```

This will generate 5 high-quality 3D visualization images showing the dual corruption pipeline from multiple perspectives.

### Dual Corruption Grid (M4Raw Data)

Generate the comprehensive dual corruption grid using real T1 MRI from M4Raw dataset:

```bash
python visualize_dual_corruption_grid.py
```

This will:
- Load an M4Raw T1-weighted brain MRI slice
- Generate a 6×9 grid showing all corruption combinations
- Highlight the typical training sample
- Save to `outputs/dual_corruption_grid.pdf` and `.png`

**Output**: A publication-ready figure clearly demonstrating MDAE's dual corruption strategy.

### Alternative: 3D Volume Visualizations (BraTS Data)

Generate 3D volume-based visualizations using BraTS datasets:

```bash
python visualize_dual_corruption.py
```

This will:
- Load a BraTS18 T1-contrast enhanced (T1ce) brain MRI sample
- Generate three publication figures for 3D volumes
- Save to `outputs/` directory

Output files:
- `dual_corruption_overview.pdf`
- `masking_ratio_comparison.pdf`
- `step_by_step_process.pdf`

### M4Raw Dataset Options

```bash
# Use different modality
python visualize_dual_corruption_grid.py --modality T2

# Use specific case and slice
python visualize_dual_corruption_grid.py --case-id 2022101101 --slice-idx 12

# High-resolution output
python visualize_dual_corruption_grid.py --format png --dpi 600

# Different patch size for masking
python visualize_dual_corruption_grid.py --patch-size 32
```

### Using Different BraTS Datasets and Modalities

```bash
# Use BraTS Pediatric dataset with T2 modality
python visualize_dual_corruption.py --dataset bratsped --modality t2

# Use specific case ID
python visualize_dual_corruption.py --case-id Brats18_2013_2_1

# Generate high-resolution PNG outputs
python visualize_dual_corruption.py --format png --dpi 600
```

### Using Synthetic Data (Fallback)

If BraTS data is unavailable, use synthetic brain-like volumes:

```bash
python visualize_dual_corruption.py --use-synthetic
```

### Command-Line Options

```bash
python visualize_dual_corruption.py [OPTIONS]

Options:
  --use-synthetic         Use synthetic data instead of real MRI (default: use real BraTS)
  --dataset {brats18,bratsped}  BraTS dataset to use (default: brats18)
  --modality {t1,t1ce,t2,flair} MRI modality (default: t1ce)
  --case-id STR           Specific case ID (default: auto-select first)
  --output-dir PATH       Output directory (default: outputs)
  --cache-dir PATH        Cache directory for samples (default: sample_data)
  --dpi INT               DPI for figures (default: 300)
  --format {pdf,png,both} Output format (default: pdf)

Examples:
  # Default: BraTS18 T1ce with PDF output
  python visualize_dual_corruption.py

  # Generate both PDF and PNG with high resolution
  python visualize_dual_corruption.py --format both --dpi 600

  # Use different modality
  python visualize_dual_corruption.py --modality flair

  # Use synthetic data
  python visualize_dual_corruption.py --use-synthetic
```

### Available BraTS Datasets

**BraTS18** (`--dataset brats18`):
- Adult brain tumor MRI
- Location: `/mnt/hanoverlimitedwus2/scratch/t-jiachentu/nnssl_data/segmentation/nnUNet_raw/Dataset042_BraTS18`
- Good for standard adult brain anatomy

**BraTSPED** (`--dataset bratsped`):
- Pediatric brain tumor MRI
- Location: `/mnt/hanoverlimitedwus2/scratch/t-jiachentu/nnssl_data/segmentation/nnUNet_raw/Dataset527_BraTSPED`
- Useful for pediatric anatomy

### MRI Modalities

- **t1**: T1-weighted (anatomical detail)
- **t1ce**: T1 contrast-enhanced (best contrast, **recommended for visualization**)
- **t2**: T2-weighted (fluid sensitivity)
- **flair**: FLAIR (lesion detection)

### M4Raw Dataset Details

**Location**: `/mnt/hanoverlimitedwus2/scratch/t-jiachentu/datasets/m4raw_dataset`

**Modalities**:
- **T1**: T1-weighted MRI (6 acquisitions: T101-T106)
- **T2**: T2-weighted MRI (6 acquisitions: T201-T206)
- **FLAIR**: FLAIR MRI (4 acquisitions: FLAIR01-FLAIR04)

**Data Format**:
- H5 files containing k-space data
- Automatically converted to image domain using inverse FFT
- RSS (root sum of squares) across coils
- 18 slices per volume (256×256 resolution)
- Normalized to [0, 1] range

**Requirements**: `h5py` and `fastmri` packages (automatically installed with requirements.txt)

## Module Documentation

### `corruption_utils.py`
Core corruption functions extracted from `NoiseConditionedMDAETrainer.py`:
- `create_blocky_mask()`: Generate 16³ patch-based spatial masks (3D)
- `create_2d_blocky_mask()`: Generate 16×16 patch-based spatial masks (2D)
- `corrupt_ddpm()`: DDPM-style diffusion corruption
- `apply_noise_corruption()`: VP schedule diffusion noise
- `dual_corruption()`: Complete dual corruption pipeline (3D)
- `dual_corruption_2d()`: Complete dual corruption pipeline (2D)

### `data_loader_utils.py`
Data loading utilities:
- `load_m4raw_sample()`: Load real 2D MRI slices from M4Raw dataset using h5py/fastmri
- `load_brats_sample()`: Load real 3D brain MRI from BraTS datasets using nibabel
- `create_synthetic_brain_volume()`: Generate synthetic brain-like data (fallback)
- `extract_2d_slices()`: Extract axial/coronal/sagittal slices
- `crop_or_pad_to_shape()`: Center crop/pad volumes to target shape
- `read_m4raw_h5()`: Read H5 files and convert k-space to image domain

### `plot_utils.py`
CVPR-quality plotting functions:
- `setup_cvpr_style()`: Configure matplotlib for publication
- `plot_2d_slice()`: Beautiful 2D slice visualization
- `plot_mask_overlay()`: Overlay masks on images
- `create_dual_corruption_grid()`: **NEW** - Grid visualization showing full corruption space
- `create_dual_corruption_overview()`: Main overview figure (3D)
- `create_masking_ratio_comparison()`: Ratio comparison figure (3D)

### `pyvista_utils.py`
**NEW**: PyVista-based 3D visualization utilities:
- `setup_pyvista_plotter()`: Configure PyVista for publication-quality rendering
- `render_3d_volume()`: Transparent volume rendering with opacity transfer functions
- `render_3d_mask_blocks()`: Visualize blocky 3D masking patterns as colored cubes
- `create_side_by_side_comparison()`: Multi-panel comparison of corruption stages
- `render_multiview()`: Multiple viewing angles (axial, coronal, sagittal, isometric)
- `create_mask_overlay_3d()`: Volume with mask overlay
- `save_publication_image()`: Export high-resolution images

### `visualize_dual_corruption_3d.py`
**NEW**: Main script for generating 3D volume visualizations using PyVista.
- Supports all BraTS datasets and modalities
- Generates 5 different 3D visualizations
- Publication-quality static images (PNG)
- Customizable corruption parameters

### `visualize_dual_corruption_grid.py`
**NEW**: Main script for generating the comprehensive dual corruption grid using M4Raw data.

### `visualize_dual_corruption.py`
Original script for 2D slice visualizations using BraTS data and matplotlib.

## Implementation Details

### Dual Corruption Process

The visualization exactly replicates the corruption process from `NoiseConditionedMDAETrainer.train_step()`:

```python
# 1. Generate blocky spatial mask
mask = create_blocky_mask(volume_shape, patch_size=16, mask_percentage=0.75)

# 2. Apply noise corruption to ENTIRE volume
x_t = alpha(t) * x_0 + sigma(t) * epsilon

# 3. Apply spatial masking
x_tilde = mask ⊙ x_t  # Zeros out masked regions

# 4. Network sees doubly-corrupted input
output = g_theta(x_tilde, t)

# 5. Loss computed ONLY on masked regions
loss = compute_loss(output, x_0, loss_mask=(1-mask))
```

### VP Schedule Parameters

Matches the MDAE paper and implementation:
- $\beta_{\min} = 10^{-4}$
- $\beta_{\max} = 0.02$
- $\sigma_{\text{data}} = 1.0$

### Masking Convention

- **Spatial mask** `m`: 1 = visible, 0 = masked
- **Loss mask**: 1 = masked (compute loss), 0 = visible (ignore)

### Patch Size

- **16³ voxel patches**: Matches blocky masking implementation
- Creates spatially contiguous masked regions

### PyVista 3D Visualization Features

**Volume Rendering**:
- Transparency-based rendering using opacity transfer functions
- Customizable opacity curves for different tissue types
- Multiple camera angles and projections
- GPU-accelerated rendering (when available)

**Blocky Mask Visualization**:
- Each 16³ patch rendered as a colored cube
- Green cubes = visible patches
- Red cubes = masked patches
- Interactive exploration of 3D spatial distribution

**Publication Quality**:
- Configurable DPI (default: 300, recommended: 600 for print)
- White background for publications
- Professional three-point lighting
- Consistent camera positioning across views

**Supported Visualizations**:
1. `volume`: Transparent brain volume rendering
2. `mask`: 3D blocky patch visualization
3. `comparison`: 4-panel side-by-side comparison
4. `multiview`: Multiple viewing angles
5. `overlay`: Volume with mask overlay

## Output Specifications

All figures are generated with CVPR publication standards:

- **Resolution**: 300 DPI (configurable)
- **Format**: PDF (vector) or PNG (raster)
- **Font**: Times New Roman (serif)
- **Font sizes**:
  - Main title: 16pt
  - Subplot titles: 14pt
  - Labels: 12pt
- **Colormap**:
  - Grayscale for clean images
  - Viridis for noisy images
  - Red/Blue overlays for masks

## Troubleshooting

### Issue: "Cannot load BraTS data"

**Solution 1**: Check if BraTS datasets are accessible:
```bash
# Check BraTS18
ls /mnt/hanoverlimitedwus2/scratch/t-jiachentu/nnssl_data/segmentation/nnUNet_raw/Dataset042_BraTS18/imagesTs

# Check BraTSPED
ls /mnt/hanoverlimitedwus2/scratch/t-jiachentu/nnssl_data/segmentation/nnUNet_raw/Dataset527_BraTSPED/imagesTs
```

**Solution 2**: Use synthetic data mode as fallback:
```bash
python visualize_dual_corruption.py --use-synthetic
```

The script will automatically fall back to synthetic data if BraTS loading fails.

### Issue: "No cases found in dataset"

**Cause**: The dataset directory may be empty or have different structure.

**Solution**: Check the available cases:
```bash
ls /mnt/hanoverlimitedwus2/scratch/t-jiachentu/nnssl_data/segmentation/nnUNet_raw/Dataset042_BraTS18/imagesTs/*.nii*
```

Or specify a different case ID:
```bash
python visualize_dual_corruption.py --case-id Brats18_XXXX_X_X
```

### Issue: "ModuleNotFoundError"

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Issue: LaTeX rendering errors

**Solution**: Disable LaTeX in `plot_utils.py`:
```python
plt.rcParams['text.usetex'] = False  # Already set by default
```

## Testing

Run unit tests for each module:

```bash
# Test corruption utilities
python corruption_utils.py

# Test data loading
python data_loader_utils.py

# Test plotting utilities
python plot_utils.py
```

All modules include `if __name__ == "__main__"` test blocks.

## File Structure

```
Visualizations/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (updated with PyVista)
│
├── # 3D Visualization Scripts (PyVista)
├── visualize_dual_corruption_3d.py    # NEW: 3D volume visualization script
├── pyvista_utils.py                   # NEW: PyVista utilities
│
├── # 2D Visualization Scripts (Matplotlib)
├── visualize_dual_corruption.py       # 2D slice visualization (BraTS)
├── visualize_dual_corruption_grid.py  # 2D grid visualization (M4Raw)
├── visualize_diffusion_row.py         # Diffusion progression
│
├── # Utilities
├── corruption_utils.py                # Corruption functions
├── data_loader_utils.py               # Data loading (BraTS & M4Raw)
├── plot_utils.py                      # Matplotlib plotting utilities
│
├── outputs/                           # Generated figures (created)
│   ├── # 3D Visualizations (PyVista)
│   ├── volume_3d_clean.png
│   ├── volume_3d_mask_pattern.png
│   ├── volume_3d_comparison_4panel.png
│   ├── volume_3d_multiview.png
│   ├── volume_3d_mask_overlay.png
│   │
│   ├── # 2D Visualizations (Matplotlib)
│   ├── dual_corruption_grid.pdf
│   ├── dual_corruption_overview.pdf
│   ├── masking_ratio_comparison.pdf
│   └── step_by_step_process.pdf
│
└── sample_data/                       # Cached samples (created)
    └── sample_0.npz
```

## Citation

If you use these visualizations, please cite the MDAE paper:

```bibtex
@inproceedings{mdae2026,
  title={Masked Diffusion Autoencoders for 3D Medical Vision Representation Learning},
  author={...},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

## License

This code is provided for research and academic purposes.

## Contact

For questions or issues, please refer to the main MDAE repository.

---

**Last updated**: November 2, 2025
**Compatible with**: NoiseConditionedMDAETrainer (nnssl v2.0)
**Visualization Tools**:
- Matplotlib - 2D slice-based visualizations
- PyVista - 3D volume visualizations (NEW)
**Data sources**:
- M4Raw dataset - multi-acquisition MRI (recommended for 2D grid visualization)
- BraTS datasets - brain tumor segmentation (for 3D volume visualization)
