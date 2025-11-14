# PyVista 3D Visualization Guide for MDAE

## Overview

We've successfully added **PyVista-based 3D volume visualization** to your MDAE visualization toolkit. This complements the existing matplotlib 2D visualizations with true 3D rendering capabilities.

## How the Visualization System Works

### MDAE Dual Corruption Pipeline

Based on `BaseMDAETrainer.py`, MDAE combines two pretraining strategies:

1. **Spatial Masking** (from MAE):
   - Creates blocky 16³ voxel patches
   - Masks 75% of patches by default
   - Creates spatially contiguous masked regions

2. **Noise Corruption** (from Corruption2Self):
   - Applies DDPM-style diffusion noise
   - Uses variance-preserving schedule
   - Corruption: `x_t = √(1-t) * x₀ + √t * ε`

3. **Combined Process**:
   ```
   1. Generate blocky spatial mask m (16³ patches)
   2. Apply diffusion noise to entire volume → x_t  
   3. Apply spatial masking: x̃ = m ⊙ x_t
   4. Network reconstructs x₀ from x̃
   5. Loss computed only on masked regions
   ```

### Visualization Tools

#### 2D Visualizations (Matplotlib)
- `visualize_dual_corruption_grid.py` - Corruption parameter grid (M4Raw data)
- `visualize_dual_corruption.py` - Pipeline stages (BraTS data)
- Output: PDF/PNG 2D slices

#### 3D Visualizations (PyVista) - NEW!
- `visualize_dual_corruption_3d.py` - 3D volume rendering
- `pyvista_utils.py` - Core 3D rendering functions
- Output: High-resolution PNG images

## Generated 3D Visualizations

### 1. Clean Volume (`volume_3d_clean.png`)
- Transparent 3D rendering of brain MRI
- Shows internal structure with opacity control
- Isometric view for depth perception

### 2. Blocky Mask Pattern (`volume_3d_mask_pattern.png`)
- 3D visualization of 16³ patch masking
- Green cubes = visible patches
- Red cubes = masked patches  
- Shows spatial distribution in 3D

### 3. 4-Panel Comparison (`volume_3d_comparison_4panel.png`)
- **Top-left**: Clean volume x₀
- **Top-right**: Spatial mask m (75% masked)
- **Bottom-left**: Noisy volume x_t
- **Bottom-right**: Doubly corrupted x̃

### 4. Multi-angle Views (`volume_3d_multiview.png`)
- Axial view (top-down)
- Coronal view (front)
- Sagittal view (side)
- 3D isometric view

### 5. Mask Overlay (`volume_3d_mask_overlay.png`)
- Volume with mask overlay
- Red regions = masked areas
- Shows which areas need reconstruction

## Usage

### Basic Usage
```bash
# Activate environment
conda activate nnseg

# Generate all 3D visualizations with synthetic data
python visualize_dual_corruption_3d.py --use-synthetic

# Use real BraTS data
python visualize_dual_corruption_3d.py --dataset brats18 --modality t1ce

# High-resolution output
python visualize_dual_corruption_3d.py --dpi 600
```

### Custom Corruption Parameters
```bash
# High masking ratio (90%)
python visualize_dual_corruption_3d.py --mask-percentage 0.9 --timestep 0.7

# Low masking ratio (25%)  
python visualize_dual_corruption_3d.py --mask-percentage 0.25 --timestep 0.3
```

### Selective Visualization
```bash
# Generate only specific visualizations
python visualize_dual_corruption_3d.py --visualizations volume mask comparison
```

## Key Features

### Volume Rendering
- **Transparency**: Uses opacity transfer functions to reveal internal structure
- **Quality**: 300 DPI default (configurable up to 600 DPI)
- **Camera**: Isometric projection for publication
- **Lighting**: Professional three-point lighting
- **Background**: White for publications

### Blocky Mask Visualization
- Each 16³ patch rendered as colored cube
- Color-coded: Green (visible) vs Red (masked)
- Interactive camera positioning
- Shows true 3D spatial distribution

### Publication Quality
- Off-screen rendering (no display required)
- Configurable DPI and window size
- Consistent styling across all visualizations
- Ready for CVPR/medical imaging papers

## Technical Details

### Dependencies
- **PyVista** ≥0.43.0: 3D visualization
- **VTK** ≥9.2.0: Rendering backend
- **imageio**: Image I/O
- **PyTorch**: Data processing
- **nibabel**: Medical imaging (NIfTI files)

### Default Parameters (Matching BaseMDAETrainer)
- Masking ratio: 75%
- Block size: 16³ voxels
- Diffusion timestep: 0.5
- SDE type: DDPM
- Volume size: 128³ (resized for consistency)
- Number of patches: 8×8×8 = 512

### Rendering Settings
- Window size: 1920×1080 (16:9)
- Off-screen: True (headless)
- Lighting: Three-point professional
- Opacity: Sigmoid transfer function
- Colormap: Grayscale (clinical standard)

## File Structure

```
Visualizations/
├── pyvista_utils.py                   # NEW: 3D rendering utilities
├── visualize_dual_corruption_3d.py    # NEW: 3D visualization script
├── outputs/
│   ├── volume_3d_clean.png           # NEW: Transparent volume
│   ├── volume_3d_mask_pattern.png    # NEW: 3D blocky mask
│   ├── volume_3d_comparison_4panel.png # NEW: Side-by-side
│   ├── volume_3d_multiview.png       # NEW: Multiple angles
│   └── volume_3d_mask_overlay.png    # NEW: Volume + mask
```

## Comparison: 2D vs 3D

### 2D Visualizations (Matplotlib)
✓ Fast rendering  
✓ Vector graphics (PDF)  
✓ Easy to annotate  
✓ Grid layouts  
- Limited to slices

### 3D Visualizations (PyVista)
✓ True 3D structure  
✓ Volumetric rendering  
✓ Multiple viewing angles  
✓ Spatial mask distribution  
- Raster only (PNG)

**Recommendation**: Use both! 2D for overview figures, 3D for detailed spatial analysis.

## Example Workflow

```bash
# 1. Activate environment
conda activate nnseg

# 2. Generate 2D grid (M4Raw data)
python visualize_dual_corruption_grid.py

# 3. Generate 3D volumes (BraTS data)
python visualize_dual_corruption_3d.py --dpi 600

# 4. Generate 2D pipeline stages
python visualize_dual_corruption.py --format both

# Result: Comprehensive visualization suite for paper!
```

## Troubleshooting

### VTK Warning: "bad X server connection"
- **Expected**: Off-screen rendering works without display
- **Safe to ignore**: Visualizations still generated correctly

### ModuleNotFoundError: torch
- **Solution**: Activate nnseg environment: `conda activate nnseg`

### Out of memory
- **Solution**: Reduce volume size or use synthetic data
  ```bash
  python visualize_dual_corruption_3d.py --use-synthetic
  ```

## Future Enhancements

Potential additions:
- [ ] Animated GIF showing corruption progression
- [ ] Interactive HTML widgets for notebooks
- [ ] Multiple corruption levels in single figure
- [ ] Custom color schemes for different modalities

## Citation

If using these visualizations, cite:
- MDAE paper (your work)
- PyVista: Sullivan et al., JOSS 2019

---

**Created**: November 2, 2025  
**Environment**: nnseg conda environment  
**Status**: Production-ready ✓
