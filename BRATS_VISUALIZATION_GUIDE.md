# BRATS Dataset 3D MRI Visualization Guide

This guide explains how 3D MRI data is loaded and visualized for the BRATS (Brain Tumor Segmentation) dataset, along with the visualization tools created for exploring the dataset.

## Overview of BRATS Dataset

### Data Structure

The BRATS dataset contains 3D brain MRI scans with tumor segmentation labels:

- **Image Data**: 4D volumes `(240, 240, 155, 4)`
  - Dimensions: Height × Width × Depth × Channels
  - 4 MRI modalities per sample:
    1. **T1**: T1-weighted (anatomical detail)
    2. **T1ce**: T1 contrast-enhanced (best contrast for tumors)
    3. **T2**: T2-weighted (fluid sensitivity)
    4. **FLAIR**: Fluid-attenuated inversion recovery (lesion detection)

- **Label Data**: 3D volumes `(240, 240, 155)`
  - Segmentation masks with 4 classes:
    - `0`: Normal tissue (background)
    - `1`: Edema (green)
    - `2`: Non-enhancing tumor (yellow)
    - `3`: Enhancing tumor (red)

### File Format

- Format: NIfTI (`.nii.gz`) - standard neuroimaging format
- Library: `nibabel` for reading/writing
- Compression: gzip compression for efficient storage

## How 3D MRI Data is Loaded

### Loading Process

```python
import nibabel as nib

# Load image data (4 modalities)
img_obj = nib.load('path/to/BRATS_001.nii.gz')
img_data = img_obj.get_fdata()  # Shape: (240, 240, 155, 4)

# Load label data (segmentation)
label_obj = nib.load('path/to/labels/BRATS_001.nii.gz')
label_data = label_obj.get_fdata()  # Shape: (240, 240, 155)
```

### Data Preprocessing

1. **Normalization**: Percentile-based normalization to [0, 1] range
   ```python
   vmin = slice_data.min()
   vmax = np.percentile(slice_data, 99)  # 99th percentile
   normalized = (slice_data - vmin) / (vmax - vmin)
   ```

2. **Slice Extraction**: Extract 2D slices from 3D volume for visualization
   - **Axial**: Horizontal slices (most common for brain imaging)
   - **Coronal**: Frontal slices
   - **Sagittal**: Side slices

3. **Label Overlay**: Combine grayscale MRI with colored segmentation masks
   - Edema: Semi-transparent green
   - Non-enhancing tumor: Semi-transparent yellow
   - Enhancing tumor: Semi-transparent red

## Visualization Tools

### 1. Individual Sample Visualization (`visualize_brats_samples.py`)

Comprehensive visualization tool for exploring individual BRATS samples.

**Features:**
- All 4 modalities visualization
- Segmentation mask display
- Multi-view rendering (axial, coronal, sagittal)
- Slice progression through volume
- Label overlay on MRI images

**Usage:**
```bash
# Visualize all samples
python visualize_brats_samples.py

# Visualize specific samples
python visualize_brats_samples.py --samples BRATS_001 BRATS_002

# Custom output directory
python visualize_brats_samples.py --output-dir my_visualizations
```

**Generated Files (per sample):**
- `{sample}_all_modalities_axial.png` - All 4 modalities in axial view
- `{sample}_all_modalities_coronal.png` - All 4 modalities in coronal view
- `{sample}_slice_progression_t1ce.png` - 9 slices showing progression
- `{sample}_multiview_t1ce.png` - Axial, coronal, sagittal views

### 2. Cross-Sample Comparison (`create_brats_comparison.py`)

Create comparison visualizations across multiple samples for analysis.

**Features:**
- Side-by-side sample comparison
- Single modality across all samples
- Complete grid showing all modalities and samples
- Consistent color scheme and scaling

**Usage:**
```bash
python create_brats_comparison.py
```

**Generated Files:**
- `comparison_all_samples_t1ce.png` - T1ce comparison across samples
- `comparison_all_samples_t2.png` - T2 comparison across samples
- `comparison_complete_grid.png` - All modalities, all samples

### 3. Interactive Exploration (Jupyter Notebook)

For interactive exploration, use the provided Jupyter notebook:

**Location:** `support/Visualize-3D-MRI-Scans-Brain-case/playground.ipynb`

**Features:**
- Interactive layer slider
- Live modality selection
- Real-time segmentation class filtering
- Widget-based exploration

**Usage:**
```bash
cd support/Visualize-3D-MRI-Scans-Brain-case
jupyter notebook playground.ipynb
```

## Generated Visualizations Summary

### Directory Structure
```
brats_visualizations/
├── BRATS_001/
│   ├── BRATS_001_all_modalities_axial.png
│   ├── BRATS_001_all_modalities_coronal.png
│   ├── BRATS_001_multiview_t1ce.png
│   └── BRATS_001_slice_progression_t1ce.png
├── BRATS_002/
│   ├── BRATS_002_all_modalities_axial.png
│   ├── BRATS_002_all_modalities_coronal.png
│   ├── BRATS_002_multiview_t1ce.png
│   └── BRATS_002_slice_progression_t1ce.png
├── BRATS_003/
│   ├── BRATS_003_all_modalities_axial.png
│   ├── BRATS_003_all_modalities_coronal.png
│   ├── BRATS_003_multiview_t1ce.png
│   └── BRATS_003_slice_progression_t1ce.png
├── comparison_all_samples_t1ce.png
├── comparison_all_samples_t2.png
└── comparison_complete_grid.png
```

**Total Files Generated:** 15 high-resolution visualizations

## Visualization Types Explained

### 1. All Modalities View
Shows all 4 MRI modalities side-by-side with segmentation:
- **Row 1**: T1, T1ce, T2, FLAIR (raw images)
- **Row 2**: Segmentation mask, T1ce + overlay

**Use Case:** Understanding different modalities and their tumor contrast

### 2. Slice Progression
Shows 9 evenly-spaced slices through the volume:
- Progression from bottom to top of brain
- Same modality (T1ce) for consistency
- Labels overlaid on each slice

**Use Case:** Understanding 3D tumor distribution and size

### 3. Multi-View
Shows same slice location in three orientations:
- **Axial**: Top-down view (most common)
- **Coronal**: Front-to-back view
- **Sagittal**: Left-to-right view

**Use Case:** Understanding tumor position in 3D space

### 4. Cross-Sample Comparison
Shows same view across all samples:
- Consistent slice selection (center)
- Same modality for fair comparison
- Same normalization and color scheme

**Use Case:** Comparing tumor characteristics across patients

## Key Insights from Visualizations

### Data Characteristics Observed:

1. **Multi-Modal Information:**
   - T1ce provides best tumor boundary definition
   - FLAIR shows edema extent clearly
   - Different modalities capture different tissue properties

2. **Tumor Segmentation:**
   - All samples show clear tumor segmentation
   - Multiple tumor components (edema, enhancing, non-enhancing)
   - Varying tumor sizes and locations across samples

3. **3D Structure:**
   - Tumors are truly 3D structures
   - Extent varies significantly across slices
   - Location varies (e.g., BRATS_001 appears more superior)

4. **Data Quality:**
   - Clean, well-aligned images
   - Consistent voxel spacing
   - Good contrast quality across all modalities

## Technical Details

### Color Scheme
- **MRI Images**: Grayscale (standard clinical convention)
- **Edema**: Green with 30% opacity
- **Non-enhancing tumor**: Yellow with 50% opacity
- **Enhancing tumor**: Red with 70% opacity

### Image Quality
- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG (lossless compression)
- **Size**: Optimized for both viewing and printing

### Normalization Strategy
- Percentile-based (99th percentile) to handle outliers
- Per-slice normalization for consistent visualization
- Maintains relative intensity relationships

## Extending the Visualizations

### Adding New Visualization Types

The `BRATSVisualizer` class can be easily extended:

```python
from visualize_brats_samples import BRATSVisualizer

visualizer = BRATSVisualizer('./data')
image_data, label_data = visualizer.load_sample('BRATS_001')

# Create custom visualization
# ... your visualization code ...
```

### Customizing Parameters

Modify these parameters in the scripts:
- `modality_idx`: Change which modality to visualize (0-3)
- `slice_idx`: Select specific slice location
- `view`: Choose 'axial', 'coronal', or 'sagittal'
- `percentile`: Adjust normalization percentile (default: 99)

## Requirements

**Python Packages:**
- `nibabel`: NIfTI file reading
- `numpy`: Array operations
- `matplotlib`: Plotting
- `seaborn`: Styling

**Installation:**
```bash
pip install nibabel numpy matplotlib seaborn
```

## References

### BRATS Dataset
The Brain Tumor Segmentation (BRATS) challenge provides standardized brain MRI data with tumor segmentation labels.

**Key Papers:**
- Menze et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)" IEEE TMI 2015
- Bakas et al. "Advancing The Cancer Genome Atlas glioma MRI collections" Nature Scientific Data 2017

### MRI Modalities
- **T1**: Anatomical imaging, gray/white matter contrast
- **T1ce**: Enhanced with gadolinium contrast agent, highlights blood-brain barrier breakdown
- **T2**: Fluid-sensitive, shows edema
- **FLAIR**: Suppresses CSF signal, highlights periventricular lesions

## Conclusion

These visualization tools provide comprehensive exploration of the BRATS 3D MRI dataset:

1. **Loading**: Using `nibabel` to read NIfTI format medical images
2. **Processing**: Normalization and slice extraction from 3D volumes
3. **Visualization**: Multi-modal, multi-view, and cross-sample comparisons
4. **Analysis**: Understanding tumor characteristics and data quality

The generated visualizations can be used for:
- Dataset quality assessment
- Model development and debugging
- Result presentation
- Educational purposes

---

**Created:** November 2025
**Tools Used:** Python, nibabel, matplotlib, numpy
**Dataset:** BRATS (Brain Tumor Segmentation Challenge)
