#!/usr/bin/env python3
"""
Create cross-sample 3D comparison for BRATS dataset.

This script generates side-by-side 3D visualizations comparing all BRATS samples.
"""

import numpy as np
import nibabel as nib
import pyvista as pv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from pyvista_utils import (
    setup_pyvista_plotter,
    render_3d_volume,
    save_publication_image
)


def load_and_normalize(image_path, modality_idx=1):
    """Load and normalize a BRATS sample."""
    img_data = nib.load(str(image_path)).get_fdata()
    modality_volume = img_data[:, :, :, modality_idx]

    # Normalize
    vmin = modality_volume.min()
    vmax = np.percentile(modality_volume, 99)
    normalized = np.clip((modality_volume - vmin) / (vmax - vmin + 1e-8), 0, 1)

    return normalized


def extract_tumor_mesh(label_path, label_id, smoothing=50):
    """Extract 3D tumor mesh."""
    label_data = nib.load(str(label_path)).get_fdata()
    mask = (label_data == label_id).astype(np.float32)

    if mask.sum() == 0:
        return None

    # Create PyVista grid
    grid = pv.ImageData()
    grid.dimensions = np.array(mask.shape)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)
    grid.point_data['values'] = mask.flatten(order='F')

    # Extract surface
    mesh = grid.contour([0.5], scalars='values')

    # Smooth
    if smoothing > 0:
        mesh = mesh.smooth(n_iter=smoothing)

    return mesh


def main():
    """Main execution."""
    print("Creating cross-sample 3D comparison...")

    data_dir = Path('./support/Visualize-3D-MRI-Scans-Brain-case/data')
    output_dir = Path('./brats_3d_visualizations')
    output_dir.mkdir(exist_ok=True)

    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'

    sample_names = ['BRATS_001', 'BRATS_002', 'BRATS_003']

    # 1. T1ce volume comparison (3 samples side-by-side)
    print("\n1. Creating T1ce volume cross-sample comparison...")
    plotter = setup_pyvista_plotter(
        window_size=(2400, 800),
        off_screen=True,
        shape=(1, 3)
    )

    for idx, sample_name in enumerate(sample_names):
        plotter.subplot(0, idx)

        image_path = images_dir / f"{sample_name}.nii.gz"
        volume = load_and_normalize(image_path, modality_idx=1)

        render_3d_volume(
            volume,
            plotter,
            opacity_style='brain',
            transparency_level='medium',
            cmap='gray',
            title=sample_name,
            camera_position='iso',
            preset='brain_mri',
            modality='t1ce'
        )

    # Link views
    plotter.link_views()

    save_path = output_dir / 'cross_sample_t1ce_3d_comparison.png'
    save_publication_image(plotter, str(save_path), dpi=300)
    plotter.close()
    print(f"  Saved: {save_path}")

    # 2. Tumor segmentation comparison (3 samples side-by-side)
    print("\n2. Creating tumor segmentation cross-sample comparison...")
    plotter = setup_pyvista_plotter(
        window_size=(2400, 800),
        off_screen=True,
        shape=(1, 3)
    )

    tumor_colors = {
        1: [0.0, 1.0, 0.0],  # Green - Edema
        2: [1.0, 1.0, 0.0],  # Yellow - Non-enhancing
        3: [1.0, 0.0, 0.0]   # Red - Enhancing
    }

    for idx, sample_name in enumerate(sample_names):
        print(f"  Processing {sample_name}...")
        plotter.subplot(0, idx)

        label_path = labels_dir / f"{sample_name}.nii.gz"

        # Extract and render each tumor component
        for label_id in [1, 2, 3]:
            mesh = extract_tumor_mesh(label_path, label_id, smoothing=50)

            if mesh is not None and mesh.n_points > 0:
                plotter.add_mesh(
                    mesh,
                    color=tumor_colors[label_id],
                    opacity=0.7,
                    smooth_shading=True,
                    show_edges=False
                )

        # Set camera
        plotter.view_isometric()

        # Add title
        plotter.add_text(
            sample_name,
            position='upper_edge',
            font_size=16,
            color='black',
            font='times'
        )

    # Link views
    plotter.link_views()

    save_path = output_dir / 'cross_sample_tumor_3d_comparison.png'
    save_publication_image(plotter, str(save_path), dpi=300)
    plotter.close()
    print(f"  Saved: {save_path}")

    # 3. Volume + tumor overlay comparison
    print("\n3. Creating volume+tumor cross-sample comparison...")
    plotter = setup_pyvista_plotter(
        window_size=(2400, 800),
        off_screen=True,
        shape=(1, 3)
    )

    for idx, sample_name in enumerate(sample_names):
        print(f"  Processing {sample_name}...")
        plotter.subplot(0, idx)

        # Render volume
        image_path = images_dir / f"{sample_name}.nii.gz"
        volume = load_and_normalize(image_path, modality_idx=1)

        render_3d_volume(
            volume,
            plotter,
            opacity_style='brain',
            transparency_level='high',
            cmap='gray',
            title=None,
            camera_position='iso',
            preset='brain_mri',
            modality='t1ce'
        )

        # Overlay tumors
        label_path = labels_dir / f"{sample_name}.nii.gz"
        for label_id in [1, 2, 3]:
            mesh = extract_tumor_mesh(label_path, label_id, smoothing=50)
            if mesh is not None and mesh.n_points > 0:
                plotter.add_mesh(
                    mesh,
                    color=tumor_colors[label_id],
                    opacity=0.7,
                    smooth_shading=True,
                    show_edges=False
                )

        # Set camera
        plotter.view_isometric()

        # Add title
        plotter.add_text(
            sample_name,
            position='upper_edge',
            font_size=16,
            color='black',
            font='times'
        )

    # Link views
    plotter.link_views()

    save_path = output_dir / 'cross_sample_volume_tumor_3d_comparison.png'
    save_publication_image(plotter, str(save_path), dpi=300)
    plotter.close()
    print(f"  Saved: {save_path}")

    print("\n" + "="*80)
    print("Cross-sample 3D comparison complete!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
