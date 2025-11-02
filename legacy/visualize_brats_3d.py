#!/usr/bin/env python3
"""
BRATS Dataset 3D Visualization using PyVista

This script generates true 3D volumetric visualizations of BRATS brain MRI data
with tumor segmentation using PyVista for publication-quality rendering.

Features:
- 3D transparent volume rendering of all MRI modalities
- 3D tumor segmentation mesh extraction and rendering
- Multi-modality comparison in 3D
- Multi-angle views (axial, coronal, sagittal, isometric)
- Volume + tumor overlay visualization

Usage:
    # Visualize all samples
    python visualize_brats_3d.py

    # Visualize specific sample
    python visualize_brats_3d.py --sample BRATS_001

    # High-resolution output
    python visualize_brats_3d.py --dpi 600

Author: Based on PyVista utilities
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import pyvista as pv
import warnings
warnings.filterwarnings('ignore')

# Import PyVista utilities
from pyvista_utils import (
    setup_pyvista_plotter,
    render_3d_volume,
    save_publication_image,
    create_opacity_transfer_function
)


class BRATS3DVisualizer:
    """3D visualizer for BRATS MRI data using PyVista."""

    MODALITY_NAMES = ['T1', 'T1ce', 'T2', 'FLAIR']
    LABEL_NAMES = {
        0: 'Normal',
        1: 'Edema',
        2: 'Non-enhancing tumor',
        3: 'Enhancing tumor'
    }
    LABEL_COLORS = {
        1: [0.0, 1.0, 0.0],      # Green for edema
        2: [1.0, 1.0, 0.0],      # Yellow for non-enhancing
        3: [1.0, 0.0, 0.0]       # Red for enhancing
    }

    def __init__(self, data_dir: str):
        """Initialize 3D visualizer."""
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'

    def load_sample(self, sample_name: str):
        """Load BRATS sample."""
        image_path = self.images_dir / f"{sample_name}.nii.gz"
        label_path = self.labels_dir / f"{sample_name}.nii.gz"

        print(f"Loading {sample_name}...")
        img_data = nib.load(str(image_path)).get_fdata()
        label_data = nib.load(str(label_path)).get_fdata()

        print(f"  Image shape: {img_data.shape}")
        print(f"  Label shape: {label_data.shape}")
        print(f"  Unique labels: {np.unique(label_data)}")

        return img_data, label_data

    def extract_tumor_mesh(self, label_data: np.ndarray, label_id: int, smoothing: int = 100):
        """
        Extract 3D mesh surface for a specific tumor label.

        Args:
            label_data: 3D label array
            label_id: Label ID to extract (1=edema, 2=non-enhancing, 3=enhancing)
            smoothing: Smoothing iterations for mesh (default: 100)

        Returns:
            PyVista mesh object
        """
        # Create binary mask for this label
        mask = (label_data == label_id).astype(np.float32)

        # Skip if no voxels for this label
        if mask.sum() == 0:
            return None

        # Create PyVista grid
        grid = pv.ImageData()
        grid.dimensions = np.array(mask.shape)
        grid.spacing = (1.0, 1.0, 1.0)
        grid.origin = (0.0, 0.0, 0.0)
        # Use point_data instead of cell_data for contour filter
        grid.point_data['values'] = mask.flatten(order='F')

        # Extract surface using marching cubes
        # Use contour to extract isosurface at 0.5
        mesh = grid.contour([0.5], scalars='values')

        # Smooth the mesh
        if smoothing > 0:
            mesh = mesh.smooth(n_iter=smoothing)

        return mesh

    def render_3d_modality_volume(
        self,
        image_data: np.ndarray,
        modality_idx: int,
        plotter: pv.Plotter,
        title: str = None,
        transparency_level: str = 'medium',
        camera_position: str = 'iso'
    ):
        """Render a single MRI modality as 3D volume."""
        # Extract modality
        modality_volume = image_data[:, :, :, modality_idx]

        # Normalize to better visualization range
        volume_normalized = self._normalize_volume(modality_volume)

        # Get modality name
        modality_name = self.MODALITY_NAMES[modality_idx]
        if title is None:
            title = f'{modality_name} Volume'

        # Render 3D volume
        render_3d_volume(
            volume_normalized,
            plotter,
            opacity_style='brain',
            transparency_level=transparency_level,
            cmap='gray',
            title=title,
            camera_position=camera_position,
            preset='brain_mri',
            modality=modality_name.lower()
        )

    def render_tumor_segmentation_3d(
        self,
        label_data: np.ndarray,
        plotter: pv.Plotter,
        show_labels: list = [1, 2, 3],
        opacity: float = 0.7,
        title: str = 'Tumor Segmentation 3D',
        camera_position: str = 'iso'
    ):
        """
        Render tumor segmentation as 3D colored meshes.

        Args:
            label_data: 3D label volume
            plotter: PyVista plotter
            show_labels: List of label IDs to show
            opacity: Mesh opacity (0-1)
            title: Plot title
            camera_position: Camera position
        """
        print("  Extracting tumor meshes...")

        # Extract and render each tumor component
        legend_labels = []
        for label_id in show_labels:
            if label_id not in self.LABEL_COLORS:
                continue

            mesh = self.extract_tumor_mesh(label_data, label_id, smoothing=50)

            if mesh is not None and mesh.n_points > 0:
                color = self.LABEL_COLORS[label_id]
                label_name = self.LABEL_NAMES[label_id]

                plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    show_edges=False
                )

                legend_labels.append([label_name, color])
                print(f"    Added {label_name} mesh ({mesh.n_points} points)")

        # Add legend
        if legend_labels:
            plotter.add_legend(
                legend_labels,
                bcolor='white',
                face='rectangle',
                size=(0.2, 0.15),
                loc='upper right'
            )

        # Set camera
        if camera_position == 'xy':
            plotter.view_xy()
        elif camera_position == 'xz':
            plotter.view_xz()
        elif camera_position == 'yz':
            plotter.view_yz()
        elif camera_position == 'iso':
            plotter.view_isometric()

        # Add title
        if title:
            plotter.add_text(
                title,
                position='upper_edge',
                font_size=16,
                color='black',
                font='times'
            )

    def render_volume_with_tumor_overlay(
        self,
        image_data: np.ndarray,
        label_data: np.ndarray,
        modality_idx: int = 1,  # T1ce
        plotter: pv.Plotter = None,
        title: str = None,
        volume_opacity: str = 'high',
        tumor_opacity: float = 0.7,
        camera_position: str = 'iso'
    ):
        """
        Render MRI volume with tumor segmentation overlay.

        Args:
            image_data: 4D image array (H, W, D, C)
            label_data: 3D label array
            modality_idx: Which modality to show (1=T1ce recommended)
            plotter: PyVista plotter
            title: Plot title
            volume_opacity: Volume transparency level
            tumor_opacity: Tumor mesh opacity
            camera_position: Camera position
        """
        # Extract and render volume
        modality_volume = image_data[:, :, :, modality_idx]
        volume_normalized = self._normalize_volume(modality_volume)

        modality_name = self.MODALITY_NAMES[modality_idx]
        if title is None:
            title = f'{modality_name} + Tumor Overlay'

        print(f"  Rendering {modality_name} volume with tumor overlay...")

        # Render volume
        render_3d_volume(
            volume_normalized,
            plotter,
            opacity_style='brain',
            transparency_level=volume_opacity,
            cmap='gray',
            title=None,  # Add combined title later
            camera_position=camera_position,
            preset='brain_mri',
            modality=modality_name.lower()
        )

        # Overlay tumor meshes
        print("  Overlaying tumor meshes...")
        for label_id in [1, 2, 3]:
            mesh = self.extract_tumor_mesh(label_data, label_id, smoothing=50)

            if mesh is not None and mesh.n_points > 0:
                color = self.LABEL_COLORS[label_id]
                plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=tumor_opacity,
                    smooth_shading=True,
                    show_edges=False
                )

        # Add title
        if title:
            plotter.add_text(
                title,
                position='upper_edge',
                font_size=16,
                color='black',
                font='times'
            )

    def create_multiview_3d(
        self,
        image_data: np.ndarray,
        label_data: np.ndarray,
        modality_idx: int = 1,
        save_path: Path = None,
        dpi: int = 300
    ):
        """Create multi-angle 3D views."""
        print("Creating multi-view 3D visualization...")

        views = ['xy', 'xz', 'yz', 'iso']
        view_titles = ['Axial View', 'Coronal View', 'Sagittal View', '3D Isometric']

        plotter = setup_pyvista_plotter(
            window_size=(2400, 2400),
            off_screen=True,
            shape=(2, 2)
        )

        modality_volume = image_data[:, :, :, modality_idx]
        volume_normalized = self._normalize_volume(modality_volume)
        modality_name = self.MODALITY_NAMES[modality_idx]

        for idx, (view, view_title) in enumerate(zip(views, view_titles)):
            row = idx // 2
            col = idx % 2
            plotter.subplot(row, col)

            # Render volume
            render_3d_volume(
                volume_normalized,
                plotter,
                opacity_style='brain',
                transparency_level='medium',
                cmap='gray',
                title=f'{modality_name} - {view_title}',
                camera_position=view,
                preset='brain_mri',
                modality=modality_name.lower()
            )

            # Overlay tumors
            for label_id in [1, 2, 3]:
                mesh = self.extract_tumor_mesh(label_data, label_id, smoothing=50)
                if mesh is not None and mesh.n_points > 0:
                    plotter.add_mesh(
                        mesh,
                        color=self.LABEL_COLORS[label_id],
                        opacity=0.7,
                        smooth_shading=True
                    )

        # Link views
        plotter.link_views()

        if save_path:
            save_publication_image(plotter, str(save_path), dpi=dpi)

        plotter.close()

    def create_modality_comparison_3d(
        self,
        image_data: np.ndarray,
        save_path: Path = None,
        dpi: int = 300
    ):
        """Create 3D comparison of all 4 modalities."""
        print("Creating all-modality 3D comparison...")

        plotter = setup_pyvista_plotter(
            window_size=(2400, 1200),
            off_screen=True,
            shape=(1, 4)
        )

        for modality_idx in range(4):
            plotter.subplot(0, modality_idx)

            modality_volume = image_data[:, :, :, modality_idx]
            volume_normalized = self._normalize_volume(modality_volume)
            modality_name = self.MODALITY_NAMES[modality_idx]

            render_3d_volume(
                volume_normalized,
                plotter,
                opacity_style='brain',
                transparency_level='medium',
                cmap='gray',
                title=modality_name,
                camera_position='iso',
                preset='brain_mri',
                modality=modality_name.lower()
            )

        # Link views
        plotter.link_views()

        if save_path:
            save_publication_image(plotter, str(save_path), dpi=dpi)

        plotter.close()

    def _normalize_volume(self, volume: np.ndarray, percentile: float = 99) -> np.ndarray:
        """Normalize volume using percentile method."""
        vmin = volume.min()
        vmax = np.percentile(volume, percentile)
        normalized = np.clip((volume - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return normalized


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Generate 3D visualizations of BRATS MRI data'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./support/Visualize-3D-MRI-Scans-Brain-case/data',
        help='Path to BRATS data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./brats_3d_visualizations',
        help='Output directory for 3D visualizations'
    )
    parser.add_argument(
        '--sample',
        type=str,
        default=None,
        help='Specific sample to visualize (e.g., BRATS_001). If None, visualizes all.'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for output images (default: 300)'
    )
    parser.add_argument(
        '--modality',
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help='Modality index for overlay (0=T1, 1=T1ce, 2=T2, 3=FLAIR, default: 1)'
    )

    args = parser.parse_args()

    # Setup
    print("=" * 80)
    print("BRATS 3D Visualization with PyVista")
    print("=" * 80)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualizer = BRATS3DVisualizer(args.data_dir)

    # Determine samples to process
    if args.sample:
        sample_names = [args.sample]
    else:
        # Get all available samples
        image_files = sorted(visualizer.images_dir.glob('*.nii.gz'))
        sample_names = [f.stem.replace('.nii', '') for f in image_files]

    print(f"\nFound {len(sample_names)} samples: {sample_names}\n")

    # Process each sample
    for sample_name in sample_names:
        print(f"\n{'='*70}")
        print(f"Processing: {sample_name}")
        print(f"{'='*70}")

        # Load data
        image_data, label_data = visualizer.load_sample(sample_name)

        # Create sample output directory
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(exist_ok=True)

        # 1. T1ce 3D volume
        print("\n1. Rendering T1ce 3D volume...")
        plotter = setup_pyvista_plotter(window_size=(1920, 1080), off_screen=True)
        visualizer.render_3d_modality_volume(
            image_data, 1, plotter, title=f'{sample_name} - T1ce Volume'
        )
        save_path = sample_dir / f'{sample_name}_t1ce_3d_volume.png'
        save_publication_image(plotter, str(save_path), dpi=args.dpi)
        plotter.close()

        # 2. Tumor segmentation 3D
        print("\n2. Rendering tumor segmentation 3D meshes...")
        plotter = setup_pyvista_plotter(window_size=(1920, 1080), off_screen=True)
        visualizer.render_tumor_segmentation_3d(
            label_data, plotter, title=f'{sample_name} - Tumor Segmentation 3D'
        )
        save_path = sample_dir / f'{sample_name}_segmentation_3d_mesh.png'
        save_publication_image(plotter, str(save_path), dpi=args.dpi)
        plotter.close()

        # 3. Volume + tumor overlay
        print("\n3. Rendering volume with tumor overlay...")
        plotter = setup_pyvista_plotter(window_size=(1920, 1080), off_screen=True)
        visualizer.render_volume_with_tumor_overlay(
            image_data, label_data, args.modality, plotter,
            title=f'{sample_name} - Volume + Tumor Overlay'
        )
        save_path = sample_dir / f'{sample_name}_volume_with_tumor_overlay.png'
        save_publication_image(plotter, str(save_path), dpi=args.dpi)
        plotter.close()

        # 4. Multi-view 3D
        print("\n4. Creating multi-view 3D visualization...")
        save_path = sample_dir / f'{sample_name}_multiview_3d.png'
        visualizer.create_multiview_3d(
            image_data, label_data, args.modality, save_path, args.dpi
        )

        # 5. All modalities 3D comparison
        print("\n5. Creating all-modality 3D comparison...")
        save_path = sample_dir / f'{sample_name}_all_modalities_3d_comparison.png'
        visualizer.create_modality_comparison_3d(
            image_data, save_path, args.dpi
        )

        print(f"\nCompleted {sample_name}! Outputs saved to: {sample_dir}")

    print(f"\n{'='*80}")
    print(f"All 3D visualizations complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
