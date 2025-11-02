#!/usr/bin/env python3
"""
Comprehensive BRATS Dataset Visualization Script

This script visualizes 3D MRI brain data from the BRATS dataset,
including all 4 MRI modalities and segmentation labels.

Usage:
    python visualize_brats_samples.py [--data-dir PATH] [--output-dir PATH]
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Tuple, List, Optional

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (15, 10)


class BRATSVisualizer:
    """Visualizer for BRATS 3D MRI data."""

    MODALITY_NAMES = ['T1', 'T1ce', 'T2', 'FLAIR']
    LABEL_NAMES = {
        0: 'Normal',
        1: 'Edema',
        2: 'Non-enhancing tumor',
        3: 'Enhancing tumor'
    }
    LABEL_COLORS = {
        0: [0, 0, 0, 0],        # Transparent for normal
        1: [0, 1, 0, 0.3],      # Green for edema
        2: [1, 1, 0, 0.5],      # Yellow for non-enhancing
        3: [1, 0, 0, 0.7]       # Red for enhancing
    }

    def __init__(self, data_dir: str):
        """
        Initialize BRATS visualizer.

        Args:
            data_dir: Path to BRATS data directory containing images/ and labels/
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

    def load_sample(self, sample_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a BRATS sample (image and label).

        Args:
            sample_name: Sample name (e.g., 'BRATS_001')

        Returns:
            Tuple of (image_data, label_data)
            - image_data: shape (H, W, D, C) where C=4 modalities
            - label_data: shape (H, W, D)
        """
        image_path = self.images_dir / f"{sample_name}.nii.gz"
        label_path = self.labels_dir / f"{sample_name}.nii.gz"

        # Load image
        img_obj = nib.load(str(image_path))
        img_data = img_obj.get_fdata()

        # Load label
        label_obj = nib.load(str(label_path))
        label_data = label_obj.get_fdata()

        print(f"Loaded {sample_name}:")
        print(f"  Image shape: {img_data.shape}")
        print(f"  Label shape: {label_data.shape}")
        print(f"  Unique labels: {np.unique(label_data)}")

        return img_data, label_data

    def get_center_slices(self, volume_shape: Tuple) -> Tuple[int, int, int]:
        """Get center slice indices for axial, coronal, sagittal views."""
        if len(volume_shape) == 4:
            H, W, D, _ = volume_shape
        else:
            H, W, D = volume_shape
        return H // 2, W // 2, D // 2

    def normalize_slice(self, slice_data: np.ndarray, percentile: float = 99) -> np.ndarray:
        """Normalize slice to [0, 1] using percentile."""
        vmin = slice_data.min()
        vmax = np.percentile(slice_data, percentile)
        normalized = np.clip((slice_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return normalized

    def create_label_overlay(self, slice_data: np.ndarray, label_slice: np.ndarray) -> np.ndarray:
        """
        Create colored label overlay on grayscale image.

        Args:
            slice_data: Normalized image slice [0, 1]
            label_slice: Label slice with values [0, 1, 2, 3]

        Returns:
            RGB image with label overlay
        """
        # Convert grayscale to RGB
        rgb_image = np.stack([slice_data] * 3, axis=-1)

        # Create overlay
        overlay = np.zeros((*slice_data.shape, 4))
        for label_id, color in self.LABEL_COLORS.items():
            mask = label_slice == label_id
            overlay[mask] = color

        # Blend
        alpha = overlay[..., 3:4]
        rgb_overlay = overlay[..., :3]
        result = rgb_image * (1 - alpha) + rgb_overlay * alpha

        return result

    def visualize_all_modalities_single_slice(
        self,
        image_data: np.ndarray,
        label_data: np.ndarray,
        slice_idx: Optional[int] = None,
        view: str = 'axial',
        save_path: Optional[Path] = None
    ):
        """
        Visualize all 4 modalities for a single slice.

        Args:
            image_data: Image volume (H, W, D, 4)
            label_data: Label volume (H, W, D)
            slice_idx: Slice index (if None, use center slice)
            view: 'axial', 'coronal', or 'sagittal'
            save_path: Optional path to save figure
        """
        # Determine slice index
        if slice_idx is None:
            center_h, center_w, center_d = self.get_center_slices(image_data.shape)
            if view == 'axial':
                slice_idx = center_d
            elif view == 'coronal':
                slice_idx = center_w
            else:  # sagittal
                slice_idx = center_h

        # Extract slices based on view
        if view == 'axial':
            image_slices = image_data[:, :, slice_idx, :]
            label_slice = label_data[:, :, slice_idx]
        elif view == 'coronal':
            image_slices = image_data[:, slice_idx, :, :]
            label_slice = label_data[:, slice_idx, :]
        else:  # sagittal
            image_slices = image_data[slice_idx, :, :, :]
            label_slice = label_data[slice_idx, :, :]

        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'BRATS MRI - {view.capitalize()} View (Slice {slice_idx})',
                     fontsize=20, fontweight='bold')

        # Plot each modality
        for i, (ax, modality) in enumerate(zip(axes.flat[:4], self.MODALITY_NAMES)):
            modality_slice = image_slices[..., i]
            normalized = self.normalize_slice(modality_slice)
            # Transpose for display (swap x and y for correct orientation)
            ax.imshow(normalized, cmap='gray', origin='lower', aspect='auto')
            ax.set_title(f'{modality}', fontsize=16, fontweight='bold')
            ax.axis('off')

        # Plot segmentation mask
        ax = axes.flat[4]
        mask_rgb = np.zeros((*label_slice.shape, 3))
        for label_id in [1, 2, 3]:  # Skip background
            mask = label_slice == label_id
            color = self.LABEL_COLORS[label_id][:3]
            mask_rgb[mask] = color
        ax.imshow(mask_rgb, origin='lower', aspect='auto')
        ax.set_title('Segmentation Mask', fontsize=16, fontweight='bold')
        ax.axis('off')

        # Plot overlay on T1ce (best contrast)
        ax = axes.flat[5]
        t1ce_slice = image_slices[..., 1]
        normalized_t1ce = self.normalize_slice(t1ce_slice)
        overlay = self.create_label_overlay(normalized_t1ce, label_slice)
        ax.imshow(overlay, origin='lower', aspect='auto')
        ax.set_title('T1ce + Segmentation Overlay', fontsize=16, fontweight='bold')
        ax.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.LABEL_COLORS[1][:3], label='Edema'),
            Patch(facecolor=self.LABEL_COLORS[2][:3], label='Non-enhancing tumor'),
            Patch(facecolor=self.LABEL_COLORS[3][:3], label='Enhancing tumor')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                  fontsize=14, frameon=True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def visualize_multi_slice_progression(
        self,
        image_data: np.ndarray,
        label_data: np.ndarray,
        modality_idx: int = 1,  # T1ce
        num_slices: int = 9,
        save_path: Optional[Path] = None
    ):
        """
        Show progression through slices for a single modality.

        Args:
            image_data: Image volume (H, W, D, 4)
            label_data: Label volume (H, W, D)
            modality_idx: Which modality to visualize (0=T1, 1=T1ce, 2=T2, 3=FLAIR)
            num_slices: Number of slices to show
            save_path: Optional path to save figure
        """
        H, W, D, C = image_data.shape

        # Select evenly spaced slices
        slice_indices = np.linspace(D // 4, 3 * D // 4, num_slices).astype(int)

        # Create grid
        rows = 3
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        modality_name = self.MODALITY_NAMES[modality_idx]
        fig.suptitle(f'BRATS MRI - {modality_name} Slice Progression',
                     fontsize=20, fontweight='bold')

        for idx, (ax, slice_idx) in enumerate(zip(axes.flat, slice_indices)):
            # Get slice
            image_slice = image_data[:, :, slice_idx, modality_idx]
            label_slice = label_data[:, :, slice_idx]

            # Normalize and create overlay
            normalized = self.normalize_slice(image_slice)
            overlay = self.create_label_overlay(normalized, label_slice)

            ax.imshow(overlay, origin='lower', aspect='auto')
            ax.set_title(f'Slice {slice_idx}/{D}', fontsize=12)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def visualize_all_views(
        self,
        image_data: np.ndarray,
        label_data: np.ndarray,
        modality_idx: int = 1,  # T1ce
        save_path: Optional[Path] = None
    ):
        """
        Show axial, coronal, and sagittal views of a single modality.

        Args:
            image_data: Image volume (H, W, D, 4)
            label_data: Label volume (H, W, D)
            modality_idx: Which modality to visualize
            save_path: Optional path to save figure
        """
        center_h, center_w, center_d = self.get_center_slices(image_data.shape)
        modality_name = self.MODALITY_NAMES[modality_idx]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'BRATS MRI - {modality_name} Multi-View',
                     fontsize=20, fontweight='bold')

        # Axial view
        ax = axes[0]
        image_slice = image_data[:, :, center_d, modality_idx]
        label_slice = label_data[:, :, center_d]
        normalized = self.normalize_slice(image_slice)
        overlay = self.create_label_overlay(normalized, label_slice)
        ax.imshow(overlay, origin='lower', aspect='auto')
        ax.set_title(f'Axial (slice {center_d})', fontsize=16)
        ax.axis('off')

        # Coronal view
        ax = axes[1]
        image_slice = image_data[:, center_w, :, modality_idx]
        label_slice = label_data[:, center_w, :]
        normalized = self.normalize_slice(image_slice)
        overlay = self.create_label_overlay(normalized, label_slice)
        ax.imshow(overlay, origin='lower', aspect='auto')
        ax.set_title(f'Coronal (slice {center_w})', fontsize=16)
        ax.axis('off')

        # Sagittal view
        ax = axes[2]
        image_slice = image_data[center_h, :, :, modality_idx]
        label_slice = label_data[center_h, :, :]
        normalized = self.normalize_slice(image_slice)
        overlay = self.create_label_overlay(normalized, label_slice)
        ax.imshow(overlay, origin='lower', aspect='auto')
        ax.set_title(f'Sagittal (slice {center_h})', fontsize=16)
        ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def get_available_samples(self) -> List[str]:
        """Get list of available sample names."""
        image_files = sorted(self.images_dir.glob('*.nii.gz'))
        sample_names = [f.stem.replace('.nii', '') for f in image_files]
        return sample_names


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Visualize BRATS 3D MRI dataset'
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
        default='./brats_visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--samples',
        type=str,
        nargs='+',
        default=None,
        help='Sample names to visualize (default: all)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")

    # Initialize visualizer
    visualizer = BRATSVisualizer(args.data_dir)

    # Get samples to process
    if args.samples:
        sample_names = args.samples
    else:
        sample_names = visualizer.get_available_samples()

    print(f"\nFound {len(sample_names)} samples: {sample_names}")

    # Process each sample
    for sample_name in sample_names:
        print(f"\n{'='*70}")
        print(f"Processing: {sample_name}")
        print(f"{'='*70}")

        # Load data
        image_data, label_data = visualizer.load_sample(sample_name)

        # Create sample-specific output directory
        sample_dir = output_dir / sample_name
        sample_dir.mkdir(exist_ok=True)

        # 1. All modalities - axial view
        print("\n1. Generating all modalities visualization (axial)...")
        fig = visualizer.visualize_all_modalities_single_slice(
            image_data, label_data,
            view='axial',
            save_path=sample_dir / f'{sample_name}_all_modalities_axial.png'
        )
        plt.close(fig)

        # 2. Multi-slice progression (T1ce)
        print("2. Generating slice progression visualization...")
        fig = visualizer.visualize_multi_slice_progression(
            image_data, label_data,
            modality_idx=1,  # T1ce
            save_path=sample_dir / f'{sample_name}_slice_progression_t1ce.png'
        )
        plt.close(fig)

        # 3. Multi-view (axial, coronal, sagittal)
        print("3. Generating multi-view visualization...")
        fig = visualizer.visualize_all_views(
            image_data, label_data,
            modality_idx=1,  # T1ce
            save_path=sample_dir / f'{sample_name}_multiview_t1ce.png'
        )
        plt.close(fig)

        # 4. All modalities - coronal view
        print("4. Generating all modalities visualization (coronal)...")
        fig = visualizer.visualize_all_modalities_single_slice(
            image_data, label_data,
            view='coronal',
            save_path=sample_dir / f'{sample_name}_all_modalities_coronal.png'
        )
        plt.close(fig)

        print(f"\nCompleted {sample_name}! Outputs saved to: {sample_dir}")

    print(f"\n{'='*70}")
    print(f"All visualizations complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
