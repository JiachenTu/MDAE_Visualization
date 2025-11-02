#!/usr/bin/env python3
"""
Create cross-sample comparison figures for BRATS dataset.

This script loads all 3 BRATS samples and creates comparison visualizations.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional

sns.set_style('darkgrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


class BRATSComparison:
    """Create comparison visualizations across multiple BRATS samples."""

    MODALITY_NAMES = ['T1', 'T1ce', 'T2', 'FLAIR']
    LABEL_NAMES = {
        0: 'Normal',
        1: 'Edema',
        2: 'Non-enhancing tumor',
        3: 'Enhancing tumor'
    }
    LABEL_COLORS = {
        0: [0, 0, 0, 0],
        1: [0, 1, 0, 0.3],
        2: [1, 1, 0, 0.5],
        3: [1, 0, 0, 0.7]
    }

    def __init__(self, data_dir: str):
        """Initialize comparison visualizer."""
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'images'
        self.labels_dir = self.data_dir / 'labels'

    def load_sample(self, sample_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a BRATS sample."""
        image_path = self.images_dir / f"{sample_name}.nii.gz"
        label_path = self.labels_dir / f"{sample_name}.nii.gz"

        img_data = nib.load(str(image_path)).get_fdata()
        label_data = nib.load(str(label_path)).get_fdata()

        return img_data, label_data

    def normalize_slice(self, slice_data: np.ndarray, percentile: float = 99) -> np.ndarray:
        """Normalize slice to [0, 1]."""
        vmin = slice_data.min()
        vmax = np.percentile(slice_data, percentile)
        normalized = np.clip((slice_data - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return normalized

    def create_label_overlay(self, slice_data: np.ndarray, label_slice: np.ndarray) -> np.ndarray:
        """Create colored label overlay."""
        rgb_image = np.stack([slice_data] * 3, axis=-1)
        overlay = np.zeros((*slice_data.shape, 4))

        for label_id, color in self.LABEL_COLORS.items():
            mask = label_slice == label_id
            overlay[mask] = color

        alpha = overlay[..., 3:4]
        rgb_overlay = overlay[..., :3]
        result = rgb_image * (1 - alpha) + rgb_overlay * alpha

        return result

    def compare_all_samples_single_modality(
        self,
        sample_names: List[str],
        modality_idx: int = 1,  # T1ce
        slice_idx: Optional[int] = None,
        save_path: Optional[Path] = None
    ):
        """
        Compare all samples for a single modality.

        Args:
            sample_names: List of sample names to compare
            modality_idx: Which modality (0=T1, 1=T1ce, 2=T2, 3=FLAIR)
            slice_idx: Slice index (if None, use center)
            save_path: Path to save figure
        """
        n_samples = len(sample_names)
        modality_name = self.MODALITY_NAMES[modality_idx]

        fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
        fig.suptitle(f'BRATS Dataset Comparison - {modality_name}',
                     fontsize=22, fontweight='bold')

        for i, sample_name in enumerate(sample_names):
            print(f"Loading {sample_name}...")
            image_data, label_data = self.load_sample(sample_name)

            # Get center slice if not specified
            if slice_idx is None:
                current_slice_idx = image_data.shape[2] // 2
            else:
                current_slice_idx = slice_idx

            # Extract slices
            image_slice = image_data[:, :, current_slice_idx, modality_idx]
            label_slice = label_data[:, :, current_slice_idx]

            # Normalize
            normalized = self.normalize_slice(image_slice)
            overlay = self.create_label_overlay(normalized, label_slice)

            # Plot raw image
            axes[i, 0].imshow(normalized, cmap='gray', origin='lower', aspect='auto')
            axes[i, 0].set_title(f'{sample_name} - {modality_name}', fontsize=14, fontweight='bold')
            axes[i, 0].axis('off')

            # Plot segmentation mask
            mask_rgb = np.zeros((*label_slice.shape, 3))
            for label_id in [1, 2, 3]:
                mask = label_slice == label_id
                color = self.LABEL_COLORS[label_id][:3]
                mask_rgb[mask] = color
            axes[i, 1].imshow(mask_rgb, origin='lower', aspect='auto')
            axes[i, 1].set_title(f'{sample_name} - Segmentation', fontsize=14, fontweight='bold')
            axes[i, 1].axis('off')

            # Plot overlay
            axes[i, 2].imshow(overlay, origin='lower', aspect='auto')
            axes[i, 2].set_title(f'{sample_name} - Overlay', fontsize=14, fontweight='bold')
            axes[i, 2].axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.LABEL_COLORS[1][:3], label='Edema'),
            Patch(facecolor=self.LABEL_COLORS[2][:3], label='Non-enhancing tumor'),
            Patch(facecolor=self.LABEL_COLORS[3][:3], label='Enhancing tumor')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                  fontsize=12, frameon=True)

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig

    def compare_all_modalities_all_samples(
        self,
        sample_names: List[str],
        slice_idx: Optional[int] = None,
        save_path: Optional[Path] = None
    ):
        """
        Create a large comparison grid showing all samples and all modalities.

        Args:
            sample_names: List of sample names
            slice_idx: Slice index (if None, use center)
            save_path: Path to save figure
        """
        n_samples = len(sample_names)
        n_modalities = 4

        fig, axes = plt.subplots(n_samples, n_modalities + 1, figsize=(20, 5 * n_samples))
        fig.suptitle('BRATS Dataset - Complete Comparison',
                     fontsize=24, fontweight='bold')

        for i, sample_name in enumerate(sample_names):
            print(f"Processing {sample_name}...")
            image_data, label_data = self.load_sample(sample_name)

            # Get center slice
            if slice_idx is None:
                current_slice_idx = image_data.shape[2] // 2
            else:
                current_slice_idx = slice_idx

            # Plot each modality
            for j in range(n_modalities):
                image_slice = image_data[:, :, current_slice_idx, j]
                normalized = self.normalize_slice(image_slice)

                axes[i, j].imshow(normalized, cmap='gray', origin='lower', aspect='auto')
                if i == 0:
                    axes[i, j].set_title(self.MODALITY_NAMES[j], fontsize=16, fontweight='bold')
                axes[i, j].axis('off')

            # Plot overlay (T1ce + labels)
            label_slice = label_data[:, :, current_slice_idx]
            t1ce_slice = image_data[:, :, current_slice_idx, 1]
            normalized_t1ce = self.normalize_slice(t1ce_slice)
            overlay = self.create_label_overlay(normalized_t1ce, label_slice)

            axes[i, n_modalities].imshow(overlay, origin='lower', aspect='auto')
            if i == 0:
                axes[i, n_modalities].set_title('T1ce + Labels', fontsize=16, fontweight='bold')
            axes[i, n_modalities].axis('off')

            # Add sample name as y-label
            axes[i, 0].set_ylabel(sample_name, fontsize=16, fontweight='bold', rotation=0,
                                 ha='right', va='center')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.LABEL_COLORS[1][:3], label='Edema'),
            Patch(facecolor=self.LABEL_COLORS[2][:3], label='Non-enhancing tumor'),
            Patch(facecolor=self.LABEL_COLORS[3][:3], label='Enhancing tumor')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                  fontsize=14, frameon=True)

        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")

        return fig


def main():
    """Main execution."""
    data_dir = './support/Visualize-3D-MRI-Scans-Brain-case/data'
    output_dir = Path('./brats_visualizations')
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("BRATS Dataset Cross-Sample Comparison")
    print("="*70)

    comparator = BRATSComparison(data_dir)
    sample_names = ['BRATS_001', 'BRATS_002', 'BRATS_003']

    # 1. Compare all samples for T1ce modality
    print("\n1. Creating T1ce comparison across all samples...")
    fig = comparator.compare_all_samples_single_modality(
        sample_names,
        modality_idx=1,
        save_path=output_dir / 'comparison_all_samples_t1ce.png'
    )
    plt.close(fig)

    # 2. Compare all samples for T2 modality
    print("\n2. Creating T2 comparison across all samples...")
    fig = comparator.compare_all_samples_single_modality(
        sample_names,
        modality_idx=2,
        save_path=output_dir / 'comparison_all_samples_t2.png'
    )
    plt.close(fig)

    # 3. Complete comparison (all modalities, all samples)
    print("\n3. Creating complete comparison grid...")
    fig = comparator.compare_all_modalities_all_samples(
        sample_names,
        save_path=output_dir / 'comparison_complete_grid.png'
    )
    plt.close(fig)

    print("\n" + "="*70)
    print("Comparison visualizations complete!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
