#!/usr/bin/env python3
"""
MDAE Dual Corruption Visualization Script

This script generates publication-quality figures showing the MDAE dual corruption
strategy for CVPR submission using real BraTS brain MRI data.

Usage:
    # Use real BraTS data (default)
    python visualize_dual_corruption.py

    # Use synthetic data instead
    python visualize_dual_corruption.py --use-synthetic

    # Specify dataset and modality
    python visualize_dual_corruption.py --dataset brats18 --modality t1ce

Author: Based on NoiseConditionedMDAETrainer implementation
Date: October 2025
"""

import argparse
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Import our utilities
from data_loader_utils import load_brats_sample, create_synthetic_brain_volume
from corruption_utils import dual_corruption
from plot_utils import (
    setup_cvpr_style,
    create_dual_corruption_overview,
    create_masking_ratio_comparison
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate CVPR-quality visualizations of MDAE dual corruption strategy"
    )
    parser.add_argument(
        '--use-synthetic',
        action='store_true',
        help='Use synthetic data instead of real MRI (default: use real BraTS data)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='brats18',
        choices=['brats18', 'bratsped'],
        help='BraTS dataset to use (default: brats18)'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='t1ce',
        choices=['t1', 't1ce', 't2', 'flair'],
        help='MRI modality to use (default: t1ce - T1 contrast-enhanced)'
    )
    parser.add_argument(
        '--case-id',
        type=str,
        default=None,
        help='Specific case ID to load (default: auto-select first available)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for generated figures (default: outputs)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='sample_data',
        help='Cache directory for loaded samples (default: sample_data)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures (default: 300)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='pdf',
        choices=['pdf', 'png', 'both'],
        help='Output format (default: pdf)'
    )

    args = parser.parse_args()

    # Setup
    print("=" * 70)
    print("MDAE Dual Corruption Visualization")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    setup_cvpr_style()

    # Load volume
    print("\n1. Loading volume...")
    if not args.use_synthetic:
        try:
            volume, metadata = load_brats_sample(
                case_id=args.case_id,
                dataset=args.dataset,
                modality=args.modality,
                cache_dir=str(cache_dir),
                target_shape=(128, 128, 128)  # Resize for consistent visualization
            )
            print(f"   ✓ Loaded BraTS {args.dataset.upper()} sample")
            print(f"   Case ID: {metadata['case_id']}")
            print(f"   Modality: {metadata['modality'].upper()}")
            print(f"   Shape: {volume.shape}")
            print(f"   Original shape: {metadata['original_shape']}")
        except Exception as e:
            print(f"   ✗ Failed to load BraTS data: {e}")
            print(f"   Falling back to synthetic data...")
            args.use_synthetic = True

    if args.use_synthetic:
        volume = create_synthetic_brain_volume(shape=(128, 128, 128), seed=42)
        print(f"   ✓ Created synthetic brain volume")
        print(f"   Shape: {volume.shape}")

    # Generate visualizations
    print("\n2. Generating dual corruption overview figure...")
    fig1 = create_dual_corruption_overview(
        volume,
        mask_percentages=[0.25, 0.50, 0.90],
        timesteps=[0.3, 0.5, 0.7],
        patch_size=16,
        save_path=None  # Will save manually with correct format
    )

    # Save in requested format(s)
    base_name1 = "dual_corruption_overview"
    if args.format in ['pdf', 'both']:
        save_path = output_dir / f"{base_name1}.pdf"
        fig1.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format='pdf')
        print(f"   ✓ Saved: {save_path}")

    if args.format in ['png', 'both']:
        save_path = output_dir / f"{base_name1}.png"
        fig1.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format='png')
        print(f"   ✓ Saved: {save_path}")

    plt.close(fig1)

    # Generate masking ratio comparison
    print("\n3. Generating masking ratio comparison figure...")
    fig2 = create_masking_ratio_comparison(
        volume,
        mask_percentages=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        patch_size=16,
        save_path=None
    )

    base_name2 = "masking_ratio_comparison"
    if args.format in ['pdf', 'both']:
        save_path = output_dir / f"{base_name2}.pdf"
        fig2.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format='pdf')
        print(f"   ✓ Saved: {save_path}")

    if args.format in ['png', 'both']:
        save_path = output_dir / f"{base_name2}.png"
        fig2.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format='png')
        print(f"   ✓ Saved: {save_path}")

    plt.close(fig2)

    # Generate step-by-step process visualization
    print("\n4. Generating step-by-step process figure...")
    fig3 = create_step_by_step_process(
        volume,
        mask_percentage=0.75,
        timestep=0.5,
        patch_size=16
    )

    base_name3 = "step_by_step_process"
    if args.format in ['pdf', 'both']:
        save_path = output_dir / f"{base_name3}.pdf"
        fig3.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format='pdf')
        print(f"   ✓ Saved: {save_path}")

    if args.format in ['png', 'both']:
        save_path = output_dir / f"{base_name3}.png"
        fig3.savefig(save_path, dpi=args.dpi, bbox_inches='tight', format='png')
        print(f"   ✓ Saved: {save_path}")

    plt.close(fig3)

    # Summary
    print("\n" + "=" * 70)
    print("✓ All visualizations generated successfully!")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print(f"Generated files:")
    for f in output_dir.glob(f"*.{args.format if args.format != 'both' else '*'}"):
        print(f"  - {f.name}")

    print("\nThese figures are ready for CVPR submission!")


def create_step_by_step_process(
    clean_volume: torch.Tensor,
    mask_percentage: float = 0.75,
    timestep: float = 0.5,
    patch_size: int = 16
) -> plt.Figure:
    """
    Create detailed step-by-step visualization of corruption process.

    Args:
        clean_volume: Clean 3D volume
        mask_percentage: Masking ratio to use
        timestep: Diffusion timestep
        patch_size: Patch size for blocky masking

    Returns:
        fig: Matplotlib figure
    """
    from plot_utils import plot_2d_slice, plot_mask_overlay, add_equation_text
    import matplotlib.gridspec as gridspec

    # Apply dual corruption
    result = dual_corruption(
        clean_volume,
        mask_percentage=mask_percentage,
        timestep=timestep,
        patch_size=patch_size
    )

    # Get center slice
    slice_idx = clean_volume.shape[0] // 2

    # Extract all components
    clean_slice = clean_volume[slice_idx].cpu().numpy()
    mask_slice = result['spatial_mask'][slice_idx].cpu().numpy()
    noisy_slice = result['noisy_volume'][slice_idx].cpu().numpy()
    corrupted_slice = result['doubly_corrupted'][slice_idx].cpu().numpy()
    noise_slice = result['noise'][slice_idx].cpu().numpy()
    loss_mask_slice = result['loss_mask'][slice_idx].cpu().numpy()

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.2)

    # Step 1: Clean volume
    ax1 = fig.add_subplot(gs[0, 0])
    plot_2d_slice(clean_slice, ax1, title="Step 1: Clean Volume", cmap='gray')
    add_equation_text(ax1, "$x_0$ (normalized)", position=(0.5, -0.12))

    # Step 2: Sample masking ratio
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.5, f"$p_{{mask}} = {mask_percentage:.2f}$\n\nSampled from\n$\\mathcal{{U}}(0.01, 0.99)$",
             ha='center', va='center', fontsize=20, transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax2.axis('off')
    ax2.set_title("Step 2: Sample Masking Ratio", fontweight='bold')

    # Step 3: Generate mask
    ax3 = fig.add_subplot(gs[0, 2])
    plot_mask_overlay(clean_slice, mask_slice, ax3, title="Step 3: Generate Blocky Mask",
                      mask_color='red', mask_alpha=0.5)
    add_equation_text(ax3, f"$m$ ({patch_size}³ patches)", position=(0.5, -0.12))

    # Step 4: Sample timestep and noise
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.text(0.5, 0.5, f"$t = {timestep:.2f}$\n$\\epsilon \\sim \\mathcal{{N}}(0, I)$\n\n"
             f"$\\alpha(t) = {result['alpha_t']:.3f}$\n$\\sigma(t) = {result['sigma_t']:.3f}$",
             ha='center', va='center', fontsize=16, transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor='orange', linewidth=2))
    ax4.axis('off')
    ax4.set_title("Step 4: Sample Timestep & Noise", fontweight='bold')

    # Step 5: Apply noise corruption
    ax5 = fig.add_subplot(gs[1, 1])
    plot_2d_slice(noisy_slice, ax5, title="Step 5: Apply Noise Corruption", cmap='viridis')
    add_equation_text(ax5, "$x_t = \\alpha(t) x_0 + \\sigma(t) \\epsilon$", position=(0.5, -0.12))

    # Step 6: Apply spatial masking
    ax6 = fig.add_subplot(gs[1, 2])
    plot_mask_overlay(noisy_slice, mask_slice, ax6, title="Step 6: Apply Spatial Masking",
                      mask_color='blue', mask_alpha=0.6)
    add_equation_text(ax6, "$\\tilde{x} = m \\odot x_t$", position=(0.5, -0.12))

    # Step 7: Loss computation regions
    ax7 = fig.add_subplot(gs[2, 0])
    plot_mask_overlay(clean_slice, mask_slice, ax7, title="Reconstruction Target",
                      mask_color='green', mask_alpha=0.5)
    add_equation_text(ax7, "$x_0$ (masked regions only)", position=(0.5, -0.12))

    # Step 8: Network prediction (placeholder)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.text(0.5, 0.5, "Network\nReconstruction\n\n$\\hat{x} = g_\\theta(\\tilde{x}, t)$",
             ha='center', va='center', fontsize=18, transform=ax8.transAxes,
             bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax8.axis('off')
    ax8.set_title("Step 7: Network Prediction", fontweight='bold')

    # Step 9: Loss computation
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.text(0.5, 0.5, "Loss Computation\n\n"
             "$\\mathcal{L} = \\frac{w(t)}{|\\Omega|} \\sum_{j \\in \\Omega} ||\\hat{x}_j - x_{0,j}||^2$\n\n"
             f"$|\\Omega| = {loss_mask_slice.sum():.0f}$ voxels",
             ha='center', va='center', fontsize=14, transform=ax9.transAxes,
             bbox=dict(boxstyle='round,pad=1', facecolor='lightcoral', edgecolor='red', linewidth=2))
    ax9.axis('off')
    ax9.set_title("Step 8: Compute Loss", fontweight='bold')

    # Main title
    fig.suptitle(
        "MDAE Dual Corruption: Step-by-Step Process",
        fontsize=18,
        fontweight='bold',
        y=0.98
    )

    return fig


if __name__ == "__main__":
    main()
