"""
Plotting utilities for CVPR-quality MDAE visualizations.

This module provides functions to create publication-ready figures showing
the MDAE dual corruption process.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


# CVPR-quality figure aesthetics
CVPR_DPI = 300
CVPR_FIGSIZE_SINGLE = (10, 8)
CVPR_FIGSIZE_WIDE = (20, 6)
CVPR_FONT_SIZE = 12
CVPR_TITLE_SIZE = 14
CVPR_LABEL_SIZE = 11


def setup_cvpr_style():
    """Setup matplotlib style for CVPR-quality figures."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': CVPR_FONT_SIZE,
        'axes.titlesize': CVPR_TITLE_SIZE,
        'axes.labelsize': CVPR_LABEL_SIZE,
        'xtick.labelsize': CVPR_LABEL_SIZE,
        'ytick.labelsize': CVPR_LABEL_SIZE,
        'legend.fontsize': CVPR_LABEL_SIZE,
        'figure.titlesize': CVPR_TITLE_SIZE + 2,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'text.usetex': False,  # Set to True if LaTeX is available
        'figure.dpi': CVPR_DPI,
        'savefig.dpi': CVPR_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def plot_2d_slice(
    slice_2d: np.ndarray,
    ax: plt.Axes,
    title: str = "",
    cmap: str = 'gray',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = False,
    xlabel: str = "",
    ylabel: str = ""
):
    """
    Plot a 2D slice with CVPR aesthetics.

    Args:
        slice_2d: 2D numpy array [H, W]
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap name
        vmin, vmax: Value range for colormap
        show_colorbar: Whether to show colorbar
        xlabel, ylabel: Axis labels
    """
    if isinstance(slice_2d, torch.Tensor):
        slice_2d = slice_2d.cpu().numpy()

    # Compute value range if not provided
    if vmin is None:
        vmin = np.percentile(slice_2d, 1)
    if vmax is None:
        vmax = np.percentile(slice_2d, 99)

    # Plot
    im = ax.imshow(slice_2d, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title and labels
    if title:
        ax.set_title(title, fontweight='bold', pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Add colorbar if requested
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=CVPR_LABEL_SIZE-1)

    return im


def plot_mask_overlay(
    slice_2d: np.ndarray,
    mask_2d: np.ndarray,
    ax: plt.Axes,
    title: str = "",
    mask_color: str = 'blue',
    mask_alpha: float = 0.4,
    show_colorbar: bool = False
):
    """
    Plot 2D slice with mask overlay.

    Args:
        slice_2d: 2D image slice [H, W]
        mask_2d: 2D binary mask [H, W] where 0=masked, 1=visible
        ax: Matplotlib axes
        title: Plot title
        mask_color: Color for masked regions
        mask_alpha: Transparency of mask overlay
        show_colorbar: Whether to show colorbar
    """
    if isinstance(slice_2d, torch.Tensor):
        slice_2d = slice_2d.cpu().numpy()
    if isinstance(mask_2d, torch.Tensor):
        mask_2d = mask_2d.cpu().numpy()

    # Plot base image
    vmin, vmax = np.percentile(slice_2d, [1, 99])
    im = ax.imshow(slice_2d, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')

    # Create mask overlay (highlight MASKED regions, where mask_2d=0)
    masked_regions = (mask_2d == 0).astype(float)

    # Create colored overlay
    cmap_mask = LinearSegmentedColormap.from_list('mask', ['none', mask_color])
    ax.imshow(masked_regions, cmap=cmap_mask, alpha=mask_alpha, aspect='auto')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set title
    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    # Add colorbar if requested
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=CVPR_LABEL_SIZE-1)

    return im


def plot_difference_map(
    slice1: np.ndarray,
    slice2: np.ndarray,
    ax: plt.Axes,
    title: str = "Difference",
    cmap: str = 'RdBu_r',
    show_colorbar: bool = True
):
    """
    Plot difference between two slices.

    Args:
        slice1, slice2: 2D slices to compare
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap (diverging recommended)
        show_colorbar: Whether to show colorbar
    """
    if isinstance(slice1, torch.Tensor):
        slice1 = slice1.cpu().numpy()
    if isinstance(slice2, torch.Tensor):
        slice2 = slice2.cpu().numpy()

    diff = slice1 - slice2

    # Symmetric range around zero
    vmax = np.percentile(np.abs(diff), 99)
    vmin = -vmax

    im = ax.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=CVPR_LABEL_SIZE-1)

    return im


def add_arrow_annotation(
    ax: plt.Axes,
    text: str,
    xy: Tuple[float, float],
    xytext: Tuple[float, float],
    color: str = 'black',
    arrowstyle: str = '->'
):
    """Add arrow annotation to plot."""
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        xycoords='axes fraction',
        textcoords='axes fraction',
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color,
            lw=2,
            connectionstyle='arc3,rad=0.3'
        ),
        fontsize=CVPR_FONT_SIZE,
        ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, alpha=0.8)
    )


def add_equation_text(
    ax: plt.Axes,
    equation: str,
    position: Tuple[float, float] = (0.5, -0.15),
    fontsize: Optional[int] = None,
    color: str = 'black'
):
    """Add equation text below subplot."""
    if fontsize is None:
        fontsize = CVPR_FONT_SIZE

    ax.text(
        position[0], position[1],
        equation,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha='center',
        va='top',
        color=color,
        family='monospace'
    )


def create_dual_corruption_overview(
    clean_volume: torch.Tensor,
    mask_percentages: List[float] = [0.25, 0.50, 0.90],
    timesteps: List[float] = [0.3, 0.5, 0.7],
    patch_size: int = 16,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create overview figure showing dual corruption process.

    This creates a multi-row figure where each row shows:
    1. Original clean volume
    2. Spatial mask
    3. Noisy volume (after diffusion)
    4. Doubly corrupted (masked + noisy)
    5. Reconstruction target (masked regions only)

    Args:
        clean_volume: Clean 3D volume [D, H, W]
        mask_percentages: List of masking percentages to visualize
        timesteps: List of diffusion timesteps for each example
        patch_size: Patch size for blocky masking
        slice_idx: Slice index to visualize (None = center)
        save_path: Path to save figure (None = don't save)

    Returns:
        fig: Matplotlib figure
    """
    from corruption_utils import dual_corruption

    setup_cvpr_style()

    n_examples = len(mask_percentages)
    assert len(timesteps) == n_examples, "Must provide timestep for each mask percentage"

    # Get center slice if not specified
    if slice_idx is None:
        slice_idx = clean_volume.shape[0] // 2

    # Create figure with grid
    fig = plt.figure(figsize=(20, 4 * n_examples + 2))
    gs = gridspec.GridSpec(n_examples, 5, figure=fig, hspace=0.3, wspace=0.15)

    # Column titles
    col_titles = [
        "Clean Volume $x_0$",
        "Spatial Mask $m$",
        "Noisy Volume $x_t$",
        "Doubly Corrupted $\\tilde{x}$",
        "Reconstruction Target"
    ]

    for row_idx, (mask_pct, t) in enumerate(zip(mask_percentages, timesteps)):
        # Apply dual corruption
        result = dual_corruption(
            clean_volume,
            mask_percentage=mask_pct,
            timestep=t,
            patch_size=patch_size
        )

        # Extract slices
        clean_slice = clean_volume[slice_idx].cpu().numpy()
        mask_slice = result['spatial_mask'][slice_idx].cpu().numpy()
        noisy_slice = result['noisy_volume'][slice_idx].cpu().numpy()
        corrupted_slice = result['doubly_corrupted'][slice_idx].cpu().numpy()
        target_slice = clean_slice * (1 - mask_slice)  # Only masked regions

        # Plot each stage
        axes = []

        # 1. Clean volume
        ax = fig.add_subplot(gs[row_idx, 0])
        plot_2d_slice(clean_slice, ax, cmap='gray')
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=CVPR_TITLE_SIZE, fontweight='bold', pad=15)
        ax.set_ylabel(f"p_mask={mask_pct:.0%}, t={t:.1f}", fontsize=CVPR_FONT_SIZE, fontweight='bold')
        axes.append(ax)

        # 2. Spatial mask
        ax = fig.add_subplot(gs[row_idx, 1])
        plot_mask_overlay(clean_slice, mask_slice, ax, mask_color='red', mask_alpha=0.6)
        if row_idx == 0:
            ax.set_title(col_titles[1], fontsize=CVPR_TITLE_SIZE, fontweight='bold', pad=15)
        axes.append(ax)

        # 3. Noisy volume
        ax = fig.add_subplot(gs[row_idx, 2])
        plot_2d_slice(noisy_slice, ax, cmap='viridis')
        if row_idx == 0:
            ax.set_title(col_titles[2], fontsize=CVPR_TITLE_SIZE, fontweight='bold', pad=15)
        axes.append(ax)

        # 4. Doubly corrupted
        ax = fig.add_subplot(gs[row_idx, 3])
        plot_mask_overlay(noisy_slice, mask_slice, ax, mask_color='blue', mask_alpha=0.7)
        if row_idx == 0:
            ax.set_title(col_titles[3], fontsize=CVPR_TITLE_SIZE, fontweight='bold', pad=15)
        axes.append(ax)

        # 5. Reconstruction target
        ax = fig.add_subplot(gs[row_idx, 4])
        plot_2d_slice(target_slice, ax, cmap='gray')
        if row_idx == 0:
            ax.set_title(col_titles[4], fontsize=CVPR_TITLE_SIZE, fontweight='bold', pad=15)
        axes.append(ax)

    # Add main title
    fig.suptitle(
        "MDAE Dual Corruption Strategy: Spatial Masking + Diffusion Noise",
        fontsize=CVPR_TITLE_SIZE + 4,
        fontweight='bold',
        y=0.98
    )

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=CVPR_DPI, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def create_masking_ratio_comparison(
    clean_volume: torch.Tensor,
    mask_percentages: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    patch_size: int = 16,
    slice_idx: Optional[int] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create figure comparing different masking ratios.

    Args:
        clean_volume: Clean 3D volume [D, H, W]
        mask_percentages: List of masking percentages
        patch_size: Patch size for blocky masking
        slice_idx: Slice index to visualize
        save_path: Path to save figure

    Returns:
        fig: Matplotlib figure
    """
    from corruption_utils import dual_corruption

    setup_cvpr_style()

    # Get center slice
    if slice_idx is None:
        slice_idx = clean_volume.shape[0] // 2

    clean_slice = clean_volume[slice_idx].cpu().numpy()

    # Create figure
    n_ratios = len(mask_percentages)
    fig, axes = plt.subplots(2, n_ratios, figsize=(4 * n_ratios, 8))

    for col_idx, mask_pct in enumerate(mask_percentages):
        # Apply dual corruption
        result = dual_corruption(
            clean_volume,
            mask_percentage=mask_pct,
            timestep=0.5,  # Fixed timestep
            patch_size=patch_size
        )

        mask_slice = result['spatial_mask'][slice_idx].cpu().numpy()
        corrupted_slice = result['doubly_corrupted'][slice_idx].cpu().numpy()

        # Top row: Mask overlay
        ax_mask = axes[0, col_idx]
        plot_mask_overlay(clean_slice, mask_slice, ax_mask, mask_color='red', mask_alpha=0.5)
        ax_mask.set_title(f"{mask_pct:.0%} Masked", fontweight='bold')

        # Bottom row: Doubly corrupted
        ax_corrupt = axes[1, col_idx]
        plot_2d_slice(corrupted_slice, ax_corrupt, cmap='gray')

        # Add actual masking percentage
        actual_pct = 1 - mask_slice.mean()
        ax_corrupt.set_xlabel(f"Actual: {actual_pct:.1%}", fontsize=CVPR_LABEL_SIZE)

    # Row labels
    axes[0, 0].set_ylabel("Spatial Mask", fontsize=CVPR_FONT_SIZE + 2, fontweight='bold')
    axes[1, 0].set_ylabel("Doubly Corrupted", fontsize=CVPR_FONT_SIZE + 2, fontweight='bold')

    # Main title
    fig.suptitle(
        "Stochastic Masking Ratio Comparison (16³ Blocky Patches)",
        fontsize=CVPR_TITLE_SIZE + 2,
        fontweight='bold'
    )

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=CVPR_DPI, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


def create_dual_corruption_grid(
    clean_image: torch.Tensor,
    timesteps: List[float] = None,
    mask_ratios: List[float] = None,
    patch_size: int = 16,
    highlight_cell: Tuple[int, int] = None,
    save_path: str = None
):
    """
    Create a grid visualization showing dual corruption across timesteps and masking ratios.

    Args:
        clean_image: Clean 2D image [H, W]
        timesteps: List of diffusion timesteps (columns). Default: [0, 0.005, 0.01, 0.02, 0.5, 1, 2, 3, 5] / 10
        mask_ratios: List of masking ratios (rows). Default: [0, 0.1, 0.25, 0.5, 0.75, 0.9]
        patch_size: Patch size for blocky masking
        highlight_cell: Tuple (row_idx, col_idx) to highlight with orange border. Default: (4, 5)
        save_path: Optional path to save figure

    Returns:
        fig: Matplotlib figure object
    """
    from corruption_utils import corrupt_ddpm, create_2d_blocky_mask

    # Default timesteps (matching notebook)
    if timesteps is None:
        timesteps = np.array([0, 0.005, 0.01, 0.02, 0.5, 1, 2, 3, 5]) / 10

    # Default masking ratios
    if mask_ratios is None:
        mask_ratios = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]

    # Default highlight (middle cell: ~50-75% mask, t~0.5)
    if highlight_cell is None:
        highlight_cell = (4, 4)  # Row 4 (75%), Column 4 (t=0.5)

    n_rows = len(mask_ratios)
    n_cols = len(timesteps)

    print(f"Creating dual corruption grid: {n_rows} rows × {n_cols} columns")

    # Create figure with appropriate size
    fig_width = min(24, 2.5 * n_cols)
    fig_height = min(16, 2.5 * n_rows)
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create grid of corrupted images
    corrupted_grid = []
    for mask_ratio in mask_ratios:
        row_images = []
        for t in timesteps:
            # Apply diffusion corruption
            noisy = corrupt_ddpm(clean_image, t)

            # Apply masking
            if mask_ratio > 0:
                mask = create_2d_blocky_mask(clean_image.shape, patch_size, mask_ratio)
                corrupted = noisy * mask
            else:
                corrupted = noisy

            row_images.append(corrupted.numpy())
        corrupted_grid.append(row_images)

    # Convert to numpy array for easier indexing
    corrupted_grid = np.array(corrupted_grid)  # Shape: [n_rows, n_cols, H, W]

    # Create subplot grid with adequate spacing for labels
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.15, wspace=0.10,
                          left=0.12, right=0.95, top=0.88, bottom=0.12)

    # Plot all images
    for i in range(n_rows):
        for j in range(n_cols):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(corrupted_grid[i, j], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            # Highlight specified cell
            if highlight_cell and (i, j) == highlight_cell:
                rect = plt.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='orange',
                                    facecolor='none', transform=ax.transAxes, clip_on=False)
                ax.add_patch(rect)

    # Add annotations
    # Top: timestep labels
    for j, t in enumerate(timesteps):
        ax = fig.add_subplot(gs[0, j])
        if j == 0:
            label = r'$t=0$'
        elif j == 2:
            label = r'$t_{\mathrm{data}}$'
        elif j == n_cols - 2:
            label = r'$t$'
        elif j == n_cols - 1:
            label = r'$T_{\mathrm{max}}$'
        else:
            label = f'$t={t:.2f}$'

        ax.text(0.5, 1.25, label, fontsize=12, ha='center', va='bottom',
                transform=ax.transAxes)

    # Left: masking ratio labels
    for i, ratio in enumerate(mask_ratios):
        ax = fig.add_subplot(gs[i, 0])
        label = f'{int(ratio*100)}%'
        ax.text(-0.20, 0.5, label, fontsize=12, ha='right', va='center',
                transform=ax.transAxes, rotation=0)

    # Top annotation: "Forward Corruption" arrow
    fig.text(0.5, 0.96, 'Forward Diffusion Corruption →',
             fontsize=16, ha='center', va='bottom',
             bbox=dict(facecolor='lightgrey', edgecolor='none', boxstyle='round,pad=0.4'))

    # Left annotation: "Masking Ratio"
    fig.text(0.02, 0.5, 'Masking Ratio ↓', fontsize=16, ha='center', va='center',
             rotation=90)

    # Bottom: "Data" and "Noise" labels
    ax_data = fig.add_subplot(gs[0, 2])
    ax_data.text(0.5, -0.35, 'Data', fontsize=14, ha='center', va='top',
                 transform=ax_data.transAxes, weight='bold')

    ax_noise = fig.add_subplot(gs[0, -1])
    ax_noise.text(0.5, -0.35, 'Noise', fontsize=14, ha='center', va='top',
                  transform=ax_noise.transAxes, weight='bold')

    # Main title
    fig.suptitle('MDAE Dual Corruption: Diffusion Noise × Spatial Masking',
                 fontsize=18, fontweight='bold', y=0.99)

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=CVPR_DPI, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")

    return fig


if __name__ == "__main__":
    # Test plotting utilities
    print("Testing plotting utilities...")

    setup_cvpr_style()

    # Create synthetic volume
    from data_loader_utils import create_synthetic_brain_volume

    volume = create_synthetic_brain_volume(shape=(128, 128, 128), seed=42)
    print(f"Created synthetic volume: {volume.shape}")

    # Test basic plotting
    print("\nTesting basic 2D slice plotting...")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    slice_2d = volume[64]
    plot_2d_slice(slice_2d, ax, title="Test Slice", cmap='gray', show_colorbar=True)
    plt.close()

    print("✓ Basic plotting works!")

    # Test dual corruption overview
    print("\nTesting dual corruption overview figure...")
    fig = create_dual_corruption_overview(
        volume,
        mask_percentages=[0.25, 0.75],
        timesteps=[0.3, 0.7],
        patch_size=16
    )
    plt.close()

    print("✓ Dual corruption overview works!")

    print("\n✓ All plotting utilities working correctly!")
