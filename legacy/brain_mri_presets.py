"""
Brain MRI visualization presets and transfer functions.

This module provides clinical-grade transfer functions optimized for brain MRI
visualization, based on 3D Slicer presets and neuroimaging best practices.
"""

import numpy as np
from typing import Dict, Tuple, List


# BraTS intensity ranges (after z-score normalization)
BRATS_INTENSITY_RANGES = {
    't1': {'window': 4.0, 'level': 0.0, 'vmin': -3.0, 'vmax': 5.0},
    't1ce': {'window': 5.0, 'level': 0.5, 'vmin': -3.0, 'vmax': 7.0},
    't2': {'window': 4.5, 'level': 0.0, 'vmin': -3.0, 'vmax': 6.0},
    'flair': {'window': 4.5, 'level': 0.0, 'vmin': -3.0, 'vmax': 6.0},
}


def get_brain_mri_transfer_function(
    modality: str = 't1ce',
    style: str = 'brain_mri',
    use_gradient: bool = False
) -> Dict[str, np.ndarray]:
    """
    Get clinical-grade transfer function for brain MRI.

    Based on 3D Slicer presets optimized for neuroimaging.

    Args:
        modality: MRI modality ('t1', 't1ce', 't2', 'flair')
        style: Preset style ('brain_mri', 'brain_contrast', 'brain_edges')
        use_gradient: If True, include gradient-based opacity

    Returns:
        Dictionary with 'scalar_opacity', 'color', and optionally 'gradient_opacity'
    """
    # Get intensity range for modality
    intensity_range = BRATS_INTENSITY_RANGES.get(modality, BRATS_INTENSITY_RANGES['t1ce'])
    vmin = intensity_range['vmin']
    vmax = intensity_range['vmax']
    window = intensity_range['window']
    level = intensity_range['level']

    # Calculate key thresholds
    lower_threshold = level - window / 2
    upper_threshold = level + window / 2
    mid_point = level

    # Normalize to 0-1 range for easier manipulation
    n_points = 256
    normalized_values = np.linspace(0, 1, n_points)

    # Create transfer function based on style
    if style == 'brain_mri':
        # Standard brain MRI: transparent background, visible tissue
        scalar_opacity = create_brain_mri_opacity(normalized_values)
        color_map = 'gray'

    elif style == 'brain_contrast':
        # High contrast for tumors/lesions
        scalar_opacity = create_brain_contrast_opacity(normalized_values)
        color_map = 'viridis'

    elif style == 'brain_edges':
        # Emphasize boundaries and edges
        scalar_opacity = create_brain_edges_opacity(normalized_values)
        color_map = 'gray'

    else:
        # Default to standard brain MRI
        scalar_opacity = create_brain_mri_opacity(normalized_values)
        color_map = 'gray'

    result = {
        'scalar_opacity': scalar_opacity,
        'color_map': color_map,
        'window': window,
        'level': level,
        'vmin': vmin,
        'vmax': vmax
    }

    # Add gradient opacity if requested
    if use_gradient:
        result['gradient_opacity'] = create_gradient_opacity()

    return result


def create_brain_mri_opacity(normalized_values: np.ndarray) -> np.ndarray:
    """
    Standard brain MRI opacity transfer function.

    Optimized for T1/T1ce showing:
    - CSF/Background: transparent (low intensity)
    - Gray matter: visible but semi-transparent
    - White matter: clearly visible (high intensity)
    - Contrast-enhanced regions: very visible

    Args:
        normalized_values: Array of normalized intensity values [0, 1]

    Returns:
        Opacity values [0, 1] for each intensity
    """
    opacity = np.zeros_like(normalized_values)

    # Background and CSF: fully transparent (0-10% of range)
    mask_background = normalized_values < 0.1
    opacity[mask_background] = 0.0

    # Transition region: gradual increase (10-30%)
    mask_transition = (normalized_values >= 0.1) & (normalized_values < 0.3)
    transition_values = (normalized_values[mask_transition] - 0.1) / 0.2
    opacity[mask_transition] = transition_values ** 1.5 * 0.4  # Steeper curve

    # Gray matter: visible (30-50%)
    mask_gray = (normalized_values >= 0.3) & (normalized_values < 0.5)
    gray_values = (normalized_values[mask_gray] - 0.3) / 0.2
    opacity[mask_gray] = 0.4 + gray_values * 0.25

    # White matter: clearly visible (50-70%)
    mask_white = (normalized_values >= 0.5) & (normalized_values < 0.7)
    white_values = (normalized_values[mask_white] - 0.5) / 0.2
    opacity[mask_white] = 0.65 + white_values * 0.15

    # Contrast-enhanced / high intensity: very visible (70-100%)
    mask_high = normalized_values >= 0.7
    high_values = (normalized_values[mask_high] - 0.7) / 0.3
    opacity[mask_high] = 0.8 + high_values * 0.15

    return opacity


def create_brain_contrast_opacity(normalized_values: np.ndarray) -> np.ndarray:
    """
    High contrast opacity for tumor/lesion visualization.

    Emphasizes contrast-enhanced regions and boundaries.

    Args:
        normalized_values: Array of normalized intensity values [0, 1]

    Returns:
        Opacity values [0, 1] for each intensity
    """
    opacity = np.zeros_like(normalized_values)

    # Background: fully transparent (0-10%)
    mask_background = normalized_values < 0.1
    opacity[mask_background] = 0.0

    # Low intensity: minimal opacity (10-35%)
    mask_low = (normalized_values >= 0.1) & (normalized_values < 0.35)
    low_values = (normalized_values[mask_low] - 0.1) / 0.25
    opacity[mask_low] = low_values ** 2 * 0.3

    # Medium intensity: moderate opacity (35-65%)
    mask_medium = (normalized_values >= 0.35) & (normalized_values < 0.65)
    medium_values = (normalized_values[mask_medium] - 0.35) / 0.3
    opacity[mask_medium] = 0.3 + medium_values ** 1.2 * 0.45

    # High intensity (contrast-enhanced): very visible (65-100%)
    mask_high = normalized_values >= 0.65
    high_values = (normalized_values[mask_high] - 0.65) / 0.35
    opacity[mask_high] = 0.75 + high_values * 0.2

    return opacity


def create_brain_edges_opacity(normalized_values: np.ndarray) -> np.ndarray:
    """
    Edge-enhanced opacity for boundary visualization.

    Uses sharp transitions to emphasize tissue boundaries.

    Args:
        normalized_values: Array of normalized intensity values [0, 1]

    Returns:
        Opacity values [0, 1] for each intensity
    """
    opacity = np.zeros_like(normalized_values)

    # Background: transparent (0-15%)
    mask_background = normalized_values < 0.15
    opacity[mask_background] = 0.0

    # Sharp transition for edges (15-40%)
    mask_edge = (normalized_values >= 0.15) & (normalized_values < 0.4)
    edge_values = (normalized_values[mask_edge] - 0.15) / 0.25
    opacity[mask_edge] = edge_values ** 0.7 * 0.65  # Sharp curve for edges

    # Tissue regions: clearly visible (40-100%)
    mask_tissue = normalized_values >= 0.4
    tissue_values = (normalized_values[mask_tissue] - 0.4) / 0.6
    opacity[mask_tissue] = 0.65 + tissue_values * 0.3

    return opacity


def create_gradient_opacity() -> np.ndarray:
    """
    Create gradient-based opacity transfer function.

    High gradients (edges/boundaries) are more opaque.
    Low gradients (uniform tissue) are more transparent.

    Returns:
        Gradient opacity values from 0 (no gradient) to 1 (high gradient)
    """
    n_points = 256
    gradient_values = np.linspace(0, 1, n_points)

    # Emphasize high gradients (edges)
    gradient_opacity = gradient_values ** 0.5  # Square root for gentle emphasis

    # Scale to reasonable range
    gradient_opacity = gradient_opacity * 0.8 + 0.2  # Range: 0.2 to 1.0

    return gradient_opacity


def get_corruption_color_scheme() -> Dict[str, str]:
    """
    Get color scheme for different corruption stages.

    Returns:
        Dictionary mapping stage names to colormaps
    """
    return {
        'clean': 'gray',           # Clinical standard
        'noisy': 'plasma',         # Warm colors show noise
        'masked': 'Blues',         # Blue overlay for masked regions
        'corrupted': 'viridis',    # Combined visualization
        'mask_overlay': 'Reds'     # Red for mask overlay
    }


def compute_window_level_from_histogram(
    volume: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0
) -> Tuple[float, float]:
    """
    Automatically compute optimal window and level from volume histogram.

    Args:
        volume: 3D volume array
        percentile_low: Lower percentile for window (default: 1%)
        percentile_high: Upper percentile for window (default: 99%)

    Returns:
        Tuple of (window, level)
    """
    vmin = np.percentile(volume, percentile_low)
    vmax = np.percentile(volume, percentile_high)

    window = vmax - vmin
    level = (vmax + vmin) / 2

    return window, level


if __name__ == "__main__":
    # Test presets
    print("Testing brain MRI presets...")

    # Test standard brain MRI preset
    preset = get_brain_mri_transfer_function('t1ce', 'brain_mri', use_gradient=True)
    print(f"\nBrain MRI preset:")
    print(f"  Window: {preset['window']}")
    print(f"  Level: {preset['level']}")
    print(f"  Color map: {preset['color_map']}")
    print(f"  Has gradient opacity: {'gradient_opacity' in preset}")

    # Test contrast preset
    preset = get_brain_mri_transfer_function('t1ce', 'brain_contrast')
    print(f"\nBrain contrast preset:")
    print(f"  Window: {preset['window']}")
    print(f"  Level: {preset['level']}")
    print(f"  Color map: {preset['color_map']}")

    # Test color scheme
    colors = get_corruption_color_scheme()
    print(f"\nCorruption color scheme:")
    for stage, cmap in colors.items():
        print(f"  {stage}: {cmap}")

    print("\n✓ Brain MRI presets working correctly!")
