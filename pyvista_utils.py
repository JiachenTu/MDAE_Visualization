"""
PyVista-based 3D visualization utilities for MDAE.

This module provides high-quality 3D volume rendering capabilities for visualizing
MRI volumes and the dual corruption pipeline using PyVista.
"""

import numpy as np
import torch
import pyvista as pv
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
import warnings
warnings.filterwarnings('ignore')


# Publication-quality rendering settings
PUBLICATION_DPI = 300
WINDOW_SIZE = (1920, 1080)  # 16:9 aspect ratio
LIGHTING_STYLE = 'three lights'


def setup_pyvista_plotter(
    window_size: Tuple[int, int] = WINDOW_SIZE,
    off_screen: bool = True,
    lighting: str = LIGHTING_STYLE,
    shape: Tuple[int, int] = (1, 1)
) -> pv.Plotter:
    """
    Setup PyVista plotter for publication-quality rendering.

    Args:
        window_size: Window size (width, height) in pixels
        off_screen: If True, render off-screen (for saving images)
        lighting: Lighting style ('three lights', 'light kit', etc.)
        shape: Grid shape for subplots (rows, cols)

    Returns:
        plotter: Configured PyVista plotter
    """
    plotter = pv.Plotter(
        window_size=window_size,
        off_screen=off_screen,
        shape=shape,
        border=False
    )

    # Set background color (white for publications)
    plotter.set_background('white')

    # Configure lighting
    plotter.enable_3_lights()

    return plotter


def create_pyvista_grid(volume: np.ndarray) -> pv.ImageData:
    """
    Create PyVista ImageData grid from 3D volume.

    Args:
        volume: 3D numpy array [D, H, W]

    Returns:
        PyVista ImageData grid with volume values
    """
    # Create PyVista grid
    grid = pv.ImageData()
    grid.dimensions = np.array(volume.shape) + 1
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)

    # Add volume data
    grid.cell_data['values'] = volume.flatten(order='F')

    return grid


def render_3d_volume(
    volume: Union[torch.Tensor, np.ndarray],
    plotter: pv.Plotter,
    transparency_level: str = 'medium',
    cmap: str = 'gray',
    show_edges: bool = False,
    clim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    camera_position: Optional[str] = None,
    clip_outliers: bool = False,
    opacity_mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> None:
    """
    Render a 3D volume with transparency using PyVista's built-in opacity functions.

    Args:
        volume: 3D volume tensor [D, H, W]
        plotter: PyVista plotter object
        transparency_level: Transparency level ('low', 'medium', 'high')
        cmap: Colormap name (default: 'gray')
        show_edges: If True, show volume edges
        clim: Color limits (min, max). If None, uses 1st-99th percentiles
        title: Plot title
        camera_position: Camera position ('xy', 'xz', 'yz', 'iso')
        clip_outliers: If True, clip extreme outliers for transparent rendering
        opacity_mask: Optional custom opacity mask [D, H, W] where 0=transparent, 1=opaque
    """
    # Convert to numpy if tensor
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()

    # Determine color limits from volume
    if clim is None:
        if clip_outliers:
            # For volumes with sentinel values, use robust statistics
            # Exclude extreme outliers beyond 3 standard deviations
            mean = np.mean(volume)
            std = np.std(volume)
            clim = (mean - 3*std, mean + 3*std)
        else:
            clim = (np.percentile(volume, 1), np.percentile(volume, 99))

    # Create PyVista grid
    grid = create_pyvista_grid(volume)

    # Apply opacity mask if provided by setting masked regions to NaN
    # PyVista automatically renders NaN values as transparent
    if opacity_mask is not None:
        # Convert opacity mask to numpy
        if isinstance(opacity_mask, torch.Tensor):
            opacity_mask = opacity_mask.cpu().numpy()

        # Create a copy of volume to avoid modifying original
        volume_with_transparency = volume.copy()

        # Set regions where opacity=0 to NaN (will render as transparent)
        volume_with_transparency[opacity_mask < 0.5] = np.nan

        # Create new grid with transparent regions
        grid = create_pyvista_grid(volume_with_transparency)

    # Use PyVista's built-in opacity strings for brain MRI visualization
    opacity_map = {
        'high': 'sigmoid_10',      # Very transparent
        'medium': 'sigmoid_5',     # Medium transparency
        'low': 'sigmoid_3'         # Less transparent
    }
    opacity_str = opacity_map.get(transparency_level, 'sigmoid_5')

    # Add volume to plotter
    # Note: PyVista automatically renders NaN values as transparent
    plotter.add_volume(
        grid,
        scalars='values',
        cmap=cmap,
        opacity=opacity_str,
        clim=clim,
        show_scalar_bar=False
    )

    # Add edges if requested
    if show_edges:
        edges = grid.extract_surface()
        plotter.add_mesh(edges, style='wireframe', color='black', line_width=1)

    # Set camera position
    if camera_position == 'xy':
        plotter.view_xy()
    elif camera_position == 'xz':
        plotter.view_xz()
    elif camera_position == 'yz':
        plotter.view_yz()
    elif camera_position == 'iso':
        plotter.view_isometric()
    else:
        # Default: 3/4 view
        plotter.camera_position = 'iso'

    # Add title
    if title:
        plotter.add_text(
            title,
            position='upper_edge',
            font_size=16,
            color='black',
            font='times'
        )


def render_3d_mask_blocks(
    mask: Union[torch.Tensor, np.ndarray],
    plotter: pv.Plotter,
    patch_size: int = 16,
    visible_color: str = 'lightgreen',
    masked_color: str = 'lightcoral',
    opacity: float = 0.6,
    show_visible: bool = True,
    show_masked: bool = True,
    title: Optional[str] = None,
    camera_position: Optional[str] = None
) -> None:
    """
    Render 3D blocky mask as colored cubes.

    Args:
        mask: Binary mask [D, H, W] where 1=visible, 0=masked
        plotter: PyVista plotter object
        patch_size: Size of cubic patches
        visible_color: Color for visible patches
        masked_color: Color for masked patches
        opacity: Transparency of patches
        show_visible: If True, show visible patches
        show_masked: If True, show masked patches
        title: Plot title
        camera_position: Camera position
    """
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    D, H, W = mask.shape

    # Calculate number of patches
    n_patches_D = D // patch_size
    n_patches_H = H // patch_size
    n_patches_W = W // patch_size

    # Downsample mask to patch level
    mask_patches = np.zeros((n_patches_D, n_patches_H, n_patches_W))
    for i in range(n_patches_D):
        for j in range(n_patches_H):
            for k in range(n_patches_W):
                patch = mask[
                    i*patch_size:(i+1)*patch_size,
                    j*patch_size:(j+1)*patch_size,
                    k*patch_size:(k+1)*patch_size
                ]
                # Patch is visible if majority of voxels are visible
                mask_patches[i, j, k] = (patch.mean() > 0.5)

    # Create cubes for each patch
    for i in range(n_patches_D):
        for j in range(n_patches_H):
            for k in range(n_patches_W):
                is_visible = mask_patches[i, j, k] > 0.5

                # Skip if not showing this type
                if is_visible and not show_visible:
                    continue
                if not is_visible and not show_masked:
                    continue

                # Create cube
                center = (
                    (k + 0.5) * patch_size,
                    (j + 0.5) * patch_size,
                    (i + 0.5) * patch_size
                )
                bounds = (
                    k * patch_size, (k + 1) * patch_size,
                    j * patch_size, (j + 1) * patch_size,
                    i * patch_size, (i + 1) * patch_size
                )

                cube = pv.Cube(bounds=bounds)
                color = visible_color if is_visible else masked_color

                plotter.add_mesh(
                    cube,
                    color=color,
                    opacity=opacity,
                    show_edges=True,
                    edge_color='black',
                    line_width=0.5
                )

    # Set camera position
    if camera_position == 'xy':
        plotter.view_xy()
    elif camera_position == 'xz':
        plotter.view_xz()
    elif camera_position == 'yz':
        plotter.view_yz()
    elif camera_position == 'iso':
        plotter.view_isometric()
    else:
        plotter.camera_position = 'iso'

    # Add title
    if title:
        plotter.add_text(
            title,
            position='upper_edge',
            font_size=16,
            color='black',
            font='times'
        )

    # Add legend
    legend_labels = []
    if show_visible:
        legend_labels.append(['Visible Patches', visible_color])
    if show_masked:
        legend_labels.append(['Masked Patches', masked_color])

    if legend_labels:
        plotter.add_legend(
            legend_labels,
            bcolor='white',
            face='rectangle',
            size=(0.15, 0.15)
        )


def create_side_by_side_comparison(
    volumes: List[Union[torch.Tensor, np.ndarray]],
    titles: List[str],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    patch_size: int = 16,
    transparency_level: str = 'medium',
    window_size: Tuple[int, int] = (2400, 1200),
    save_path: Optional[str] = None,
    opacity_masks: Optional[List[Optional[Union[torch.Tensor, np.ndarray]]]] = None
) -> pv.Plotter:
    """
    Create side-by-side comparison of corruption stages.

    Args:
        volumes: List of 3D volumes to compare
        titles: List of titles for each volume
        mask: Optional binary mask for one panel
        patch_size: Patch size for mask visualization
        transparency_level: Transparency level ('low', 'medium', 'high')
        window_size: Window size
        save_path: Optional path to save image
        opacity_masks: Optional list of opacity masks (one per volume, None for default opacity)

    Returns:
        plotter: PyVista plotter with all volumes
    """
    n_volumes = len(volumes)

    # Determine grid shape (prefer 2x2 for 4 volumes)
    if n_volumes == 4:
        shape = (2, 2)
    elif n_volumes <= 3:
        shape = (1, n_volumes)
    else:
        # For more volumes, use 2 rows
        cols = (n_volumes + 1) // 2
        shape = (2, cols)

    # Create plotter
    plotter = setup_pyvista_plotter(
        window_size=window_size,
        off_screen=True,
        shape=shape
    )

    # Ensure opacity_masks list matches volumes length
    if opacity_masks is None:
        opacity_masks = [None] * n_volumes

    # Pre-compute shared clim for noisy and doubly corrupted volumes to ensure consistent rendering
    shared_clim = {}
    for idx, (volume, title) in enumerate(zip(volumes, titles)):
        if 'noisy volume' in title.lower():
            vol = volume.cpu().numpy() if isinstance(volume, torch.Tensor) else volume
            shared_clim['noisy'] = (np.percentile(vol, 1), np.percentile(vol, 99))
            break

    # Add each volume to its subplot
    for idx, (volume, title) in enumerate(zip(volumes, titles)):
        row = idx // shape[1]
        col = idx % shape[1]
        plotter.subplot(row, col)

        # Special handling for spatial mask with aesthetic colors
        if 'spatial mask' in title.lower():
            # Render binary mask with light blue tint for aesthetics
            render_3d_volume(
                volume,
                plotter,
                transparency_level='low',
                cmap='Blues_r',  # Reversed Blue colormap: light for 0 (masked), dark blue for 1 (visible)
                title=title,
                camera_position='iso',
                clim=(0, 1)  # Force 0-1 range for binary mask
            )
        else:
            # Use consistent transparency levels
            if 'visible regions only' in title.lower() or 'masked regions only' in title.lower():
                trans_level = 'medium'  # Match clean volume transparency for natural background
            else:
                trans_level = transparency_level  # Default 'medium' for all other panels

            # Get opacity mask for this volume (if provided)
            opacity_mask = opacity_masks[idx] if idx < len(opacity_masks) else None

            # Determine clim for this panel - use shared clim for doubly corrupted
            panel_clim = None
            if 'doubly corrupted' in title.lower():
                panel_clim = shared_clim.get('noisy', None)  # Use same clim as noisy volume

            render_3d_volume(
                volume,
                plotter,
                transparency_level=trans_level,
                cmap='gray',
                title=title,
                camera_position='iso',
                clip_outliers=False,  # No outlier clipping for consistent rendering
                opacity_mask=opacity_mask,
                clim=panel_clim
            )

    # Link cameras for synchronized views
    if n_volumes > 1:
        plotter.link_views()

    # Save if requested
    if save_path:
        save_publication_image(plotter, save_path)

    return plotter


def save_publication_image(
    plotter: pv.Plotter,
    save_path: str,
    dpi: int = PUBLICATION_DPI,
    transparent_background: bool = False
) -> None:
    """
    Save high-quality publication image.

    Args:
        plotter: PyVista plotter object
        save_path: Output file path (supports .png, .jpg, .pdf)
        dpi: DPI for output image
        transparent_background: If True, use transparent background
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate pixel dimensions from DPI
    # Assume 8x6 inch figure
    width = int(8 * dpi)
    height = int(6 * dpi)

    # Update window size
    plotter.window_size = (width, height)

    # Save screenshot
    plotter.screenshot(
        str(save_path),
        transparent_background=transparent_background,
        return_img=False
    )

    print(f"Saved: {save_path}")


def render_multiview(
    volume: Union[torch.Tensor, np.ndarray],
    views: List[str] = ['xy', 'xz', 'yz', 'iso'],
    titles: Optional[List[str]] = None,
    transparency_level: str = 'medium',
    window_size: Tuple[int, int] = (2400, 1200),
    save_path: Optional[str] = None
) -> pv.Plotter:
    """
    Render volume from multiple viewing angles.

    Args:
        volume: 3D volume to render
        views: List of view angles
        titles: Optional custom titles
        transparency_level: Transparency level ('low', 'medium', 'high')
        window_size: Window size
        save_path: Optional path to save

    Returns:
        plotter: PyVista plotter
    """
    n_views = len(views)

    # Default titles
    if titles is None:
        view_names = {
            'xy': 'Axial View',
            'xz': 'Coronal View',
            'yz': 'Sagittal View',
            'iso': '3D Isometric View'
        }
        titles = [view_names.get(v, v) for v in views]

    # Create plotter
    if n_views == 4:
        shape = (2, 2)
    else:
        shape = (1, n_views)

    plotter = setup_pyvista_plotter(
        window_size=window_size,
        off_screen=True,
        shape=shape
    )

    # Render each view
    for idx, (view, title) in enumerate(zip(views, titles)):
        row = idx // shape[1]
        col = idx % shape[1]
        plotter.subplot(row, col)

        render_3d_volume(
            volume,
            plotter,
            transparency_level=transparency_level,
            cmap='gray',
            title=title,
            camera_position=view
        )

    # Save if requested
    if save_path:
        save_publication_image(plotter, save_path)

    return plotter


def create_mask_overlay_3d(
    volume: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    patch_size: int = 16,
    transparency_level: str = 'medium',
    window_size: Tuple[int, int] = (1920, 1080),
    save_path: Optional[str] = None
) -> pv.Plotter:
    """
    Create 3D visualization with volume and mask overlay.

    Args:
        volume: 3D volume
        mask: Binary mask
        patch_size: Patch size
        transparency_level: Transparency level ('low', 'medium', 'high')
        window_size: Window size
        save_path: Optional save path

    Returns:
        plotter: PyVista plotter
    """
    plotter = setup_pyvista_plotter(window_size=window_size, off_screen=True)

    # Render volume with transparency
    render_3d_volume(
        volume,
        plotter,
        transparency_level=transparency_level,
        cmap='gray'
    )

    # Overlay mask blocks (only show masked regions)
    render_3d_mask_blocks(
        mask,
        plotter,
        patch_size=patch_size,
        show_visible=False,
        show_masked=True,
        masked_color='red',
        opacity=0.5
    )

    # Set camera
    plotter.camera_position = 'iso'

    # Save if requested
    if save_path:
        save_publication_image(plotter, save_path)

    return plotter


if __name__ == "__main__":
    # Quick test
    print("Testing PyVista utilities...")

    # Create synthetic volume
    volume = torch.randn(64, 64, 64)
    volume = (volume - volume.min()) / (volume.max() - volume.min())

    print("Creating test visualization...")
    plotter = setup_pyvista_plotter(off_screen=True)
    render_3d_volume(volume, plotter, title="Test Volume")
    plotter.screenshot("test_pyvista.png")
    plotter.close()

    print("PyVista utilities working correctly!")
