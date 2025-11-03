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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import mathtext


# Publication-quality rendering settings
PUBLICATION_DPI = 300
WINDOW_SIZE = (1920, 1080)  # 16:9 aspect ratio
LIGHTING_STYLE = 'three lights'


def render_latex_text(text: str, fontsize: int = 14, dpi: int = 150) -> np.ndarray:
    """
    Render LaTeX text using matplotlib and return as numpy array.

    Args:
        text: LaTeX text string (can include $ symbols)
        fontsize: Font size for rendering
        dpi: DPI for rendering quality

    Returns:
        RGB numpy array of rendered text (no alpha channel needed for logo)
    """
    # Create a matplotlib figure with the text
    fig, ax = plt.subplots(figsize=(3, 0.6), dpi=dpi)
    ax.axis('off')
    # Use matplotlib's native LaTeX rendering
    ax.text(0.5, 0.5, text, fontsize=fontsize, ha='center', va='center',
            transform=ax.transAxes, family='serif')

    # Set figure background to white
    fig.patch.set_facecolor('white')

    # Render to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() for newer matplotlib versions
    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape((height, width, 4))
    plt.close(fig)

    # Convert RGBA to RGB
    rgb = buf[:, :, :3]

    return rgb


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
        # For LaTeX titles, remove dollar signs for display
        # PyVista doesn't have built-in LaTeX rendering, so we show clean text
        clean_title = title.replace('$', '').replace('\\mathbf{', '').replace('}', '').replace('\\', '')
        plotter.add_text(
            clean_title,
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


def create_diffusion_progression_3d(
    clean_volume: Union[torch.Tensor, np.ndarray],
    timesteps: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    transparency_level: str = 'medium',
    window_size: Tuple[int, int] = (4500, 900),
    sde: str = 'ddpm',
    max_sigma: float = 5.0,
    save_path: Optional[str] = None
) -> pv.Plotter:
    """
    Create horizontal progression showing diffusion process from clean to noisy.

    This visualization shows the forward diffusion process without spatial masking,
    demonstrating how a clean volume progressively becomes corrupted with noise.

    Args:
        clean_volume: Clean 3D volume [D, H, W]
        timesteps: List of timesteps to visualize (default: [0.0, 0.25, 0.5, 0.75, 1.0])
        transparency_level: Transparency level ('low', 'medium', 'high')
        window_size: Window size (width, height) for the full visualization
        sde: SDE type ('ddpm', 've', 'flow')
        max_sigma: Maximum sigma for VE SDE
        save_path: Optional path to save image

    Returns:
        plotter: PyVista plotter with all volumes rendered
    """
    from corruption_utils import apply_noise_corruption

    # Convert to tensor if needed
    if isinstance(clean_volume, np.ndarray):
        clean_volume = torch.from_numpy(clean_volume).float()

    n_steps = len(timesteps)

    # Create plotter with horizontal grid
    plotter = setup_pyvista_plotter(
        window_size=window_size,
        off_screen=True,
        shape=(1, n_steps)  # 1 row, n columns
    )

    # Compute consistent color limits across all timesteps for uniform rendering
    # Use clean volume statistics
    clim = (np.percentile(clean_volume.cpu().numpy(), 1),
            np.percentile(clean_volume.cpu().numpy(), 99))

    # Render each timestep
    for idx, t in enumerate(timesteps):
        plotter.subplot(0, idx)

        # Determine volume to render (no titles)
        if idx == 0:
            # First column: clean volume
            corrupted_volume = clean_volume
        else:
            # All other columns: apply noise corruption
            corrupted_volume, _, alpha_t, sigma_t = apply_noise_corruption(
                clean_volume, t, sde=sde, max_sigma=max_sigma
            )

        # No titles for any column
        title = ''

        # Render volume
        render_3d_volume(
            corrupted_volume,
            plotter,
            transparency_level=transparency_level,
            cmap='gray',
            title=title,
            camera_position='iso',
            clim=clim  # Use consistent color limits
        )

    # Link cameras for synchronized views
    if n_steps > 1:
        plotter.link_views()

    # Save if requested
    if save_path:
        save_publication_image(plotter, save_path)

    return plotter


def create_dual_corruption_grid_3d(
    clean_volume: Union[torch.Tensor, np.ndarray],
    diffusion_timesteps: List[float] = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    masking_ratios: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95],
    patch_size: int = 16,
    transparency_level: str = 'medium',
    window_size: Tuple[int, int] = (5400, 5400),
    sde: str = 'ddpm',
    max_sigma: float = 5.0,
    save_path: Optional[str] = None,
    save_individual_center: bool = True,
    center_save_path: Optional[str] = None
) -> pv.Plotter:
    """
    Create L-shaped grid showing both diffusion and masking corruption.

    Top row shows diffusion progression (no masking).
    Left column shows masking progression (no diffusion).
    Center shows large 4×4 dual corrupted volume combining both corruptions.

    Args:
        clean_volume: Clean 3D volume [D, H, W]
        diffusion_timesteps: Timesteps for top row (default: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
        masking_ratios: Masking ratios for left column (default: [0.0, 0.2, 0.4, 0.6, 0.8, 0.95])
        patch_size: Patch size for blocky masking
        transparency_level: Transparency level ('low', 'medium', 'high')
        window_size: Window size (width, height)
        sde: SDE type ('ddpm', 've', 'flow')
        max_sigma: Maximum sigma for VE SDE
        save_path: Optional path to save grid image
        save_individual_center: If True, save center volume as standalone image
        center_save_path: Optional path for standalone center volume (default: auto-generated)

    Returns:
        plotter: PyVista plotter with L-shaped grid
    """
    from corruption_utils import apply_noise_corruption, create_blocky_mask

    # Convert to tensor if needed
    if isinstance(clean_volume, np.ndarray):
        clean_volume = torch.from_numpy(clean_volume).float()

    n_cols = len(diffusion_timesteps)
    n_rows = len(masking_ratios)

    # Create plotter with grid layout
    plotter = setup_pyvista_plotter(
        window_size=window_size,
        off_screen=True,
        shape=(n_rows, n_cols)
    )

    # Compute consistent color limits
    clim = (np.percentile(clean_volume.cpu().numpy(), 1),
            np.percentile(clean_volume.cpu().numpy(), 99))

    # Render top row: diffusion progression (no masking)
    for col_idx, t in enumerate(diffusion_timesteps):
        plotter.subplot(0, col_idx)

        if t == 0.0:
            corrupted_volume = clean_volume
        else:
            corrupted_volume, _, _, _ = apply_noise_corruption(
                clean_volume, t, sde=sde, max_sigma=max_sigma
            )

        render_3d_volume(
            corrupted_volume,
            plotter,
            transparency_level=transparency_level,
            cmap='gray',
            title='',
            camera_position='iso',
            clim=clim
        )

    # Render left column (starting from row 1): masking progression (no diffusion)
    # Use "visible regions only" rendering with opacity masks
    for row_idx in range(1, n_rows):
        plotter.subplot(row_idx, 0)

        mask_ratio = masking_ratios[row_idx]

        if mask_ratio == 0.0:
            # No masking, show clean volume without opacity mask
            opacity_mask = None
        else:
            # Create spatial mask for this masking ratio
            volume_shape = clean_volume.shape if clean_volume.ndim == 3 else clean_volume.shape[1:]
            spatial_mask = create_blocky_mask(volume_shape, patch_size, mask_ratio)

            # Expand mask if needed
            if clean_volume.ndim == 4:
                spatial_mask = spatial_mask.unsqueeze(0)

            # Use mask as opacity: 1=visible (opaque), 0=masked (transparent)
            opacity_mask = spatial_mask.float()

        # Render clean volume with opacity mask (shows visible regions only)
        render_3d_volume(
            clean_volume,
            plotter,
            transparency_level=transparency_level,
            cmap='gray',
            title='',
            camera_position='iso',
            clim=clim,
            opacity_mask=opacity_mask
        )

    # Render dual corrupted volume spanning 4×4 grid centered at position (3.5, 3.5)
    # This shows the combination of 4th row's masking (60%) and 4th column's timestep (0.5)
    # We'll use a temporary plotter to create the volume, then extract and add to custom renderer

    from corruption_utils import dual_corruption
    import vtk

    # Calculate viewport for 4×4 spanning area centered at position (3.5, 3.5)
    # In a 6×6 grid, each cell is 1/6 of width/height
    # Center at (3.5, 3.5) means spanning from cell 1.5 to 5.5 in both dimensions
    # - x: from 1.5/6 = 0.25 to 5.5/6 ≈ 0.9167
    # - y: from (1 - 5.5/6) ≈ 0.0833 to (1 - 1.5/6) = 0.75
    viewport = [0.25, 0.0833, 0.9167, 0.75]  # [xmin, ymin, xmax, ymax]

    # Create a new renderer with this viewport
    renderer = vtk.vtkRenderer()
    renderer.SetViewport(*viewport)
    renderer.SetBackground(1.0, 1.0, 1.0)  # White background

    # Generate dual corrupted volume
    dual_result = dual_corruption(
        clean_volume,
        mask_percentage=masking_ratios[3],  # 0.6 (60% masked)
        timestep=diffusion_timesteps[3],    # t=0.5
        patch_size=patch_size
    )

    dual_corrupted = dual_result['doubly_corrupted']

    # Convert to numpy for PyVista
    if isinstance(dual_corrupted, torch.Tensor):
        dual_corrupted_np = dual_corrupted.cpu().numpy()
    else:
        dual_corrupted_np = dual_corrupted

    # Create PyVista grid
    grid = create_pyvista_grid(dual_corrupted_np)

    # Map opacity level to PyVista string
    opacity_map = {
        'high': 'sigmoid_10',
        'medium': 'sigmoid_5',
        'low': 'sigmoid_3'
    }
    opacity_str = opacity_map.get(transparency_level, 'sigmoid_5')

    # Create a temporary PyVista plotter to generate the volume actor
    temp_plotter = pv.Plotter(off_screen=True)
    temp_plotter.add_volume(
        grid,
        scalars='values',
        cmap='gray',
        opacity=opacity_str,
        clim=clim,
        show_scalar_bar=False
    )

    # Set isometric view for consistency
    temp_plotter.view_isometric()

    # Extract the volume actor from the temporary plotter
    volume_actor = temp_plotter.renderer.actors.values()
    volume_actor = list(volume_actor)[0]  # Get the first (and only) actor

    # Add the volume actor to our custom renderer
    renderer.AddActor(volume_actor)

    # Set camera for this renderer to match the isometric view
    camera = renderer.GetActiveCamera()
    camera.SetPosition(1, 1, 1)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCamera()

    # Add renderer to main plotter's render window
    plotter.ren_win.AddRenderer(renderer)

    # Close temporary plotter
    temp_plotter.close()

    # Save individual center volume as standalone image if requested
    if save_individual_center:
        # Create a standalone plotter for the center dual corrupted volume
        standalone_plotter = setup_pyvista_plotter(
            window_size=(1800, 1800),  # Square aspect ratio for single volume
            off_screen=True,
            shape=(1, 1)
        )

        # Render the dual corrupted volume
        render_3d_volume(
            dual_corrupted,
            standalone_plotter,
            transparency_level=transparency_level,
            cmap='gray',
            title='',  # No title for standalone
            camera_position='iso',
            clim=clim
        )

        # Determine save path for individual center volume
        if center_save_path is None:
            # Auto-generate path from grid save_path or use default
            if save_path:
                from pathlib import Path
                grid_path = Path(save_path)
                center_save_path = str(grid_path.parent / f"{grid_path.stem}_center{grid_path.suffix}")
            else:
                center_save_path = "outputs/dual_corrupted_center_volume.png"

        # Save standalone center volume
        save_publication_image(standalone_plotter, center_save_path)
        standalone_plotter.close()

    # Leave cells (3,3) through (5,5) empty in the regular grid
    # PyVista will show white background for unused subplots

    # Link cameras for synchronized views
    plotter.link_views()

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
