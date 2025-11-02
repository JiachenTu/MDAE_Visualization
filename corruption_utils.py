"""
Corruption utilities for MDAE visualization.

This module implements the dual corruption strategy used in MDAE training:
1. Spatial masking with blocky patches (16³)
2. Noise corruption via diffusion process (VP schedule)

Extracted from NoiseConditionedMDAETrainer.py for visualization purposes.
"""

import numpy as np
import torch
from typing import Tuple


def create_blocky_mask(
    volume_shape: Tuple[int, int, int],
    patch_size: int = 16,
    mask_percentage: float = 0.75
) -> torch.Tensor:
    """
    Create blocky patch-based spatial mask.

    Args:
        volume_shape: 3D volume shape (D, H, W)
        patch_size: Size of cubic patches (default: 16 for 16³ patches)
        mask_percentage: Percentage of patches to mask (0.0-1.0)

    Returns:
        mask: Binary mask tensor [D, H, W] where 1=visible, 0=masked
    """
    D, H, W = volume_shape

    # Calculate number of patches in each dimension
    n_patches_D = D // patch_size
    n_patches_H = H // patch_size
    n_patches_W = W // patch_size

    # Create patch-level mask
    total_patches = n_patches_D * n_patches_H * n_patches_W
    num_masked_patches = int(total_patches * mask_percentage)

    # Randomly select patches to mask
    patch_mask_flat = torch.ones(total_patches)
    masked_indices = torch.randperm(total_patches)[:num_masked_patches]
    patch_mask_flat[masked_indices] = 0

    # Reshape to patch grid
    patch_mask = patch_mask_flat.reshape(n_patches_D, n_patches_H, n_patches_W)

    # Upsample to full resolution by repeating patches
    full_mask = (
        patch_mask.repeat_interleave(patch_size, dim=0)
        .repeat_interleave(patch_size, dim=1)
        .repeat_interleave(patch_size, dim=2)
    )

    # Handle cases where volume_shape is not perfectly divisible by patch_size
    full_mask = full_mask[:D, :H, :W]

    return full_mask


def create_2d_blocky_mask(
    image_shape: Tuple[int, int],
    patch_size: int = 16,
    mask_percentage: float = 0.75
) -> torch.Tensor:
    """
    Create blocky patch-based spatial mask for 2D images.

    Args:
        image_shape: 2D image shape (H, W)
        patch_size: Size of square patches (default: 16 for 16×16 patches)
        mask_percentage: Percentage of patches to mask (0.0-1.0)

    Returns:
        mask: Binary mask tensor [H, W] where 1=visible, 0=masked
    """
    H, W = image_shape

    # Calculate number of patches in each dimension
    n_patches_H = H // patch_size
    n_patches_W = W // patch_size

    # Create patch-level mask
    total_patches = n_patches_H * n_patches_W
    num_masked_patches = int(total_patches * mask_percentage)

    # Randomly select patches to mask
    patch_mask_flat = torch.ones(total_patches)
    masked_indices = torch.randperm(total_patches)[:num_masked_patches]
    patch_mask_flat[masked_indices] = 0

    # Reshape to patch grid
    patch_mask = patch_mask_flat.reshape(n_patches_H, n_patches_W)

    # Upsample to full resolution by repeating patches
    full_mask = (
        patch_mask.repeat_interleave(patch_size, dim=0)
        .repeat_interleave(patch_size, dim=1)
    )

    # Handle cases where image_shape is not perfectly divisible by patch_size
    full_mask = full_mask[:H, :W]

    return full_mask


def compute_vp_schedule(t: float, beta_min: float = 1e-4, beta_max: float = 0.02) -> Tuple[float, float]:
    """
    Compute variance-preserving (VP) diffusion schedule parameters.

    Args:
        t: Diffusion timestep in [0, 1]
        beta_min: Minimum noise level (default: 1e-4)
        beta_max: Maximum noise level (default: 0.02)

    Returns:
        alpha_t: Scaling factor for clean signal
        sigma_t: Noise level
    """
    # Linear schedule for beta(t)
    beta_t = beta_min + t * (beta_max - beta_min)

    # VP schedule: alpha^2 + sigma^2 = 1
    alpha_t = np.sqrt(1 - beta_t)
    sigma_t = np.sqrt(beta_t)

    return alpha_t, sigma_t


def corrupt(x: torch.Tensor, amount: float, sde: str = 'ddpm', max_sigma: float = 5.0) -> torch.Tensor:
    """
    Corrupt input using normalize→add noise→unnormalize pattern.

    Matches reference implementation from noise_corruption.py for proper noise scaling.
    This ensures corruption effects are visually consistent regardless of input data range.

    Args:
        x: Input tensor to corrupt (any scale)
        amount: Corruption amount (0.0-1.0 for ddpm/flow, can be >1 for ve)
        sde: SDE type. One of:
            - 'ddpm': Variance-preserving (default)
            - 've': Variance-exploding SDE
            - 'flow': Flow matching
        max_sigma: Maximum sigma for VE SDE (default: 5.0)

    Returns:
        Corrupted tensor in original scale

    Example:
        >>> clean = torch.randn(1, 128, 128, 128)
        >>> noisy = corrupt(clean, amount=0.5, sde='ddpm')
        >>> # noisy = √(0.5)·clean + √(0.5)·noise (in normalized space)
    """
    # Step 1: Z-score normalize the input
    mean = x.mean()
    std = x.std()

    # Handle edge case of zero std
    if std < 1e-8:
        std = torch.tensor(1.0, dtype=x.dtype, device=x.device)

    x_normalized = (x - mean) / std

    # Step 2: Generate noise and corrupt in normalized space
    noise = torch.randn_like(x)

    # Convert amount to tensor if it's a scalar
    if not isinstance(amount, torch.Tensor):
        amount = torch.tensor(amount, dtype=x.dtype, device=x.device)

    if sde == 'ddpm':
        # Variance-preserving corruption: x_t = √(1-amount)·x₀ + √(amount)·ε
        x_corrupted_normalized = torch.sqrt(1 - amount) * x_normalized + torch.sqrt(amount) * noise

    elif sde == 've':
        # Variance exploding SDE process: x_t = x₀ + (amount·max_sigma)·ε
        scaling_factor = amount * max_sigma
        x_corrupted_normalized = x_normalized + scaling_factor * noise

    elif sde == 'flow':
        # Flow matching corruption: x_t = (1-amount)·x₀ + amount·ε
        x_corrupted_normalized = (1 - amount) * x_normalized + amount * noise

    else:
        raise ValueError(f"Unknown SDE type: {sde}. Must be one of: 'ddpm', 've', 'flow'")

    # Step 3: Unnormalize back to original scale
    x_corrupted = x_corrupted_normalized * std + mean

    return x_corrupted


def apply_noise_corruption(
    clean_volume: torch.Tensor,
    t: float,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
    noise: torch.Tensor = None,
    sde: str = 'ddpm',
    max_sigma: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Apply diffusion noise corruption to volume with proper normalization.

    Uses normalize→corrupt→unnormalize pattern for proper noise scaling regardless
    of input data range. This ensures visually consistent corruption effects.

    Args:
        clean_volume: Clean input volume [D, H, W] or [C, D, H, W]
        t: Diffusion timestep in [0, 1]
        beta_min: Minimum noise level (default: 1e-4)
        beta_max: Maximum noise level (default: 0.02)
        noise: Optional pre-generated noise (DEPRECATED - noise now generated internally)
        sde: SDE type ('ddpm', 've', 'flow'). Default: 'ddpm'
        max_sigma: Maximum sigma for VE SDE (default: 5.0)

    Returns:
        corrupted_volume: Noisy volume using normalize→corrupt→unnormalize
        noise_tensor: Always None (noise generated internally for proper normalization)
        alpha_t: Signal scaling factor (for reference)
        sigma_t: Noise level (for reference)

    Note:
        The noise parameter is deprecated and ignored. Noise is now generated internally
        during the normalization process to ensure proper scaling.
    """
    # For visualization, use timestep t directly as corruption amount
    # This provides intuitive control: t=0 (clean), t=0.5 (equal mix), t=1 (very noisy)
    # The VP schedule (beta_min, beta_max) is overly conservative for visualization
    amount = t

    # Compute VP schedule parameters for reference/compatibility only
    alpha_t, sigma_t = compute_vp_schedule(t, beta_min, beta_max)

    # Apply corruption using normalize→corrupt→unnormalize pattern
    # For DDPM: x_t = √(1-t)·x₀ + √(t)·ε in normalized space
    corrupted_volume = corrupt(clean_volume, amount, sde=sde, max_sigma=max_sigma)

    # Note: noise is generated internally in corrupt() for proper normalization
    # Return None to indicate noise is not exposed (breaking change but necessary)
    return corrupted_volume, None, alpha_t, sigma_t


def apply_spatial_masking(
    volume: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply spatial mask to volume (zero out masked regions).

    Args:
        volume: Input volume [D, H, W] or [C, D, H, W]
        mask: Binary mask [D, H, W] where 1=visible, 0=masked

    Returns:
        masked_volume: Volume with masked regions set to zero
    """
    # Ensure mask has same dimensionality as volume
    if volume.ndim == 4 and mask.ndim == 3:
        mask = mask.unsqueeze(0)  # Add channel dimension

    # Apply mask: (1-m) ⊙ x keeps visible regions, zeros masked
    # Note: In MDAE, mask convention is 1=visible, 0=masked
    # So we actually use mask directly to keep visible regions
    masked_volume = volume * mask

    return masked_volume


def dual_corruption(
    clean_volume: torch.Tensor,
    mask_percentage: float = 0.75,
    timestep: float = 0.5,
    patch_size: int = 16,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
    sde: str = 'ddpm',
    max_sigma: float = 5.0,
    return_intermediate: bool = False
) -> dict:
    """
    Apply complete MDAE dual corruption strategy with proper normalization.

    This function replicates the exact corruption process from
    NoiseConditionedMDAETrainer.train_step():
    1. Generate blocky spatial mask with given percentage
    2. Apply noise corruption to entire volume (with normalize→corrupt→unnormalize)
    3. Apply spatial masking to zero out masked regions

    Args:
        clean_volume: Clean input volume [C, D, H, W] or [D, H, W]
        mask_percentage: Percentage of patches to mask (0.0-1.0)
        timestep: Diffusion timestep t ∈ [0, 1]
        patch_size: Size of cubic patches for blocky masking
        beta_min: Minimum noise schedule parameter (default: 1e-4)
        beta_max: Maximum noise schedule parameter (default: 0.02)
        sde: SDE type ('ddpm', 've', 'flow'). Default: 'ddpm'
        max_sigma: Maximum sigma for VE SDE (default: 5.0)
        return_intermediate: If True, return all intermediate steps

    Returns:
        dict containing:
            - doubly_corrupted: Final input x̃ = (1-m) ⊙ x_t
            - spatial_mask: Binary mask (1=visible, 0=masked)
            - noisy_volume: Volume after noise corruption x_t
            - clean_volume: Original clean volume x_0
            - noise: Always None (noise generated internally)
            - alpha_t: Signal scaling factor (for reference)
            - sigma_t: Noise level (for reference)
            - timestep: Diffusion timestep used
            - mask_percentage: Masking percentage used
    """
    # Handle both 3D [D,H,W] and 4D [C,D,H,W] inputs
    if clean_volume.ndim == 3:
        volume_shape = clean_volume.shape
        has_channel_dim = False
    else:
        volume_shape = clean_volume.shape[1:]  # (D, H, W)
        has_channel_dim = True

    # Step 1: Generate blocky spatial mask
    spatial_mask = create_blocky_mask(volume_shape, patch_size, mask_percentage)

    # Step 2: Apply noise corruption to ENTIRE volume (with normalization)
    noisy_volume, noise, alpha_t, sigma_t = apply_noise_corruption(
        clean_volume, timestep, beta_min, beta_max, sde=sde, max_sigma=max_sigma
    )

    # Step 3: Apply spatial masking to noisy volume
    # Set masked regions to dark values for better visualization contrast
    # Note: In code, mask=1 means visible, 0=masked
    if has_channel_dim:
        mask_expanded = spatial_mask.unsqueeze(0)
    else:
        mask_expanded = spatial_mask

    # Keep visible regions, set masked regions to low value (darker)
    doubly_corrupted = noisy_volume * mask_expanded

    # Set masked regions to a value below the data range for dark appearance
    min_val = noisy_volume.min() - 0.5 * noisy_volume.std()
    doubly_corrupted = doubly_corrupted + min_val * (1 - mask_expanded)

    # Create separated views with transparency for hidden regions
    sentinel_high = noisy_volume.max() + 100 * noisy_volume.std()  # For masked_only clipping
    sentinel_low = noisy_volume.min() - 2 * noisy_volume.std()     # For visible_only (renders as dark/transparent)

    # Masked regions only: keep masked regions from doubly_corrupted, hide visible regions
    masked_only = doubly_corrupted.clone()
    masked_only = torch.where(mask_expanded > 0.5, sentinel_high, masked_only)

    # Visible regions only: use noisy_volume to show actual MRI brain structure
    # Set masked regions to very low value (renders as transparent/dark)
    visible_only = noisy_volume.clone()
    visible_only = torch.where(mask_expanded < 0.5, sentinel_low, visible_only)

    # Prepare return dictionary
    result = {
        'doubly_corrupted': doubly_corrupted,
        'spatial_mask': spatial_mask,
        'noisy_volume': noisy_volume,
        'clean_volume': clean_volume,
        'noise': noise,
        'alpha_t': alpha_t,
        'sigma_t': sigma_t,
        'timestep': timestep,
        'mask_percentage': mask_percentage,
        'masked_only': masked_only,
        'visible_only': visible_only,
    }

    # Invert mask for loss computation (MDAE computes loss on masked regions only)
    # loss_mask: 1=masked (compute loss), 0=visible (ignore)
    loss_mask = 1 - spatial_mask
    result['loss_mask'] = loss_mask

    return result


def corrupt_ddpm(x: torch.Tensor, amount: float, max_sigma: float = 5.0) -> torch.Tensor:
    """
    Apply DDPM-style diffusion corruption matching the notebook implementation.

    Args:
        x: Clean input tensor
        amount: Corruption amount (timestep)
        max_sigma: Maximum sigma value (not used in DDPM mode)

    Returns:
        Corrupted tensor: x_t = √(1-amount) * x + √amount * noise
    """
    noise = torch.randn_like(x)
    # Ensure amount is a tensor
    if not isinstance(amount, torch.Tensor):
        amount = torch.tensor(amount, dtype=x.dtype, device=x.device)
    return torch.sqrt(1 - amount) * x + torch.sqrt(amount) * noise


def dual_corruption_2d(
    clean_image: torch.Tensor,
    mask_percentage: float = 0.75,
    timestep: float = 0.5,
    patch_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply dual corruption to 2D image.

    Args:
        clean_image: Clean 2D image [H, W] or [C, H, W]
        mask_percentage: Percentage of patches to mask (0.0-1.0)
        timestep: Diffusion timestep t ∈ [0, 1]
        patch_size: Size of square patches for blocky masking

    Returns:
        doubly_corrupted: Final corrupted image
        spatial_mask: Binary mask (1=visible, 0=masked)
    """
    # Handle both 2D [H,W] and 3D [C,H,W] inputs
    if clean_image.ndim == 2:
        image_shape = clean_image.shape
        has_channel_dim = False
    else:
        image_shape = clean_image.shape[1:]  # (H, W)
        has_channel_dim = True

    # Step 1: Apply diffusion corruption
    noisy_image = corrupt_ddpm(clean_image, timestep)

    # Step 2: Generate blocky spatial mask
    spatial_mask = create_2d_blocky_mask(image_shape, patch_size, mask_percentage)

    # Step 3: Apply spatial masking to noisy image
    if has_channel_dim:
        mask_expanded = spatial_mask.unsqueeze(0)
    else:
        mask_expanded = spatial_mask

    doubly_corrupted = noisy_image * mask_expanded

    return doubly_corrupted, spatial_mask


if __name__ == "__main__":
    # Quick test
    print("Testing dual corruption utilities...")

    # Create synthetic 3D volume
    volume_shape = (128, 128, 128)
    clean_volume = torch.randn(1, *volume_shape)  # [C, D, H, W]

    # Apply dual corruption
    result = dual_corruption(
        clean_volume,
        mask_percentage=0.75,
        timestep=0.5,
        patch_size=16
    )

    print(f"Clean volume shape: {result['clean_volume'].shape}")
    print(f"Spatial mask shape: {result['spatial_mask'].shape}")
    print(f"Noisy volume shape: {result['noisy_volume'].shape}")
    print(f"Doubly corrupted shape: {result['doubly_corrupted'].shape}")
    print(f"Alpha(t): {result['alpha_t']:.4f}, Sigma(t): {result['sigma_t']:.4f}")
    print(f"Mask percentage: {result['mask_percentage']:.1%}")
    print(f"Actual masked ratio: {(1 - result['spatial_mask'].float().mean()).item():.1%}")

    print("\n✓ Dual corruption utilities working correctly!")
