"""
Data loading utilities for MDAE visualization.

This module provides functions to load 3D medical imaging samples from BraTS and
OpenMind datasets for visualization purposes.
"""

import numpy as np
import torch
import json
from pathlib import Path
from typing import Optional, Tuple, List
import os

# Try to import nibabel for loading .nii files
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. Cannot load BraTS .nii files.")

# Try to import h5py and fastmri for loading m4raw data
try:
    import h5py
    import fastmri
    from fastmri.data import transforms as fastmri_transforms
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    print("Warning: h5py or fastmri not available. Cannot load m4raw .h5 files.")


# BraTS dataset configurations
BRATS_CONFIGS = {
    'brats18': {
        'name': 'BraTS18',
        'data_root': '/mnt/hanoverlimitedwus2/scratch/t-jiachentu/nnssl_data/segmentation/nnUNet_raw/Dataset042_BraTS18',
        'test_dir': 'imagesTs',
        'case_prefix': 'Brats18_',
        'modality_suffixes': {
            't1': '_0000',
            't1ce': '_0001',  # T1 contrast-enhanced (best for visualization)
            't2': '_0002',
            'flair': '_0003'
        }
    },
    'bratsped': {
        'name': 'BraTSPED',
        'data_root': '/mnt/hanoverlimitedwus2/scratch/t-jiachentu/nnssl_data/segmentation/nnUNet_raw/Dataset027_BraTSPED',
        'test_dir': 'imagesTs',
        'case_prefix': 'BraTSPED_',
        'modality_suffixes': {
            't1': '_0000',    # T1n (native)
            't1ce': '_0001',  # T1c (contrast-enhanced)
            't2': '_0002',    # T2w
            'flair': '_0003'  # T2-FLAIR
        }
    }
}

# M4Raw dataset configuration
M4RAW_CONFIG = {
    'data_root': '/mnt/hanoverlimitedwus2/scratch/t-jiachentu/datasets/m4raw_dataset',
    'test_dir': 'multicoil_test',
    'train_dir': 'multicoil_train',
    'modalities': ['T1', 'T2', 'FLAIR'],
    'acquisitions': {
        'T1': ['T101', 'T102', 'T103', 'T104', 'T105', 'T106'],
        'T2': ['T201', 'T202', 'T203', 'T204', 'T205', 'T206'],
        'FLAIR': ['FLAIR01', 'FLAIR02', 'FLAIR03', 'FLAIR04']
    }
}


def normalize_m4raw(x):
    """
    Normalize m4raw volume to [0, 1] range.

    Args:
        x: Input array

    Returns:
        Normalized array
    """
    y = np.zeros_like(x)
    if x.ndim == 3:
        # Normalize entire volume
        x_min = x.min()
        x_max = x.max()
        if x_max > x_min:
            y = (x - x_min) / (x_max - x_min)
    else:
        # Normalize each slice separately
        for i in range(y.shape[0]):
            x_min = x[i].min()
            x_max = x[i].max()
            if x_max > x_min:
                y[i] = (x[i] - x_min) / (x_max - x_min)
    return y


def read_m4raw_h5(file_path: str) -> np.ndarray:
    """
    Read m4raw H5 file and convert k-space to image domain.

    Args:
        file_path: Path to H5 file

    Returns:
        Image array [num_slices, H, W] normalized to [0, 1]
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py and fastmri are required to load m4raw data")

    with h5py.File(file_path, 'r') as hf:
        # Load k-space data
        volume_kspace = hf['kspace'][()]

        # Convert to tensor
        slice_kspace = fastmri_transforms.to_tensor(volume_kspace)

        # Inverse FFT to image domain
        slice_image = fastmri.ifft2c(slice_kspace)

        # Compute magnitude and RSS (root sum of squares) across coils
        slice_image_abs = fastmri.complex_abs(slice_image)
        slice_image_rss = fastmri.rss(slice_image_abs, dim=1)

        # Convert to numpy and normalize
        slice_image_rss = np.abs(slice_image_rss.numpy())
        slice_image_rss = normalize_m4raw(slice_image_rss)

    return slice_image_rss


def load_m4raw_sample(
    case_id: Optional[str] = None,
    modality: str = 'T1',
    acquisition: Optional[str] = None,
    split: str = 'test',
    slice_idx: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Load an m4raw MRI sample for visualization.

    Args:
        case_id: Specific case ID (e.g., "2022101101"). If None, uses first available.
        modality: MRI modality ('T1', 'T2', 'FLAIR')
        acquisition: Specific acquisition (e.g., 'T102'). If None, uses first for modality.
        split: Data split ('test' or 'train')
        slice_idx: Which slice to extract (0-17). If None, uses middle slice.
        cache_dir: Optional directory to cache loaded samples

    Returns:
        volume: 2D image tensor [H, W]
        metadata: Dictionary with sample information
    """
    if not H5PY_AVAILABLE:
        raise ImportError("h5py and fastmri are required to load m4raw data. "
                         "Install with: pip install h5py fastmri")

    config = M4RAW_CONFIG
    modality = modality.upper()

    if modality not in config['modalities']:
        raise ValueError(f"Modality must be one of {config['modalities']}, got {modality}")

    # Determine data directory
    data_dir = Path(config['data_root']) / (config['test_dir'] if split == 'test' else config['train_dir'])

    if not data_dir.exists():
        raise FileNotFoundError(f"M4Raw directory not found: {data_dir}")

    # Determine acquisition to use
    if acquisition is None:
        acquisition = config['acquisitions'][modality][0]  # Use first acquisition

    # Find matching files
    if case_id is None:
        # Auto-select first case
        pattern = f"*_{acquisition}.h5"
        matching_files = sorted(list(data_dir.glob(pattern)))
        if not matching_files:
            raise FileNotFoundError(f"No files found matching {pattern} in {data_dir}")
        h5_path = matching_files[0]
        case_id = h5_path.stem.split('_')[0]  # Extract case ID from filename
    else:
        h5_path = data_dir / f"{case_id}_{acquisition}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"File not found: {h5_path}")

    print(f"Loading m4raw sample: {h5_path.name}")

    # Check cache
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"m4raw_{case_id}_{acquisition}.npz"
        if cache_path.exists():
            print(f"  Loading from cache: {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            volume = torch.from_numpy(cached['volume']).float()
            metadata = cached['metadata'].item()
            return volume, metadata

    # Load H5 file
    volume_3d = read_m4raw_h5(str(h5_path))  # Shape: [num_slices, H, W]
    print(f"  Loaded volume shape: {volume_3d.shape}")
    print(f"  Value range: [{volume_3d.min():.3f}, {volume_3d.max():.3f}]")

    # Select slice
    num_slices = volume_3d.shape[0]
    if slice_idx is None:
        slice_idx = num_slices // 2  # Middle slice
    elif slice_idx < 0 or slice_idx >= num_slices:
        raise ValueError(f"slice_idx must be in [0, {num_slices-1}], got {slice_idx}")

    slice_2d = volume_3d[slice_idx]
    print(f"  Selected slice {slice_idx}/{num_slices-1}")

    # Convert to torch tensor
    volume = torch.from_numpy(slice_2d).float()

    # Prepare metadata
    metadata = {
        'case_id': case_id,
        'modality': modality,
        'acquisition': acquisition,
        'split': split,
        'slice_idx': slice_idx,
        'num_slices': num_slices,
        'file_path': str(h5_path),
        'shape': tuple(volume.shape)
    }

    # Cache if requested
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"m4raw_{case_id}_{acquisition}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            volume=volume.numpy(),
            metadata=np.array(metadata, dtype=object)
        )
        print(f"  Saved to cache: {cache_path}")

    return volume, metadata


def load_brats_sample(
    case_id: Optional[str] = None,
    dataset: str = 'brats18',
    modality: str = 't1ce',
    cache_dir: Optional[str] = None,
    target_shape: Optional[Tuple[int, int, int]] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Load a BraTS MRI sample for visualization.

    Args:
        case_id: Specific case ID (e.g., "Brats18_2013_0_1"). If None, uses first available.
        dataset: Dataset name ('brats18' or 'bratsped')
        modality: MRI modality ('t1', 't1ce', 't2', 'flair')
        cache_dir: Optional directory to cache loaded samples
        target_shape: Optional target shape for cropping/padding (D, H, W)

    Returns:
        volume: 3D volume tensor [D, H, W]
        metadata: Dictionary with sample information

    Raises:
        ImportError: If nibabel is not available
        FileNotFoundError: If data files not found
        ValueError: If invalid dataset or modality specified
    """
    if not NIBABEL_AVAILABLE:
        raise ImportError(
            "nibabel is required to load BraTS data. "
            "Install with: pip install nibabel"
        )

    # Validate inputs
    if dataset not in BRATS_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(BRATS_CONFIGS.keys())}")

    config = BRATS_CONFIGS[dataset]

    if modality not in config['modality_suffixes']:
        raise ValueError(
            f"Unknown modality: {modality}. "
            f"Choose from: {list(config['modality_suffixes'].keys())}"
        )

    # Check cache first
    if cache_dir is not None and case_id is not None:
        cache_path = Path(cache_dir) / f"{dataset}_{case_id}_{modality}.npz"
        if cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            volume = torch.from_numpy(data['volume']).float()
            metadata = dict(data['metadata'].item())
            return volume, metadata

    # Build data path
    data_root = Path(config['data_root'])
    test_dir = data_root / config['test_dir']

    if not test_dir.exists():
        raise FileNotFoundError(
            f"BraTS data directory not found: {test_dir}\n"
            f"Please ensure {dataset} data is available."
        )

    # Find case
    if case_id is None:
        # Use first available case
        modality_suffix = config['modality_suffixes'][modality]
        all_files = sorted(test_dir.glob(f"{config['case_prefix']}*{modality_suffix}.nii*"))

        if not all_files:
            raise FileNotFoundError(f"No {dataset} cases found in {test_dir}")

        # Extract case ID from first file
        first_file = all_files[0]
        case_id = first_file.stem.replace(modality_suffix, '').replace('.nii', '')
        print(f"Auto-selected case: {case_id}")

    # Load the volume
    modality_suffix = config['modality_suffixes'][modality]
    file_pattern = f"{case_id}{modality_suffix}.nii*"
    matching_files = list(test_dir.glob(file_pattern))

    if not matching_files:
        raise FileNotFoundError(
            f"Could not find file matching pattern: {test_dir}/{file_pattern}"
        )

    nii_path = matching_files[0]
    print(f"Loading BraTS volume: {nii_path.name}")

    # Load with nibabel
    nii_img = nib.load(str(nii_path))
    volume_data = nii_img.get_fdata()

    # Convert to torch tensor
    volume = torch.from_numpy(volume_data).float()

    # Store original shape before any processing
    original_shape = tuple(volume.shape)
    print(f"  Original shape: {original_shape}")
    print(f"  Value range: [{volume.min():.2f}, {volume.max():.2f}]")

    # Handle different volume formats (some BraTS have 4D with singleton channel)
    if volume.ndim == 4:
        # Take first channel if 4D
        volume = volume[..., 0]  # (D, H, W, 1) -> (D, H, W)
        original_shape = tuple(volume.shape)  # Update after channel removal

    # Ensure 3D
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape: {volume.shape}")

    # Optional: crop/pad to target shape
    if target_shape is not None:
        volume = crop_or_pad_to_shape(volume, target_shape)
        print(f"  Resized to: {volume.shape}")

    # Normalize (z-score normalization as in MDAE)
    # Note: Some BraTS volumes may have background=0, handle carefully
    foreground_mask = volume > volume.quantile(0.01)  # Exclude background
    if foreground_mask.any():
        foreground_voxels = volume[foreground_mask]
        mean_val = foreground_voxels.mean()
        std_val = foreground_voxels.std()
        volume = (volume - mean_val) / (std_val + 1e-8)
    else:
        # Fallback to simple normalization
        volume = (volume - volume.mean()) / (volume.std() + 1e-8)

    print(f"  After normalization: [{volume.min():.2f}, {volume.max():.2f}]")

    # Prepare metadata
    metadata = {
        'case_id': case_id,
        'dataset': dataset,
        'modality': modality,
        'file_path': str(nii_path),
        'original_shape': original_shape,
        'shape': tuple(volume.shape),
        'affine': nii_img.affine.tolist() if hasattr(nii_img, 'affine') else None,
    }

    # Save to cache if specified
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{dataset}_{case_id}_{modality}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            volume=volume.numpy(),
            metadata=np.array(metadata, dtype=object)
        )
        print(f"  Saved to cache: {cache_path}")

    return volume, metadata


def crop_or_pad_to_shape(
    volume: torch.Tensor,
    target_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """
    Crop or pad volume to target shape (center crop/pad).

    Args:
        volume: Input volume [D, H, W]
        target_shape: Target shape (D, H, W)

    Returns:
        resized_volume: Volume with target shape
    """
    D, H, W = volume.shape
    tD, tH, tW = target_shape

    # Calculate crop/pad amounts for each dimension
    def get_crop_or_pad(current_size, target_size):
        if current_size > target_size:
            # Crop
            start = (current_size - target_size) // 2
            end = start + target_size
            return slice(start, end)
        else:
            # Will need padding
            return slice(None)

    # Apply cropping if needed
    d_slice = get_crop_or_pad(D, tD)
    h_slice = get_crop_or_pad(H, tH)
    w_slice = get_crop_or_pad(W, tW)

    volume = volume[d_slice, h_slice, w_slice]

    # Apply padding if needed
    pad_d = max(0, tD - volume.shape[0])
    pad_h = max(0, tH - volume.shape[1])
    pad_w = max(0, tW - volume.shape[2])

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # Pad symmetrically
        padding = (
            pad_w // 2, pad_w - pad_w // 2,  # W
            pad_h // 2, pad_h - pad_h // 2,  # H
            pad_d // 2, pad_d - pad_d // 2,  # D
        )
        volume = torch.nn.functional.pad(volume, padding, mode='constant', value=0)

    return volume


def load_openmind_sample(
    sample_idx: int = 0,
    data_root: str = "/mnt/hanoverlimitedwus2/mri/scratch/t-jiachentu/nnssl_data/nnssl_preprocessed/Dataset745_OpenMind",
    config_name: str = "onemmiso",
    cache_dir: Optional[str] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Load a preprocessed OpenMind sample for visualization.

    Args:
        sample_idx: Index of sample to load
        data_root: Root directory of preprocessed Dataset745_OpenMind
        config_name: Configuration name (e.g., 'onemmiso', 'noresample')
        cache_dir: Optional directory to cache loaded samples

    Returns:
        volume: 3D volume tensor [D, H, W]
        metadata: Dictionary with sample information
    """
    # Try to load from cache first
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"sample_{sample_idx}.npz"
        if cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            data = np.load(cache_path)
            volume = torch.from_numpy(data['volume']).float()
            metadata = dict(data['metadata'].item())
            return volume, metadata

    # Load from pretrain_data JSON
    pretrain_json_path = Path(data_root) / f"pretrain_data__{config_name}.json"

    if not pretrain_json_path.exists():
        raise FileNotFoundError(
            f"Pretrain data JSON not found: {pretrain_json_path}\n"
            f"Please ensure OpenMind data is preprocessed and accessible."
        )

    print(f"Loading pretrain data from: {pretrain_json_path}")
    with open(pretrain_json_path, 'r') as f:
        pretrain_data = json.load(f)

    # Get sample information
    if sample_idx >= len(pretrain_data):
        raise ValueError(f"Sample index {sample_idx} out of range (max: {len(pretrain_data)-1})")

    sample_info = pretrain_data[sample_idx]
    print(f"Sample {sample_idx}: {sample_info}")

    # Extract file path (this will vary based on nnssl data structure)
    # The JSON typically contains 'image' path
    if isinstance(sample_info, dict):
        file_path = sample_info.get('image', sample_info.get('data', None))
    else:
        # If sample_info is just a string path
        file_path = sample_info

    if file_path is None:
        raise ValueError(f"Could not extract file path from sample info: {sample_info}")

    # Load the actual volume file
    full_path = Path(file_path)
    if not full_path.exists():
        # Try relative to data_root
        full_path = Path(data_root) / file_path

    if not full_path.exists():
        raise FileNotFoundError(f"Volume file not found: {full_path}")

    print(f"Loading volume from: {full_path}")

    # Load based on file extension
    if full_path.suffix == '.npz':
        data = np.load(full_path)
        volume = data['data']  # Assuming key is 'data'
    elif full_path.suffix == '.npy':
        volume = np.load(full_path)
    else:
        raise ValueError(f"Unsupported file format: {full_path.suffix}")

    # Convert to torch tensor
    volume = torch.from_numpy(volume).float()

    # Handle different volume formats
    if volume.ndim == 4:  # [C, D, H, W]
        # Take first channel if multiple channels
        volume = volume[0]
    elif volume.ndim != 3:
        raise ValueError(f"Unexpected volume shape: {volume.shape}")

    # Normalize volume (z-score normalization as in MDAE)
    volume = (volume - volume.mean()) / (volume.std() + 1e-8)

    # Prepare metadata
    metadata = {
        'sample_idx': sample_idx,
        'file_path': str(full_path),
        'shape': tuple(volume.shape),
        'data_root': data_root,
        'config_name': config_name,
    }

    # Save to cache if specified
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"sample_{sample_idx}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            volume=volume.numpy(),
            metadata=np.array(metadata, dtype=object)
        )
        print(f"Saved to cache: {cache_path}")

    return volume, metadata


def create_synthetic_brain_volume(
    shape: Tuple[int, int, int] = (128, 128, 128),
    num_structures: int = 5,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Create a synthetic 3D brain-like volume for visualization.

    Useful when real OpenMind data is not accessible.

    Args:
        shape: 3D volume shape (D, H, W)
        num_structures: Number of brain-like structures to add
        seed: Random seed for reproducibility

    Returns:
        volume: Synthetic 3D volume [D, H, W]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    D, H, W = shape

    # Start with Gaussian noise
    volume = torch.randn(D, H, W) * 0.1

    # Add ellipsoidal structures (brain-like)
    center_D, center_H, center_W = D // 2, H // 2, W // 2

    for i in range(num_structures):
        # Random ellipsoid parameters
        radius_D = np.random.randint(D // 8, D // 4)
        radius_H = np.random.randint(H // 8, H // 4)
        radius_W = np.random.randint(W // 8, W // 4)

        # Random center offset
        offset_D = np.random.randint(-D // 6, D // 6)
        offset_H = np.random.randint(-H // 6, H // 6)
        offset_W = np.random.randint(-W // 6, W // 6)

        center = [center_D + offset_D, center_H + offset_H, center_W + offset_W]

        # Create coordinate grids
        dd, hh, ww = torch.meshgrid(
            torch.arange(D), torch.arange(H), torch.arange(W),
            indexing='ij'
        )

        # Compute ellipsoid
        ellipsoid = (
            ((dd - center[0]) / radius_D) ** 2 +
            ((hh - center[1]) / radius_H) ** 2 +
            ((ww - center[2]) / radius_W) ** 2
        )

        # Add structure with smooth edges
        intensity = np.random.uniform(0.5, 2.0)
        volume += intensity * torch.exp(-ellipsoid / 2)

    # Add texture
    volume += torch.randn(D, H, W) * 0.2

    # Normalize
    volume = (volume - volume.mean()) / (volume.std() + 1e-8)

    return volume


def get_center_slice_indices(
    volume_shape: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """
    Get indices for center slices in each dimension.

    Args:
        volume_shape: 3D volume shape (D, H, W)

    Returns:
        center_d: Center index in depth dimension
        center_h: Center index in height dimension
        center_w: Center index in width dimension
    """
    D, H, W = volume_shape
    return D // 2, H // 2, W // 2


def extract_2d_slices(
    volume: torch.Tensor,
    slice_type: str = 'axial'
) -> torch.Tensor:
    """
    Extract 2D slice from 3D volume for visualization.

    Args:
        volume: 3D volume [D, H, W]
        slice_type: Type of slice ('axial', 'coronal', 'sagittal')

    Returns:
        slice_2d: 2D slice [H, W] or [D, W] or [D, H]
    """
    D, H, W = volume.shape
    center_d, center_h, center_w = get_center_slice_indices(volume.shape)

    if slice_type == 'axial':
        # Horizontal slice (most common for brain)
        return volume[center_d, :, :]
    elif slice_type == 'coronal':
        # Frontal slice
        return volume[:, center_h, :]
    elif slice_type == 'sagittal':
        # Side slice
        return volume[:, :, center_w]
    else:
        raise ValueError(f"Unknown slice type: {slice_type}")


if __name__ == "__main__":
    # Test data loading
    print("=" * 70)
    print("Testing Data Loading Utilities")
    print("=" * 70)

    # Test 1: Try to load real BraTS data
    print("\n1. Attempting to load BraTS MRI sample...")
    try:
        volume, metadata = load_brats_sample(
            dataset='brats18',
            modality='t1ce',
            cache_dir='sample_data'
        )
        print(f"   ✓ Successfully loaded BraTS sample!")
        print(f"   Case: {metadata['case_id']}")
        print(f"   Dataset: {metadata['dataset']}")
        print(f"   Modality: {metadata['modality']}")
        print(f"   Shape: {metadata['shape']}")
        print(f"   Value range: [{volume.min():.2f}, {volume.max():.2f}]")

        # Test slice extraction
        print("\n   Extracting axial slice...")
        axial_slice = extract_2d_slices(volume, 'axial')
        print(f"   Axial slice shape: {axial_slice.shape}")

    except Exception as e:
        print(f"   ✗ Could not load BraTS data: {e}")
        print(f"   This is expected if BraTS data is not accessible")

    # Test 2: Try to load OpenMind sample (fallback)
    print("\n2. Attempting to load OpenMind sample...")
    try:
        volume, metadata = load_openmind_sample(sample_idx=0)
        print(f"   ✓ Loaded OpenMind sample")
        print(f"   Shape: {volume.shape}")
        print(f"   Metadata: {metadata}")
    except Exception as e:
        print(f"   ✗ Could not load OpenMind sample: {e}")
        print(f"   (This is expected if data is not accessible)")

    # Test 3: Create synthetic volume (always works)
    print("\n3. Creating synthetic brain volume...")
    synthetic_volume = create_synthetic_brain_volume(shape=(128, 128, 128), seed=42)
    print(f"   ✓ Synthetic volume shape: {synthetic_volume.shape}")
    print(f"   Value range: [{synthetic_volume.min():.2f}, {synthetic_volume.max():.2f}]")

    # Test slice extraction
    axial_slice = extract_2d_slices(synthetic_volume, 'axial')
    print(f"   Axial slice shape: {axial_slice.shape}")

    print("\n" + "=" * 70)
    print("✓ Data loading utilities working correctly!")
    print("=" * 70)
