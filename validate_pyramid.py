#!/usr/bin/env python3
"""
Validation script to check if pyramid levels contain valid data
"""

import sys
import numpy as np
import tensorstore as ts
from pathlib import Path

def validate_pyramid_level(zarr_path, level):
    """
    Validate a specific pyramid level by checking for non-zero data
    """
    level_path = Path(zarr_path) / str(level)
    
    if not level_path.exists():
        return False, f"Level {level} directory does not exist"
    
    try:
        # Open the zarr array
        spec = {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(level_path),
            },
        }
        
        zarr_array = ts.open(spec).result()
        shape = zarr_array.shape
        
        # Sample a few chunks to check for non-zero data
        sample_size = min(100, shape[-1], shape[-2])  # Sample 100x100 or smaller
        
        # Check center region
        c_mid = shape[0] // 2 if shape[0] > 1 else 0
        z_mid = shape[1] // 2 if shape[1] > 1 else 0
        y_start = max(0, shape[2] // 2 - sample_size // 2)
        y_end = min(shape[2], y_start + sample_size)
        x_start = max(0, shape[3] // 2 - sample_size // 2)
        x_end = min(shape[3], x_start + sample_size)
        
        sample_data = zarr_array[c_mid, z_mid, y_start:y_end, x_start:x_end].read().result()
        
        # Check statistics
        non_zero_count = np.count_nonzero(sample_data)
        total_count = sample_data.size
        mean_val = np.mean(sample_data)
        max_val = np.max(sample_data)
        min_val = np.min(sample_data)
        
        is_valid = non_zero_count > 0 and max_val > 0
        
        stats = {
            'shape': shape,
            'non_zero_ratio': non_zero_count / total_count,
            'mean': mean_val,
            'max': max_val,
            'min': min_val,
            'sample_size': sample_data.shape
        }
        
        return is_valid, stats
        
    except Exception as e:
        return False, f"Error reading level {level}: {e}"

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_pyramid.py <path_to_zarr_directory>")
        print("Example: python validate_pyramid.py /path/to/your/zarr/array")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    
    if not Path(zarr_path).exists():
        print(f"Error: Zarr directory '{zarr_path}' does not exist")
        sys.exit(1)
    
    print(f"Validating pyramid: {zarr_path}")
    print("=" * 60)
    
    # Find all pyramid levels
    zarr_dir = Path(zarr_path)
    levels = []
    for item in zarr_dir.iterdir():
        if item.is_dir() and item.name.isdigit():
            levels.append(int(item.name))
    
    levels.sort()
    
    if not levels:
        print("❌ No pyramid levels found")
        sys.exit(1)
    
    print(f"Found pyramid levels: {levels}")
    print()
    
    all_valid = True
    
    for level in levels:
        is_valid, result = validate_pyramid_level(zarr_path, level)
        
        if is_valid:
            stats = result
            print(f"✅ Level {level}: VALID")
            print(f"   Shape: {stats['shape']}")
            print(f"   Non-zero ratio: {stats['non_zero_ratio']:.3f}")
            print(f"   Mean: {stats['mean']:.3f}")
            print(f"   Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            print(f"   Sample size: {stats['sample_size']}")
        else:
            print(f"❌ Level {level}: INVALID - {result}")
            all_valid = False
        print()
    
    if all_valid:
        print("🎉 All pyramid levels are valid!")
    else:
        print("⚠️  Some pyramid levels have issues")
        sys.exit(1)

if __name__ == "__main__":
    main() 