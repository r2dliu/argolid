#!/usr/bin/env python3
"""
Test script for pyramid generation with ThreadPoolExecutor vs ProcessPoolExecutor
"""

import os
import time
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, 'src/python')

from argolid.volume_generator import PyramidGenerator3D
import concurrent.futures
from multiprocessing import get_context

def test_thread_vs_process_pyramid(zarr_path, num_levels=3):
    """
    Test pyramid generation with both ThreadPoolExecutor and ProcessPoolExecutor
    """
    
    # Test with ThreadPoolExecutor (current implementation)
    print("=" * 50)
    print("Testing with ThreadPoolExecutor")
    print("=" * 50)
    
    pyramid_gen = PyramidGenerator3D(zarr_path, base_scale_key=0)
    
    start_time = time.time()
    try:
        pyramid_gen.generate_pyramid(num_levels)
        thread_time = time.time() - start_time
        print(f"✅ ThreadPoolExecutor completed in {thread_time:.2f} seconds")
        
        # Check if files were created and have non-zero size
        for level in range(1, num_levels + 1):
            level_path = Path(zarr_path) / str(level)
            if level_path.exists():
                # Check for some data files
                data_files = list(level_path.rglob("*.zarray"))
                if data_files:
                    print(f"   Level {level}: ✅ Created successfully")
                else:
                    print(f"   Level {level}: ❌ No data files found")
            else:
                print(f"   Level {level}: ❌ Directory not created")
                
    except Exception as e:
        print(f"❌ ThreadPoolExecutor failed: {e}")
        thread_time = None

def test_with_process_executor(zarr_path, num_levels=3):
    """
    Test the old ProcessPoolExecutor approach for comparison
    """
    print("\n" + "=" * 50)
    print("Testing with ProcessPoolExecutor (for comparison)")
    print("=" * 50)
    
    pyramid_gen = PyramidGenerator3D(zarr_path, base_scale_key=0)
    
    # Temporarily modify the method to use ProcessPoolExecutor
    def generate_pyramid_with_processes(self, num_levels: int) -> None:
        self._create_zattr_file(num_levels)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count() // 2, mp_context=get_context("spawn")
        ) as executor:
            executor.map(self.downsample_pyramid, range(1, num_levels+1))
    
    # Monkey patch for testing
    pyramid_gen.generate_pyramid_old = pyramid_gen.generate_pyramid
    pyramid_gen.generate_pyramid = lambda num_levels: generate_pyramid_with_processes(pyramid_gen, num_levels)
    
    start_time = time.time()
    try:
        pyramid_gen.generate_pyramid(num_levels)
        process_time = time.time() - start_time
        print(f"✅ ProcessPoolExecutor completed in {process_time:.2f} seconds")
        
        # Check if files were created and have non-zero size
        for level in range(1, num_levels + 1):
            level_path = Path(zarr_path) / str(level)
            if level_path.exists():
                data_files = list(level_path.rglob("*.zarray"))
                if data_files:
                    print(f"   Level {level}: ✅ Created successfully")
                else:
                    print(f"   Level {level}: ❌ No data files found")
            else:
                print(f"   Level {level}: ❌ Directory not created")
                
    except Exception as e:
        print(f"❌ ProcessPoolExecutor failed: {e}")
        process_time = None

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_pyramid_generation.py <path_to_zarr_directory>")
        print("Example: python test_pyramid_generation.py /path/to/your/zarr/array")
        sys.exit(1)
    
    zarr_path = sys.argv[1]
    
    if not os.path.exists(zarr_path):
        print(f"Error: Zarr directory '{zarr_path}' does not exist")
        sys.exit(1)
    
    # Check if base scale exists
    base_scale_path = Path(zarr_path) / "0"
    if not base_scale_path.exists():
        print(f"Error: Base scale directory '{base_scale_path}' does not exist")
        sys.exit(1)
    
    print(f"Testing pyramid generation on: {zarr_path}")
    print(f"CPU count: {os.cpu_count()}")
    print(f"Max workers: {os.cpu_count() // 2}")
    
    # Test with ThreadPoolExecutor
    test_thread_vs_process_pyramid(zarr_path, num_levels=3)
    
    # Optionally test with ProcessPoolExecutor for comparison
    # Uncomment the line below if you want to compare
    # test_with_process_executor(zarr_path, num_levels=3)

if __name__ == "__main__":
    main() 