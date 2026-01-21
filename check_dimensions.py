#!/usr/bin/env python3
"""
Verify that motion files in data/Comp_v6_KLD01/train have 263 dimensions.
"""

import os
import numpy as np
from glob import glob
from collections import Counter

def check_train_dimensions(base_path="data/Comp_v6_KLD01/train/motions"):
    """
    Check all .npy files in the train/motions directory and report their dimensions.
    """
    
    if not os.path.exists(base_path):
        print(f"❌ Path not found: {base_path}")
        print(f"   Please provide the correct path to data/Comp_v6_KLD01/train/motions")
        return
    
    # Find all .npy files recursively
    motion_files = sorted(glob(os.path.join(base_path, "**", "*.npy"), recursive=True))
    
    if not motion_files:
        print(f"❌ No .npy files found in {base_path}")
        return
    
    print(f"Found {len(motion_files)} motion files in {base_path}\n")
    
    # Track dimensionalities
    dim_counter = Counter()
    shape_examples = {}
    
    # Check each file
    for i, fpath in enumerate(motion_files):
        try:
            data = np.load(fpath)
            shape = data.shape
            
            # Get the feature dimension (should be last axis)
            if data.ndim == 2:
                frames, features = shape
                dim_counter[features] += 1
                
                # Store examples for each dimension
                if features not in shape_examples:
                    shape_examples[features] = (fpath, shape)
                    
            else:
                dim_counter[f"unexpected_ndim_{data.ndim}"] += 1
                if f"unexpected_ndim_{data.ndim}" not in shape_examples:
                    shape_examples[f"unexpected_ndim_{data.ndim}"] = (fpath, shape)
        
        except Exception as e:
            print(f"⚠️  Error loading {fpath}: {e}")
    
    # Report results
    print("=" * 60)
    print("DIMENSION SUMMARY")
    print("=" * 60)
    
    for dim, count in sorted(dim_counter.items()):
        percentage = (count / len(motion_files)) * 100
        status = "✅" if dim == 263 else "❌"
        print(f"{status} {dim}-dimensional: {count} files ({percentage:.1f}%)")
        
        if dim in shape_examples:
            example_path, example_shape = shape_examples[dim]
            rel_path = os.path.relpath(example_path, base_path)
            print(f"   Example: {rel_path} → shape {example_shape}")
    
    print()
    
    # Final verdict
    if dim_counter.get(263, 0) == len(motion_files):
        print("✅ ALL FILES ARE 263-DIMENSIONAL ✅")
    else:
        print("❌ DIMENSION MISMATCH DETECTED ❌")
        non_263_count = sum(count for dim, count in dim_counter.items() if dim != 263)
        print(f"   {non_263_count} files are NOT 263-dimensional")
    
    print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check motion file dimensions")
    parser.add_argument("--path", type=str,
                       default="data/Comp_v6_KLD01/train/motions",
                       help="Path to the motions directory")
    
    args = parser.parse_args()
    check_train_dimensions(args.path)
