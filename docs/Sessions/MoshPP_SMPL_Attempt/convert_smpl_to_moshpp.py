#!/usr/bin/env python3
"""
Convert SMPL model from NPZ to sparse PKL format for MoSh++ compatibility.

This script converts the SMPL model files from .npz format to .pkl format
with scipy sparse matrices for J_regressor, which is required by MoSh++.

Usage:
    python convert_smpl_to_moshpp.py --input data/smpl/smpl/SMPL_NEUTRAL.npz --output data/smpl_mosh_fixed/smpl/neutral/model.pkl
"""

import argparse
import pickle
import numpy as np
from scipy.sparse import coo_matrix
from pathlib import Path


def convert_smpl_npz_to_pkl(input_path: str, output_path: str):
    """Convert SMPL NPZ model to sparse PKL format for MoSh++."""

    print(f"Loading SMPL model from: {input_path}")

    # Load the NPZ model
    model_npz = np.load(input_path)

    print(f"Model keys: {list(model_npz.keys())}")

    # Create model dictionary
    model_dict = {}

    # Copy all keys from NPZ to dict
    for key in model_npz.keys():
        model_dict[key] = model_npz[key]

    # Convert J_regressor to sparse COO format if it's dense
    if "J_regressor" in model_dict:
        J_regressor = model_dict["J_regressor"]
        print(f"J_regressor shape: {J_regressor.shape}")
        print(f"J_regressor type: {type(J_regressor)}")

        if not hasattr(J_regressor, "row"):
            # Convert dense to sparse COO
            print("Converting J_regressor to sparse COO format...")
            J_sparse = coo_matrix(J_regressor)
            model_dict["J_regressor"] = J_sparse
            print(f"Converted to sparse matrix: {J_sparse.shape}, nnz={J_sparse.nnz}")
        else:
            print("J_regressor is already sparse ✓")

    # Ensure bs_style is set to 'lbs' (required by MoSh++)
    if "bs_style" not in model_dict:
        model_dict["bs_style"] = "lbs"
        print("Added bs_style='lbs'")

    # Ensure bs_type is set
    if "bs_type" not in model_dict:
        model_dict["bs_type"] = "lrotmin"
        print("Added bs_type='lrotmin'")

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as pickle with protocol 4 (Python 3.4+ compatible)
    print(f"Saving converted model to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(model_dict, f, protocol=4)

    print(f"✓ Successfully converted SMPL model to: {output_path}")

    # Verify the saved model
    print("\nVerifying saved model...")
    with open(output_path, "rb") as f:
        loaded_model = pickle.load(f)

    print(f"Loaded model keys: {list(loaded_model.keys())}")
    print(f"J_regressor type: {type(loaded_model['J_regressor'])}")
    print(f"posedirs shape: {loaded_model['posedirs'].shape}")
    print(f"kintree_table shape: {loaded_model['kintree_table'].shape}")

    if hasattr(loaded_model["J_regressor"], "row"):
        print("✓ J_regressor is sparse matrix with .row attribute")
    else:
        print("✗ J_regressor is NOT sparse matrix")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert SMPL model from NPZ to sparse PKL format for MoSh++"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/smpl/smpl/SMPL_NEUTRAL.npz",
        help="Path to input SMPL NPZ model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/smpl_mosh_fixed/smpl/neutral/model.pkl",
        help="Path to output SMPL PKL model file",
    )

    args = parser.parse_args()

    success = convert_smpl_npz_to_pkl(args.input, args.output)

    if success:
        print("\n✓ Conversion completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Copy prior files to: {Path(args.output).parent.parent}/")
        print(f"2. Test with MoSh++ using surface_model.type='smpl'")
    else:
        print("\n✗ Conversion failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
