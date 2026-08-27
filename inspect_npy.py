#!/usr/bin/env python3
"""
Inspect a .npy file (typically a label dictionary).
Usage: python inspect_npy.py /path/to/file.npy
"""

import sys
import numpy as np


def inspect_npy(filepath):
    """Load and display information about a .npy file."""
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"File: {filepath}")
    print(f"Type of loaded object: {type(data)}")

    # If it's a dictionary (common for label maps)
    if isinstance(data, dict):
        print(f"Number of keys: {len(data)}")
        # Show first few key-value pairs
        print("\nFirst 10 entries (key -> value):")
        for i, (k, v) in enumerate(data.items()):
            if i >= 10:
                break
            print(f"  {k!r} -> {v!r}")
        # Value statistics
        values = list(data.values())
        unique_vals = set(values)
        print(f"\nNumber of unique values: {len(unique_vals)}")
        print(f"Value range: min={min(values)} , max={max(values)}")
        # Check if values are contiguous from 0
        if sorted(unique_vals) == list(range(len(unique_vals))):
            print("Values are contiguous from 0 to N-1 (good for classification).")
        else:
            print("Values are NOT contiguous from 0.")
        # Check if any key corresponds to multiple values
        rev_map = {}
        dup = False
        for k, v in data.items():
            if v in rev_map:
                print(f"Warning: value {v} appears for keys {rev_map[v]} and {k}")
                dup = True
            else:
                rev_map[v] = k
        if not dup:
            print("No duplicate values (one-to-one mapping).")
    # If it's a NumPy array
    elif isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        print(f"First 10 elements: {data.flat[:10]}")
    else:
        print(f"Object content: {data}")

    # Additional: show first few keys in sorted order
    if isinstance(data, dict):
        keys_sample = sorted(data.keys())[:5]
        print(f"\nExample keys (sorted): {keys_sample}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_npy.py <path_to_npy_file>")
        sys.exit(1)
    inspect_npy(sys.argv[1])