"""
Test vectorized CTCF calculation performance.

Compares loop-based vs vectorized implementation.
"""

import numpy as np
import time
import sys
sys.path.insert(0, str(__file__).rsplit('\\', 2)[0])


def test_ctcf_vectorized_speed():
    """Test that vectorized CTCF is significantly faster than loop-based."""
    from cellquant_enterprise.core.quantification.ctcf import (
        calculate_ctcf_vectorized,
        calculate_ctcf_single
    )

    # Create test data
    np.random.seed(42)
    n_cells = 1000
    img_size = 512

    # Create mock masks with n_cells cells
    masks = np.zeros((img_size, img_size), dtype=np.int32)
    cell_size = int(np.sqrt((img_size * img_size) / (n_cells * 4)))

    cell_id = 1
    for y in range(0, img_size - cell_size, cell_size * 2):
        for x in range(0, img_size - cell_size, cell_size * 2):
            if cell_id > n_cells:
                break
            masks[y:y+cell_size, x:x+cell_size] = cell_id
            cell_id += 1

    actual_cells = len(np.unique(masks)) - 1
    print(f"Created {actual_cells} cells in {img_size}x{img_size} image")

    # Create mock marker image
    marker_img = np.random.randint(100, 1000, (img_size, img_size), dtype=np.uint16)
    background = 150.0

    # Time vectorized version
    start = time.time()
    result = calculate_ctcf_vectorized(marker_img, masks, background, marker_name="Test")
    vectorized_time = time.time() - start
    print(f"Vectorized: {vectorized_time:.3f}s for {result.n_cells} cells")

    # Time loop-based version
    start = time.time()
    labels = np.unique(masks)[1:]
    loop_results = []
    for cell_id in labels:
        cell_pixels = marker_img[masks == cell_id]
        area = len(cell_pixels)
        ctcf, integ, mean_int = calculate_ctcf_single(cell_pixels, area, background)
        loop_results.append(ctcf)
    loop_time = time.time() - start
    print(f"Loop-based: {loop_time:.3f}s for {len(labels)} cells")

    # Calculate speedup
    speedup = loop_time / vectorized_time
    print(f"Speedup: {speedup:.1f}x")

    # Verify results match
    loop_ctcf = np.array(loop_results)
    vec_ctcf = result.ctcf

    max_diff = np.max(np.abs(loop_ctcf - vec_ctcf))
    print(f"Max difference: {max_diff:.6f}")

    assert speedup > 5, f"Expected at least 5x speedup, got {speedup:.1f}x"
    assert max_diff < 1e-6, f"Results don't match: max diff = {max_diff}"

    print("\nVectorized CTCF test PASSED!")
    return speedup


if __name__ == "__main__":
    test_ctcf_vectorized_speed()
