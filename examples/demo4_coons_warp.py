# RasterFairy Standalone Demo 4 - Coons Warp
#
# Description:
# This demo shows Coons Warp preprocessing of a point cloud before
# arranging it into a rectangular grid. It is based on
# "examples/legacy/Raster Fairy Demo 2.ipynb" and demonstrates how
# coonswarp.rectifyCloud can be used to improve the distribution of
# points prior to gridding with rasterfairy.transformPointCloud2D.
#
# Dependencies:
# - matplotlib
# - numpy
# - rasterfairy
#
# Running the script:
# Execute this script from the command line using Python 3:
#   python examples/demo4_coons_warp.py
#
# Output:
# The script will generate and save three PNG images in the
# `examples/demo_output/` directory:
# - coons_warp_initial.png: The initial scatter plot of the generated data.
# - coons_warp_warped.png: The scatter plot after Coons Warp preprocessing.
# - coons_warp_gridded.png: The scatter plot after RasterFairy has
#                           arranged the warped data into a grid.
#

import matplotlib.pyplot as plt
import numpy as np
import rasterfairy
from rasterfairy import coonswarp
from rasterfairy import prime # Added based on user feedback
import os

def generate_data():
    """
    Generates a 2D point cloud with a non-optimal distribution and colors.
    This data generation is based on the first cell of
    `examples/legacy/Raster Fairy Demo 2.ipynb`.

    Returns:
        xy (np.array): The 2D coordinates of the points.
        colors (np.array): The colors for each point.
        totalDataPoints (int): The total number of points.
    """
    side = 85
    totalDataPoints = side * side
    np.random.seed(0) # For reproducibility
    xy = np.random.uniform(low=-1, high=+1, size=(totalDataPoints, 2))
    xy = np.cumsum(xy, axis=0)
    xy -= xy.min(axis=0)
    xy /= xy.max(axis=0)
    xy *= (side - 1)

    # Generate colors based on xy coordinates
    colors_raw = np.zeros((totalDataPoints, 3))
    colors_raw[:, :2] += xy
    colors_raw -= np.min(colors_raw)
    colors_raw /= np.max(colors_raw)

    return xy, colors_raw, totalDataPoints

def display_scatter_plot(xy_coords, colors, title="Scatter Plot", filename="scatter_plot.png", marker_size=7.5):
    """
    Creates and saves a scatter plot.
    Handles output directory creation and sets a black background for plots.
    """
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    fig = plt.figure(figsize=(10.0, 10.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('black') # Black background as in the notebook
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.autoscale_view(True, True, True)
    ax.invert_yaxis() # Consistent with original notebook
    ax.scatter(xy_coords[:, 0], xy_coords[:, 1], c=colors, edgecolors='none', marker='s', s=marker_size)
    ax.set_title(title)
    plt.savefig(filename)
    plt.close(fig)
    print(f"Plot saved as {filename}")

def get_rectangular_arrangement(num_points):
    """
    Gets the most square rectangular arrangement for a number of points.
    """
    arrangements = rasterfairy.getRectArrangements(num_points)
    if not arrangements:
        raise ValueError("No rectangular arrangements found for the given number of points.")
    # The first arrangement is typically the most square
    return arrangements[0]

def transform_to_grid(xy_coords, arrangement):
    """
    Transforms 2D point cloud to a target grid arrangement using RasterFairy.
    """
    grid_xy, (width, height) = rasterfairy.transformPointCloud2D(xy_coords, target=arrangement)
    return grid_xy, width, height

def main_block():
    """Main execution logic for the RasterFairy Coons Warp demo."""
    print("Starting RasterFairy standalone demo 4 - Coons Warp...")

    # 1. Generate Data
    print("Step 1: Generating Data...")
    xy, colors, totalDataPoints = generate_data()
    print(f"{totalDataPoints} data points generated.")
    print(f"Shape of initial xy: {xy.shape}")

    # 2. Initial Scatter Plot
    print("\nStep 2: Generating Initial Scatter Plot...")
    display_scatter_plot(xy, colors, title="Initial Point Cloud",
                         filename="examples/demo_output/coons_warp_initial.png", marker_size=7.5)

    # 3. Coons Warp Preprocessing
    print("\nStep 3: Applying Coons Warp Preprocessing...")
    warped_xy = coonswarp.rectifyCloud(xy, perimeterSubdivisionSteps=32, autoPerimeterOffset=True)
    print(f"Shape of warped_xy: {warped_xy.shape}")

    # 4. Warped Scatter Plot
    print("\nStep 4: Generating Warped Scatter Plot...")
    display_scatter_plot(warped_xy, colors, title="Warped Point Cloud",
                         filename="examples/demo_output/coons_warp_warped.png", marker_size=7.5)

    # 5. Transform Warped Cloud to Grid
    print("\nStep 5: Transforming Warped Cloud to Grid...")
    try:
        target_arrangement = get_rectangular_arrangement(totalDataPoints)
        print(f"Selected target arrangement (width, height): {target_arrangement}")
    except ValueError as e:
        print(f"Error getting arrangement: {e}")
        return

    grid_xy, grid_width, grid_height = transform_to_grid(warped_xy, target_arrangement)
    print(f"Warped point cloud transformed to a {grid_width}x{grid_height} grid.")
    print(f"Shape of grid_xy: {grid_xy.shape}")

    # 6. Final Grid Scatter Plot
    print("\nStep 6: Generating Final Grid Scatter Plot...")
    display_scatter_plot(grid_xy, colors, title=f"Gridded Warped Point Cloud ({grid_width}x{grid_height})",
                         filename="examples/demo_output/coons_warp_gridded.png", marker_size=9)

    print("\n...RasterFairy standalone demo 4 - Coons Warp finished.")
    print("Output images saved in examples/demo_output/:")
    print("- coons_warp_initial.png")
    print("- coons_warp_warped.png")
    print("- coons_warp_gridded.png")

if __name__ == "__main__":
    main_block()
