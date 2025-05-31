"""
RasterFairy Standalone Warp Demo
--------------------------------

This script demonstrates the warping functionality of the RasterFairy library
by transforming a 2D point cloud (generated via t-SNE from random 3D data)
into different shapes using Coons Patch, similar to the
`Raster Fairy Warp Demo.ipynb` notebook.

It generates its own sample data (2000 3D points, then t-SNE to 2D).
The script outputs several PNG images:
1.  The initial t-SNE embedding.
2.  The point cloud warped to a default rectangular shape.
3.  The point cloud warped to a circular target grid.
4.  The source grid used for the circular warp.
5.  The target circular grid.

Dependencies:
- matplotlib
- numpy
- scikit-learn (for TSNE)
- rasterfairy

Running the script:
Execute this script from the command line using Python 3.
You can specify the output file prefix and some warping parameters.

Example:
  python examples/raster_fairy_warp_standalone.py --output_prefix plots/my_warp_demo

Command-line Arguments:
  --output_prefix:                Prefix for the output PNG image files.
                                  If it contains a directory path,
                                  the directory will be created if it doesn't exist.
                                  (default: rasterfairy_warp)
  --perimeter_subdivision_steps:  Perimeter subdivision steps, primarily for
                                  rectifyCloud (rectangular warp) and also available
                                  if modifying getCloudGrid calls.
                                  (default: 4)
  --padding_scale:                Padding scale for rectifyCloud and getCloudGrid.
                                  (default: 1.05)
  --perimeter_offset:             Perimeter offset for getCloudGrid (used in circular warp).
                                  (default: 64)
  --smoothing:                    Smoothing factor for getCloudGrid (used in circular warp).
                                  (default: 0.5)
"""
# RasterFairy Standalone Warp Demo
#
# This script demonstrates the warping functionality of the RasterFairy library
# by transforming a 2D point cloud into different shapes using Coons Patch.
#
# Dependencies:
# - matplotlib
# - numpy
# - scikit-learn (for TSNE)
# - rasterfairy
# - Pillow (PIL) # For image saving from matplotlib if needed directly

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import rasterfairy
from rasterfairy import coonswarp # Specific import for warp functions
import argparse
import os

# Placeholder for PIL/Pillow import for saving plots if matplotlib doesn't handle it directly
# from PIL import Image
# import io

def generate_data(n_points=2000):
    """Generates n_points random 3D points and their 2D t-SNE embedding."""
    print(f"Generating {n_points} random 3D data points...")
    data_points_3d = np.random.uniform(low=0.0, high=1.0, size=(n_points, 3))

    print("Computing 2D t-SNE embedding...")
    # It's good practice to set random_state for TSNE for reproducibility
    tsne_embedding_2d = TSNE(n_components=2, random_state=42, perplexity=30.0, n_iter=300).fit_transform(data_points_3d)
    # Adjusted perplexity and n_iter to common starting values, notebook uses defaults
    # which might vary. random_state=42 is a common choice for reproducibility.

    print("Data generation complete.")
    return data_points_3d, tsne_embedding_2d, n_points

def display_scatter_plot(xy_coords, colors_3d, title="Scatter Plot", filename="scatter_plot.png", marker_size=7.5, face_color='white', invert_y=False):
    """Creates and saves a scatter plot with customizable styles."""
    print(f"Creating scatter plot: {title}, saving to {filename}...")
    fig = plt.figure(figsize=(10.0, 10.0)) # Standard figure size
    ax = fig.add_subplot(1, 1, 1)

    if face_color:
        ax.set_facecolor(face_color)

    # Hide spines and ticks for a cleaner look, similar to notebook
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    ax.autoscale_view(True, True, True)

    if invert_y:
        ax.invert_yaxis()

    ax.scatter(xy_coords[:, 0], xy_coords[:, 1], c=colors_3d, edgecolors='none', marker='s', s=marker_size)
    ax.set_title(title)

    try:
        plt.savefig(filename)
        print(f"Plot saved as {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")

    plt.close(fig) # Close the figure to free memory

def warp_to_rectangle(xy_coords, perimeter_subdivision_steps=4, auto_perimeter_offset=False, padding_scale=1.05):
    """Warps the 2D point cloud towards a rectangular shape using rectifyCloud."""
    print("Warping point cloud to rectangle...")
    print(f"  Perimeter Subdivision Steps: {perimeter_subdivision_steps}")
    print(f"  Auto Perimeter Offset: {auto_perimeter_offset}")
    print(f"  Padding Scale: {padding_scale}")

    warped_xy = coonswarp.rectifyCloud(
        xy_coords,
        perimeterSubdivisionSteps=perimeter_subdivision_steps,
        autoPerimeterOffset=auto_perimeter_offset,
        paddingScale=padding_scale
    )
    print("Rectangular warping complete.")
    return warped_xy

def warp_to_circle(xy_coords, perimeter_subdivision_steps=2, auto_perimeter_offset=False, perimeter_offset=64, padding_scale=1.05, smoothing=0.5):
    """Warps the 2D point cloud towards a circular shape using warpCloud."""
    print("Warping point cloud to circle...")
    print(f"  Source Grid - Perimeter Subdivision Steps: {perimeter_subdivision_steps}")
    print(f"  Source Grid - Auto Perimeter Offset: {auto_perimeter_offset}")
    print(f"  Source Grid - Perimeter Offset: {perimeter_offset}")
    print(f"  Source Grid - Padding Scale: {padding_scale}")
    print(f"  Source Grid - Smoothing: {smoothing}")

    source_grid = coonswarp.getCloudGrid(
        xy_coords,
        perimeterSubdivisionSteps=perimeter_subdivision_steps,
        autoPerimeterOffset=auto_perimeter_offset,
        perimeterOffset=perimeter_offset,
        paddingScale=padding_scale,
        smoothing=smoothing
    )

    print("Generating circular target grid...")
    target_grid_circular = coonswarp.getCircularGrid(xy_coords) # Takes original xy for density matching

    print("Performing warp to circle...")
    warped_xy_circular = coonswarp.warpCloud(xy_coords, source_grid, target_grid_circular)

    print("Circular warping complete.")
    return warped_xy_circular, source_grid, target_grid_circular

def main_block(args):
    """Main execution logic for the RasterFairy Warp demo."""
    print("Starting RasterFairy standalone warp demo...")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Perimeter subdivision steps (shared): {args.perimeter_subdivision_steps}")
    print(f"Padding scale (shared): {args.padding_scale}")
    print(f"Perimeter offset (for circular warp source grid): {args.perimeter_offset}")
    print(f"Smoothing (for circular warp source grid): {args.smoothing}")

    # 1. Generate Data
    data_points_3d, xy_coords, num_data_points = generate_data(n_points=2000)
    print(f"{num_data_points} data points generated.")
    print(f"Shape of 3D data: {data_points_3d.shape}")
    print(f"Shape of t-SNE embedded 2D coordinates: {xy_coords.shape}")

    # 2. Initial Scatter Plot
    initial_plot_filename = f"{args.output_prefix}_initial_tsne.png"
    display_scatter_plot(xy_coords, data_points_3d,
                         title="Initial t-SNE Embedding (Notebook Style)",
                         filename=initial_plot_filename,
                         marker_size=7.5,
                         face_color='black',
                         invert_y=True)

    # 3. Default Rectangular Warp
    warped_rect_xy = warp_to_rectangle(
        xy_coords,
        perimeter_subdivision_steps=args.perimeter_subdivision_steps, # Uses shared arg
        padding_scale=args.padding_scale # Uses shared arg
    )
    rect_warp_plot_filename = f"{args.output_prefix}_warped_rectangle.png"
    display_scatter_plot(warped_rect_xy, data_points_3d,
                         title="Warped to Rectangle (Notebook Style)",
                         filename=rect_warp_plot_filename,
                         marker_size=7.5,
                         face_color='black',
                         invert_y=True)

    # 4. Circular Target Warp
    # Notebook uses perimeterSubdivisionSteps=2 for this getCloudGrid call.
    # We will use the shared args.perimeter_subdivision_steps for now, or a fixed value.
    # Let's use a fixed value of 2 for perimeterSubdivisionSteps for getCloudGrid in circular warp
    # to match the notebook, as the argparse default is 4 for rectifyCloud.

    warped_circ_xy, source_g, target_g_circ = warp_to_circle(
        xy_coords,
        perimeter_subdivision_steps=2, # Fixed to 2 as per notebook for this specific grid
        perimeter_offset=args.perimeter_offset,
        padding_scale=args.padding_scale, # Uses shared arg
        smoothing=args.smoothing
    )
    circ_warp_plot_filename = f"{args.output_prefix}_warped_circle.png"
    display_scatter_plot(warped_circ_xy, data_points_3d,
                         title="Warped to Circle (Notebook Style)",
                         filename=circ_warp_plot_filename,
                         marker_size=7.5,
                         face_color='black',
                         invert_y=True)

    # Plotting source and target grids (optional, simplified version)
    # The notebook plots these with white color and small alpha.
    # We can adapt display_scatter_plot or make a new one if colors are an issue.
    # For now, let's try with display_scatter_plot and a fixed color.
    source_grid_filename = f"{args.output_prefix}_source_grid_for_circle.png"
    display_scatter_plot(source_g, np.array([[1.0,1.0,1.0]] * len(source_g)), # white color
                         title="Source Grid for Circular Warp",
                         filename=source_grid_filename,
                         marker_size=4, face_color='black', invert_y=True)

    target_grid_circ_filename = f"{args.output_prefix}_target_grid_circular.png"
    display_scatter_plot(target_g_circ, np.array([[1.0,1.0,1.0]] * len(target_g_circ)), # white color
                         title="Target Circular Grid",
                         filename=target_grid_circ_filename,
                         marker_size=4, face_color='black', invert_y=True)


    print("...RasterFairy standalone warp demo finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RasterFairy Standalone Warp Demo")
    parser.add_argument('--output_prefix', type=str, default="rasterfairy_warp",
                        help='Prefix for output PNG image files.')
    parser.add_argument('--perimeter_subdivision_steps', type=int, default=4, # Default for rectifyCloud
                        help='Perimeter subdivision steps for rectifyCloud and getCloudGrid.')
    parser.add_argument('--padding_scale', type=float, default=1.05,
                        help='Padding scale for rectifyCloud and getCloudGrid.')
    parser.add_argument('--perimeter_offset', type=int, default=64,
                        help='Perimeter offset for getCloudGrid.')
    parser.add_argument('--smoothing', type=float, default=0.5,
                        help='Smoothing factor for getCloudGrid.')
    # Add more arguments as functionality is implemented

    args = parser.parse_args()

    output_dir_part = os.path.dirname(args.output_prefix)
    if output_dir_part and not os.path.exists(output_dir_part):
        os.makedirs(output_dir_part)
        print(f"Created output directory: {output_dir_part}")

    main_block(args)
