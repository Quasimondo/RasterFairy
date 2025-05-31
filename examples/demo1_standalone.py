# RasterFairy Standalone Demo 1
#
# This script demonstrates the basic functionality of the RasterFairy library
# by transforming a 2D point cloud (generated via t-SNE from random 3D data)
# into a regular rectangular grid.
#
# Dependencies:
# - matplotlib
# - numpy
# - scikit-learn (for TSNE)
# - rasterfairy
# - Pillow (PIL)
#
# Running the script:
# Execute this script from the command line using Python 3:
#   python demo1_standalone.py
#
# Output:
# The script will generate and save two PNG images in the current directory:
# - tsne_embedding.png: The initial scatter plot of the t-SNE embedded data.
# - rasterfairy_grid.png: The scatter plot of the data after RasterFairy
#                         has arranged it into a rectangular grid.
#

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import rasterfairy
import colorsys
# PIL is imported as Pillow in modern Python, check if 'from PIL import Image' works
# If not, the subtask might need to install Pillow or adjust the import.
# For now, let's assume Pillow is available.
from PIL import Image
import io # For saving plots to image files if not directly displaying

def generate_data():
    # generate a set of 4900 random 3D points
    totalDataPoints = 4900
    dataPoints = generate_color_harmonies(totalDataPoints, "tetradic")
    # create a t-sne embedding in 2D
    xy = TSNE().fit_transform(dataPoints)
    return dataPoints, xy, totalDataPoints

def generate_color_harmonies(n_colors, harmony_type="triadic"):
    """Generate colors based on color theory harmonies"""
    
    base_hue = np.random.random()
    colors = []
    
    if harmony_type == "triadic":
        # Colors 120° apart on color wheel
        hues = [(base_hue + i/3) % 1.0 for i in range(3)]
    elif harmony_type == "complementary":
        # Colors 180° apart
        hues = [base_hue, (base_hue + 0.5) % 1.0]
    elif harmony_type == "analogous":
        # Colors close to each other
        hues = [(base_hue + i*0.083) % 1.0 for i in range(-2, 3)]  # ±30°
    elif harmony_type == "tetradic":
        # Rectangle on color wheel
        hues = [(base_hue + i*0.25) % 1.0 for i in range(4)]
    
    # Generate variations of these harmonious hues
    for _ in range(n_colors):
        hue = np.random.choice(hues)
        # Add slight variation
        hue = (hue + np.random.uniform(-0.05, 0.05)) % 1.0
        
        sat = np.random.uniform(0.6, 1.0)
        val = np.random.uniform(0.7, 1.0)
        
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(rgb)
    
    return np.array(colors)

def display_scatter_plot(xy_coords, colors, title="Scatter Plot", filename="scatter_plot.png", marker_size=7.5):
    """Creates and saves a scatter plot."""
    fig = plt.figure(figsize=(10.0, 10.0))
    ax = fig.add_subplot(1, 1, 1)
    # Ensure visibility on standard backgrounds, original notebook used black.
    # ax.set_facecolor('white') # Or remove for default
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.autoscale_view(True, True, True)
    # The original notebook inverted y-axis, keeping it for consistency
    ax.invert_yaxis()
    ax.scatter(xy_coords[:, 0], xy_coords[:, 1], c=colors, edgecolors='none', marker='s', s=marker_size)
    ax.set_title(title)
    plt.savefig(filename)
    # plt.show() # Commented out as we are saving for a standalone script
    plt.close(fig) # Close the figure to free memory
    print(f"Plot saved as {filename}")


def get_rectangular_arrangement(num_points):
    """Gets the most square rectangular arrangement for a number of points."""
    arrangements = rasterfairy.getRectArrangements(num_points)
    print(f"{len(arrangements)} possible arrangements. {arrangements}")
    if not arrangements:
        raise ValueError("No rectangular arrangements found for the given number of points.")
    # The first arrangement is the most square
    return arrangements[0]
    
def transform_to_grid(xy_coords, arrangement):
    """Transforms 2D point cloud to a grid arrangement."""
    grid_xy, (width, height) = rasterfairy.transformPointCloud2D(xy_coords, target=arrangement)
    return grid_xy, width, height

# Placeholder for where this function will be called
# display_scatter_plot(grid_xy, dataPoints, title="RasterFairy Grid", filename="rasterfairy_grid.png", marker_size=9)

def main_block():
    """Main execution logic for the RasterFairy demo."""
    print("Starting RasterFairy standalone demo 1...")

    # 1. Generate Data
    dataPoints, xy, totalDataPoints = generate_data()
    print(f"{totalDataPoints} data points generated.")
    print(f"Shape of t-SNE embedded xy: {xy.shape}")

    # 2. Initial Scatter Plot
    display_scatter_plot(xy, dataPoints, title="Initial t-SNE Embedding", filename="tsne_embedding.png", marker_size=7.5)

    # 3. Get Rectangular Arrangement
    try:
        target_arrangement = get_rectangular_arrangement(totalDataPoints)
        print(f"Selected target arrangement (width, height): {target_arrangement}")
    except ValueError as e:
        print(f"Error: {e}")
        return # Exit if no arrangement found

    # 4. Transform Point Cloud to Grid
    grid_xy, grid_width, grid_height = transform_to_grid(xy, target_arrangement)
    print(f"Point cloud transformed to a {grid_width}x{grid_height} grid.")
    print(f"Shape of grid_xy: {grid_xy.shape}")

    # 5. Grid Scatter Plot
    display_scatter_plot(grid_xy, dataPoints, title=f"RasterFairy Grid ({grid_width}x{grid_height})", filename="rasterfairy_grid.png", marker_size=9)


    # 6. Get Circular Arrangement
    try:
        bestr,bestrp,bestc = rasterfairy.getBestCircularMatch(totalDataPoints)
        circular_arrangement = rasterfairy.getCircularArrangement(bestr,bestrp)
        print(f"Circular arrangement {circular_arrangement}")
        target_arrangement = rasterfairy.arrangementToRasterMask(circular_arrangement)
        print(f"Target arrangement {target_arrangement}")
    except ValueError as e:
        print(f"Error: {e}")
        return # Exit if no arrangement found

    # 7. Transform Point Cloud to Grid
    grid_xy, grid_width, grid_height = transform_to_grid(xy, target_arrangement)
    print(f"Point cloud transformed to a {grid_width}x{grid_height} grid.")
    print(f"Shape of grid_xy: {grid_xy.shape}")

    # 8. Grid Scatter Plot
    display_scatter_plot(grid_xy, dataPoints, title=f"RasterFairy Grid ({grid_width}x{grid_height})", filename="rasterfairy_circular_grid.png", marker_size=9)


    # 9. Get Triangular Arrangement
    try:
        #Note that this returns a list even though it is always a single item:
        triangular_arrangement = rasterfairy.getTriangularArrangement(totalDataPoints)[0]
        print(f"Triangular arrangement {triangular_arrangement}")
        target_arrangement = rasterfairy.arrangementToRasterMask(triangular_arrangement)
        print(f"Target arrangement {target_arrangement}")
    except ValueError as e:
        print(f"Error: {e}")
        return # Exit if no arrangement found

    # 10. Transform Point Cloud to Grid
    grid_xy, grid_width, grid_height = transform_to_grid(xy, target_arrangement)
    print(f"Point cloud transformed to a {grid_width}x{grid_height} grid.")
    print(f"Shape of grid_xy: {grid_xy.shape}")

    # 11. Grid Scatter Plot
    display_scatter_plot(grid_xy, dataPoints, title=f"RasterFairy Grid ({grid_width}x{grid_height})", filename="rasterfairy_triangular_grid.png", marker_size=9)




    print("...RasterFairy standalone demo 1 finished.")
    print("Output images: tsne_embedding.png, rasterfairy_grid.png, rasterfairy_circular_grid.png, rasterfairy_triangular_grid.png")

if __name__ == "__main__":
    main_block()
