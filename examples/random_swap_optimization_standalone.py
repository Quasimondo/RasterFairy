"""
RasterFairy Standalone Demo - Random Swap Optimization

This script demonstrates the SwapOptimizer functionality of the RasterFairy
library. It shows how to:
1. Generate a 2D point cloud (via t-SNE).
2. Arrange it into an initial rectangular grid using RasterFairy.
3. Optimize the grid using SwapOptimizer to improve the assignment of
   points to grid cells, minimizing distances between original points
   and their new grid locations.
4. Continue optimization for more iterations.
5. Apply 'shaking' (a technique similar to simulated annealing) to potentially
   escape local minima and find better solutions.
6. Optimize from a randomly shuffled grid assignment to explore different
   parts of the solution space.

Output images are saved in the current directory.
This script is an adaptation of the operations shown in the
'random_swap_optimization.ipynb' notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import colorsys
from PIL import Image # Not strictly necessary for current savefig use, but harmless
import io # Not strictly necessary for current savefig use, but harmless
import rasterfairy
from rasterfairy import rfoptimizer

def generate_color_harmonies(n_colors, harmony_type="tetradic"):
    """
    Generate 'n_colors' based on color theory harmonies.

    Parameters:
    - n_colors (int): Number of colors to generate.
    - harmony_type (str): Type of color harmony ("triadic", "complementary",
                          "analogous", "tetradic").

    Returns:
    - numpy.ndarray: Array of RGB colors.
    """
    base_hue = np.random.random()
    colors = []

    if harmony_type == "triadic":
        hues = [(base_hue + i/3) % 1.0 for i in range(3)]
    elif harmony_type == "complementary":
        hues = [base_hue, (base_hue + 0.5) % 1.0]
    elif harmony_type == "analogous":
        hues = [(base_hue + i*0.083) % 1.0 for i in range(-2, 3)]
    elif harmony_type == "tetradic":
        hues = [(base_hue + i*0.25) % 1.0 for i in range(4)]

    for _ in range(n_colors):
        hue = np.random.choice(hues)
        hue = (hue + np.random.uniform(-0.05, 0.05)) % 1.0
        sat = np.random.uniform(0.6, 1.0)
        val = np.random.uniform(0.7, 1.0)
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(rgb)

    return np.array(colors)

def generate_data():
    """
    Generates a dataset of 3D points and their 2D t-SNE embedding.

    Returns:
    - dataPoints (numpy.ndarray): The original 3D data points (used as colors).
    - xy (numpy.ndarray): The 2D t-SNE embedding of dataPoints.
    - totalDataPoints (int): The number of data points generated.
    """
    totalDataPoints = 4900 # Corresponds to a 70x70 grid
    dataPoints = generate_color_harmonies(totalDataPoints, "tetradic")
    # Create a t-SNE embedding in 2D.
    # random_state is set for reproducibility.
    xy = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000).fit_transform(dataPoints)
    return dataPoints, xy, totalDataPoints

def display_scatter_plot(xy_coords, colors, title="Scatter Plot", filename="scatter_plot.png", marker_size=7.5):
    """
    Creates, displays, and saves a scatter plot.

    Parameters:
    - xy_coords (numpy.ndarray): Array of 2D coordinates for the points.
    - colors (numpy.ndarray): Array of colors for each point.
    - title (str): Title of the plot.
    - filename (str): Filename to save the plot.
    - marker_size (float): Size of the markers in the scatter plot.
    """
    fig = plt.figure(figsize=(10.0, 10.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.autoscale_view(True, True, True)
    ax.invert_yaxis() # Consistent with original notebook
    ax.scatter(xy_coords[:, 0], xy_coords[:, 1], c=colors, edgecolors='none', marker='s', s=marker_size)
    ax.set_title(title, color='white')
    plt.savefig(filename)
    plt.close(fig)
    print(f"Plot saved as {filename}")

def main_block():
    """
    Main execution block for the RasterFairy Random Swap Optimization demo.

    Steps include:
    1. Generating initial data and t-SNE embedding.
    2. Creating an initial grid assignment with RasterFairy.
    3. Optimizing the grid with SwapOptimizer (initial, continued, with shaking).
    4. Optimizing from a randomly shuffled grid assignment.
    """
    print("Starting random swap optimization standalone demo...")

    # 1. Generate Data
    # dataPoints are the original high-dimensional data (here, 3D colors).
    # xy is the 2D t-SNE embedding of dataPoints.
    # totalDataPoints is the number of points.
    dataPoints, xy, totalDataPoints = generate_data()
    print(f"{totalDataPoints} data points generated.")
    print(f"Shape of t-SNE embedded xy: {xy.shape}")

    # 2. Initial Scatter Plot of t-SNE Embedding
    # This shows the raw 2D representation of the data before any grid assignment.
    display_scatter_plot(xy, dataPoints, title="Initial t-SNE Embedding (Swap Opt)", filename="tsne_embedding_swap_opt.png", marker_size=7.5)

    # 3. Get Rectangular Arrangement
    # Determine possible grid dimensions for the totalDataPoints.
    # RasterFairy provides arrangements, the first one being the most square.
    print("\nGetting rectangular arrangements...")
    arrangements = rasterfairy.getRectArrangements(totalDataPoints)
    print(f"{len(arrangements)} possible arrangements found: {arrangements}")
    if not arrangements:
        print("Error: No rectangular arrangements found for the given number of points.")
        return
    target_arrangement = arrangements[0] # e.g., (70, 70) for 4900 points
    print(f"Selected target arrangement (width, height): {target_arrangement}")

    # 4. Transform Point Cloud to Grid (Initial Assignment)
    # This function assigns each point in 'xy' to a cell in the 'target_arrangement' grid.
    # 'grid_xy' contains the 2D coordinates of the centers of these grid cells.
    # The assignment is initial and not yet optimized for spatial coherence of 'xy'.
    print("\nTransforming point cloud to grid...")
    grid_xy, (width, height) = rasterfairy.transformPointCloud2D(xy, target=target_arrangement)
    print(f"Point cloud transformed to a {width}x{height} grid.")
    print(f"Shape of grid_xy: {grid_xy.shape}") # Should be (totalDataPoints, 2)

    # 5. Grid Scatter Plot (Initial RasterFairy assignment)
    # Shows the points assigned to grid cells, colored by their original 'dataPoints' color.
    # This is the state *before* SwapOptimizer is used.
    display_scatter_plot(grid_xy, dataPoints, title=f"Initial RasterFairy Grid ({width}x{height}) (Swap Opt)", filename="rasterfairy_grid_initial_swap_opt.png", marker_size=9)

    # --- Swap Optimization Steps ---
    print("\n--- Starting Swap Optimization Steps ---")

    # 6. Initialize SwapOptimizer
    # This optimizer will attempt to improve the assignment of points to grid cells.
    optimizer = rfoptimizer.SwapOptimizer()

    # 7. Run Initial Optimization
    iterations = 100000
    print(f"\nRunning initial optimization for {iterations} iterations...")
    # optimizer.optimize arguments:
    # - xy: The original 2D points (t-SNE embedding).
    # - grid_xy: The target grid cell coordinates.
    # - width, height: Dimensions of the grid.
    # - iterations: Number of optimization iterations.
    # It returns 'swapTable': an array of indices. If swapTable[new_idx] = old_idx,
    # it means the point originally at dataPoints[old_idx] (and xy[old_idx])
    # should now be placed at grid_xy[new_idx].
    # The optimizer prints progress by default (verbose=True).
    swapTable = optimizer.optimize(xy, grid_xy, width, height, iterations)

    # 8. Apply SwapTable and Plot Optimized Grid
    # 'optimized_xy' reorders 'grid_xy' according to the 'swapTable' to show the optimized assignment.
    # The colors 'dataPoints' remain fixed but are now plotted at new grid locations.
    print("\nApplying swap table and plotting optimized grid...")
    optimized_xy = grid_xy[swapTable]
    display_scatter_plot(optimized_xy, dataPoints, title=f"Optimized RasterFairy Grid ({width}x{height}, {iterations} iter) (Swap Opt)", filename="rasterfairy_grid_optimized_swap_opt.png", marker_size=9)

    # 9. Continue Optimization
    # Further optimize from the current state.
    continue_iterations = 100000
    print(f"\nContinuing optimization for another {continue_iterations} iterations...")
    swapTable = optimizer.continueOptimization(continue_iterations)

    # 10. Apply New SwapTable and Plot Continued Optimized Grid
    print("\nApplying new swap table and plotting continued optimized grid...")
    continued_optimized_xy = grid_xy[swapTable]
    total_iterations_after_continue = iterations + continue_iterations
    display_scatter_plot(continued_optimized_xy, dataPoints, title=f"Continued Optimization ({width}x{height}, {total_iterations_after_continue} total iter) (Swap Opt)", filename="rasterfairy_grid_continued_swap_opt.png", marker_size=9)

    # 11. Optimization with Shaking
    # 'shakeIterations' introduces perturbations during optimization,
    # helping to escape local minima (similar to simulated annealing).
    shake_iterations_additional = 100000
    shake_amount = 5 # Number of shake operations per main iteration
    print(f"\nPerforming optimization with shaking (shakeAmount={shake_amount}) for {shake_iterations_additional} iterations...")
    swapTable = optimizer.continueOptimization(shake_iterations_additional, shakeIterations=shake_amount)

    # 12. Apply New SwapTable and Plot Shaken Optimized Grid
    print("\nApplying new swap table and plotting shaken optimized grid...")
    shaken_optimized_xy = grid_xy[swapTable]
    total_iterations_after_shake = total_iterations_after_continue + shake_iterations_additional
    display_scatter_plot(shaken_optimized_xy, dataPoints, title=f"Optimization with Shaking ({width}x{height}, {total_iterations_after_shake} total iter, shake={shake_amount}) (Swap Opt)", filename="rasterfairy_grid_shaken_swap_opt.png", marker_size=9)

    # --- Optimization from Random Start ---
    print("\n--- Starting Optimization from Random Arrangement ---")

    # 13. Prepare for Random Start Optimization
    print("\nPreparing for optimization from a random starting arrangement...")
    # Shuffle the current swapTable to create a random assignment of points to grid cells.
    # This means the optimizer will start from a completely different configuration.
    # Note: The swapTable from the previous step is modified in-place by np.random.shuffle.
    np.random.shuffle(swapTable)
    shuffled_grid_for_opt = grid_xy[swapTable] # Apply this shuffled assignment for visualization
    # Display the shuffled grid - this is the starting point for the next optimization.
    display_scatter_plot(shuffled_grid_for_opt, dataPoints, title=f"Shuffled Grid (Random Start Point) ({width}x{height}) (Swap Opt)", filename="rasterfairy_grid_shuffled_for_random_opt.png", marker_size=9)

    # 14. Initialize a New Optimizer for Random Start
    # Using a new optimizer instance ensures a fresh state (e.g., no learned parameters from previous runs).
    print("\nInitializing new optimizer for random start optimization...")
    random_start_optimizer = rfoptimizer.SwapOptimizer()

    # 15. Run Optimization from Random Arrangement
    random_start_iterations = 100000
    print(f"\nRunning optimization from random start for {random_start_iterations} iterations...")
    # Provide the shuffled 'swapTable' as the initial state for the optimization.
    final_swapTable_random = random_start_optimizer.optimize(xy, grid_xy, width, height, random_start_iterations, swapTable=swapTable)

    # 16. Apply Final SwapTable and Plot
    print("\nApplying final swap table from random start optimization and plotting...")
    final_optimized_xy_random = grid_xy[final_swapTable_random]
    display_scatter_plot(final_optimized_xy_random, dataPoints, title=f"Optimized from Random Start ({width}x{height}, {random_start_iterations} iter) (Swap Opt)", filename="rasterfairy_grid_random_start_swap_opt.png", marker_size=9)

    print("\n...random swap optimization standalone demo finished.")
    print("All output images:")
    print("- tsne_embedding_swap_opt.png")
    print("- rasterfairy_grid_initial_swap_opt.png")
    print("- rasterfairy_grid_optimized_swap_opt.png")
    print("- rasterfairy_grid_continued_swap_opt.png")
    print("- rasterfairy_grid_shaken_swap_opt.png")
    print("- rasterfairy_grid_shuffled_for_random_opt.png")
    print("- rasterfairy_grid_random_start_swap_opt.png")

if __name__ == "__main__":
    main_block()
