# RasterFairy Image Similarity Demo
#
# This script demonstrates how to arrange a large set of images by similarity
# on a grid using:
# - CIFAR-10 dataset for source images
# - DreamSim for feature extraction
# - t-SNE for dimensionality reduction
# - RasterFairy for grid arrangement
#
# Dependencies:
# - matplotlib
# - numpy
# - scikit-learn (for TSNE)
# - rasterfairy
# - Pillow (PIL)
# - torch
# - torchvision
# - dreamsim
#
# Install dreamsim with: pip install dreamsim
#
# Running the script:
# Execute this script from the command line using Python 3:
#   python image_similarity_demo.py
#
# Output:
# The script will generate a large bitmap showing images arranged by similarity
#

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import rasterfairy
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from dreamsim import dreamsim
import os
from tqdm import tqdm
import random
import pickle
from pathlib import Path

def get_cache_filename(cache_dir="./feature_cache"):
    """Get the filename for the feature cache."""
    Path(cache_dir).mkdir(exist_ok=True)
    return os.path.join(cache_dir, "dreamsim_features.pkl")

def load_feature_cache(cache_filename):
    """Load existing feature cache from disk."""
    if os.path.exists(cache_filename):
        try:
            print(f"Loading existing feature cache from {cache_filename}")
            with open(cache_filename, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded cache with {len(cache)} existing features")
            return cache
        except Exception as e:
            print(f"Error loading cache: {e}. Starting with empty cache.")
            return {}
    else:
        print("No existing feature cache found. Starting fresh.")
        return {}

def save_feature_cache(cache, cache_filename):
    """Save feature cache to disk."""
    try:
        print(f"Saving feature cache with {len(cache)} features to {cache_filename}")
        with open(cache_filename, 'wb') as f:
            pickle.dump(cache, f)
        print("Feature cache saved successfully")
    except Exception as e:
        print(f"Error saving cache: {e}")

def extract_features_dreamsim_cached(images, indices, cache_dir="./feature_cache"):
    """Extract feature vectors using DreamSim with caching by dataset index."""
    print("Initializing DreamSim model...")
    
    # Initialize DreamSim model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load feature cache
    cache_filename = get_cache_filename(cache_dir)
    feature_cache = load_feature_cache(cache_filename)
    
    # Check which features we need to compute
    missing_indices = []
    cached_features = {}
    
    for i, idx in enumerate(indices):
        if idx in feature_cache:
            cached_features[i] = feature_cache[idx]
        else:
            missing_indices.append((i, idx))
    
    print(f"Found {len(cached_features)} cached features")
    print(f"Need to compute {len(missing_indices)} new features")
    
    # Only load model if we have new features to compute
    if missing_indices:
        # Load DreamSim model
        model, preprocess = dreamsim(pretrained=True, device=device)
        
        print(f"Computing features for {len(missing_indices)} new images...")
        
        # Process missing images
        for i, dataset_idx in tqdm(missing_indices, desc="Computing new features"):
            img = images[i]
            
            # Resize CIFAR-10 images (32x32) to what DreamSim expects
            img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Preprocess image
            tensor = preprocess(img_resized).to(device)
            
            # Extract features for single image
            with torch.no_grad():
                embedding = model.embed(tensor)
                feature_vector = embedding.cpu().numpy()
                
                # Store in both caches
                cached_features[i] = feature_vector
                feature_cache[dataset_idx] = feature_vector
        
        # Save updated cache
        save_feature_cache(feature_cache, cache_filename)
    else:
        print("All features found in cache!")
    
    # Reconstruct features array in correct order
    features = []
    for i in range(len(images)):
        features.append(cached_features[i])
    
    # Stack all features
    features = np.vstack(features)
    print(f"Feature extraction complete. Shape: {features.shape}")
    
    return features

def clear_feature_cache(cache_dir="./feature_cache"):
    """Clear the feature cache (useful for testing or if you want to start fresh)."""
    cache_filename = get_cache_filename(cache_dir)
    if os.path.exists(cache_filename):
        os.remove(cache_filename)
        print(f"Feature cache cleared: {cache_filename}")
    else:
        print("No feature cache to clear")

def get_cache_info(cache_dir="./feature_cache"):
    """Get information about the current cache."""
    cache_filename = get_cache_filename(cache_dir)
    if os.path.exists(cache_filename):
        cache = load_feature_cache(cache_filename)
        print(f"Cache contains {len(cache)} features")
        print(f"Cache file size: {os.path.getsize(cache_filename) / (1024*1024):.2f} MB")
        
        # Show some statistics
        if cache:
            sample_feature = next(iter(cache.values()))
            print(f"Feature vector shape: {sample_feature.shape}")
            indices = list(cache.keys())
            print(f"Index range: {min(indices)} to {max(indices)}")
    else:
        print("No cache file exists")


def download_cifar10(data_dir="./data"):
    """Download CIFAR-10 dataset."""
    print("Downloading CIFAR-10 dataset...")
    
    # Simple transform to convert PIL to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Download training set
    trainset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    
    print(f"CIFAR-10 downloaded. Training set size: {len(trainset)}")
    return trainset

def select_random_subset(dataset, num_samples=6000, seed=0xc0ffee):
    """Select a random subset of images from the dataset."""
    print(f"Selecting random subset of {num_samples} images...")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Get random indices
    total_samples = len(dataset)
    if num_samples > total_samples:
        num_samples = total_samples
        print(f"Requested more samples than available. Using all {total_samples} samples.")
    
    indices = random.sample(range(total_samples), num_samples)
    
    # Extract images and labels
    images = []
    labels = []
    
    print("Loading selected images...")
    for i in tqdm(indices):
        img_tensor, label = dataset[i]
        # Convert tensor back to PIL Image for easier handling
        img_pil = transforms.ToPILImage()(img_tensor)
        images.append(img_pil)
        labels.append(label)
    
    return images, labels, indices

def compute_embedding(features, method='tsne', random_state=0xc0ffee, **kwargs):
    """Compute 2D embedding of the features using various dimensionality reduction methods."""
    
    if method.lower() == 'tsne':
        print(f"Computing t-SNE embedding...")
        perplexity = kwargs.get('perplexity', 30)
        n_iter = kwargs.get('n_iter', 1000)
        
        from sklearn.manifold import TSNE
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=n_iter,
            verbose=1
        )
        
    elif method.lower() == 'umap':
        print(f"Computing UMAP embedding...")
        n_neighbors = kwargs.get('n_neighbors', 15)
        min_dist = kwargs.get('min_dist', 0.1)
        metric = kwargs.get('metric', 'euclidean')
        
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
                verbose=True
            )
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            raise
            
    elif method.lower() == 'pca':
        print(f"Computing PCA embedding...")
        from sklearn.decomposition import PCA
        reducer = PCA(
            n_components=2,
            random_state=random_state
        )
        
    elif method.lower() == 'isomap':
        print(f"Computing Isomap embedding...")
        n_neighbors = kwargs.get('n_neighbors', 5)
        
        from sklearn.manifold import Isomap
        reducer = Isomap(
            n_components=2,
            n_neighbors=n_neighbors
        )
        
    elif method.lower() == 'lle':
        print(f"Computing LLE embedding...")
        n_neighbors = kwargs.get('n_neighbors', 5)
        
        from sklearn.manifold import LocallyLinearEmbedding
        reducer = LocallyLinearEmbedding(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        
    elif method.lower() == 'spectral':
        print(f"Computing Spectral embedding...")
        n_neighbors = kwargs.get('n_neighbors', 10)
        
        from sklearn.manifold import SpectralEmbedding
        reducer = SpectralEmbedding(
            n_components=2,
            n_neighbors=n_neighbors,
            random_state=random_state
        )
        
    else:
        raise ValueError(f"Unknown method: {method}. Available methods: tsne, umap, pca, isomap, lle, spectral")
    
    xy = reducer.fit_transform(features)
    print(f"{method.upper()} embedding complete. Shape: {xy.shape}")
    
    return xy

def get_rectangular_arrangement(num_points):
    """Get the most square rectangular arrangement for the images."""
    print(f"Finding rectangular arrangement for {num_points} points...")
    
    arrangements = rasterfairy.getRectArrangements(num_points)
    print(f"Found {len(arrangements)} possible arrangements: {arrangements}")
    
    if not arrangements:
        raise ValueError("No rectangular arrangements found for the given number of points.")
    
    # The first arrangement is the most square
    arrangement = arrangements[0]
    print(f"Selected arrangement: {arrangement[0]}x{arrangement[1]}")
    
    return arrangement

def transform_to_grid(xy_coords, arrangement):
    """Transform 2D coordinates to grid positions."""
    print("Transforming point cloud to grid...")
    
    grid_xy, (width, height) = rasterfairy.transformPointCloud2D(
        xy_coords, 
        target=arrangement
    )
    
    print(f"Grid transformation complete: {width}x{height}")
    
    return grid_xy, width, height

def create_embedding_bitmap(images, xy_coords, output_width, output_height, 
                          image_size=32, output_filename="embedding_layout.jpg"):
    """Create a bitmap showing images at their original embedding coordinates."""
    print(f"Creating embedding bitmap: {output_width}x{output_height}")
    
    # Normalize coordinates to fit the output dimensions
    # Add some padding to ensure all images fit within bounds
    padding = image_size // 2
    
    x_coords = xy_coords[:, 0]
    y_coords = xy_coords[:, 1]
    
    # Normalize x coordinates
    x_min, x_max = x_coords.min(), x_coords.max()
    x_range = x_max - x_min
    if x_range > 0:
        x_normalized = ((x_coords - x_min) / x_range) * (output_width - 2 * padding) + padding
    else:
        x_normalized = np.full_like(x_coords, output_width // 2)
    
    # Normalize y coordinates
    y_min, y_max = y_coords.min(), y_coords.max()
    y_range = y_max - y_min
    if y_range > 0:
        y_normalized = ((y_coords - y_min) / y_range) * (output_height - 2 * padding) + padding
    else:
        y_normalized = np.full_like(y_coords, output_height // 2)
    
    print(f"Coordinate ranges - X: {x_min:.2f} to {x_max:.2f}, Y: {y_min:.2f} to {y_max:.2f}")
    print(f"Normalized ranges - X: {x_normalized.min():.0f} to {x_normalized.max():.0f}, Y: {y_normalized.min():.0f} to {y_normalized.max():.0f}")
    
    # Create blank canvas
    canvas = Image.new('RGB', (output_width, output_height), color='black')
    
    # Place each image on the canvas
    for i, (x, y) in enumerate(tqdm(zip(x_normalized, y_normalized), desc="Placing images in embedding space")):
        if i < len(images):
            img = images[i]
            
            # Resize image to fit
            img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            
            # Calculate pixel position (center the image on the coordinate)
            pixel_x = int(x - image_size // 2)
            pixel_y = int(y - image_size // 2)
            
            # Ensure coordinates are within bounds
            if (0 <= pixel_x <= output_width - image_size and 
                0 <= pixel_y <= output_height - image_size):
                
                # For overlapping images, we can either paste directly (last image wins)
                # or use alpha blending. For now, let's just paste directly.
                canvas.paste(img_resized, (pixel_x, pixel_y))
    
    # Save the result
    print(f"Saving embedding bitmap as {output_filename}")
    canvas.save(output_filename, quality=95)
    
    return canvas

def create_embedding_bitmap_with_alpha(images, xy_coords, output_width, output_height, 
                                     image_size=32, alpha=0.7, output_filename="embedding_layout_alpha.jpg"):
    """Create a bitmap with alpha blending for better visualization of overlaps."""
    print(f"Creating embedding bitmap with alpha blending: {output_width}x{output_height}")
    
    # Normalize coordinates
    padding = image_size // 2
    
    x_coords = xy_coords[:, 0]
    y_coords = xy_coords[:, 1]
    
    # Normalize x coordinates
    x_min, x_max = x_coords.min(), x_coords.max()
    x_range = x_max - x_min
    if x_range > 0:
        x_normalized = ((x_coords - x_min) / x_range) * (output_width - 2 * padding) + padding
    else:
        x_normalized = np.full_like(x_coords, output_width // 2)
    
    # Normalize y coordinates
    y_min, y_max = y_coords.min(), y_coords.max()
    y_range = y_max - y_min
    if y_range > 0:
        y_normalized = ((y_coords - y_min) / y_range) * (output_height - 2 * padding) + padding
    else:
        y_normalized = np.full_like(y_coords, output_height // 2)
    
    # Create blank canvas
    canvas = Image.new('RGBA', (output_width, output_height), color=(0, 0, 0, 255))
    
    # Place each image with alpha blending
    for i, (x, y) in enumerate(tqdm(zip(x_normalized, y_normalized), desc="Placing images with alpha")):
        if i < len(images):
            img = images[i]
            
            # Resize image and convert to RGBA
            img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            img_rgba = img_resized.convert('RGBA')
            
            # Apply alpha to the image
            alpha_int = int(alpha * 255)
            img_alpha = Image.new('RGBA', img_rgba.size, (255, 255, 255, alpha_int))
            img_with_alpha = Image.composite(img_rgba, Image.new('RGBA', img_rgba.size, (0, 0, 0, 0)), img_alpha)
            
            # Calculate pixel position
            pixel_x = int(x - image_size // 2)
            pixel_y = int(y - image_size // 2)
            
            # Ensure coordinates are within bounds
            if (0 <= pixel_x <= output_width - image_size and 
                0 <= pixel_y <= output_height - image_size):
                
                # Paste with alpha blending
                canvas.paste(img_with_alpha, (pixel_x, pixel_y), img_with_alpha)
    
    # Convert back to RGB for saving as JPEG
    final_canvas = Image.new('RGB', (output_width, output_height), color='black')
    final_canvas.paste(canvas, (0, 0), canvas)
    
    # Save the result
    print(f"Saving alpha-blended embedding bitmap as {output_filename}")
    final_canvas.save(output_filename, quality=95)
    
    return final_canvas



def create_image_grid(images, grid_positions, grid_width, grid_height, 
                     image_size=32, output_filename="image_similarity_grid.jpg"):
    """Create a large bitmap with images arranged according to grid positions."""
    print(f"Creating image grid: {grid_width}x{grid_height}")
    
    # Calculate output image dimensions
    output_width = grid_width * image_size
    output_height = grid_height * image_size
    
    print(f"Output image size: {output_width}x{output_height}")
    
    # Create blank canvas
    canvas = Image.new('RGB', (output_width, output_height), color='black')
    
    # Place each image on the canvas
    for i, (x, y) in enumerate(tqdm(grid_positions, desc="Placing images")):
        if i < len(images):
            img = images[i]
            
            # Resize image to fit grid cell
            img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            
            # Calculate pixel position
            pixel_x = int(x * image_size)
            pixel_y = int(y * image_size)
            
            # Ensure coordinates are within bounds
            if 0 <= pixel_x < output_width - image_size and 0 <= pixel_y < output_height - image_size:
                canvas.paste(img_resized, (pixel_x, pixel_y))
    
    # Save the result
    print(f"Saving image grid as {output_filename}")
    canvas.save(output_filename, quality=95)
    
    return canvas

def create_visualization_plots(xy_original, xy_grid, labels, grid_width, grid_height, method='tsne'):
    """Create visualization plots showing the transformation."""
    # Get CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create color map for classes
    colors = plt.cm.tab10(np.array(labels) / 9.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original embedding
    scatter1 = ax1.scatter(xy_original[:, 0], xy_original[:, 1], 
                          c=colors, s=2, alpha=0.7)
    ax1.set_title(f'Original {method.upper()} Embedding\n(Colored by CIFAR-10 Class)', fontsize=14)
    ax1.set_xlabel(f'{method.upper()} 1')
    ax1.set_ylabel(f'{method.upper()} 2')
    
    # Grid arrangement
    scatter2 = ax2.scatter(xy_grid[:, 0], xy_grid[:, 1], 
                          c=colors, s=4, alpha=0.8, marker='s')
    ax2.set_title(f'RasterFairy Grid Arrangement\n{grid_width}x{grid_height}', fontsize=14)
    ax2.set_xlabel('Grid X')
    ax2.set_ylabel('Grid Y')
    ax2.set_aspect('equal')
    
    # Add legend
    handles = [plt.scatter([], [], c=plt.cm.tab10(i/9.0), s=50, label=class_names[i]) 
               for i in range(10)]
    ax1.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'demo_output/cifar10_comparison_{method}_vs_grid.jpg', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved as demo_output/cifar10_comparison_{method}_vs_grid.jpg")

def main():
    """Main execution logic for the image similarity demo."""
    print("Starting RasterFairy Image Similarity Demo...")
    
    # Configuration
    NUM_IMAGES = 6000  # Adjust this number as needed
    IMAGE_SIZE = 48    # Size of each image in the final grid (pixels)
    EMBEDDING_METHODS = ['tsne', 'umap', 'pca', 'isomap', 'lle', 'spectral']
    EMBEDDING_METHOD = EMBEDDING_METHODS[0]  # Options: 'tsne', 'umap', 'pca', 'isomap', 'lle', 'spectral'
    CACHE_DIR = "./feature_cache"  # Directory to store cached features
    CREATE_ALPHA_VERSION = False  # Whether to create alpha-blended version
    PRE_WARP = True
    
    # Show cache info at start
    print("\n=== Cache Info (Before) ===")
    get_cache_info(CACHE_DIR)
    print("=" * 30)
    
    os.makedirs("demo_output",exist_ok=True)
    
    try:
        # Step 1: Download CIFAR-10 dataset
        dataset = download_cifar10()
        
        # Step 2: Select random subset
        images, labels, indices = select_random_subset(dataset, NUM_IMAGES)
        
        # Step 3: Extract features using DreamSim with caching
        features = extract_features_dreamsim_cached(images, indices, CACHE_DIR)
        
        # Show cache info after feature extraction
        print("\n=== Cache Info (After) ===")
        get_cache_info(CACHE_DIR)
        print("=" * 30)
        
        
        for EMBEDDING_METHOD in EMBEDDING_METHODS:
            # Step 4: Compute embedding using selected method
            if EMBEDDING_METHOD.lower() == 'tsne':
                xy_original = compute_embedding(features, method='tsne', perplexity=30)
            elif EMBEDDING_METHOD.lower() == 'umap':
                xy_original = compute_embedding(features, method='umap', n_neighbors=15, min_dist=0.1)
            elif EMBEDDING_METHOD.lower() == 'pca':
                xy_original = compute_embedding(features, method='pca')
            elif EMBEDDING_METHOD.lower() == 'isomap':
                xy_original = compute_embedding(features, method='isomap', n_neighbors=5)
            elif EMBEDDING_METHOD.lower() == 'lle':
                xy_original = compute_embedding(features, method='lle', n_neighbors=5)
            elif EMBEDDING_METHOD.lower() == 'spectral':
                xy_original = compute_embedding(features, method='spectral', n_neighbors=10)
            else:
                print(f"Unknown method {EMBEDDING_METHOD}, falling back to t-SNE")
                xy_original = compute_embedding(features, method='tsne')
            
            # Step 5: Get rectangular arrangement
            arrangement = get_rectangular_arrangement(len(images))
            
            # Step 6: Transform to grid
            #optional - pre-warp cloud towards a more rectangular shape:
            if PRE_WARP:
                xy_rectified = rasterfairy.coonswarp.rectifyCloud(xy_original, perimeterSubdivisionSteps=32, autoPerimeterOffset=True)
                xy_grid, grid_width, grid_height = transform_to_grid(xy_rectified, arrangement)
            else:    
                xy_grid, grid_width, grid_height = transform_to_grid(xy_original, arrangement)
            
            
            
            # Calculate output dimensions for consistency
            output_width = grid_width * IMAGE_SIZE
            output_height = grid_height * IMAGE_SIZE
            
            # Step 7: Create embedding layout bitmap (using same dimensions as grid)
            create_embedding_bitmap(
                images, xy_original, output_width, output_height,
                image_size=IMAGE_SIZE,
                output_filename=f"demo_output/cifar10_embedding_layout_{EMBEDDING_METHOD}.jpg"
            )
            
            # Step 7b: Optionally create alpha-blended version
            if CREATE_ALPHA_VERSION:
                create_embedding_bitmap_with_alpha(
                    images, xy_original, output_width, output_height,
                    image_size=IMAGE_SIZE, alpha=0.6,
                    output_filename=f"demo_output/cifar10_embedding_layout_{EMBEDDING_METHOD}_alpha.jpg"
                )
            
            # Step 8: Create visualization comparison
            create_visualization_plots(xy_original, xy_grid, labels, grid_width, grid_height, EMBEDDING_METHOD)
            
            # Step 9: Create final image grid
            canvas = create_image_grid(
                images, xy_grid, grid_width, grid_height, 
                image_size=IMAGE_SIZE,
                output_filename=f"demo_output/cifar10_similarity_grid_{EMBEDDING_METHOD}.jpg"
            )
            
            print("\nDemo completed successfully!")
            print("Generated files:")
            print(f"- demo_output/cifar10_similarity_grid_{EMBEDDING_METHOD}.jpg: Images arranged in regular grid by similarity")
            print(f"- demo_output/cifar10_embedding_layout_{EMBEDDING_METHOD}.jpg: Images at original embedding coordinates")
            if CREATE_ALPHA_VERSION:
                print(f"- demo_output/cifar10_embedding_layout_{EMBEDDING_METHOD}_alpha.jpg: Same as above with alpha blending")
            print(f"- demo_output/cifar10_comparison_{EMBEDDING_METHOD}_vs_grid.jpg: Visualization of the transformation")
            
            # Print some statistics
            print(f"\nStatistics:")
            print(f"- Embedding method: {EMBEDDING_METHOD.upper()}")
            print(f"- Total images processed: {len(images)}")
            print(f"- Grid dimensions: {grid_width}x{grid_height}")
            print(f"- Output image size: {output_width}x{output_height}")
            print(f"- Feature vector dimension: {features.shape[1]}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
