"""
Mangrove Carbon Stock Analysis & Visualization

Description: 
    This script processes multi-temporal Sentinel-2 imagery to calculate:
    1. NDVI (Normalized Difference Vegetation Index)
    2. Estimated Carbon Stock in Mangrove ecosystems (Empirical Model)
    Outputs include GeoTIFF files, a Dashboard Visualization (PNG), and a Time-series Animation (GIF).
"""

import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from rasterio.warp import transform_bounds 
from matplotlib.ticker import FuncFormatter 

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# ⚠️ IMPORTANT: Update these paths to match your local machine before running!
CONFIG = {
    'input_files': {
        # Format: 'Year': r"PATH_TO_YOUR_SENTINEL_DATA.tif"
        '2019': r"path/to/your/data/2019/S2_2019_Image.tif", 
        '2022': r"path/to/your/data/2022/S2_2022_Image.tif",
        '2024': r"path/to/your/data/2024/S2_2024_Image.tif"
    },
    # Directory where results (Images, GIFs, TIFs) will be saved
    'output_dir': r"path/to/your/output_folder",
    
    'portfolio_title': "Mangrove Carbon Stock Estimation (Budeng Village)",
    'threshold_ndvi': 0.4  # Vegetation threshold for mangroves
}

# Ensure output directory exists
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ==========================================
# UTILITY & PROCESSING FUNCTIONS
# ==========================================
def calculate_indices(red, nir):
    """Calculates NDVI while handling division by zero errors."""
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (nir - red) / (nir + red)
    
    # Clip values to valid range [-1, 1] to remove noise
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ndvi

def estimate_carbon(ndvi, threshold):
    """
    Estimates carbon stock based on NDVI using an allometric equation.
    Formula: AGB = 150 * (NDVI^2.5) -> Carbon = AGB * 0.47
    """
    # Mask non-mangrove areas
    mask = np.where(ndvi > threshold, ndvi, np.nan)
    
    # Calculate Above Ground Biomass (AGB) and Carbon
    agb = 150 * (mask ** 2.5)
    carbon_stock = agb * 0.47
    return carbon_stock

def process_raster(filepath):
    """Reads raster data and processes bands into derivative products."""
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}...")
    
    with rasterio.open(filepath) as src:
        # Transform bounds for map plotting
        left, bottom, right, top = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
        extent = [left, right, bottom, top]
        profile = src.profile

        # Read Bands (Assuming standard Sentinel-2 ordering)
        # IMPORTANT: Divide by 10000.0 to convert integer to Reflectance (0.0 - 1.0)
        blue  = src.read(1).astype('float32') / 10000.0
        green = src.read(2).astype('float32') / 10000.0
        red   = src.read(3).astype('float32') / 10000.0
        nir   = src.read(4).astype('float32') / 10000.0

    # Calculate Indices
    ndvi = calculate_indices(red, nir)
    carbon = estimate_carbon(ndvi, CONFIG['threshold_ndvi'])

    # Create Color Composites for Visualization
    # True Color (RGB)
    tci = np.dstack((red, green, blue))
    tci = np.clip(tci * 3.0, 0, 1) # Brightness adjustment
    
    # False Color (NIR-Red-Green) - Good for vegetation analysis
    fcc = np.dstack((nir, red, green))
    fcc = np.clip(fcc * 2.5, 0, 1)

    return {
        'tci': tci, 'fcc': fcc, 'ndvi': ndvi, 
        'carbon': carbon, 'extent': extent, 'profile': profile
    }

def save_geotiff(data, profile, output_path):
    """Saves a numpy array as a GeoTIFF file."""
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999)
    with rasterio.open(output_path, 'w', **profile) as dst:
        data_save = np.nan_to_num(data, nan=-9999)
        dst.write(data_save.astype(rasterio.float32), 1)
    print(f"Saved: {output_path}")

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================
def dms_formatter(x, pos):
    """Formats coordinates to Degree Minutes."""
    d = int(x)
    m = int((x - d) * 60)
    return f"{d}°{abs(m):02d}'"

def setup_axis(ax, title):
    """Applies styling and formatting to map axes."""
    ax.set_title(title, fontsize=10, pad=6)
    ax.xaxis.set_major_formatter(FuncFormatter(dms_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(dms_formatter))
    ax.tick_params(labelsize=8)
    ax.locator_params(axis='x', nbins=3)
    ax.locator_params(axis='y', nbins=3)

def create_dashboard(results_dict):
    """Generates a static multi-year comparison dashboard."""
    print("Generating Dashboard...")
    years = sorted(results_dict.keys())
    fig, axes = plt.subplots(4, 3, figsize=(16, 20))
    plt.subplots_adjust(wspace=0.15, hspace=0.3)

    for i, year in enumerate(years):
        data = results_dict[year]
        extent = data['extent']

        # Define Layers
        layers = [
            (data['fcc'], f"False Color - {year}", None, None, None),
            (data['tci'], f"True Color - {year}", None, None, None),
            (data['ndvi'], f"NDVI - {year}", 'RdYlGn', -0.2, 1.0),
            (data['carbon'], f"Carbon Stock - {year}", 'Greens', 0, 100)
        ]

        for row, (img, title, cmap, vmin, vmax) in enumerate(layers):
            ax = axes[row, i]
            if cmap:
                im = ax.imshow(img, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
                # Store reference for colorbars (only needed once per row)
                if i == 2:
                    if 'NDVI' in title: im_ndvi = im
                    if 'Carbon' in title: im_carb = im
            else:
                ax.imshow(img, extent=extent)
            
            setup_axis(ax, title)

    # Add Colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.35, 0.015, 0.12])
    fig.colorbar(im_ndvi, cax=cbar_ax1, label='NDVI Index')
    
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.015, 0.12])
    fig.colorbar(im_carb, cax=cbar_ax2, label='Ton C/Ha')

    # Titles & Watermarks
    plt.suptitle("MULTI-TEMPORAL SPATIAL ANALYSIS OF MANGROVE ECOSYSTEMS", 
                 fontsize=20, weight='bold', y=0.95, color='#2c3e50')
    fig.text(0.5, 0.02, CONFIG['portfolio_title'], ha='center', fontsize=24, 
             weight='bold', color='#145A32', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    out_path = os.path.join(CONFIG['output_dir'], "Dashboard_Mangrove_Analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Dashboard saved: {out_path}")

def create_gif(results_dict):
    """Generates a time-series animation of carbon stock."""
    print("Rendering GIF...")
    frames = []
    years = sorted(results_dict.keys())
    
    for year in years:
        fig, ax = plt.subplots(figsize=(8, 8))
        data = results_dict[year]
        
        im = ax.imshow(data['carbon'], cmap='Greens', vmin=0, vmax=80)
        ax.set_title(f"{CONFIG['portfolio_title']}\nYear: {year}", 
                     fontsize=14, weight='bold', pad=15, color='#145A32')
        ax.axis('off')
        
        # Convert plot to image buffer
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)

    if frames:
        out_path = os.path.join(CONFIG['output_dir'], "Mangrove_Carbon_TimeSeries.gif")
        imageio.mimsave(out_path, frames, duration=1.5, loop=0)
        print(f"Animation saved: {out_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    processed_results = {}

    print("Starting Analysis Pipeline...")

    # 1. Processing Loop
    for year, path in CONFIG['input_files'].items():
        if os.path.exists(path):
            # Process Data
            res = process_raster(path)
            processed_results[year] = res
            
            # Export Individual GeoTIFFs
            save_geotiff(res['ndvi'], res['profile'], 
                         os.path.join(CONFIG['output_dir'], f"NDVI_{year}.tif"))
            save_geotiff(res['carbon'], res['profile'], 
                         os.path.join(CONFIG['output_dir'], f"Carbon_{year}.tif"))
        else:
            print(f"File not found: {path}")

    # 2. Visualization Generation
    if processed_results:
        create_dashboard(processed_results)
        create_gif(processed_results)
    
    print("\nAll processes completed successfully.")