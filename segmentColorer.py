import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import binary_erosion, ball
from tifffile import imread
import glob
import os

def segment_and_plot_every_50(directory_path, erosion_radius=0):
    # Load all .tif files from the directory in sorted order
    file_list = sorted(glob.glob(os.path.join(directory_path, "*.tif")))

    for time_idx, file_path in enumerate(file_list):

        if (time_idx + 1) % 100 != 0: continue

        # Load the 3D volume data
        volume = imread(file_path)
        
        # Label the connected components in the binary volume
        #labeled_volume = measure.label(volume, connectivity=1)
        
        print("Vol")

        # Apply binary erosion to the labeled volume to separate objects
        #eroded_volume = binary_erosion(volume > 0, selem=ball(erosion_radius)).astype(int)
        
        print("eroded")

        # Re-label the eroded volume to preserve separate segments
        eroded_labeled_volume = measure.label(volume, connectivity=2)

        print("labeled.")

        # Every 50 volumes, create a plot of the segmented volume
        if (time_idx + 1) % 100 == 0:
            # Maximum Intensity Projection (MIP) across all z slices
            mip_projection = np.max(eroded_labeled_volume, axis=0)  # Max across z-axis (axis=0 for z-dimension)
            
            print("maxed...")

            # Plotting the Maximum Intensity Projection (MIP)
            plt.figure(figsize=(8, 8))
            plt.imshow(mip_projection, cmap='gray')  # Display in grayscale
            plt.title(f"Max Intensity Projection at Timepoint {time_idx+1}")
            plt.axis('off')
            plt.show()

# Usage:
# Replace 'path_to_directory' with the path to your directory containing the .tif files.
segment_and_plot_every_50(
    directory_path='Experiment19NewClass',
    erosion_radius=0  # Adjust this to control the erosion strength
)