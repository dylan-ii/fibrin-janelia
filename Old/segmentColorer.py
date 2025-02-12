import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import binary_erosion, ball
from tifffile import imread
import glob
import os

def segment_and_plot_every_50(directory_path, erosion_radius=0):
    file_list = sorted(glob.glob(os.path.join(directory_path, "*.tif")))

    for time_idx, file_path in enumerate(file_list):

        if (time_idx + 1) % 100 != 0: continue
        volume = imread(file_path)
        
        #labeled_volume = measure.label(volume, connectivity=1)
        
        print("Vol")
        #eroded_volume = binary_erosion(volume > 0, selem=ball(erosion_radius)).astype(int)
        
        print("eroded")
        eroded_labeled_volume = measure.label(volume, connectivity=2)

        print("labeled.")

        if (time_idx + 1) % 100 == 0:
            mip_projection = np.max(eroded_labeled_volume, axis=0)
            
            print("maxed...")

            # Plotting the Maximum Intensity Projection (MIP)
            plt.figure(figsize=(8, 8))
            plt.imshow(mip_projection, cmap='gray')
            plt.title(f"Max Intensity Projection at Timepoint {time_idx+1}")
            plt.axis('off')
            plt.show()

segment_and_plot_every_50(
    directory_path='Experiment19NewClass',
    erosion_radius=0 
)
