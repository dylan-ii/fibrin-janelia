import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, color
from tifffile import imread, imwrite
import glob
import pandas as pd
import trackpy as tp

def segment_and_track(directory_path, output_segmented_dir, min_size=250):
    os.makedirs(output_segmented_dir, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(directory_path, "*.tif")))

    all_features = []

    for time_idx, file_path in enumerate(file_list):
        volume = imread(file_path)
        
        labeled_volume = measure.label(volume, connectivity=1)
        
        segment_sizes = np.bincount(labeled_volume.ravel())
        
        large_segments = np.isin(labeled_volume, np.where(segment_sizes >= min_size)[0])
        
        filtered_volume = labeled_volume * large_segments
        
        segmented_filename = os.path.join(output_segmented_dir, f"segmented_{time_idx+1}.tif")
        imwrite(segmented_filename, filtered_volume.astype(np.uint16))
        
        props = measure.regionprops(filtered_volume)
        for prop in props:
            if prop.area >= min_size:
                y, x, z = prop.centroid
                all_features.append([x, y, z, time_idx])

    features_df = pd.DataFrame(all_features, columns=['x', 'y', 'z', 'frame'])
    
    tracked_features = tp.link_df(features_df, search_range=10, memory=3)
    
    num_tracked_objects = tracked_features.groupby('frame')['particle'].nunique()
    
    plt.figure(figsize=(10, 6))
    plt.plot(num_tracked_objects.index, num_tracked_objects.values, marker='o')
    plt.title("Number of Tracked Objects Over Time")
    plt.xlabel("Timepoint")
    plt.ylabel("Number of Tracked Objects")
    plt.show()

segment_and_track(
    directory_path='Experiment19NewClass',
    output_segmented_dir='Experiment19Segmented',
    #output_images_dir='Experiment19SegmentedImages',
    min_size=250
)
