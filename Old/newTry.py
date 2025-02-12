import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from tifffile import imread
import glob

def analyze_connected_components(directory_path, min_size=50):
    num_segments = []
    avg_segment_sizes = []
    
    file_list = sorted(glob.glob(os.path.join(directory_path, "*.tif")))
    
    for i, file_path in enumerate(file_list):
        volume = imread(file_path)
        
        labeled_volume = measure.label(volume, connectivity=1)
        
        props = measure.regionprops(labeled_volume)
        
        segment_sizes = [prop.area for prop in props if prop.area >= min_size]
        num_segments_current = len(segment_sizes)
        avg_segment_size_current = np.mean(segment_sizes) if segment_sizes else 0
        
        num_segments.append(num_segments_current)
        avg_segment_sizes.append(avg_segment_size_current)
        
        print(f"Timepoint {i+1}/{len(file_list)}: File '{os.path.basename(file_path)}' "
              f"Number of Segments = {num_segments_current}, Average Segment Size = {avg_segment_size_current:.2f}")
    
    timepoints = range(len(file_list))
    
    plt.figure(figsize=(12, 6))
    
    # plot number of segments over time
    plt.subplot(1, 2, 1)
    plt.plot(timepoints, num_segments, marker='o', color='b')
    plt.title("Number of Segments Over Time")
    plt.xlabel("Timepoint")
    plt.ylabel("Number of Segments")
    
    # plot average segment size over time
    plt.subplot(1, 2, 2)
    plt.plot(timepoints, avg_segment_sizes, marker='o', color='r')
    plt.title("Average Segment Size Over Time")
    plt.xlabel("Timepoint")
    plt.ylabel("Average Segment Size")
    
    plt.tight_layout()
    plt.show()

analyze_connected_components('AravindNewClassifier', min_size=100)
