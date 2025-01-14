import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.morphology import skeletonize_3d, remove_small_objects
from skimage.io import imread
import glob

def analyze_connected_components(directory_path, min_size, skeletonize_min_size=2):
    # Lists to store results
    num_segments = []
    avg_segment_sizes = []
    all_fiber_lengths = []
    
    # Load all .tif files from the directory in sorted order
    file_list = sorted(glob.glob(os.path.join(directory_path, "*.tif")))
    
    for i, file_path in enumerate(file_list):
        if i%10 != 0:  # Adjust the interval as needed
            continue

        print(f"Processing Timepoint {i+1}/{len(file_list)}: File '{os.path.basename(file_path)}'")

        # Load the 3D volume data
        volume = imread(file_path)
        print(f"Loaded volume shape: {volume.shape}, unique values: {np.unique(volume)}")
        
        # Remove small objects
        cleaned_volume = remove_small_objects(volume, min_size=skeletonize_min_size)
        print(f"Cleaned volume shape: {cleaned_volume.shape}, unique values: {np.unique(cleaned_volume)}")
        
        skeleton = skeletonize_3d(cleaned_volume)

        # Label connected components
        labeled_volume = measure.label(skeleton, connectivity=1)
        print(f"Labeled volume shape: {labeled_volume.shape}, "
              f"Number of unique labels: {len(np.unique(labeled_volume))}")
        
        # Count segment sizes
        segment_sizes = np.bincount(labeled_volume.ravel())
        print(f"Segment sizes (including background): {segment_sizes}")
        

        # Filter out background and small segments
        valid_labels = np.where(segment_sizes >= min_size)
        #valid_labels = valid_labels[valid_labels != 0]  # Exclude background
        num_segments_current = len(valid_labels)
        avg_segment_size_current = np.mean(segment_sizes[valid_labels]) if num_segments_current > 0 else 0
        
        print(type(skeleton))
        print(skeleton.shape)

        print(f"Filtered valid labels: {valid_labels}")
        print(f"Number of valid segments: {num_segments_current}")
        print(f"Average segment size: {avg_segment_size_current}")

        # Visualize skeleton
        plt.figure(figsize=(8, 8))
        plt.imshow(skeleton.max(axis=0), cmap='gray')
        plt.title(f"Skeletonized Volume (Timepoint {i+1})")
        plt.axis('off')
        plt.show()

        # Visualize labeled volume
        plt.figure(figsize=(8, 6))
        plt.imshow(labeled_volume.max(axis=0), cmap='tab20', interpolation='nearest')
        plt.title(f"Labeled Components (Timepoint {i+1})")
        plt.axis('off')
        plt.colorbar(label="Label")
        plt.show()

        # Visualize the skeletonized volume
        plt.figure(figsize=(8, 6))
        plt.imshow(skeleton.max(axis=0), cmap='gray', interpolation='nearest')
        plt.title(f"Skeletonized Fibers (Timepoint {i+1})")
        plt.axis('off')
        plt.show()
        
        # Plot distribution of fiber voxel counts
        plt.figure(figsize=(8, 6))
        plt.hist(fiber_lengths, bins=20, color='skyblue', edgecolor='black')
        
        # Highlight the maximum fiber length
        max_fiber_length = max(fiber_lengths) if fiber_lengths else 0
        plt.axvline(x=max_fiber_length, color='red', linestyle='dashed', linewidth=2)
        
        # Annotate the maximum fiber length
        plt.text(
            max_fiber_length, 1, f'Max: {max_fiber_length}', 
            color='red', verticalalignment='bottom', horizontalalignment='right', 
            fontsize=12
        )
        
        plt.title(f"Fiber Length Distribution (Timepoint {i+1})")
        plt.xlabel("Fiber Length (voxels)")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
        
        # Store results
        num_segments.append(num_segments_current)
        avg_segment_sizes.append(avg_segment_size_current)
        all_fiber_lengths.append(fiber_lengths)
        
        print(f"Timepoint {i+1}/{len(file_list)} Summary: "
              f"Number of Segments = {num_segments_current}, "
              f"Average Segment Size = {avg_segment_size_current:.2f}, "
              f"Number of Fibers = {len(fiber_lengths)}")
    
    timepoints = range(len(file_list))
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(timepoints, num_segments, marker='o', color='b')
    plt.title("Number of Segments Over Time")
    plt.xlabel("Timepoint")
    plt.ylabel("Number of Segments")
    
    plt.subplot(1, 2, 2)
    plt.plot(timepoints, avg_segment_sizes, marker='o', color='r')
    plt.title("Average Segment Size Over Time")
    plt.xlabel("Timepoint")
    plt.ylabel("Average Segment Size")
    
    plt.tight_layout()
    plt.show()

analyze_connected_components('AravindNewClassifier', min_size=10)