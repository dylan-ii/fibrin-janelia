import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d, remove_small_objects
from skimage.io import imread
import matplotlib.colors as mcolors
from skimage.measure import label, regionprops
from skimage import measure
from scipy.spatial import cKDTree
import networkx as nx

def process_volume(file_path, min_size, skeletonize_min_size, angle_threshold=35):
    volume = imread(file_path)
    print(f"Loaded volume shape: {volume.shape}, unique values: {np.unique(volume)}")

    cleaned_volume = remove_small_objects(volume, min_size=min_size)
    print(f"Cleaned volume shape: {cleaned_volume.shape}, unique values: {np.unique(cleaned_volume)}")

    skeleton = skeletonize_3d(cleaned_volume)

    # had a function to filter, but didn't work
    # when re-implemented properly, this'll likely save loads of processing time
    filtered_skeleton = skeleton 

    graph = skeleton_to_graph(filtered_skeleton)

    branches = split_branches_by_angle(graph, filtered_skeleton, angle_threshold)

    labeled_volume = np.zeros_like(skeleton, dtype=np.int32)
    for i, branch in enumerate(branches, start=1):
        for node in branch:
            labeled_volume[node] = i

    unique_labels = np.unique(labeled_volume)
    print(f"Labeled volume shape: {labeled_volume.shape}, unique labels: {len(unique_labels) - 1}")

    skeleton_voxel_count = np.count_nonzero(skeleton)
    labeled_voxel_count = np.count_nonzero(labeled_volume)
    print(f"Skeleton voxel count: {skeleton_voxel_count}, Labeled voxel count: {labeled_voxel_count}")
    
    segment_sizes = np.bincount(labeled_volume.ravel())
    valid_labels = np.where(segment_sizes >= skeletonize_min_size)[0]
    valid_labels = valid_labels[valid_labels != 0]

    fiber_lengths = segment_sizes[valid_labels]
    print(f"Number of valid fibers: {len(fiber_lengths)}")

    return skeleton, labeled_volume, fiber_lengths

def filter_skeleton_by_size(skeleton, min_size):
    labeled_skeleton, num_labels = measure.label(skeleton, connectivity=3, return_num=True)
    props = regionprops(labeled_skeleton)
    valid_labels = {prop.label for prop in props if prop.area >= min_size}
    filtered_skeleton = np.isin(labeled_skeleton, list(valid_labels)).astype(np.bool_)
    return filtered_skeleton

def skeleton_to_graph(skeleton):
    nz = np.argwhere(skeleton)
    tree = cKDTree(nz)
    neighbors = tree.query_ball_point(nz, r=1.5)  # r=1.5 ensures adjacency

    G = nx.Graph()
    for idx, point in enumerate(nz):
        G.add_node(tuple(point))
        for neighbor_idx in neighbors[idx]:
            if neighbor_idx > idx:
                G.add_edge(tuple(point), tuple(nz[neighbor_idx]))
    return G

def split_branches_by_angle(graph, skeleton, angle_threshold):
    branches = []
    visited = set()

    def is_junction(node):
        return len(list(graph.neighbors(node))) > 2

    def compute_angle(v1, v2):
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    for node in graph.nodes:
        if node in visited or is_junction(node):
            continue

        branch = []
        queue = [node]
        while queue:
            current = queue.pop()
            if current in visited or is_junction(current):
                break

            branch.append(current)
            visited.add(current)

            neighbors = list(graph.neighbors(current))
            for neighbor in neighbors:
                if neighbor not in visited:
                    vector1 = np.array(current) - np.array(neighbor)
                    if branch:
                        vector2 = np.array(branch[-1]) - np.array(current)
                        if compute_angle(vector1, vector2) > angle_threshold:
                            break
                    queue.append(neighbor)
        if branch:
            branches.append(branch)

    return branches

def visualize_results(skeleton, labeled_volume, fiber_lengths, timepoint):
    # plot for labeled volume (max projection)
    plt.figure(figsize=(8, 6))
    plt.imshow(labeled_volume.max(axis=0), cmap='tab20', interpolation='nearest')
    plt.title(f"Labeled Components (Timepoint {timepoint})")
    plt.axis('off')
    plt.colorbar(label="Label")
    plt.show()
    
    # plot for labeled volume (y-slice) with expanded colors
    mid_slice = labeled_volume.shape[1] // 2  # Middle y-slice
    slice_data = labeled_volume[:, mid_slice, :]  # Extract y-slice
    unique_labels = np.unique(slice_data)
    num_labels = len(unique_labels)
    
    colors = plt.cm.viridis(np.linspace(0, 1, num_labels))
    cmap = mcolors.ListedColormap(colors, name='expanded_cmap')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(slice_data, cmap=cmap, interpolation='nearest')
    plt.title(f"Labeled Volume Y-Slice (Timepoint {timepoint}, Slice {mid_slice})")
    plt.axis('off')
    plt.colorbar(label="Label")
    plt.show()
    
    # plot for skeleton (max projection)
    plt.figure(figsize=(8, 6))
    plt.imshow(skeleton.max(axis=0), cmap='gray', interpolation='nearest')
    plt.title(f"Skeletonized Volume (Timepoint {timepoint})")
    plt.axis('off')
    plt.show()
    
    # plot distribution of fiber lengths
    plt.figure(figsize=(8, 6))
    plt.hist(fiber_lengths, bins=20, color='skyblue', edgecolor='black')
    max_fiber_length = max(fiber_lengths) if fiber_lengths.size > 0 else 0
    plt.axvline(x=max_fiber_length, color='red', linestyle='dashed', linewidth=2)
    plt.text(max_fiber_length, 1, f'Max: {max_fiber_length}', color='red', fontsize=12)
    plt.title(f"Fiber Length Distribution (Timepoint {timepoint})")
    plt.xlabel("Fiber Length (voxels)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def analyze_time_series(folder_path, min_size, skeletonize_min_size):
    file_list = sorted(
        [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.tif', '.tiff'))]
    )
    if not file_list:
        print("No image files found in the specified folder.")
        return
    
    num_segments = []
    avg_segment_sizes = []
    all_fiber_lengths = []

    for i, file_path in enumerate(file_list):
        if i%25 != 0 and i != 127 : continue

        print(f"Processing timepoint {i+1}/{len(file_list)}...")
        skeleton, labeled_volume, fiber_lengths = process_volume(file_path, min_size, skeletonize_min_size)
        
        num_segments.append(len(fiber_lengths))
        avg_segment_sizes.append(np.mean(fiber_lengths) if fiber_lengths.size > 0 else 0)
        all_fiber_lengths.append(fiber_lengths)
        
        # plots for each time point here
        #visualize_results(skeleton, labeled_volume, fiber_lengths, i+1)
    
    timepoints = range(len(num_segments))
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

analyze_time_series('AravindNewClassifier', min_size=800, skeletonize_min_size=17)