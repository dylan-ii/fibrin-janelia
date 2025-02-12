import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_erosion

NUM_EROSIONS = 1 # keep to 1

# ellipsoid element params
RADIUS_X = 3 
RADIUS_Y = 3
RADIUS_Z = 0.85

# functionality for ellipsoid structuring element
def create_ellipsoid_strel(radius_x, radius_y, radius_z):
    x, y, z = np.ogrid[-radius_x:radius_x+1, -radius_y:radius_y+1, -radius_z:radius_z+1]
    strel = (x**2 / radius_x**2 + y**2 / radius_y**2 + z**2 / radius_z**2) <= 1
    return strel.astype(int)

STRUCTURING_ELEMENT = create_ellipsoid_strel(RADIUS_X, RADIUS_Y, RADIUS_Z)

# function for loading and removing small objects from volume
def load_and_preprocess_volume(file_path, min_size):
    volume = imread(file_path)
    
    for _ in range(NUM_EROSIONS):
        eroded_volume = binary_erosion(volume, structure=STRUCTURING_ELEMENT)
        volume[volume != eroded_volume] = 0
    
    cleaned_volume = remove_small_objects(volume, min_size=min_size)
    return cleaned_volume

# main function for segmentation
def segment_fibers(volume, min_size=0):
    labeled_volume = label(volume, connectivity=3)
    if min_size > 0:
        labeled_volume = remove_small_objects(labeled_volume, min_size=min_size)
    return labeled_volume

# function for calculations of fiber properties
def calculate_fiber_properties(labeled_volume):
    props = regionprops(labeled_volume)
    fiber_data = []
    print(len(props))
    for prop in props:
        voxel_count = prop.area
        coords = prop.coords
        end_to_end_length = np.linalg.norm(coords[0] - coords[-1])
        fiber_data.append({
            'label': prop.label,
            'voxel_count': voxel_count,
            'end_to_end_length': end_to_end_length
        })
    print("Fiber properties calculated.")
    return fiber_data

# function for 3d visualizations of labeled fibers
def plot_3d_fibers(labeled_volume, fiber_data, title="3D Fiber Visualization"):
    print("Plotting 3D fibers...")
    
    if not fiber_data:
        print("No fibers to plot.")
        return
    
    selected_fibers = np.random.choice([f['label'] for f in fiber_data], min(10000, len(fiber_data)), replace=False)
    
    binary_volume = np.isin(labeled_volume, selected_fibers)
    z, y, x = np.nonzero(binary_volume)
    points = np.vstack((x, y, z)).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([1, 1, 1])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=600)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 5.0
    
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

# function for determining gel ratio
# currently set to never return True, which would stop segmentation at the time gel is determined
def identify_gel_point(fiber_data):
    voxel_counts = sorted([fiber['voxel_count'] for fiber in fiber_data], reverse=True)
    if len(voxel_counts) < 2:
        return False, 1
    gel_ratio = voxel_counts[1] / voxel_counts[0]
    print(gel_ratio)
    #return voxel_counts[0] > 5 * voxel_counts[1], gel_ratio
    return False, gel_ratio

# main function for processing and visualization
def analyze_time_series(folder_path, save_path, min_size):
    file_list = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.tif', '.tiff'))])
    if not file_list: return
    
    save_dir = os.path.join(save_path, 'labeled_volumes')
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = {'num_fibers': [], 'avg_voxels': [], 'avg_lengths': [], 'gel_ratios': []}
    
    for i, file_path in enumerate(file_list):
        print(f"Processing timepoint {i+1}/{len(file_list)}...")
        volume = load_and_preprocess_volume(file_path, min_size)
        labeled_volume = segment_fibers(volume, min_size)
        imsave(os.path.join(save_dir, f'labeled_volume_{i:04d}.tif'), labeled_volume.astype(np.uint16))
        
        fiber_data = calculate_fiber_properties(labeled_volume)
        metrics['num_fibers'].append(len(fiber_data))
        metrics['avg_voxels'].append(np.mean([f['voxel_count'] for f in fiber_data]) if fiber_data else 0)
        metrics['avg_lengths'].append(np.mean([f['end_to_end_length'] for f in fiber_data]) if fiber_data else 0)
        _, gel_ratio = identify_gel_point(fiber_data)
        metrics['gel_ratios'].append(gel_ratio)
        
        #plot_3d_fibers(labeled_volume, fiber_data)
        if metrics['gel_ratios'][-1] <= 0.0001:  # Example threshold
            print(f"Gelation detected at timepoint {i+1}!")
            break
    
    # plot results
    timepoints = np.arange(1, len(metrics['num_fibers']) + 1)
    
    plt.figure()
    plt.plot(timepoints, metrics['num_fibers'], marker='o', linestyle='-')
    plt.xlabel("Timepoint")
    plt.ylabel("Number of Labels")
    plt.title("Number of Labels Over Time")
    plt.show()
    
    plt.figure()
    plt.plot(timepoints, metrics['avg_lengths'], marker='o', linestyle='-', color='r')
    plt.xlabel("Timepoint")
    plt.ylabel("Average End-to-End Length")
    plt.title("Average End-to-End Length Over Time")
    plt.show()
    
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Timepoint")
    ax1.set_ylabel("Avg Voxel Count", color='b')
    ax1.plot(timepoints, metrics['avg_voxels'], marker='o', linestyle='-', color='b', label="Avg Voxel Count")
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Gel Ratio", color='g')
    ax2.plot(timepoints, metrics['gel_ratios'], marker='s', linestyle='--', color='g', label="Gel Ratio")
    ax2.tick_params(axis='y', labelcolor='g')
    
    plt.title("Avg Voxel Count and Gel Ratio Over Time")
    fig.tight_layout()
    plt.show()

# main script call, be sure to update directory to folder where your classified tifs are
# min_size has appropriate values between 125-500, with more noise being closer to lower values
# advisable to stay nearer to 500 min size for computational speed in tracking, but once things are finalized,
# may be more reasonable to go lower size and let some more noise slip in
analyze_time_series('20241024/AravindClass', "20241024", min_size=500)