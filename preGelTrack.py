import time
import gc
import os

import numpy as np
import pandas as pd
import open3d as o3d

from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from sklearn.decomposition import PCA

import imageio
import cv2
from glob import glob

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes

class TrackingConfig:
    def __init__(self):

        # tracking parameters
        self.min_score = 0.4 # minimum score to be considered in tracking cost matrix
        self.max_gap = 2 # max gap in frames to considered dropped objects
        self.min_track_length = 0 # minimum number of frames tracked to count an object as tracked
        self.max_distance = 200.0 # maximum initial centroid voxel distance for tracking
        self.require_3d_overlap = False # old param don't use
        self.min_drop_size = 50  # minimum size for drop handling
        self.gel_point_frame = 40  # frame to start reducing max_distance
        self.max_distance_decay_rate = 0.985  # decay rate per frame
        
        # score component weights
        self.centroid_weight = 1
        self.size_weight = 0.3
        self.length_weight = 0.3
        self.orientation_weight = 0
        self.diameter_weight = 0.3

        # fiber recovery parameters
        self.min_avg_distance = 75 # dist to be considered when sampling drop points
        self.sample_rate = 0.5 # % of voxels to sample from dropped obj
        self.min_size_ratio = 0 # minimum size ratio for recovered track
        self.min_orient_sim = 0 # minimum dot for orientation

        # video visualization parameters
        self.num_visualize = 15000
        self.fps = 5
        self.output_path = "tracked_fibers.mp4"

        # plot visualization parameters
        self.time_window = 5
        self.component_grid_path = "temporal_component_grid.png"
        self.rolling_percentiles_path = "rolling_percentiles.png"

    def get_current_max_distance(self, current_frame):
        """Compute dynamic max distance based on frame number"""
        if current_frame <= self.gel_point_frame:
            return self.max_distance
        decay_frames = current_frame - self.gel_point_frame
        return self.max_distance * (self.max_distance_decay_rate ** decay_frames)

# main tracking function
def track_fibers(labeled_volumes, config=TrackingConfig()):
    print("\nInitiating tracking with optimizations...")

    precomputed = precompute_regions_and_overlaps(labeled_volumes, config)
    precomputed_regions = precomputed['regions']

    all_tracks = {}
    current_tracks = {}
    next_id = 1
    dropped_frames = []
    recovery_events = {}

    component_data = {
        'centroid': {'values': [], 'times': []},
        'size': {'values': [], 'times': []},
        'length': {'values': [], 'times': []},
        'orientation': {'values': [], 'times': []},
        'diameter': {'values': [], 'times': []},
        'total': {'values': [], 'times': []}
    }

    with tqdm(total=len(labeled_volumes), desc="Tracking progress") as pbar:
        for t in range(len(labeled_volumes)):
            current_max_distance = config.get_current_max_distance(t)
            regions = precomputed_regions[t]
            
            if t == 0:
                for region in regions:
                    all_tracks[next_id] = [(t, region)]
                    current_tracks[region['label']] = next_id
                    next_id += 1
            else:
                prev_regions = precomputed_regions[t-1]
                current_regions = precomputed_regions[t]

                if len(prev_regions) == 0:
                    new_tracks = {}
                    for curr_region in current_regions:
                        all_tracks[next_id] = [(t, curr_region)]
                        new_tracks[curr_region['label']] = next_id
                        next_id += 1
                    current_tracks = new_tracks
                    pbar.update(1)
                    pbar.set_postfix_str(f"Tracks: {len(all_tracks)}")
                    continue

                prev_centroids = np.vstack([r['centroid'] for r in prev_regions])
                prev_tree = cKDTree(prev_centroids)
                cost_matrix = np.zeros((len(prev_regions), len(current_regions)))

                with tqdm(total=len(current_regions), desc=f"Frame {t} cost matrix", leave=False) as matrix_pbar:
                    for j, curr_region in enumerate(current_regions):
                        distances, indices = prev_tree.query(
                            curr_region['centroid'], 
                            distance_upper_bound=current_max_distance
                        )
                        distances = np.atleast_1d(distances)
                        indices = np.atleast_1d(indices)
                        valid_pairs = [(i, d) for i, d in zip(indices, distances) if d <= current_max_distance]

                        for i, centroid_dist in valid_pairs:
                            prev_region = prev_regions[i]
                            centroid_score = 1 - (centroid_dist / current_max_distance)
                            size_ratio = min(prev_region['area'], curr_region['area']) / max(prev_region['area'], curr_region['area'])
                            len_prev = prev_region['end_to_end_length']
                            len_curr = curr_region['end_to_end_length']
                            length_sim = min(len_prev, len_curr)/max(len_prev, len_curr) if max(len_prev, len_curr) > 0 else 0
                            diam_prev = prev_region['diameter']
                            diam_curr = curr_region['diameter']
                            diam_sim = min(diam_prev, diam_curr)/max(diam_prev, diam_curr) if max(diam_prev, diam_curr) > 0 else 0
                            orient_prev = prev_region['orientation']
                            orient_curr = curr_region['orientation']
                            orient_sim = abs(np.dot(orient_prev, orient_curr)) if (np.linalg.norm(orient_prev)) > 0 and np.linalg.norm(orient_curr) > 0 else 0
                            #length_factor = ((len_prev + len_curr)/2) / max(len_prev, len_curr, 1e-6)
                            orientation_term = orient_sim# * length_factor
                            
                            score = (
                                centroid_score * config.centroid_weight +
                                size_ratio * config.size_weight +
                                length_sim * config.length_weight +
                                orientation_term * config.orientation_weight +
                                diam_sim * config.diameter_weight
                            )

                            if score >= config.min_score:
                                cost_matrix[i, j] = score

                            component_data['centroid']['values'].append(centroid_score)
                            component_data['centroid']['times'].append(t)
                            component_data['size']['values'].append(size_ratio)
                            component_data['size']['times'].append(t)
                            component_data['length']['values'].append(length_sim)
                            component_data['length']['times'].append(t)
                            component_data['orientation']['values'].append(orientation_term)
                            component_data['orientation']['times'].append(t)
                            component_data['diameter']['values'].append(diam_sim)
                            component_data['diameter']['times'].append(t)
                            component_data['total']['values'].append(score)
                            component_data['total']['times'].append(t)
                        matrix_pbar.update(1)

                row_ind, col_ind = linear_sum_assignment(-cost_matrix)
                matches = {}
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] > 0:
                        prev_label = prev_regions[i]['label']
                        curr_label = current_regions[j]['label']
                        if prev_label in current_tracks:
                            matches[curr_label] = current_tracks[prev_label]

                previous_track_map = current_tracks.copy()
                new_tracks = {}
                for j, curr_region in enumerate(current_regions):
                    curr_label = curr_region['label']
                    if curr_label in matches:
                        track_id = matches[curr_label]
                        all_tracks[track_id].append((t, curr_region))
                        new_tracks[curr_label] = track_id
                    else:
                        all_tracks[next_id] = [(t, curr_region)]
                        new_tracks[curr_label] = next_id
                        next_id += 1

                current_region_trees = {}
                for region in current_regions:
                    coords = region['coords']
                    if len(coords) == 0:
                        current_region_trees[region['label']] = None
                    else:
                        current_region_trees[region['label']] = cKDTree(coords)

                # fiber drop recovery logic
                lost_ids = set(previous_track_map.values()) - set(new_tracks.values())
                recovered_ids = set()

                for lost_id in lost_ids:
                    track_history = all_tracks[lost_id]
                    last_entry = track_history[-1]
                    prev_region = last_entry[1]
                    if prev_region['area'] < config.min_drop_size:
                        continue

                    coords = prev_region['coords']
                    sample_size = max(1, int(config.sample_rate * len(coords)))
                    sampled_coords = coords[np.random.choice(len(coords), sample_size, replace=False)]

                    candidates = []
                    for curr_region in current_regions:
                        centroid_dist = np.linalg.norm(curr_region['centroid'] - prev_region['centroid'])
                        if centroid_dist > current_max_distance:
                            continue

                        curr_tree = current_region_trees.get(curr_region['label'], None)
                        if curr_tree is None:
                            continue

                        distances, _ = curr_tree.query(sampled_coords)
                        avg_distance = np.mean(distances)
                        if avg_distance > config.min_avg_distance:
                            continue

                        size_ratio = min(prev_region['area'], curr_region['area']) / max(prev_region['area'], curr_region['area'])
                        if size_ratio < config.min_size_ratio:
                            continue

                        orient_prev = prev_region['orientation']
                        orient_curr = curr_region['orientation']
                        orient_sim = abs(np.dot(orient_prev, orient_curr)) if (np.linalg.norm(orient_prev) > 0 and np.linalg.norm(orient_curr) > 0) else 0
                        if orient_sim < config.min_orient_sim:
                            continue

                        candidates.append((curr_region, avg_distance))

                    if candidates:
                        candidates.sort(key=lambda x: x[1])
                        best_candidate = candidates[0][0]
                        curr_label = best_candidate['label']

                        if curr_label in new_tracks:
                            existing_track_id = new_tracks[curr_label]
                            if len(all_tracks[existing_track_id]) == 1 and all_tracks[existing_track_id][0][0] == t:
                                del all_tracks[existing_track_id]
                                new_tracks[curr_label] = lost_id
                                all_tracks[lost_id].append((t, best_candidate))
                                recovered_ids.add(lost_id)
                                recovery_events[t] = recovery_events.get(t, 0) + 1
                        else:
                            new_tracks[curr_label] = lost_id
                            all_tracks[lost_id].append((t, best_candidate))
                            recovered_ids.add(lost_id)
                            recovery_events[t] = recovery_events.get(t, 0) + 1

                for lost_id in lost_ids - recovered_ids:
                    dropped_frames.append(all_tracks[lost_id][-1][0])

                current_tracks = new_tracks

            pbar.update(1)
            pbar.set_postfix_str(f"Tracks: {len(all_tracks)}")

    # plot for drop events
    if dropped_frames:
        plt.figure(figsize=(10,6))
        plt.hist(dropped_frames, bins=np.arange(-0.5, len(labeled_volumes)+0.5, 1), 
                 edgecolor='black', density=True)
        plt.xlabel('Frame Number')
        plt.ylabel('Probability Density')
        plt.title('Distribution of Track Drop Events')
        plt.savefig('track_drop_distribution.png')
        plt.close()

    # plot for drop recovery events
    if recovery_events:
        frames = sorted(recovery_events.keys())
        counts = [recovery_events[frame] for frame in frames]
        plt.figure(figsize=(10,6))
        plt.bar(frames, counts, width=1.0, edgecolor='black')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Drop Recovery Events')
        plt.title('Drop Recovery Events Over Time')
        plt.savefig('drop_recovery_events.png')
        plt.close()

    for key in component_data:
        component_data[key]['values'] = np.array(component_data[key]['values'])
        component_data[key]['times'] = np.array(component_data[key]['times'])

    plot_temporal_component_distributions(component_data, config)
    plot_rolling_percentiles(component_data, config)

    return filter_tracks(all_tracks, config)

# precomputing functionality, runs before tracking to keep efficient
def precompute_regions_and_overlaps(labeled_volumes, config):

    print("Precomputing regions...")
    precomputed_regions = []

    for vol in tqdm(labeled_volumes, desc="Processing volumes"):
        regions = []
        for r in tqdm(regionprops(vol), desc="Analyzing regions", leave=False):
            coords = r.coords
            if len(coords) < 2:
                end_to_end_length = 0.0
                orientation = np.zeros(3)
                diameter = 0.0
            else:
                pca = PCA(n_components=3)
                try:
                    pca.fit(coords)
                except:
                    orientation = np.zeros(3)
                    end_to_end_length = 0.0
                    diameter = 0.0
                else:
                    # end to end length and orientation
                    main_axis = pca.components_[0]
                    projections = np.dot(coords, main_axis)
                    min_idx = np.argmin(projections)
                    max_idx = np.argmax(projections)
                    p1 = coords[min_idx]
                    p2 = coords[max_idx]
                    end_to_end_length = np.linalg.norm(p2 - p1)
                    orientation = (p2 - p1) / end_to_end_length if end_to_end_length > 0 else np.zeros(3)

                    # cross-sectional diameter
                    rotation_matrix = np.vstack([main_axis, np.random.randn(3), np.random.randn(3)]).T
                    q, _ = np.linalg.qr(rotation_matrix)
                    projected_coords = coords @ q[:, 1:]
                    
                    # convex hull of projected points
                    try:
                        hull = ConvexHull(projected_coords)
                        hull_points = projected_coords[hull.vertices]
                        pairwise_dists = np.sqrt(np.sum((hull_points[:, np.newaxis] - hull_points[np.newaxis, :])**2, axis=-1))
                        diameter = np.max(pairwise_dists)
                    except:
                        diameter = np.sqrt(r.area)

            regions.append({
                'label': r.label,
                'centroid': np.array(r.centroid),
                'area': r.area,
                'coords': r.coords,
                'end_to_end_length': end_to_end_length,
                'orientation': orientation,
                'diameter': diameter
            })
        precomputed_regions.append(regions)
        del vol
        gc.collect()

    return {'regions': precomputed_regions}

# function for filtering tracked objects based on frames tracked (see trackerconfig at top to set frame #)
def filter_tracks(all_tracks, config):
    """Filter tracks based on length and temporal consistency"""
    filtered = {}
    for track_id, history in all_tracks.items():
        times = [t for t, _ in history]
        if len(history) >= config.min_track_length:
            if len(times) == 1 or max(np.diff(times)) <= config.max_gap:
                filtered[track_id] = history
    print(f"Filtered {len(all_tracks) - len(filtered)} tracks")
    return filtered

# function to handle video visualization
def visualize_tracks(all_tracks, config=TrackingConfig()):
    """Original visualization code adapted for dictionary-based region storage"""
    print("\nPreparing visualization with parameters:")
    print(f"- Output FPS: {config.fps}")
    
    valid_tracks = [tid for tid, track in all_tracks.items() 
                    if len(track) >= config.min_track_length]
        
    print(f"- Visualizing {len(valid_tracks)} tracks")

    if not valid_tracks:
        print("No valid tracks to visualize!")
        return
    
    selected_tracks = valid_tracks
    
    max_time = max(t for track in all_tracks.values() for t, _ in track)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, visible=False)

    render_option = vis.get_render_option()
    render_option.point_size = 1.0
    
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    for t in tqdm(range(max_time + 1), desc="Rendering frames"):
        vis.clear_geometries()
        points = []
        colors = []
        
        for track_id in selected_tracks:
            track = all_tracks[track_id]
            for entry in track:
                if entry[0] == t:
                    region = entry[1]
                    pts = region['coords'][:, [2, 1, 0]]
                    points.append(pts)
                    color = plt.cm.tab10(track_id % 10)[:3]
                    colors.extend([color] * len(pts))
        
        if points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.vstack(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
            vis.add_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()
        image = (np.asarray(vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
        frame_filename = os.path.join(temp_dir, f"frame_{t:04d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    vis.destroy_window()
    
    print("\nCompiling video...")
    frame_paths = sorted(glob(os.path.join(temp_dir, "frame_*.png")))
    if not frame_paths:
        print("No frames were saved; video compilation aborted.")
        return
    
    first_frame = cv2.imread(frame_paths[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(config.output_path, fourcc, config.fps, (width, height))
    
    for frame_path in tqdm(frame_paths, desc="Encoding video"):
        frame = cv2.imread(frame_path)
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Video saved to {config.output_path}")
    
    for f in frame_paths:
        os.remove(f)
    os.rmdir(temp_dir)

# function for loading in original, segmented volumes (always run segment before this script)
def load_labeled_volumes(labeled_volumes_dir):
    """Load labeled volumes from TIFF files"""
    file_paths = sorted(glob(os.path.join(labeled_volumes_dir, '*.tif')))
    return [np.array(imageio.mimread(fp)) for fp in tqdm(file_paths, desc="Loading volumes")]

# function for plotting track duration distribution and calling video visualization
def analyze_tracks(labeled_volumes_dir, config=TrackingConfig()):
    """Main analysis workflow"""
    start_time = time.time()
    labeled_volumes = load_labeled_volumes(labeled_volumes_dir)
    if not labeled_volumes:
        print("No labeled volumes found!")
        return
    
    all_tracks = track_fibers(labeled_volumes, config)
    
    durations = [len(track) for track in all_tracks.values()]
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=np.arange(0.5, max(durations)+1.5, 1), edgecolor='black')
    plt.xlabel('Track Duration (frames)')
    plt.ylabel('Count')
    plt.title('Distribution of Track Durations')
    plt.savefig('track_durations.png')
    plt.close()
    
    if all_tracks:
        visualize_tracks(all_tracks, config)
    
    print(f"\nTotal analysis time: {(time.time() - start_time)/60:.1f} minutes")

# function for plotting score distribution over time
def plot_temporal_component_distributions(component_data, config):
    """Create 2D histograms showing component distributions over time"""
    plt.figure(figsize=(20, 15))
    
    if component_data['total']['times'].size > 0:
        max_time = component_data['total']['times'].max()
        time_bins = np.arange(0, max_time+2, config.time_window)
    else:
        print("No temporal data to plot")
        plt.close()
        return
    
    for idx, (name, data) in enumerate(component_data.items(), 1):
        if data['values'].size == 0:
            continue
            
        ax = plt.subplot(3, 2, idx)

        if name != "total":
            counts, xedges, yedges, im = ax.hist2d(
                data['times'], 
                data['values'],
                bins=[time_bins, np.linspace(0, 1, 25)],
                cmap='viridis',
                cmin=1
            )
        
        else:
            counts, xedges, yedges, im = ax.hist2d(
                data['times'], 
                data['values'],
                bins=[time_bins, np.linspace(0, 1.9, 25)],
                cmap='viridis',
                cmin=1
            )
        
        ax.set_xlabel('Time Window')
        ax.set_ylabel(name.capitalize())
        ax.set_title(f'{name.capitalize()} Score Distribution')
        plt.colorbar(im, ax=ax, label='Counts')
        
    plt.tight_layout()
    plt.savefig(config.component_grid_path)
    plt.close()

# function for plotting line graph of component scores over time
def plot_rolling_percentiles(component_data, config):
    """Show rolling window statistics for component scores"""
    plt.figure(figsize=(12, 8))
    
    for name, data in component_data.items():
        #if name == 'total': continue

        if data['values'].size == 0:
            continue
            
        df = pd.DataFrame({
            'time': data['times'],
            'value': data['values']
        })
        
        if df.empty:
            continue
            
        try:
            rolling = df.groupby('time')['value'].agg(
                median=lambda x: x.quantile(0.5),
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75)
            ).rolling(3, center=True).mean()
        except Exception as e:
            print(f"Error calculating percentiles for {name}: {e}")
            continue
            
        if rolling.empty:
            continue
            
        plt.plot(rolling.index, rolling['median'], label=name)
        plt.fill_between(rolling.index, rolling['q25'], rolling['q75'], alpha=0.2)
    
    plt.xlabel('Time')
    plt.ylabel('Score Value')
    plt.title('Rolling Window Percentiles (3-frame window)')
    plt.legend()
    plt.savefig(config.rolling_percentiles_path)
    plt.close()

# main script call, be sure to update to correct labeled_volumes directory
if __name__ == "__main__":
    config = TrackingConfig()
    analyze_tracks("20241024/labeled_volumes", config)
