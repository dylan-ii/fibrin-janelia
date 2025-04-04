import numpy as np
import random
from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter

# declare fiber class which is main building block of sim
class Fiber:
    def __init__(self, center, theta, phi, length, radius, color, growth_rate_l, growth_rate_r, 
                 max_length=30.0, max_radius=0.2):
        # class inits with vec centerpos, theta (xy orientation), phi (azimuthal), length, radius
        # these values will then be varied for the object by independent functions at each time step
        self.center = np.array(center, dtype=np.float64)
        self.theta = theta 
        self.phi = phi
        self.length = length
        self.radius = radius
        self.color = color
        self.active = True
        self.growth_rate_l = growth_rate_l
        self.growth_rate_r = growth_rate_r
        self.max_length = max_length
        self.max_radius = max_radius
        self.junction = None
        self.junction_type = None
        self.free_endpoints = None

    # varies radius and length in order for fiber to grow through sim
    def grow(self):
        if self.active:
            # case where fiber has already merged, handling here attempts to only grow along endpoints which aren't in the merger (although this is totally broken at present)
            if self.free_endpoints is not None:
                ep1, ep2 = self.free_endpoints
                new_length = min(np.linalg.norm(ep1 - ep2) + self.growth_rate_l, self.max_length)
                self.length = new_length
                self.radius = min(self.radius + self.growth_rate_r, self.max_radius)
                self.center = (ep1 + ep2) / 2.0
            # conventional growth pre-merge with values clamped to declared maxlength/radius
            else:
                self.length = min(self.length + self.growth_rate_l, self.max_length)
                self.radius = min(self.radius + self.growth_rate_r, self.max_radius)

    # drifts cenpos of fiber, according to random amounts whose magnitude is inversely related to fiber length (in order to slow drift down as the sim goes on)
    # scalar values here are completely arbitrary and were played with until it looked good
    def drift(self, time_step):
        drift_factor = 1 / (self.length/4)
        drift = np.random.normal(scale=5 * drift_factor, size=3)
        self.center += drift

    # similarly drift xy and azimuthal angles random amounts, with a factor that is also inversely related to fiber length in order to slow as sim goes on
    def rotate(self):
        angle_factor = 1.0 / (self.length + 1.0)
        delta_theta = np.random.normal(scale=0.5 * angle_factor)
        delta_phi = np.random.normal(scale=0.5 * angle_factor)
        self.theta = (self.theta + delta_theta) % (2 * np.pi)
        self.phi = (self.phi + delta_phi) % (2 * np.pi)

    # helper function which just uses trig to determine endpoints for plotting based on fiber values
    def get_endpoints(self):
        direction = np.array([
            np.sin(self.theta) * np.cos(self.phi),
            np.sin(self.theta) * np.sin(self.phi),
            np.cos(self.theta)
        ])
        half_length = self.length / 2.0
        return (self.center + direction * half_length,
                self.center - direction * half_length)

# use union find (aka disjoint-set) structure to analyze connected components to see when fibers' voxels overlap
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def union(self, i, j):
        ri = self.find(i)
        rj = self.find(j)
        if ri != rj:
            self.parent[rj] = ri

# class which lets us translate fiber objects into voxel grids for plotting and collision detection
# separating space into discrete voxel grid assists with computation speed and makes collision detection much easier
class VoxelGrid:
    def __init__(self, voxel_size=2.0):
        self.voxel_size = voxel_size
        self.grid = defaultdict(list)
    def get_voxel_index(self, point):
        return tuple((point // self.voxel_size).astype(int))
    def add_fiber(self, fiber):
        e1, e2 = fiber.get_endpoints()
        num_samples = max(2, int(np.linalg.norm(e1 - e2) // self.voxel_size))
        samples = np.linspace(e1, e2, num_samples)
        radius = fiber.radius
        voxels = set()
        for pt in samples:
            min_vox = self.get_voxel_index(pt - radius)
            max_vox = self.get_voxel_index(pt + radius)
            for x in range(min_vox[0], max_vox[0] + 1):
                for y in range(min_vox[1], max_vox[1] + 1):
                    for z in range(min_vox[2], max_vox[2] + 1):
                        voxels.add((x, y, z))
        for vox in voxels:
            self.grid[vox].append(fiber)

# helper function which determines average of closest two points between segments given by endpoints (p0, p1) and (q0, q1)
# uses lin algebra approach found here https://math.stackexchange.com/questions/846054/closest-points-on-two-line-segments
def closest_point_between_segments(p0, p1, q0, q1):
    u = p1 - p0
    v = q1 - q0
    w0 = p0 - q0
    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w0)
    e = np.dot(v, w0)
    D = a * c - b * b
    sc, tc = 0.0, 0.0

    SMALL_NUM = 1e-8
    if D < SMALL_NUM:
        sc = 0.0
        tc = e / c if c > SMALL_NUM else 0.0
    else:
        sc = (b * e - c * d) / D
        tc = (a * e - b * d) / D
    sc = np.clip(sc, 0.0, 1.0)
    tc = np.clip(tc, 0.0, 1.0)
    closest_p = p0 + sc * u
    closest_q = q0 + tc * v
    return (closest_p + closest_q) / 2.0

# handles fibers after merger detection, categorizes their endpoints as free or near the junction point in order to label them "X","L" or "T"
def merge_fibers(fiber_group, step):
    junction_points = []
    endpoints = []
    for fiber in fiber_group:
        e1, e2 = fiber.get_endpoints()
        endpoints.extend([e1, e2])
        
    n = len(fiber_group)
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = fiber_group[i].get_endpoints()
            q1, q2 = fiber_group[j].get_endpoints()
            junction_points.append(closest_point_between_segments(p1, p2, q1, q2))
    if junction_points:
        junction_avg = np.mean(junction_points, axis=0)
    else:
        junction_avg = np.mean([f.center for f in fiber_group], axis=0)

    # set an arbitrary value for distance that is scaled by fiber length to determine the threshold for what is a 'free' endpoint
    attach_thresh = 0.2 * np.mean([f.length for f in fiber_group])
    free_eps = []
    for ep in endpoints:
        if np.linalg.norm(ep - junction_avg) > attach_thresh:
            free_eps.append(ep)
    
    if len(free_eps) >= 2:
        free_eps = np.array(free_eps)
        median_val = np.median(free_eps[:, 0])
        group1 = free_eps[free_eps[:, 0] <= median_val]
        group2 = free_eps[free_eps[:, 0] > median_val]
        if len(group1) == 0 or len(group2) == 0:
            group1 = free_eps[:len(free_eps)//2]
            group2 = free_eps[len(free_eps)//2:]
        ep1 = np.mean(group1, axis=0)
        ep2 = np.mean(group2, axis=0)
    else:
        ep1, ep2 = fiber_group[0].get_endpoints()
        
    # very very heuristic method of determing junction type where I simply look at the endpoints of each segment, and find how many of them are "free"
    # as in not near the closest point of the two segments. this cannot tell I junctions apart from L and likely overestimates number of X. 

    num_free = len(free_eps)
    if num_free == 2 or num_free == 1:
        junction_type = "L"
    elif num_free == 3:
        junction_type = "T"
    elif num_free >= 4:
        junction_type = "X"
    else:
        junction_type = "Unknown"

    new_color = np.mean([f.color for f in fiber_group], axis=0)
    
    # previously set this up to elminate one of the two fibers after merging, but now just keeps things
    # important to note that because of this, it *will* double (more like 50) count junctions which stay over time
    # issue can be easily fixed by just deactivating one of the fibers or checking if a junction has formed in that position before
    # also note that because it doesn't pause all fiber movement at the moment, a T junction could form for a frame and then the fibers move, and many more X junctions would form in the sim
    # in that case the fix is also pretty simple, but just note that raw counts of junctions from this code as it's written are incorrect for this reason
    for fiber in fiber_group:
        #fiber.active = False # can simply be uncommented to kill fibers after merging. note that this also stops the scatter which labels "X","T", or "L" (which I need to fix)
        fiber.junction = junction_avg
        fiber.junction_type = junction_type
        fiber.free_endpoints = (ep1, ep2)
        fiber.color = tuple(new_color)
    
    print(f"Step {step}: Bound {n} fibers together; junction type: {junction_type}")

def simulate_and_animate(num_fibers, num_steps, voxel_size=3.0):
    fibers = []
    colors = [plt.cm.jet(i/num_fibers) for i in range(num_fibers)]
    for i in range(num_fibers):
        # init fiber params
        center = np.random.normal(0, 45, 3)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)

        # init fiber w rates, Fiber class structure: Fiber(center, theta, phi, length, radius, color, lengthGrowthRate, radiusGrowthRate)
        fiber = Fiber(center, theta, phi, 0.5, 0.07, colors[i], 0.7, 0.01)
        fibers.append(fiber)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(-75, 75)
    ax.set_ylim(-75, 75)
    ax.set_zlim(-75, 75)

    # seemingly no amount of extra args to ffmpeg prevents runtime broker from keeping the video open and annoyingly refusing to close between runs..
    writer = FFMpegWriter(fps=5, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    writer.fig = fig

    with writer.saving(fig, 'simulation.mp4', dpi=100):
        for step in range(num_steps):
            print(f"Processing step {step+1}/{num_steps}")
            voxel_grid = VoxelGrid(voxel_size)
            
            for fiber in fibers:
                fiber.junction = None

            for fiber in fibers:
                if fiber.active:
                    fiber.drift(step)
                    fiber.rotate()
                    fiber.grow()
                    voxel_grid.add_fiber(fiber)
            
            # fiber collision detection via our unionfind class
            n = len(fibers)
            uf = UnionFind(n)
            fiber_to_index = {fiber: i for i, fiber in enumerate(fibers)}
            for voxel, occupants in voxel_grid.grid.items():
                if len(occupants) >= 2:
                    indices = [fiber_to_index[f] for f in occupants if f.active]
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            uf.union(indices[i], indices[j])
            groups = defaultdict(list)
            for i, fiber in enumerate(fibers):
                if fiber.active:
                    groups[uf.find(i)].append(fiber)
            for group in groups.values():
                if len(group) >= 2:
                    merge_fibers(group, step)

            # simple plotting from here on out
            ax.clear()
            ax.set_xlim(-75, 75)
            ax.set_ylim(-75, 75)
            ax.set_zlim(-75, 75)
            for fiber in fibers:
                if not fiber.active:
                    continue
                e1, e2 = fiber.get_endpoints()
                x, y, z = zip(*[e1, e2])
                ax.plot(x, y, z, color=fiber.color, linewidth=fiber.radius*2)
                if fiber.junction is not None:
                    ax.scatter([fiber.junction[0]], [fiber.junction[1]], [fiber.junction[2]],
                               color=fiber.color, s=5, marker='o')
                    ax.text(fiber.junction[0], fiber.junction[1], fiber.junction[2],
                            fiber.junction_type, color='k', fontsize=8)
            writer.grab_frame()

        writer.finish()
    plt.close(fig)
    del writer
    print("Simulation complete. Video saved.")

simulate_and_animate(num_fibers=200, num_steps=100)
