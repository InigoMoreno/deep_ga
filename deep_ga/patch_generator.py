import cv2
from tensorflow import keras
import numpy as np
import math
from numba import cuda


@cuda.jit
def gpu_occlusion(patch, height):
    W, H = patch.shape

    cx = round((W - 1) / 2)
    cy = round((H - 1) / 2)
    c = (cx, cy, patch[cx, cy] + height)

    x, y = cuda.grid(2)
    if x < W and y < H:
        p = (x, y, patch[x, y])
        to = (c[0] - p[0], c[1] - p[1], c[2] - p[2])
        dist = math.sqrt(math.pow(to[0], 2) +
                         math.pow(to[1], 2) + math.pow(to[1], 2))
        dir = (to[0] / dist, to[1] / dist, to[2] / dist)
        for h in range(0, math.ceil(dist)):
            step_pos_ray = (p[0] + h * dir[0], p[1] +
                            h * dir[1], p[2] + h * dir[2])
            sx, sy = round(step_pos_ray[0]), round(step_pos_ray[1])
            if (sx, sy) == (x, y):
                continue
            if step_pos_ray[2] <= patch[sx, sy]:
                patch[x, y] = np.nan
                break


def raycast_occlusion(arr, height):
    an_array = cuda.to_device(arr)
    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    gpu_occlusion[blockspergrid, threadsperblock](an_array, height)
    return np.array(an_array)


def get_patch(dem, x, y, p, displacement=(0, 0), skipChecks=False):
    """Get patch from an elevation map

    Args:
        dem (numpy array): Elevation Map
        x (double): position in x
        y (double): position in y
        p (dict): parameters

    Returns:
        (numpy array): Patch of the elevation map
    """
    # compute patch
    patch = cv2.getRectSubPix(
        dem, (p["mapLengthPixels"], p["mapLengthPixels"]),
        ((-displacement[1] + y) / p["resolution"],
         (-displacement[0] + x) / p["resolution"]))

    if skipChecks:
        return patch

    # discard patch if there are two many holes
    nanPercentage = np.count_nonzero(np.isnan(patch)) / p["mapLengthPixels"]**2
    if nanPercentage > p["maxNanPercentage"]:
        return None

    # inpaint holes and move to zero
    patch = cv2.inpaint(patch, np.uint8(np.isnan(patch)),
                        inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    patch -= np.nanmin(patch)

    # compute slope
    slopeX = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3) / p["resolution"]
    slopeY = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3) / p["resolution"]
    slope = cv2.magnitude(slopeX, slopeY)

    # discard patch if there are not enough slopes
    _, slopeThresh = cv2.threshold(
        slope, p["minSlopeThreshold"], 0, cv2.THRESH_TOZERO)
    slopePercentage = np.count_nonzero(slopeThresh) / p["mapLengthPixels"]**2
    if slopePercentage < p["minSlopePercentage"]:
        return None
    if slopePercentage > p["maxSlopePercentage"]:
        return None

    return patch


def get_batch(batch_size, dem, p, seed=None):
    """Get a batch of patches for training

    Args:
        batch_size (int): Size of the batch
        dem (numpy array): elevation map
        p (dict): parameters
        seed (int, optional): Seed to use for randomness. Defaults to None.

    Returns:
        (tuple): tuple containing
            patches_a (numpy array): left patches of the batch
            patches_b (numpy array): right patches of the batch
            distances (double): distances between left and right patches
    """
    # initialize random
    random = np.random.RandomState(seed)

    # initialize data
    patches_a = np.empty(
        (batch_size, p["mapLengthPixels"], p["mapLengthPixels"]))
    patches_b = np.empty(
        (batch_size, p["mapLengthPixels"], p["mapLengthPixels"]))
    distances = np.empty((batch_size,))

    for i in range(batch_size):
        # find first patch
        patch_a = None
        while patch_a is None:
            xa = random.uniform(0, dem.shape[0])
            ya = random.uniform(0, dem.shape[1])
            patch_a = get_patch(dem, xa, ya, p)
        if "augment_a" in p.keys() and p["augment_a"] is not None:
            patch_a = p["augment_a"].augment_image(patch_a)
        if "raycastHeight" in p.keys() and p["raycastHeight"] is not None:
            patch_a = raycast_occlusion(patch_a, p["raycastHeight"])
        patches_a[i, :, :] = patch_a - np.nanmean(patch_a)

        # find second patch close to first
        patch_b = None
        while patch_b is None:
            if p["booleanDist"]:
                if random.choice([True, False]):
                    dx, dy = 0, 0
                else:
                    angle = random.uniform(0, 2 * math.pi)
                    dx = p["stdPatchShift"] * math.cos(angle)
                    dy = p["stdPatchShift"] * math.sin(angle)
            else:
                dx, dy = random.normal(
                    scale=p["stdPatchShift"] / p["resolution"], size=2)
            patch_b = get_patch(dem, xa + dx, ya + dy, p)
        if "augment_b" in p.keys() and p["augment_b"] is not None:
            patch_b = p["augment_b"].augment_image(patch_b)
        patches_b[i, :, :] = patch_b - np.nanmean(patch_b)

        # compute output function
        distances[i] = np.linalg.norm([dx, dy])

    return (patches_a, patches_b, distances)


def get_batch_local_global(batch_size, dems, gps, global_dem, displacement, p, seed=None):
    """Get a batch of local and global patches for training

    Args:
        batch_size (int): Size of the batch
        dems (numpy array): local elevation maps
        gps (numpy array): position information of local elevation maps
        global_dem (numpy array): global elevation map
        displacement (tuple): tuple containing displacement between gps data and dem 
        params (dict): parameters
        seed (int, optional): Seed to use for randomness. Defaults to None.

    Returns:
        (tuple): tuple containing
            patches_local (numpy array): local patches of the batch
            patches_global (numpy array): global patches of the batch
            distances (double): distances between local and global patches
    """
    # initialize random
    random = np.random.RandomState(seed)

    # initialize data
    local_patches = np.empty(
        (batch_size, p["local_mapLengthPixels"], p["local_mapLengthPixels"]))
    global_patches = np.empty(
        (batch_size, p["mapLengthPixels"], p["mapLengthPixels"]))
    distances = np.empty((batch_size,))

    for i in range(batch_size):
        # find local patch
        # we asume the dems have already been filtered
        gt_global_patch = None
        while gt_global_patch is None:
            idx = random.randint(dems.shape[0])
            local_patch = dems[idx, :, :]
            local_patch -= np.nanmin(local_patch)
            local_patches[i, :, :] = local_patch
            gt_global_patch = get_patch(
                global_dem, gps[idx, 1], gps[idx, 2], p, displacement)

        # find global patch close to local
        global_patch = None
        while global_patch is None:

            if p["booleanDist"]:
                if random.choice([True, False]):
                    dx, dy = 0, 0
                else:
                    angle = random.uniform(0, 2 * math.pi)
                    dx = p["stdPatchShift"] * math.cos(angle)
                    dy = p["stdPatchShift"] * math.sin(angle)
            else:
                dx, dy = random.normal(
                    scale=p["stdPatchShift"] / p["resolution"], size=2)
            global_patch = get_patch(
                global_dem, gps[idx, 1] + dx, gps[idx, 2] + dy, p, displacement)
        if "augment_b" in p.keys() and p["augment_b"] is not None:
            global_patch = p["augment_b"].augment_image(global_patch)
        global_patches[i, :, :] = global_patch - np.nanmean(global_patch)

        # compute output function
        distances[i] = np.linalg.norm([dx, dy])

    if "augment_a" in p.keys() and p["augment_a"] is not None:
        local_patches = p["augment_a"].augment_batch_(
            local_patches.astype("float32"))
    return (local_patches, global_patches, distances)


class PatchDataGenerator(keras.utils.Sequence):
    """Generates patch data for Keras"""

    def __init__(self, size, dem, batch_size, params):
        """Initialize

        Args:
            size (int): Total size of data (data per epoch)
            dem (numpy array): Depth elevation map
            batch_size (int): Size of each batch
            params (dict): parameters
        """
        self.size = size
        self.dem = dem
        self.batch_size = batch_size
        self.params = params

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.size / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        Xa, Xb, y = get_batch(self.batch_size, self.dem, self.params)
        return [Xa, Xb], y


class LocalGlobalPatchDataGenerator(keras.utils.Sequence):
    """Generates patch data for Keras"""

    def __init__(self, size, batch_size, dems, gps, global_dem, displacement, params):
        """Initialize

        Args:
            size (int): Total size of data (data per epoch)
            batch_size (int): Size of each batch
            dems (numpy array): local elevation maps
            gps (numpy array): position information of local elevation maps
            global_dem (numpy array): global elevation map
            displacement (tuple): tuple containing displacement between gps data and dem 
            params (dict): parameters
        """
        self.size = size
        self.dems = dems
        self.gps = gps
        self.global_dem = global_dem
        self.displacement = displacement
        self.batch_size = batch_size
        self.params = params

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.size / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        Xa, Xb, y = get_batch_local_global(
            self.batch_size, self.dems, self.gps, self.global_dem, self.displacement, self.params)
        return [Xa, Xb], y
