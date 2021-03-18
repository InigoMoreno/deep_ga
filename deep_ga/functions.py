import numpy as np
import cv2
from plyfile import PlyData, PlyElement


def ply_to_image(ply_file, resolution=1):
    """Get dem and image out of a point cloud

    Args:
        ply_file (string): Path to a .ply file containing the point cloud
        resolution (int, optional): Resolution of the DEM and image. Defaults to 1.

    Returns:
        (tuple): tuple containing
            dem (numpy array): Depth Elevation Map
            img (numpy array): RGB-image containing diffuse values
            displacement (tuple): x,y,z displacement used to generate dem and image
    """
    points = PlyData.read(ply_file).elements[0]
    x = points.data['x']
    y = points.data['y']
    z = points.data['z']
    r = points.data['diffuse_red']
    g = points.data['diffuse_green']
    b = points.data['diffuse_blue']

    n_points = points.count
    displacement = (np.min(x, 0), np.min(y, 0), np.min(z, 0))
    x -= np.min(x, 0)
    y -= np.min(y, 0)
    z -= np.min(z, 0)

    bins = (np.arange(0, np.max(x), step=resolution),
            np.arange(0, np.max(y), step=resolution))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sum_heights = np.histogram2d(x, y, bins=bins, weights=z)[0]
        sum_reds = np.histogram2d(x, y, bins=bins, weights=r)[0]
        sum_greens = np.histogram2d(x, y, bins=bins, weights=g)[0]
        sum_blues = np.histogram2d(x, y, bins=bins, weights=b)[0]
        count_points = np.histogram2d(x, y, bins=bins)[0]

    dem = sum_heights/count_points
    img = np.dstack((sum_reds / count_points, sum_greens /
                    count_points, sum_blues / count_points))

    return (dem, img, displacement)


def get_patch(dem, x, y, p):
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
        dem, (p["mapLengthPixels"], p["mapLengthPixels"]), (x, y))

    # discard patch if there are two many holes
    nanPercentage = np.count_nonzero(np.isnan(patch))/p["mapLengthPixels"]**2
    if nanPercentage > p["maxNanPercentage"]:
        return None

    # inpaint holes and move to zero
    patch = cv2.inpaint(patch, np.uint8(np.isnan(patch)),
                        inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    patch -= np.min(patch)

    # compute slope
    slopeX = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)/p["resolution"]
    slopeY = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)/p["resolution"]
    slope = cv2.magnitude(slopeX, slopeY)

    # discard patch if there are not enough slopes
    _, slopeThresh = cv2.threshold(
        slope, p["minSlopeThreshold"], 0, cv2.THRESH_TOZERO)
    slopePercentage = np.count_nonzero(slopeThresh)/p["mapLengthPixels"]**2
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

    cov = (p["stdEvaluationFunction"]/p["resolution"])**2

    for i in range(batch_size):
        # find first patch
        patch_a = None
        while patch_a is None:
        xa = random.uniform(0, dem.shape[0])
        ya = random.uniform(0, dem.shape[1])
        patch_a = get_patch(dem, xa, ya, p)
        patches_a[i, :, :] = patch_a

        # find second patch close to first
        patch_b = None
        while patch_b is None:
        xb = xa + random.normal(scale=p["stdPatchShift"]/p["resolution"])
        yb = ya + random.normal(scale=p["stdPatchShift"]/p["resolution"])
        patch_b = get_patch(dem, xb, yb, p)
        patches_b[i, :, :] = patch_b

        # compute output function
        distances[i] = np.linalg.norm([xb-xa, yb-ya])

    return (patches_a, patches_b, distances)
