import cv2
from tensorflow import keras
import numpy as np


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
