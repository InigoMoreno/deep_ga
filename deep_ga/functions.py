import numpy as np
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
