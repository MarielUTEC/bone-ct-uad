import numpy as np


def inject_sphere(volume, center=None, radius=8, intensity=2500):
    """
    Injects a synthetic high-density spherical foreign object into a 3D CT volume.

    Parameters
    ----------
    volume : np.ndarray
        3D CT volume.
    center : tuple or None
        Center of the sphere as (z, y, x). If None, the center of the volume is used.
    radius : int
        Radius of the synthetic object in voxels.
    intensity : float
        HU-like intensity assigned to the synthetic object.

    Returns
    -------
    corrupted : np.ndarray
        Volume with injected synthetic object.
    mask : np.ndarray
        Binary mask of the injected object.
    """
    corrupted = volume.copy()
    mask = np.zeros_like(volume, dtype=np.uint8)

    z_dim, y_dim, x_dim = volume.shape

    if center is None:
        center = (z_dim // 2, y_dim // 2, x_dim // 2)

    cz, cy, cx = center

    zz, yy, xx = np.ogrid[:z_dim, :y_dim, :x_dim]
    sphere = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

    corrupted[sphere] = intensity
    mask[sphere] = 1

    return corrupted, mask