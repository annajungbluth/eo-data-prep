import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import tempfile
import zipfile
from satpy import Scene
import pathlib

import warnings
warnings.simplefilter("ignore")

def random_date(start, end):
    """
    Generate a random datetime between two datetime objects.
    """
    delta = end - start
    random_days = np.random.randint(0, delta.days + 1)
    return start + timedelta(days=random_days)

def random_time(start, end):
    """
    Generate a random time between two time objects.
    """
    start_minutes = (start.hour * 60) + (start.minute)
    end_minutes = (end.hour * 60) + (end.minute)
    random_minutes = np.random.randint(start_minutes, end_minutes + 1)
    return datetime(2000, 1, 1, random_minutes // 60, random_minutes % 60, 0)

def random_datetime(start, end):
    """
    Generate a random datetime between two datetime objects.
    """
    random_date_value = random_date(start, end)
    random_time_value = random_time(datetime(2000, 1, 1, 0, 0, 0), datetime(2000, 1, 1, 23, 59, 00))
    return datetime(random_date_value.year, random_date_value.month, random_date_value.day,
                    random_time_value.hour, random_time_value.minute, random_time_value.second)

def create_fov_mask(shape, fov_radius, patch_shape=None):
    """
    Function to create mask for specified field of view.
    """
    # Create coordinate grids
    y, x = np.ogrid[:shape[0], :shape[1]]
    # Calculate center points
    center_y, center_x = shape[0] // 2, shape[1] // 2
    # Calculate distance from center for each point
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Normalize distances by max possible distance (corner to center)
    max_dist = np.sqrt((center_x)**2 + (center_y)**2)
    normalized_dist = dist_from_center / max_dist
    # Create mask for specified field of view
    mask = normalized_dist <= fov_radius

    # If specified, ensure the mask also covers the patch size
    if patch_shape is not None:
        patch_shape_half_x = patch_shape[0] // 2 + 1
        patch_shape_half_y = patch_shape[1] // 2 + 1
        # Create a square mask for the patch size
        patch_mask = np.ones(shape, dtype=bool)
        # Mask out the corner area to keep everything beyond patch_size_half
        patch_mask[:patch_shape_half_x, :] = False
        patch_mask[-patch_shape_half_x:, :] = False
        patch_mask[:, :patch_shape_half_y] = False
        patch_mask[:, -patch_shape_half_y:] = False
        # Combine the two masks
        mask = mask & patch_mask
    return mask

def check_quality_flags_goes(ds):
    """
    Function to check quality flags in GOES data.
    0 --> good pixel quality
    1 --> conditionally usable pixel quality
    2 --> out of range pixel quality
    3 --> no value pixel quality
    4 --> focal plane temperature threshold exceeded pixel quality
    """
    # Check each channel individually - exit early if bad quality found
    for i in range(1, 17):
        if (ds[f'DQF_C{i:02d}'] > 0).any().item():
            return False
    return True

def check_quality_flags_msg(ds, cutoff=10000):
    """
    Function to check quality in MSG data.

    Args:
        ds (xarray.Dataset): The dataset to check.
        cutoff (int): The maximum number of NaN values allowed. If exceeded, the function will return False. 
            There will be NaN values close to the disk edge, even after filtering for valid lat/lon values.
            From experimentation, these cases might have around 8000 to 10000 NaN values close to the disk edge
            even if the data on disk is valid. To not filter out all edge disk images, we emperically set the default cutoff to 10000.
    Returns:
        bool: True if the dataset passes the quality check, False otherwise.
    """
    channels=[
            'IR_016',
            'IR_039',
            'IR_087',
            'IR_097',
            'IR_108',
            'IR_120',
            'IR_134',
            'VIS006',
            'VIS008',
            'WV_062',
            'WV_073']
    # create a mask where the latitude value is inf
    mask_lat = ~np.isinf(ds.latitude)
    mask_lon = ~np.isinf(ds.longitude)
    # combine both masks to find all points with valid lat/lon values
    mask = mask_lat & mask_lon
    # loop through each channel to check for NaN values
    for channel in channels:
        # Check for NaN only where mask is True, i.e. where the lat/lon values are valid
        nan_in_valid_region = np.isnan(ds[channel].values[mask])
        if np.count_nonzero(nan_in_valid_region) > cutoff:
            return False
        else: 
            return True

def check_quality_flags_himawari(ds):
    """
    Function to check quality flags in Himawari data.
    """
    pass

class CenterWeightedCropDatasetEditor():
    def __init__(self, patch_shape, satellite, fov_radius=0.6, max_attempts=10):
        self.satellite = satellite
        self.patch_shape = patch_shape
        self.fov_radius = fov_radius
        self.max_attempts = max_attempts
    def __call__(self, ds):
        assert ds['x'].shape[0] >= self.patch_shape[0], 'Invalid dataset shape: %s' % str(ds['x'].shape)
        assert ds['y'].shape[0] >= self.patch_shape[1], 'Invalid dataset shape: %s' % str(ds['y'].shape)

        # get x/y grid
        x_grid, y_grid = np.meshgrid(np.arange(0, ds.x.shape[0], 1), np.arange(0, ds.y.shape[0], 1))

        # create mask for valid coordinates within desired field of view
        # NOTE: This masks from the center to the image edge, rather than disk edge
        valid_mask = create_fov_mask(
            shape=(ds.x.shape[0], ds.y.shape[0]), 
            fov_radius=self.fov_radius,
            patch_shape=self.patch_shape
            )

        # get coordinate pairs for valid points
        coords_on_disk = np.column_stack((x_grid[valid_mask], y_grid[valid_mask]))
        del x_grid, y_grid

        attempts = 0
        while attempts <= self.max_attempts:
            # pick random x/y index
            random_idx = np.random.randint(0, len(coords_on_disk))
            x, y = tuple(coords_on_disk[random_idx])
            # define patch boundaries
            xmin = x - self.patch_shape[0] // 2
            ymin = y - self.patch_shape[1] // 2
            xmax = x + self.patch_shape[0] // 2
            ymax = y + self.patch_shape[1] // 2

            # crop patch
            patch_ds = ds.sel({'x': slice(ds['x'][xmin], ds['x'][xmax - 1]),
                                'y': slice(ds['y'][ymin], ds['y'][ymax - 1])})
            # check data quality flags
            if self.satellite.lower() == 'goes':
                quality = check_quality_flags_goes(patch_ds)
            elif self.satellite.lower() == 'msg':
                quality = check_quality_flags_msg(patch_ds)
            elif self.satellite.lower() == 'himawari':
                quality = check_quality_flags_himawari(patch_ds)
            else:
                raise ValueError(f"Unknown satellite type: {self.satellite}")

            if quality == False:
                # logger.info('Found patch with bad quality flags, trying again ...')
                # try new set of indices
                attempts += 1
                continue   
            else:
                # exit loop and return patch
                return patch_ds, xmin, ymin

        logger.info('Could not find patch without bad quality flags after %d cropping attempts' % self.max_attempts)
        return None
    
def read_zipped_msg(filename, channels=None):
    """
    Function to read a zipped MSG file and return an xarray Dataset.

    Args:
        filename (str): Path to the zipped MSG file.
        channels (list, optional): List of channels to load. If None, defaults to a
            predefined list of channels.
    Returns:
        xarray.Dataset: Dataset containing the MSG data for the specified channels.
    """
    if channels is None:
        channels = [
            'IR_016',
            'IR_039',
            'IR_087',
            'IR_097',
            'IR_108',
            'IR_120',
            'IR_134',
            'VIS006',
            'VIS008',
            'WV_062',
            'WV_073'
        ]
    
    zf = zipfile.ZipFile(filename)

    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        seviri_file = list(pathlib.Path(tempdir).glob("MSG*-NA.nat"))[0]
        scn = Scene([seviri_file], reader="seviri_l1b_native")
        scn.load(channels)
        msg_ds = scn.to_xarray().load()

    return msg_ds
