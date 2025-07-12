#!/home/users/annaju/miniforge3/envs/jasmin-env/bin/python
import pathlib
import pandas as pd
import numpy as np
import xarray as xr
import argparse
from loguru import logger
from google.cloud import storage
import os

import s3fs
import argparse
import os

import fsspec
import goes2go
from tqdm import tqdm

from pyproj import Proj
from scipy.interpolate import make_splrep

def get_abi_proj(dataset: xr.Dataset) -> Proj:
    """
    Return a pyproj projection from the information contained within an ABI file
    """
    return Proj(
        proj="geos",
        h=dataset.goes_imager_projection.perspective_point_height,
        lon_0=dataset.goes_imager_projection.longitude_of_projection_origin,
        lat_0=dataset.goes_imager_projection.latitude_of_projection_origin,
        sweep=dataset.goes_imager_projection.sweep_angle_axis,
    )

def get_abi_x_y(
    lat: np.ndarray, lon: np.ndarray, dataset: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the x, y coordinates in the ABI projection for given latitudes and
        longitudes
    """
    p = get_abi_proj(dataset)
    x, y = p(lon, lat)
    return (
        x / dataset.goes_imager_projection.perspective_point_height,
        y / dataset.goes_imager_projection.perspective_point_height,
    )

def get_goes_image(
    file: str,)-> xr.Dataset:
    """
    Get the GOES image for a given timestamp.
    
    Args:
        file (str): The path to the GOES image.
    
    Returns:
        xr.Dataset: The GOES dataset for the specified timestamp.
    """
    # Create filesystem object inside worker process to avoid fork-safety issues
    fs = fsspec.filesystem('s3', anon=True)
    fsspec_caching = {
        "cache_type": "blockcache",  # block cache stores blocks of fixed size and uses eviction using a LRU strategy.
        "block_size": 8 * 1024 * 1024 # size in bytes per block, adjust depends on the file size but the recommended size is in the MB}
    }
    ds = xr.open_dataset(fs.open(file, **fsspec_caching), engine="h5netcdf")
    return ds

def get_goes_patch(
    lat: float, lon: float, dataset: xr.Dataset, patch_size: int
) -> xr.Dataset:
    """
    Get a patch of GOES data centered around a given latitude and longitude.
    """
    x, y = get_abi_x_y(np.array([lat]), np.array([lon]), dataset)
    x_dif = dataset.x.diff('x').values[0]
    y_dif = dataset.y.diff('y').values[0]
    return dataset.sel(
       x=slice(x[0] - (abs(x_dif) * patch_size / 2), x[0] + (abs(x_dif) * patch_size / 2)),
      y=slice(y[0] + (abs(y_dif) * patch_size / 2), y[0] - (abs(y_dif) * patch_size / 2)),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, required=True, help="The row to process from the GOES file")
    parser.add_argument("--GOES_file", type=str, default="jasmin.goes_intense_ibtracs.NA-EP.list.v04r01.csv", help="Path to the file containing all IBTrACS GOES times to process")
    parser.add_argument("--patch_size", type=int, default=1024, help="Size of the patch to extract from the GOES dataset")        
    args = parser.parse_args()
    
    # Load the IBTrACS dataset
    logger.info(f"Loading GOES IBTrACS dataset from {args.GOES_file}...")
    ibtracs = pd.read_csv(args.GOES_file)

    # Extract row to process
    row = ibtracs.iloc[args.num]
    SID = row['SID']
    lat = row['LAT']
    lon = row['LON']
    file = row['file']
    time_str = pd.to_datetime(row['start']).strftime('%Y%m%d%H%M%S')

    # Create output directory if it doesn't exist
    save_path = pathlib.Path(f"/work/scratch-nopw2/annaju/goes_temp/{SID}")
    # save_path = pathlib.Path(f"./{SID}")
    save_path.mkdir(parents=True, exist_ok=True)

    patch_filename = f'{SID}_{time_str}_patch.nc'
    save_file_name = save_path / patch_filename

    logger.info(f"Loading GOES image {file}...")
    ds = get_goes_image(file=file)
    logger.info(f"Extracting patch ...")
    ds_patch = get_goes_patch(lat, lon, ds, args.patch_size)

    # Create encoding for both data variables and coordinates to avoid warnings
    encoding = {}
    
    # Handle data variables
    for var in ds_patch.data_vars:
        encoding[var] = {
            "zlib": True,
            "complevel": 9,
            "dtype": "float32",
        }
    
    # Handle coordinates (especially x, y to avoid serialization warnings)
    for coord in ds_patch.coords:
        if coord in ['x', 'y']:
            encoding[coord] = {
                "dtype": "float32",
                "_FillValue": None  # This prevents the warning
            }

    # Save the patched dataset with the specified encoding
    logger.info(f"Saving patched dataset to {save_file_name} ...")
    ds_patch.to_netcdf(save_file_name, encoding=encoding)

    # Upload to GCP
    logger.info(f"Uploading file to GCP...")

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/users/annaju/esl-3d-clouds-extremes-baa3a73d57dc.json"  # TODO: Add credentials
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("2025-esl-3dclouds-extremes-datasets")
    blob = bucket.blob(f'pre-training/cyclones/goes/{SID}/{patch_filename}')
    blob.upload_from_filename(f"{save_path}/{patch_filename}")

    # remove local file
    (save_file_name).unlink()

    logger.info("Finished successfully ...")