#!/home/users/annaju/miniforge3/envs/jasmin-env/bin/python
import pathlib
import pandas as pd
import numpy as np
import xarray as xr
import argparse
from loguru import logger
from google.cloud import storage
import os
import ast

import s3fs
import argparse
import os

import fsspec
import goes2go
from tqdm import tqdm
from satpy import Scene

from pyproj import Proj
from scipy.interpolate import make_splrep

def reduce_file_size(ds, compression_level=9):
    """
    Reduce the file size of the dataset by converting to float32 and compressing.
    
    Args:
        ds (xarray.Dataset): The dataset to reduce.
        compression_level (int): Compression level for saving the dataset.
    
    Returns:
        xarray.Dataset: Reduced dataset.
    """
    # Reduce file size by converting to float32
    ds = ds.astype("float32")
    # Remove unnecessary variables
    ds = ds.drop_vars(["FLDK"])

    encoding = {}

    # Add data variable compression
    for var in ds.data_vars:
        if ds[var].dtype in ['float64', 'float32']:
            encoding[var] = {'dtype': 'float32', 'zlib': True, 'complevel': compression_level, 'shuffle': True}
    # Add coordinate compression
    for coord in ds.coords:
        if ds[coord].dtype in ['float64', 'float32']:
            encoding[coord] = {'dtype': 'float32', 'zlib': True, 'complevel': compression_level, 'shuffle': True}

    return ds, encoding

def get_ahi_proj(dataset: xr.Dataset) -> Proj:
    """
    Return a pyproj projection from the information contained within an AHI file
    """

    height = ast.literal_eval(dataset.B01.orbital_parameters).get('projection_altitude')
    lon_0 = ast.literal_eval(dataset.B01.orbital_parameters).get('projection_longitude')
    lat_0 = ast.literal_eval(dataset.B01.orbital_parameters).get('projection_latitude')
    return Proj(
        proj="geos",
        h=height,
        lon_0=lon_0,
        lat_0=lat_0,
    )

def get_ahi_x_y(
    lat: np.ndarray, lon: np.ndarray, dataset: xr.Dataset
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the x, y coordinates in the AHI projection for given latitudes and
        longitudes
    """
    p = get_ahi_proj(dataset)
    x, y = p(lon, lat)
    return (
        x / ast.literal_eval(dataset.B01.orbital_parameters).get('projection_altitude'),
        y / ast.literal_eval(dataset.B01.orbital_parameters).get('projection_altitude'),
    )

def get_himawari_image(
    files: str,)-> xr.Dataset:
    """
    Get the HIMAWARI image for a given timestamp.

    Args:
        files (str): The path to the HIMAWARI files.

    Returns:
        xr.Dataset: The HIMAWARI dataset for the specified timestamp.
    """
    # Create filesystem object inside worker process to avoid fork-safety issues
    fs = fsspec.filesystem('s3', anon=True)
    fsspec_caching = {
        "cache_type": "blockcache",  # block cache stores blocks of fixed size and uses eviction using a LRU strategy.
        "block_size": 8 * 1024 * 1024 # size in bytes per block, adjust depends on the file size but the recommended size is in the MB}
    }
    # Load with satpy
    scn = Scene(
        [f's3://{f}' for f in files], # select all files at one time
        reader="ahi_hsd", 
        reader_kwargs=dict(storage_options = {'anon': True}), 
    )
    # load available datasets
    scn.load(scn.all_dataset_names())

    # Resample to 2km resolution
    new_scn = scn.resample(scn.coarsest_area(), resampler='native')

    # Convert to xarray
    ds = new_scn.to_xarray()
    return ds

def get_himawari_patch(
    lat: float, lon: float, dataset: xr.Dataset, patch_size: int
) -> xr.Dataset:
    """
    Get a patch of HIMAWARI data centered around a given latitude and longitude.
    """
    x, y = get_ahi_x_y(np.array([lat]), np.array([lon]), dataset)
    x_dif = dataset.x.diff('x').values[0]
    y_dif = dataset.y.diff('y').values[0]
    return dataset.sel(
       x=slice(x[0] - (abs(x_dif) * patch_size / 2), x[0] + (abs(x_dif) * patch_size / 2)),
      y=slice(y[0] + (abs(y_dif) * patch_size / 2), y[0] - (abs(y_dif) * patch_size / 2)),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, required=True, help="The row to process from the GOES file")
    parser.add_argument("--HIMAWARI_file", type=str, default="jasmin.himawari_intense_ibtracs.SP-WP.list.v04r01.csv", help="Path to the file containing all IBTrACS GOES times to process")
    parser.add_argument("--patch_size", type=int, default=1024, help="Size of the patch to extract from the GOES dataset")        
    args = parser.parse_args()
    
    # Load the IBTrACS dataset
    logger.info(f"Loading HIMAWARI IBTrACS dataset from {args.HIMAWARI_file}...")
    ibtracs = pd.read_csv(args.HIMAWARI_file)

    # Extract row to process
    row = ibtracs.iloc[args.num]
    SID = row['SID']
    lat = row['LAT']
    lon = row['LON']
    files = ast.literal_eval(row['files'])
    time_str = pd.to_datetime(row['start']).strftime('%Y%m%d%H%M%S')

    # Create output directory if it doesn't exist
    save_path = pathlib.Path(f"/work/scratch-nopw2/annaju/goes_temp/{SID}")
    # save_path = pathlib.Path(f"./{SID}")
    save_path.mkdir(parents=True, exist_ok=True)

    patch_filename = f'{time_str}_{SID}_patch.nc'
    save_file_name = save_path / patch_filename

    logger.info(f"Loading {len(files)} HIMAWARI images ...")
    ds = get_himawari_image(files=files)
    logger.info(f"Extracting patch ...")
    ds_patch = get_himawari_patch(lat, lon, ds, args.patch_size)

    # Reduce file size
    ds_patch, encoding = reduce_file_size(ds_patch, compression_level=9)

    # Save the patched dataset with the specified encoding
    logger.info(f"Saving patched dataset to {save_file_name} ...")
    ds_patch.to_netcdf(save_file_name, encoding=encoding)

    # Upload to GCP
    logger.info(f"Uploading file to GCP...")

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/users/annaju/esl-3d-clouds-extremes-baa3a73d57dc.json"  # TODO: Add credentials
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("2025-esl-3dclouds-extremes-datasets")
    blob = bucket.blob(f'pre-training/cyclones/himawari/{SID}/{patch_filename}')
    blob.upload_from_filename(f"{save_path}/{patch_filename}")

    # remove local file
    (save_file_name).unlink()

    logger.info("Finished successfully ...")