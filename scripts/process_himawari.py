#!/home/users/annaju/miniforge3/envs/jasmin-env/bin/python
import pathlib
from goes2go.himawari_data import _himawari_file_df
import pandas as pd
import numpy as np
import xarray as xr
import argparse
from loguru import logger
from tqdm import tqdm
from google.cloud import storage
from satpy import Scene
from datetime import datetime, timedelta
from process_utils import random_datetime, CenterWeightedCropDatasetEditor
from multiprocessing import Pool, cpu_count, set_start_method
import os

import s3fs

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

def download_himawari(dt, patch_size, fov_radius):
    """
    Download HIMAWARI L1b data. 
    """
    try:
        # Find himawari files
        ahi_files = _himawari_file_df(
            "noaa-himawari8", 
            "FLDK", 
            dt, 
            dt + timedelta(minutes=60),
            ignore_missing=True)
        
        # Check that we have HIMAWARI files for the selected date
        if len(ahi_files) == 0:
            logger.warning(f"Process {os.getpid()}: No HIMAWARI files found for date {dt}.")
            return None
        
        # Select a random file/datetime from the availabel files
        unique_times = ahi_files.time.unique()
        selected_time = np.random.choice(unique_times)

        logger.info(f"Loading HIMAWARI file for {selected_time} ...")
        
        # Load with satpy
        scn = Scene(
            [f's3://{f}' for f in ahi_files.groupby("time").get_group(selected_time).file], # select all files at one time
            reader="ahi_hsd", 
            reader_kwargs=dict(storage_options = {'anon': True}), 
        )

        # load available datasets
        scn.load(scn.all_dataset_names())

        # Resample to 2km resolution
        new_scn = scn.resample(scn.coarsest_area(), resampler='native')

        # Convert to xarray
        ds = new_scn.to_xarray()

        meas_time_str = pd.to_datetime(selected_time).strftime('%Y%m%d%H%M%S')
        
        logger.info(f"Cropping HIMAWARI file for {selected_time} ...")

        # Crop dataset into patch
        crop = CenterWeightedCropDatasetEditor(
            patch_shape=(patch_size, patch_size), 
            fov_radius=fov_radius,
            satellite = 'himawari')
        result = crop(ds)
        
        if result is None: # i.e. if no valid patch was found
            logger.warning(f"Could not find valid patch for {selected_time}")
            return None
        else:
            patch_ds, xmin, ymin = result
            return (patch_ds, xmin, ymin, meas_time_str)
    
    except:
        logger.error(f"Could not process HIMAWARI file for {dt} ...")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--start", type=str, default="2015-07-07", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, default="2022-12-12", help="End date in YYYY-MM-DD format")
    parser.add_argument("--patch_size", type=int, default=1024, help="Size of the patch to crop from the dataset")
    parser.add_argument("--fov_radius", type=float, default=0.6, help="Field of view radius for cropping")

    args = parser.parse_args()  

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    logger.info(f"Using random seed: {args.seed}")

    # Create output directory if it doesn't exist
    save_path = pathlib.Path("/work/scratch-nopw2/annaju/himawari_temp/")
    save_path.mkdir(parents=True, exist_ok=True)

    # Convert start and end dates to datetime objects
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # Select random datetime
    dt = random_datetime(start, end)

    result = download_himawari(dt, 
                              patch_size=args.patch_size, 
                              fov_radius=args.fov_radius)
    
    if result is not None:
        patch_ds, xmin, ymin, dt_str = result

        # Reduce file size
        patch_ds, encoding = reduce_file_size(patch_ds, compression_level=9)

        logger.info(f"Saving patch for {dt_str} at coordinates ({xmin}, {ymin})...")

        # Save patch to netcdf file
        patch_filename = f"{dt_str}_patch_{xmin}_{ymin}.nc"
        patch_ds.to_netcdf(f"{save_path}/{patch_filename}", encoding=encoding)

        # Upload to GCP
        logger.info(f"Uploading file to GCP...")

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/users/annaju/esl-3d-clouds-extremes-baa3a73d57dc.json"  # TODO: Add credentials
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("2025-esl-3dclouds-extremes-datasets")
        blob = bucket.blob(f'pre-training/himawari/l1b/{patch_filename}')
        blob.upload_from_filename(f"{save_path}/{patch_filename}")

        # remove local file
        (save_path / patch_filename).unlink()

        logger.info("Finished successfully ...")

    else:
        logger.warning(f"No valid patch found for number {args.number} and {dt}. Skipping upload ...")
