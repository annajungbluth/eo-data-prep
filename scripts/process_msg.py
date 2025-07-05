import pathlib
from datetime import datetime
import numpy as np
import xarray as xr
import argparse
from satpy import Scene
from loguru import logger
import pandas as pd
import tempfile
import zipfile
from process_utils import CenterWeightedCropDatasetEditor, read_zipped_msg

CHANNELS = [
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
    ds = ds.drop_vars([f"{channel}_acq_time" for channel in CHANNELS])

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

def load_and_patch_msg(file_path, patch_size, fov_radius):
    """
    Load and patch MSG file using Satpy.
    
    Args:
        file_path (str): Path to the MSG file.
        patch_size (int): Size of the patch to crop from the dataset.
        fov_radius (float): Field of view radius for cropping.
    
    Returns:
        xarray.Dataset: Patched dataset.
    """
    ds = read_zipped_msg(file_path)

    # Crop dataset into patch
    crop = CenterWeightedCropDatasetEditor(
        patch_shape=(patch_size, patch_size), 
        fov_radius=fov_radius,
        satellite = 'msg')
    
    result = crop(ds)

    if result is None: # i.e. if no valid patch was found
        logger.warning(f"Could not find valid patch ...")
        return None

    return result
    
if __name__ == "__main__":
    # Add argument parser for command line arguments
    parser = argparse.ArgumentParser(
        description="""Load and patch MSG files, then upload to GCP"""
    )
    parser.add_argument("number", help="The file to select from our compiled list of files", type=int)
    parser.add_argument("--patch_size", type=int, default=1024, help="Size of the patch to crop from the dataset")
    parser.add_argument("--fov_radius", type=float, default=0.6, help="Field of view radius for cropping")

    args = parser.parse_args()

    logger.info(f"Processing file number {args.number}...")

    # save_path = pathlib.Path("/work/scratch-nopw2/annaju/msg-tmp/")
    save_path = pathlib.Path(".")
    save_path.mkdir(parents=True, exist_ok=True)

    df_selected_files = pd.read_csv("msg-sample-50000.csv")
    # Extract relevant file and datetime based on provided number
    selected_file = df_selected_files.iloc[args.number]["path"]

    result = load_and_patch_msg(selected_file, 
                                  patch_size=args.patch_size, 
                                  fov_radius=args.fov_radius,
                                  )
    if result is not None:
        patch_ds, xmin, ymin = result
        # Extract datetime from the file name
        dt_str = pathlib.Path(selected_file).stem.split('-')[-2].split('.')[0]

        # Reduce file size
        patch_ds, encoding = reduce_file_size(patch_ds)
 
        # Save patch to netcdf file
        patch_filename = f"{dt_str}_patch_{xmin}_{ymin}.nc"
        patch_ds.to_netcdf(f"{save_path}/{patch_filename}", encoding=encoding)
        # TODO: Reduce the file size, at the moment, each file is 24 MB

    # TODO: Add upload to GCP











