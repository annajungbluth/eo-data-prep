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
    timestamp: str,)-> xr.Dataset:
    """
    Get the GOES image for a given timestamp.
    
    Args:
        timestamp (str): The timestamp in ISO format (e.g., '2023-10-01T12:00:00Z').
    
    Returns:
        xr.Dataset: The GOES dataset for the specified timestamp.
    """
    # Download the GOES data for the specified timestamp:
    goes_df = goes2go.goes_timerange(
        start = pd.to_datetime(timestamp), 
        end = pd.to_datetime(timestamp) + pd.Timedelta(minutes=10),
        download = False, 
        product = 'ABI-L2-MCMIP', 
        domain = 'F'
    )
      # Create filesystem object inside worker process to avoid fork-safety issues
    fs = fsspec.filesystem('s3', anon=True)
    fsspec_caching = {
        "cache_type": "blockcache",  # block cache stores blocks of fixed size and uses eviction using a LRU strategy.
        "block_size": 8 * 1024 * 1024 # size in bytes per block, adjust depends on the file size but the recommended size is in the MB}
    }
    ds = xr.open_dataset(fs.open(goes_df.file[0], **fsspec_caching), engine="h5netcdf")
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

def filter_tcs(
    ibtracs: pd.DataFrame, 
    start_year: int, 
    end_year: int, 
):
    try:
        ibtracs['SEASON'] = ibtracs['SEASON'].astype(int)
    except:
        ibtracs = ibtracs[1:]
        ibtracs['SEASON'] = ibtracs['SEASON'].astype(int)
    ibtracs_sel = ibtracs[(ibtracs['SEASON'] >= start_year) &(ibtracs['SEASON'] < end_year)].reset_index(drop=True)
    return ibtracs_sel.SID.unique()

def get_selected_tc(
    ibtracs: pd.DataFrame, 
    SID: str, 
):
    """
    Function to select the tropical cyclone track from the IBTrACS dataset.
    Args:
        ibtracs (pd.DataFrame): The IBTrACS dataset.
        SID (str): The storm ID of the tropical cyclone.
    Returns:
        tuple: A tuple containing the longitudes, latitudes, and timestamps of the tropical cyclone track.
    """
    ibtracs_sel = ibtracs[ibtracs.SID==SID].reset_index(drop=True)
    longitudes = ibtracs_sel['LON'].values
    latitudes = ibtracs_sel['LAT'].values
    timestamps = ibtracs_sel['ISO_TIME'].values
    return longitudes, latitudes, timestamps
    

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
    encoding = {}

    # Add data variable compression
    for var in ds.data_vars:
        if ds[var].dtype in ['float64', 'float32']:
            encoding[var] = {'dtype': 'float32'} #, 'zlib': True, 'complevel': compression_level, 'shuffle': True}
    # Add coordinate compression
    for coord in ds.coords:
        if ds[coord].dtype in ['float64', 'float32']:
            encoding[coord] = {'dtype': 'float32'}#, 'zlib': True, 'complevel': compression_level, 'shuffle': True}

    return ds, encoding

def save_goes_patch(
    ds_patch: xr.Dataset, 
    filename: str = 'goes_patch.nc', 
):
    """
    Save the GOES patch to a NetCDF file.
    """
    ds_patch, encoding = reduce_file_size(ds_patch)
    ds_patch.to_netcdf(filename, encoding=encoding)
    print(f"GOES patch saved to {filename}")
    
def interopolate_track(
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        timestamps: np.ndarray
):
    """
    Interpolate the track of a tropical cyclone.
    
    Args:
        latitudes (np.ndarray): Array of latitudes.
        longitudes (np.ndarray): Array of longitudes.
        timestamps (np.ndarray): Array of timestamps.
    
    Returns:
        np.ndarray: Interpolated latitudes, longitudes, and timestamps.
    """
    # Implement interpolation logic here
    latitudes_intp = make_splrep((timestamps).astype(np.datetime64), latitudes, s=0.2)
    longitudes_intp = make_splrep((timestamps).astype(np.datetime64), longitudes, s=0.2)
    # For now, just return the input arrays
    return latitudes_intp, longitudes_intp

def get_available_goes_times(
        start: pd.Timestamp,
        end: pd.Timestamp, 
        reduction_factor: int = 1
):
    """
    Get available GOES times within a specified range.
    Args:
        start (pd.Timestamp): Start time.
        end (pd.Timestamp): End time.
        reduction_factor (int): Factor to reduce the number of times.
    Returns:
        pd.DataFrame: DataFrame containing available GOES times.
    """
    goes_df = goes2go.goes_timerange(start = start, end = end, satellite='noaa-goes-16',
                            download = False, product = 'ABI-L2-MCMIP', domain = 'F')
    
    goes_df['mid_time'] = goes_df.start + (goes_df.end - goes_df.start)/2

    goes_df = goes_df[::reduction_factor].reset_index(drop=True)
    return goes_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("SID", type=str, default="2024181N09320", help="Storm ID to filter the IBTrACS dataset")
    parser.add_argument("SID_num", type=int, help="Index of the storm ID to process")
    parser.add_argument("SID_file", type=str, default="/home/users/annaju/eo-data-prep/scripts/goes_intense_ibtracs.NA-EP.list.v04r01.SIDs.csv", help="Path to the file containing storm IDs")
    parser.add_argument("ibtracs_file", default="/home/users/annaju/eo-data-prep/scripts/goes_intense_ibtracs.NA-EP.list.v04r01", type=str, help="Path to the IBTrACS dataset file")
    parser.add_argument("patch_size", type=int, default=1024, help="Size of the patch to extract from the GOES dataset")        
    parser.add_argument("reduction_factor", type=int, default=1, help="Factor to reduce the number of GOES times")
    args = parser.parse_args()
    
    # Load the IBTrACS dataset
    logger.info(f"Loading IBTrACS dataset from {args.ibtracs_file}...")
    ibtracs = pd.read_csv(args.ibtracs_file)

    SIDs = pd.read_csv(args.SID_file, header=None).squeeze().tolist()
    SID = SIDs[args.SID_num]

    # Get the longitudes, latitudes, and timestamps for the selected tropical cyclone
    longitudes, latitudes, timestamps = get_selected_tc(ibtracs, SID=SID)
    if len(timestamps) == 0:
        raise ValueError(f"No data found for SID {SID}")
    
    # Get available GOES times based on the timestamps of the tropical cyclone
    logger.info(f"Getting available GOES times for SID {SID}...")
    start_time  = pd.to_datetime(timestamps[0])
    end_time = pd.to_datetime(timestamps[-1])
    goes_files = get_available_goes_times(start=start_time, end=end_time, reduction_factor=args.reduction_factor)
    logger.info(f"Found {len(goes_files)} GOES files for the specified time range.")
    if goes_files.empty:
        raise ValueError(f"No GOES files found for the specified time range: {start_time} to {end_time}")
   
    # Interpolate the track of the tropical cyclone
    logger.info(f"Interpolating track for SID {SID}...")
    f_lat, f_lon = interopolate_track(latitudes, longitudes, timestamps)

     # Create output directory if it doesn't exist
    save_path = pathlib.Path(f"/work/scratch-nopw2/annaju/goes_temp/{SID}")
    save_path.mkdir(parents=True, exist_ok=True)

    # TODO: make loop use multiprocessing
    for i, goes_time in tqdm(enumerate(goes_files.mid_time.values)):

        # interpolate latitudes and longitudes for the goes times
        # need to discard ns in goes_time
        int_lat = f_lat(goes_time.astype('datetime64[s]'))
        int_lon = f_lon(goes_time.astype('datetime64[s]'))
        
        #check if get goes image is returning a dataset and if not to move to the next iteration
        time_str = goes_time.strftime('%Y%m%d%H%M%S')
        patch_filename = f'{SID}_{time_str}_patch.nc'
        save_file_name = save_path / patch_filename
        if save_file_name.exists():
            logger.info(f"File {save_file_name} already exists, skipping...")
            continue
        try:
            ds = get_goes_image(goes_files.start.values[i])
        except:
            print(f"Skipped timestamp {goes_time} — Error: {e}")
            continue
        ds_patch = get_goes_patch(int_lat, int_lon, ds, args.patch_size)
        
        #save_goes_patch(ds_patch, filename=f'goes_patch_{i}.nc')
        ds_patch.to_netcdf(save_file_name)

        # Upload to GCP
        logger.info(f"Uploading file to GCP...")

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/home/users/annaju/esl-3d-clouds-extremes-baa3a73d57dc.json"  # TODO: Add credentials
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("2025-esl-3dclouds-extremes-datasets")
        blob = bucket.blob(f'pre-training/cyclones/{SID}/{patch_filename}')
        blob.upload_from_filename(f"{save_path}/{patch_filename}")

        # remove local file
        (save_file_name).unlink()

        logger.info("Finished successfully ...")