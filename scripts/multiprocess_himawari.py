from goes2go.himawari_data import _himawari_file_df
import pandas as pd
import numpy as np
import xarray as xr
import argparse
from loguru import logger
from tqdm import tqdm
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

def download_himawari(args_tuple):
    """
    Download HIMAWARI L1b data. 
    """
    start, end, output_dir, patch_size, fov_radius, seed_offset = args_tuple
    
    # Set unique random seed for this process
    np.random.seed(seed_offset)
    
    try:
        # Generate random datetime within the specified range
        dt = random_datetime(start, end)
        # logger.info(f"Process {os.getpid()}: Selected random datetime: {dt} ...")
        
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
            
        patch_ds, xmin, ymin = result

        # Save patch to netcdf file
        patch_filename = f"{meas_time_str}_patch_{xmin}_{ymin}.nc"

        # Compress to reduce file size
        patch_ds, encoding = reduce_file_size(patch_ds, compression_level=9)
        # Save with compression
        logger.info(f"Compressing and saving patch for {selected_time} ...")
        patch_ds.to_netcdf(f"{output_dir}/{patch_filename}", encoding=encoding)
        
        del patch_ds  # Free memory
        # logger.info(f"Process {os.getpid()}: Saved patch to {output_dir}/{patch_filename} ...")

        # Return results as dictionary
        return {
            'patch_filename': patch_filename,
            'measurement_time': meas_time_str,
            'xmin': xmin,
            'ymin': ymin,
        }
        
    except Exception as e:
        logger.error(f"Process {os.getpid()}: Error processing: {e}")
        return None

if __name__ == "__main__":
    # Set multiprocessing start method to spawn to avoid fork-safety issues
    set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--num_files", type=int, default=1, help="Number of files to download")
    parser.add_argument("--num_processes", type=int, default=None, help="Number of processes to use (default: CPU count)")
    parser.add_argument("--output_dir", type=str, default='.', help="Directory to save the downloaded data")
    parser.add_argument("--start", type=str, default="2015-07-07", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, default="2022-12-12", help="End date in YYYY-MM-DD format")
    parser.add_argument("--patch_size", type=int, default=1024, help="Size of the patch to crop from the dataset")
    parser.add_argument("--fov_radius", type=float, default=0.6, help="Field of view radius for cropping")

    args = parser.parse_args()  

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert start and end dates to datetime objects
    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # Determine number of processes
    num_processes = args.num_processes if args.num_processes else min(cpu_count() // 2, args.num_files)
    logger.info(f"Using {num_processes} process(es) to download {args.num_files} file(s)")

    # Generate unique seeds for each process
    seeds = np.random.randint(0, 100000, size=args.num_files)
    
    # Prepare arguments for multiprocessing
    process_args = [
        (start, end, args.output_dir, args.patch_size, args.fov_radius, seed)
        for seed in seeds
    ]

    # Download with progress bar
    successful_results = []
    
    with Pool(processes=num_processes) as pool:
        # Use imap for real-time progress updates
        with tqdm(total=args.num_files, desc="Downloading HIMAWARI files") as pbar:
            for result in pool.imap(download_himawari, process_args):
                if result is not None:
                    successful_results.append(result)
                pbar.update(1)
    
    logger.info(f"Successfully downloaded {len(successful_results)} out of {args.num_files} files")
    
    # Create DataFrame with results
    if successful_results:
        # results_df = pd.DataFrame(successful_results)
        
        # # Save results to CSV
        # csv_filename = f"{output_dir}/himawari_index.csv"
        # results_df.to_csv(csv_filename, index=False)

        # Print summary
        print(f"\nDownload Summary:")
        print(f"Total files requested: {args.num_files}")
        print(f"Successfully downloaded: {len(successful_results)}")
        print(f"Failed downloads: {args.num_files - len(successful_results)}")
        # print(f"Results saved to: {csv_filename}")
        
    else:
        logger.error("No files were successfully downloaded!")