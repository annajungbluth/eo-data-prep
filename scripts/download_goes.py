import goes2go
import pandas as pd
import numpy as np
import xarray as xr
import argparse
from loguru import logger
from tqdm import tqdm
from datetime import datetime, timedelta
from process_utils import random_datetime, CenterWeightedCropDatasetEditor
from multiprocessing import Pool, cpu_count, set_start_method
import os
import contextlib
import warnings

@contextlib.contextmanager
def suppress_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield

import fsspec

def download_mcmip(args_tuple):
    """
    Download GOES MCMIP data. 
    """
    start, end, output_dir, patch_size, fov_radius, seed_offset = args_tuple
    
    # Set unique random seed for this process
    np.random.seed(seed_offset)
    
    # Create filesystem object inside worker process to avoid fork-safety issues
    fs = fsspec.filesystem('s3', anon=True)
    fsspec_caching = {
    "cache_type": "blockcache",  # block cache stores blocks of fixed size and uses eviction using a LRU strategy.
    "block_size": 8
    * 1024
    * 1024,  # size in bytes per block, adjust depends on the file size but the recommended size is in the MB
    }

    try:
        # Generate random datetime within the specified range
        dt = random_datetime(start, end)
        # logger.info(f"Process {os.getpid()}: Selected random datetime: {dt} ...")
        
        # Get the list of ABI files for the random datetime
        # Will raise an error if files are not available
        with suppress_warnings():
            abi_files = goes2go.goes_timerange(
                start=dt, 
                end=dt + timedelta(minutes=30),
                download=False,
                domain='F',
                product="ABI-L2-MCMIP",
            )
        
        # Check that we have MCMIP files for the selected date
        if len(abi_files) == 0:
            logger.warning(f"Process {os.getpid()}: No MCMIP files found for date {dt}.")
            return None
        
        # Load the first file into an xarray dataset
        ds = xr.open_dataset(fs.open(abi_files['file'][0], **fsspec_caching), engine="h5netcdf")
        meas_time_str = datetime.strftime(abi_files['start'][0], '%Y%m%d%H%M%S')
        # logger.info(f"Process {os.getpid()}: Loaded MCMIP file: {abi_files['file'][0]} ...")
        
        # Crop dataset into patch
        crop = CenterWeightedCropDatasetEditor(
            patch_shape=(patch_size, patch_size), 
            fov_radius=fov_radius,
            satellite = 'goes')
        result = crop(ds)
        
        if result is None: # i.e. if no valid patch was found
            logger.warning(f"Could not find valid patch for {dt}")
            return None
            
        patch_ds, xmin, ymin = result

        # Save patch to netcdf file
        patch_filename = f"{meas_time_str}_patch_{xmin}_{ymin}.nc"
        patch_ds.to_netcdf(f"{output_dir}/{patch_filename}")
        del patch_ds  # Free memory
        # logger.info(f"Process {os.getpid()}: Saved patch to {output_dir}/{patch_filename} ...")

        # Return results as dictionary
        return {
            'patch_filename': patch_filename,
            'measurement_time': meas_time_str,
            'source_file': abi_files['file'][0],
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
    parser.add_argument("--start", type=str, default="2018-01-01", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date in YYYY-MM-DD format")
    parser.add_argument("--patch_size", type=int, default=1024, help="Size of the patch to crop from the dataset")
    parser.add_argument("--fov_radius", type=float, default=0.6, help="Field of view radius for cropping")

    args = parser.parse_args()  

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # Create output directory if it doesn't exist
    output_dir = f"{args.output_dir}/mcmip-{args.seed}"
    os.makedirs(output_dir, exist_ok=True)

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
        (start, end, output_dir, args.patch_size, args.fov_radius, seed)
        for seed in seeds
    ]

    # Download with progress bar
    successful_results = []
    
    with Pool(processes=num_processes) as pool:
        # Use imap for real-time progress updates
        with tqdm(total=args.num_files, desc="Downloading MCMIP files") as pbar:
            for result in pool.imap(download_mcmip, process_args):
                if result is not None:
                    successful_results.append(result)
                pbar.update(1)
    
    logger.info(f"Successfully downloaded {len(successful_results)} out of {args.num_files} files")
    
    # Create DataFrame with results
    if successful_results:
        results_df = pd.DataFrame(successful_results)
        
        # Save results to CSV
        csv_filename = f"{output_dir}/mcmip_index.csv"
        results_df.to_csv(csv_filename, index=False)

        # Print summary
        print(f"\nDownload Summary:")
        print(f"Total files requested: {args.num_files}")
        print(f"Successfully downloaded: {len(successful_results)}")
        print(f"Failed downloads: {args.num_files - len(successful_results)}")
        print(f"Results saved to: {csv_filename}")
        
    else:
        logger.error("No files were successfully downloaded!")