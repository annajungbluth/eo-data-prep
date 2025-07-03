#!/home/users/wkjones/miniforge3/envs/tobac_flow/bin/python
import pathlib
from datetime import datetime
import numpy as np
from numpy import ma
import xarray as xr
from cloudsatipy import open_cloudsat

import argparse
parser = argparse.ArgumentParser(
    description="""Geoprocess and combine multiple cloudsat products, then upload to GCP"""
)
parser.add_argument("orbit", help="Orbit number to process", type=int)

def construct_cloud_type_mask(cldclass_ds):
    cloud_type_mask = xr.DataArray(
        np.zeros_like(cldclass_ds.Height.values), cldclass_ds.Height.coords, cldclass_ds.Height.dims, name="CloudTypeMask", 
    )
    cloud_type_quality = xr.DataArray(
        np.zeros_like(cldclass_ds.Height.values), cldclass_ds.Height.coords, cldclass_ds.Height.dims, name="CloudTypeQuality", 
    )

    for cloud_layer in cldclass_ds.Ncloud.values:
        if np.any(cldclass_ds.CloudLayerType[cloud_layer]).item():
            raveled_idx, n_cloud_bins = get_layer_indices(
                cldclass_ds.Height, 
                cldclass_ds.CloudLayerTop[...,cloud_layer], 
                cldclass_ds.CloudLayerBase[...,cloud_layer], 
            )
            cloud_type_mask.data.ravel()[raveled_idx] = np.repeat(
                cldclass_ds.CloudLayerType[...,cloud_layer].values, n_cloud_bins
            )
            cloud_type_quality.data.ravel()[raveled_idx] = np.repeat(
                cldclass_ds.CloudTypeQuality[...,cloud_layer].values, n_cloud_bins
            )
        else:
            break

    attrs = cldclass_ds.CloudLayerType.attrs.copy()

    attrs["Flag_values"] = ["Deep", "Ns", "Cu", "Sc", "St", "Ac", "As", "High"][::-1]

    cloud_type_mask = cloud_type_mask.assign_attrs(attrs)

    quality_attrs = cldclass_ds.CloudTypeQuality.attrs.copy()
    
    cloud_type_quality = cloud_type_quality.assign_attrs(quality_attrs)
    cloud_type_quality[...,0] = 0
    
    return cloud_type_mask, cloud_type_quality


def find_nearest_height_bins(heights, layer_height):
    return np.nanargmin(
        np.abs(heights - (layer_height*1e3)).fillna(np.inf).values, axis=1
    )

def get_layer_indices(heights, layer_top, layer_base):
    top_bin = find_nearest_height_bins(heights, layer_top)
    base_bin = find_nearest_height_bins(heights, layer_base)
    
    n_cloud_bins = base_bin - top_bin + 1

    nbin_idx = np.repeat(top_bin, n_cloud_bins) + repeat_ranges(n_cloud_bins)
    nray_idx = np.repeat(heights.Nray.values, n_cloud_bins)

    raveled_idx = np.ravel_multi_index([nray_idx.astype(int), nbin_idx.astype(int)], heights.shape)
    
    return raveled_idx, n_cloud_bins

def repeat_ranges(n_repeats):
    repeat_range = np.repeat(np.ones(n_repeats.size), n_repeats)
    repeat_range[np.cumsum(n_repeats)[:-1]] = -n_repeats[:-1]+1
    repeat_range = np.cumsum(repeat_range)-1
    return repeat_range

def main() -> None:
    args = parser.parse_args()
    orbit = args.orbit

    print(f'Processing cloudsat data for orbit {orbit}')

    cloudsat_path = pathlib.Path("/gws/nopw/j04/eo_shared_data_vol1/satellite/cloudsat/Data/")

    geoprof_file = sorted(list((cloudsat_path/"2b-geoprof").rglob(f'*_{orbit:05d}_CS_*')))
    assert len(geoprof_file), "Geoprof data missing"

    date = datetime.strptime(geoprof_file[0].name[:13], "%Y%j%H%M%S")

    cldclass_file = sorted(list((cloudsat_path/"2b-cldclass-lidar"/"R05"/date.strftime("%Y/%j")).glob(f'*_{orbit:05d}_CS_*')))
    assert len(cldclass_file), "cldclass data missing"

    cwc_file = sorted(list((cloudsat_path/"2b-cwc-ro"/"R05"/date.strftime("%Y/%j")).glob(f'*_{orbit:05d}_CS_*')))
    assert len(cwc_file), "cwc data missing"
    
    flxhr_file = sorted(list((cloudsat_path/"2b-flxhr-lidar"/"R05"/date.strftime("%Y/%j")).glob(f'*_{orbit:05d}_CS_*')))
    assert len(flxhr_file), "flxhr-lidar data missing"
    
    ice_file = sorted(list((cloudsat_path/"2c-ice"/"R05"/date.strftime("%Y/%j")).glob(f'*_{orbit:05d}_CS_*')))
    assert len(ice_file), "ice data missing"

    dataset = open_cloudsat(geoprof_file[0], variable=[
        "Data_quality", "Data_status", "Data_targetID", "RayStatus_validity", "SurfaceHeightBin", "SurfaceHeightBin_fraction", "CPR_Cloud_mask", "Radar_Reflectivity", "Navigation_land_sea_flag"
    ])
    dataset["Radar_Reflectivity"] = dataset.Radar_Reflectivity.where(dataset.CPR_Cloud_mask>=30)

    cldclass_ds = open_cloudsat(cldclass_file[0])
    cloud_type_mask, cloud_type_quality = construct_cloud_type_mask(cldclass_ds)
    dataset = xr.merge([dataset, cloud_type_mask, cloud_type_quality])
    
    dataset = xr.merge([
        dataset,
        open_cloudsat(cwc_file[0], variable=[
            "Data_quality", 
            "Data_status", 
            "RO_liq_effective_radius", 
            "RO_liq_effective_radius_uncertainty", 
            "RO_ice_effective_radius", 
            "RO_ice_effective_radius_uncertainty", 
            "RO_liq_number_conc", 
            "RO_liq_num_conc_uncertainty", 
            "RO_ice_number_conc", 
            "RO_ice_num_conc_uncertainty", 
            "RO_liq_distrib_width_param", 
            "RO_liq_distrib_width_param_uncertainty", 
            "RO_ice_distrib_width_param", 
            "RO_ice_distrib_width_param_uncertainty", 
            "RO_liq_water_content", 
            "RO_liq_water_content_uncertainty", 
            "RO_ice_water_content", 
            "RO_ice_water_content_uncertainty", 
            "RO_liq_water_path", 
            "RO_liq_water_path_uncertainty", 
            "RO_ice_water_path", 
            "RO_ice_water_path_uncertainty", 
            "RO_ice_phase_fraction", 
            "RO_CWC_status", 
        ])
    ])

    dataset = xr.merge([
        dataset,
        open_cloudsat(flxhr_file[0], variable=[
            "FD", 
            "FD_NC", 
            "FD_NA", 
            "FU", 
            "FU_TOA", 
            "FD_TOA_IncomingSolar", 
            "FU_NC", 
            "FU_NC_TOA", 
            "FU_NA", 
            "FU_NA_TOA", 
            "QR", 
            "RH", 
            "COD", 
            "TOACRE", 
            "BOACRE", 
        ]).drop_vars("SurfaceHeightBin")
    ])

    dataset = xr.merge([
        dataset,
        open_cloudsat(ice_file[0], variable=[
            "re", 
            "IWC", 
            "re_uncertainty", 
            "IWC_uncertainty", 
            "ice_water_path", 
            "ice_water_path_uncertainty", 
            "optical_depth", 
            "optical_depth_uncertainty",
        ])
    ])

    save_name = f'{geoprof_file[0].name[:22]}_merged.nc'
    save_path = pathlib.Path("/work/scratch-nopw2/wkjones/cloudsat_temp/")
    save_path.mkdir(parents=True, exist_ok=True)

    dataset["Profile_time"] = dataset.Profile_time.drop_attrs()

    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in dataset.data_vars:
        dataset[var].encoding.update(comp)
    
    dataset.to_netcdf(save_path/save_name)

    print("Uploading to GCP")
    
    from google.cloud import storage
    import os
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "" # TODO: Add credentials
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("") # TODO: ADD project
    blob = bucket.blob(f'cloudsat/merged/{save_name}')
    blob.upload_from_filename(save_path/save_name)

    (save_path/save_name).unlink()

    print(f'Finished processing for {save_name}')
    
if __name__ == "__main__":
    main()