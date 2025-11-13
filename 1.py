#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


from earth2studio.data import ARCO


bbox = (-3.10, 54.80, -2.80, 55.05)  
start_time = "2014-01-01 00:00"
end_time = "2015-12-31 23:00"

out_dir = Path("outputs_rain")


def guess_latlon_names(arr):
    
    lat_dim = [d for d in arr.dims if "lat" in d.lower()]
    lon_dim = [d for d in arr.dims if "lon" in d.lower()]
    
    return lat_dim[0], lon_dim[0]


def cut_box(data, region):
   

    west, south, east, north = region
    lat, lon = guess_latlon_names(data)

    
    if data[lon].min() >= 0:
        if west < 0:
            west += 360
        if east < 0:
            east += 360

    if data[lat][0] > data[lat][-1]:
        lat_slice = slice(north, south)
    else:
        lat_slice = slice(south, north)

    lon_slice = slice(west, east)

    return data.sel({lat: lat_slice, lon: lon_slice})


def calc_area_avg_mm_hr(data_arr):
   

    lat_name, _ = guess_latlon_names(data_arr)

    weights = np.cos(np.deg2rad(data_arr[lat_name]))
    weights = weights / weights.mean()  # normalized it so the weights wont skew hopefully

    rainfall_mm = data_arr * 1000.0  # m to mm

    avg_dims = []
    for dim in rainfall_mm.dims:
        if dim != "time":
            avg_dims.append(dim)

    weighted_mean = (rainfall_mm * weights).mean(dim=avg_dims)

    return weighted_mean


def run_fcn_precip():
    

    from earth2studio.models.px import FCN
    from earth2studio.models.dx import PrecipitationAFNO
    from earth2studio.io import ZarrBackend
    import earth2studio.run as runner

    
    input_data = ARCO()

   
    px_model = FCN.load_model(FCN.load_default_package())

   
    dx_model = PrecipitationAFNO.load_model(PrecipitationAFNO.load_default_package())

    zarr_path = out_dir / "fcn_precip_2014_2015.zarr"
    backend = ZarrBackend(file_name=str(zarr_path))

    
    nsteps = 3000

    
    backend = runner.diagnostic([start_time], nsteps, px_model, dx_model, input_data, backend)

    ds = xr.open_zarr(zarr_path)

    
    precip_var = None
    for var in ds.data_vars:
        if "precip" in var.lower() or var in ("tp", "total_precipitation"):
            precip_var = var
            break

    if not precip_var:
        print("DEBUG: Dataset vars were:", list(ds.data_vars))  # just double checking idk not sure
        raise RuntimeError("Rainfall variable not found in model output.")

    
    tp_6h = ds[precip_var].sel(time=slice(start_time, end_time))

    # Convert from 6h totals to hourly rate 
    tp_hourly = tp_6h / 6.0

    # Crop to region of interest
    tp_hourly_cropped = cut_box(tp_hourly, bbox)

    return tp_hourly_cropped


def main():
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    print(" Running FCN rainfall model chain...")

    rain_data = run_fcn_precip()

    hourly_rain = calc_area_avg_mm_hr(rain_data).to_series()
    hourly_rain.name = "fcn_rain_mm_hourly"

  
    daily_rain = hourly_rain.resample("1D").sum()
    daily_rain.name = "fcn_rain_mm_daily"

    # Save results to CSV
    out_csv = out_dir / "carlisle_daily_mean_rainfall_2014_2015.csv"
    daily_rain.to_csv(out_csv, header=True)

    print(f" Done! Output written to: {out_csv.resolve()}")


if __name__ == "__main__":
    
    main()
