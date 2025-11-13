from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import xarray as xr
import fsspec
import gcsfs

OUTPUT_CSV = "carlisle_daily_mean_mm_2014_2015.csv"
TIME_START, TIME_END = "2014-01-01", "2015-12-31"


NORTH, WEST, SOUTH, EAST = 55.05, -3.10, 54.80, -2.80

# Public ARCO ERA5 (Google Cloud)
ARCO_GCS = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

def _open_zarr_gcs(url: str) -> xr.Dataset:
    
    fs = gcsfs.GCSFileSystem(token="anon", asynchronous=False)
    mapper = fs.get_mapper(url)
    return xr.open_zarr(mapper, consolidated=True)

def _normalise_coords(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})
    return ds

def _to_0360(lon):
    lon = np.asarray(lon)
    return np.where(lon < 0, lon + 360.0, lon)

def slice_bbox(ds: xr.Dataset, north: float, west: float, south: float, east: float) -> xr.Dataset:
    ds = _normalise_coords(ds)
    if float(ds.lon.max()) > 180.0:
        west, east = _to_0360(west), _to_0360(east)
    return ds.sel(lat=slice(float(north), float(south)),
                  lon=slice(float(west), float(east)))

def find_precip_var(ds: xr.Dataset) -> str:
    candidates = [
        "tp", "total_precipitation", "precipitation",
        "precipitation_amount", "total_precipitation_sum",
        "precipitation_sum"
    ]
    for name in candidates:
        if name in ds.data_vars:
            return name
    for name, da in ds.data_vars.items():
        attrs = " ".join(str(da.attrs.get(k, "")) for k in
                         ("long_name", "standard_name", "short_name", "shortName", "name")).lower()
        if "total precipitation" in attrs or ("precipitation" in attrs and "total" in attrs):
            return name
    for name in ds.data_vars:
        if "precip" in name.lower():
            return name
    raise KeyError("Could not find a precipitation variable (tp/total_precipitation/…)"
                   f"\nAvailable vars (first 40): {list(ds.data_vars)[:40]}")

def rain_mm_daily(tp: xr.DataArray) -> xr.DataArray:
    # Convert m to mm and sum per day
    return (tp * 1000.0).resample(time="1D").sum().rename("rain_mm_daily")

def run(save_path: str = OUTPUT_CSV) -> str:
    ds = _open_zarr_gcs(ARCO_GCS)
    ds = _normalise_coords(ds)
    precip_name = find_precip_var(ds)
    print(f"[INFO] Using precipitation variable: {precip_name}")

 
    out_written = False
    months = pd.period_range(TIME_START, TIME_END, freq="M")

    for p in months:
        start = str(p.start_time.normalize())
        end   = str(p.end_time.normalize())
        print(f"[LOAD] {p.strftime('%Y-%m')}")

        dsm = ds.sel(time=slice(start, end))
        dsm = slice_bbox(dsm, NORTH, WEST, SOUTH, EAST)

        daily_mean = rain_mm_daily(dsm[precip_name]).mean(dim=("lat", "lon"))
        df = daily_mean.to_pandas().to_frame("rain_mm_daily")
        df.index.name = "date"

       
        df.to_csv(save_path, mode="a", header=not out_written)
        out_written = True
        print(f"[OK] Wrote {len(df)} rows for {p.strftime('%Y-%m')}")

    print(f"[DONE] Saved {save_path}")
    return save_path

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(
            "\n[ERROR] Run failed.\n"
            f"Reason: {e}\n\n"
            "Checklist:\n"
            "  - Internet access required.\n"
            "  - In your venv, install: pip install -U xarray zarr fsspec gcsfs numpy pandas\n"
            "  - If behind a firewall/VPN, allow Google Cloud Storage.\n",
            file=sys.stderr
        )
        raise

