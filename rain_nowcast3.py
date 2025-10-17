from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import torch
import xarray as xr
from earth2studio.data import prep_data_array
from earth2studio.models.dx import PrecipitationAFNOv2
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array

from era5_local import ERA5Local
import inspect, earth2studio
print("e2studio:", getattr(earth2studio, "__version__", "unknown"))
print("prep_data_array signature:", inspect.signature(prep_data_array))

def main():
    # 1) Target times: every 6h in Nov 2015 (00/06/12/18)
    times_dt = pd.date_range("2015-11-01 00:00", "2015-11-30 18:00", freq="6H").to_pydatetime().tolist()
    times = to_time_array(times_dt)
    print(f"[INFO] times: {times_dt[0]} .. {times_dt[-1]} ({len(times_dt)} steps)")

    # 2) Model
    package = PrecipitationAFNOv2.load_default_package()
    model = PrecipitationAFNOv2.load_model(package)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    model = model.to(device).eval()
    print(f"[INFO] device={device}")

    # 3) Data source
    data = ERA5Local(root="data/era5")
    required_vars = [str(x) for x in model.input_coords()["variable"]]
    print("[INFO] model needs variables:", required_vars)

    # 4) Stream per-time processing to avoid OOM
    preds = []
    os.makedirs("outputs", exist_ok=True)

    for i, t in enumerate(times_dt):
        # ---- read one time slice from ERA5 (xarray) ----
        x_raw_1t = data(to_time_array([t]), required_vars)  # (time=1, variable, lat, lon)

        # ---- standardize on CPU (saves VRAM) ----
        # --- figure out the input transform from the package (robust across versions) ---
        input_tf = None
        for key in ("input_transform", "transform", "input_tf"):
            if isinstance(package, dict) and key in package:
                input_tf = package[key]
                break
            if hasattr(package, key):
                input_tf = getattr(package, key)
                break

        # standardlize
        x1, c1 = prep_data_array(x_raw_1t, device="cpu")

        #remap to model grid
        x1, c1 = map_coords(x1, c1, model.input_coords())

        # ensure dims and move to inference device
        if "batch" not in c1:
            x1 = x1.unsqueeze(0)
            c1 = OrderedDict([("batch", np.arange(x1.shape[0]))] + list(c1.items()))
        if "lead_time" not in c1:
            x1 = x1.unsqueeze(2)
            c1 = OrderedDict([
                ("batch", c1["batch"]),
                ("time", c1["time"]),
                ("lead_time", np.array([0], dtype=np.int32)),
                ("variable", c1["variable"]),
                ("lat", c1["lat"]),
                ("lon", c1["lon"]),
            ])
        x1 = x1.to(device, non_blocking=True)

        # inference
        with torch.no_grad():
            y1, ycoords = model(x1, c1)  # typically (batch=1, time=1, lead=1, lat, lon)
        print(f"[DBG] t={t}  y1[min,max,mean]="
              f"{float(y1.min()):.3e},{float(y1.max()):.3e},{float(y1.mean()):.3e}, "
              f"nan%={(torch.isnan(y1).float().mean().item() * 100):.2f}%")

        # build DataArray robustly from ycoords, then squeeze singleton dims
        y_np = y1.detach().cpu().numpy()
        dims_list = list(ycoords.keys())

        # if tensor has fewer dims than coord keys, drop likely singleton names
        while y_np.ndim < len(dims_list):
            for cand in ("variable", "batch", "lead_time"):
                if cand in dims_list:
                    dims_list.remove(cand)
                    break
            else:
                dims_list.pop()

        da_full = xr.DataArray(
            y_np,
            dims=tuple(dims_list),
            coords={k: ycoords[k] for k in dims_list},
            name="precip_6h_m",
        )

        # squeeze size-1 dims except time (keep time if present)
        for d in ("batch", "lead_time", "variable"):
            if d in da_full.dims and da_full.sizes[d] == 1:
                da_full = da_full.isel({d: 0})

        # ensure we have a time dimension; if not, add one with current t
        if "time" not in da_full.dims:
            da_full = da_full.expand_dims(time=[np.datetime64(t)])

        # now expected shape: (time=1, lat, lon)
        preds.append(da_full)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[INFO] processed {i+1}/{len(times_dt)}")

    # 5) concat along time -> (time, lat, lon)
    da_all = xr.concat(preds, dim="time").sortby("time")

    # 6) units: m/6h -> mm/6h, then aggregate to daily (mm/day)
    da_mm6 = (da_all * 1000.0).assign_attrs(units="mm/6h")
    daily_mm = da_mm6.resample(time="1D").sum(skipna=True).assign_attrs(units="mm/day")  # (time, lat, lon)
    # set the area carslile
    # 54.895°N, -2.936°E
    lat_carlisle = 54.895
    lon_carlisle = -2.936

    # The sub-areas that you want to draw and save
    lat_min, lat_max = 48.0, 60.0
    lon_min, lon_max = -12.0, 5.0

    # Unify the longitude to [-180,180], and then perform regional cropping
    def to_lon180(ds):
        if "lon" in ds.coords:
            lon = ds["lon"].values
            if lon.max() > 180:  # 0..360 -> -180..180
                lon2 = ((lon + 180) % 360) - 180
                order = np.argsort(lon2)
                ds = ds.assign_coords(lon=("lon", lon2)).isel(lon=order)
        return ds

    daily_mm = to_lon180(daily_mm).sortby(["lat", "lon"])

    # (lat_max, lat_min)
    lat_slice = slice(lat_max, lat_min) if (daily_mm.lat[0] > daily_mm.lat[-1]) else slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)
    daily_mm_reg = daily_mm.sel(lat=lat_slice, lon=lon_slice)

    # Regional data is saved separately
    os.makedirs("outputs/region", exist_ok=True)
    daily_mm_reg.to_netcdf("outputs/region/precip_daily_2015-11_region.nc")

    # Take the daily precipitation of the nearest grid point of "Carlisle" and export it as a CSV file
    carlisle_series = daily_mm.sel(lat=lat_carlisle, lon=lon_carlisle, method="nearest")
    df_carlisle = carlisle_series.to_dataframe(name="precip_mm_day").reset_index()
    df_carlisle["date"] = df_carlisle["time"].dt.date
    df_carlisle = df_carlisle[["date", "precip_mm_day"]]
    df_carlisle.to_csv("outputs/precip_daily_2015-11_carlisle_from_model.csv", index=False)
    print("[OK] wrote outputs/precip_daily_2015-11_carlisle_from_model.csv")

    #  save daily field
    ds_out = daily_mm.to_dataset(name="precip_mm_day")
    out_nc = "outputs/precip_daily_2015-11_afnov2.nc"
    ds_out.to_netcdf(out_nc)
    print(f"[OK] Saved {out_nc}")

    #  maps: monthly total + daily maps (lon/lat axes)
    monthly_total = daily_mm.sum("time")  # (lat, lon)
    os.makedirs("outputs/maps", exist_ok=True)

    def plot_map(da2d: xr.DataArray, title: str, fname: str, units: str = "mm", vmax=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.2, 5.6))
        if vmax is None:
            pcm = plt.pcolormesh(da2d["lon"], da2d["lat"], da2d, shading="auto")
        else:
            pcm = plt.pcolormesh(da2d["lon"], da2d["lat"], da2d, shading="auto", vmin=0, vmax=vmax)
        cb = plt.colorbar(pcm, pad=0.02)
        cb.set_label(units)
        plt.xlabel("Longitude (°)")
        plt.ylabel("Latitude (°)")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    plot_map(
        monthly_total,
        "AFNOv2 precip — 2015-11 TOTAL (mm)",
        "outputs/maps/precip_2015-11_monthly_total.png",
        units="mm",
        vmax=300,
    )

    # daily
    for t in pd.to_datetime(daily_mm["time"].values):
        da_day = daily_mm.sel(time=t)
        day_str = pd.Timestamp(t).strftime("%Y-%m-%d")
        plot_map(
            da_day,
            f"AFNOv2 precip — {day_str} (mm/day)",
            f"outputs/maps/precip_{day_str}.png",
            units="mm/day",
            vmax=20,
        )

    print(" - outputs/maps/precip_2015-11_monthly_total.png")
    print(" - outputs/maps/precip_YYYY-MM-DD.png (daily maps)")


if __name__ == "__main__":
    main()
