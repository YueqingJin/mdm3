from collections import OrderedDict
import os, inspect
import numpy as np
import pandas as pd
import torch
import xarray as xr

# Adapted for earth2studio 0.10.0a0
from earth2studio.data import prep_data_array
from earth2studio.models.dx import PrecipitationAFNOv2
from earth2studio.utils.coords import map_coords
from earth2studio.utils.time import to_time_array

import geopandas as gpd
import regionmask
from shapely.ops import unary_union

import earth2studio
from era5_local import ERA5Local
REGION = (-12.0, 5.0, 48.0, 60.0)   # (lon_min, lon_max, lat_min, lat_max), UK box; set to None to disable cropping
BATCH = 8                           # Number of 6h steps per batch (16 -> 8 or 4); reduce to 4 if memory limited
USE_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if USE_MPS else "cpu"))
DTYPE = torch.float16 if (device.type in ("cuda", "mps")) else torch.float32


# Find output inverse transform
def _find_inverse_transform(obj, depth=0, max_depth=4):
    if obj is None or depth > max_depth:
        return None
    if hasattr(obj, "inverse") and callable(getattr(obj, "inverse")):
        return obj
    try:
        from collections.abc import Mapping
        if isinstance(obj, Mapping):
            for v in obj.values():
                r = _find_inverse_transform(v, depth + 1, max_depth)
                if r is not None:
                    return r
    except Exception:
        pass
    try:
        for name in dir(obj):
            if "transform" in name.lower() or name.lower().endswith("tf"):
                try:
                    r = _find_inverse_transform(getattr(obj, name), depth + 1, max_depth)
                    if r is not None:
                        return r
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _get_output_transform(package):
    for k in ("output_transform", "transform_out", "output_tf", "target_transform"):
        if isinstance(package, dict) and k in package:
            return package[k]
        if hasattr(package, k):
            return getattr(package, k)
    return None


#  Station coordinates (can be overridden by environment variables)
def get_station_coords(station_id: str):
    lat_env = os.getenv("STATION_LAT"); lon_env = os.getenv("STATION_LON")
    if lat_env and lon_env:
        return float(lat_env), float(lon_env)
    if str(station_id) == "76007":  # Eden at Sheepmount (NRFA)
        return 54.90486, -2.952852
    raise ValueError(
        f"No default coordinates for station {station_id}; "
        f"set STATION_LAT/LON to override."
    )

----------
def save_series_nc_csv(da: xr.DataArray, nc_path: str, csv_path: str, round_cols: dict | None = None):
    da1 = da.reset_coords(drop=True)
    os.makedirs(os.path.dirname(nc_path), exist_ok=True)
    da1.to_netcdf(nc_path)
    df = da1.to_dataframe().reset_index()
    if round_cols:
        for col, n in round_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(n)
    df.to_csv(csv_path, index=False)
    print(f"[OK] wrote {nc_path}")
    print(f"[OK] wrote {csv_path}")


# 6h -> 1h split
def load_climo_weights(csv_path="outputs/weights_hourly_carlisle_2005-2014.csv"):
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} does not exist, using equal weights.")
        return pd.DataFrame([(sh, i, 1/6.0) for sh in [0, 6, 12, 18] for i in range(6)],
                            columns=["start_hour", "h_in6", "weight"])
    tbl = pd.read_csv(csv_path)
    if "start_hour" not in tbl.columns or "h_in6" not in tbl.columns:
        tbl = pd.read_csv(csv_path, index_col=[0, 1]).reset_index()
        if tbl.columns.size >= 3:
            tbl.columns = ["start_hour", "h_in6"] + list(tbl.columns[2:])
    val_col = "weight" if "weight" in tbl.columns else ("tp" if "tp" in tbl.columns else None)
    if val_col is None:
        raise ValueError(f"weights CSV missing 'weight' or 'tp' column: {csv_path}")
    return tbl.rename(columns={val_col: "weight"})[["start_hour", "h_in6", "weight"]].copy()


def split_6h_to_hourly(da_6h: xr.DataArray, weights_tbl: pd.DataFrame) -> xr.DataArray:
    times_6h = pd.to_datetime(da_6h["time"].values)
    hourly_list = []
    W = {}
    for sh in [0, 6, 12, 18]:
        w = (weights_tbl[weights_tbl["start_hour"] == sh]
             .sort_values("h_in6")["weight"].to_numpy())
        if len(w) != 6:
            w = np.ones(6) / 6.0
        W[sh] = w / w.sum()
    for t0 in times_6h:
        sh = (pd.Timestamp(t0).hour // 6) * 6
        w = W.get(sh, np.ones(6) / 6.0)
        slab = da_6h.sel(time=t0)
        hours = [t0 + pd.Timedelta(hours=i) for i in range(6)]
        for i, hi in enumerate(hours):
            hourly_list.append(
                xr.DataArray(
                    slab.values * w[i],
                    dims=("lat", "lon"),
                    coords={"lat": slab["lat"], "lon": slab["lon"], "time": hi}
                )
            )
    da_hourly = xr.concat(hourly_list, dim="time").sortby("time")
    return da_hourly.assign_attrs(units="mm/h")


#Longitude to [-180,180)
def to_lon180(ds):
    if "lon" in ds.coords:
        lon = ds["lon"].values
        if float(np.nanmax(lon)) > 180:
            lon2 = ((lon + 180) % 360) - 180
            order = np.argsort(lon2)
            ds = ds.assign_coords(lon=("lon", lon2)).isel(lon=order)
    return ds

def main():
    print("torch:", torch.__version__)
    print("mps available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    print("e2studio:", getattr(earth2studio, "__version__", "unknown"))
    print("prep_data_array signature:", inspect.signature(prep_data_array))

    #Run a single month:
    OUT_TAG = "2014-12"

    #  6h time axis for this month (00:00 to 18:00)
    last_day = pd.Period(OUT_TAG, freq="M").days_in_month
    START, END = f"{OUT_TAG}-01 00:00", f"{OUT_TAG}-{last_day:02d} 18:00"
    times_dt = pd.date_range(START, END, freq="6H").to_pydatetime().tolist()
    print(f"[RUN] {OUT_TAG}: {times_dt[0]} .. {times_dt[-1]}  ({len(times_dt)} steps)")

    # Models  data source
    package = PrecipitationAFNOv2.load_default_package()
    model = PrecipitationAFNOv2.load_model(package)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    model = model.to(device).eval()
    tf_out = _get_output_transform(package) or _find_inverse_transform(package) or _find_inverse_transform(model)

    data = ERA5Local(root="data/era5")
    required_vars = [str(x) for x in model.input_coords()["variable"]]
    print("[INFO] model needs variables:", required_vars)

    #Directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/series", exist_ok=True)
    os.makedirs("outputs/region", exist_ok=True)
    os.makedirs("outputs/maps", exist_ok=True)

    # AOI
    AOI_DIR = "data/era5/nrfa_76007"
    AOI_SHP = f"{AOI_DIR}/NRFA_catchments.shp"
    station_id = "76007"
    os.environ["SHAPE_RESTORE_SHX"] = "YES"

    gdf = gpd.read_file(AOI_SHP)
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
    if "ID_STRING" in gdf.columns:
        sel = gdf[gdf["ID_STRING"].astype(str).str.fullmatch(rf"0*{station_id}")]
    elif "ID" in gdf.columns:
        sel = gdf[gdf["ID"].astype(str).str.fullmatch(rf"0*{station_id}")]
    else:
        mask = np.zeros(len(gdf), dtype=bool)
        for c in gdf.columns:
            if c == "geometry": continue
            mask |= gdf[c].astype(str).str.contains(rf"\b0*{station_id}\b", case=False, na=False)
        sel = gdf[mask]
    if sel.empty:
        raise RuntimeError(f"Station ID {station_id} not found in {AOI_SHP}.")
    geom = unary_union(sel.geometry)
    rgn = regionmask.Regions(outlines=[geom])

    station_lat, station_lon = get_station_coords(station_id)
    print(f"[INFO] station {station_id} -> lat={station_lat:.5f}, lon={station_lon:.5f}")

    #Inference (this month)
    preds = []
    BATCH = 16
    for k in range(0, len(times_dt), BATCH):
        batch_times_dt = times_dt[k:k + BATCH]

        # Input preparation (normalization + mapping on CPU)
        x_raw_bt = data(to_time_array(batch_times_dt), required_vars)
        x_bt, c0 = prep_data_array(x_raw_bt, device="cpu")
        x_bt, c0 = map_coords(x_bt, c0, model.input_coords())
        print("[CHK] mapped NaN ratio:", float(torch.isnan(x_bt).float().mean()))

        nvar = int(x_raw_bt.sizes["variable"])
        # Prefer names from coord; if length mismatch, fall back to required_vars
        try:
            names_from_coord = [str(v) for v in np.atleast_1d(x_raw_bt.coords["variable"].values)]
        except Exception:
            names_from_coord = []
        if len(names_from_coord) == nvar:
            var_names = names_from_coord
        else:
            var_names = [required_vars[i] if i < len(required_vars) else f"var{i}" for i in range(nvar)]

        bad = []
        for i in range(nvar):
            slab0 = x_raw_bt.isel(variable=i, time=0).values  # Select by index to avoid dtype/label mismatches
            if np.isnan(slab0).all():
                bad.append(var_names[i])

        if bad:
            raise RuntimeError(
                f"[{OUT_TAG}] these variables are all NaN after alignment for this month: {bad} — "
                f"most likely ERA5 files for {OUT_TAG} are missing or times are misaligned."
            )

        x_bt = x_bt.unsqueeze(0).unsqueeze(2).to(device, non_blocking=True, dtype=torch.float32)

        # Coordinate dict
        lead_arr = np.array([21600], dtype=np.int32)
        c_bt = OrderedDict([
            ("batch", np.array([0])),
            ("time", np.array(batch_times_dt, dtype="datetime64[ns]")),
            ("lead_time", lead_arr),
            ("variable", c0["variable"]),
            ("lat", c0["lat"]),
            ("lon", c0["lon"]),
        ])

        # Inference (no half precision)
        with torch.no_grad():
            y_bt, ycoords = model(x_bt, c_bt)

        if tf_out is not None:
            y_bt = tf_out.inverse(y_bt, ycoords)
        y_bt = torch.clamp(y_bt, min=0)

        # xarray
        y_np = np.squeeze(y_bt.detach().cpu().numpy())
        da_bt = xr.DataArray(
            y_np, dims=("time", "lat", "lon"),
            coords={"time": pd.to_datetime(batch_times_dt),
                    "lat": ycoords["lat"], "lon": ycoords["lon"]},
            name="precip_6h_m",
        )
        preds.append(da_bt)
        print(f"[INFO] {OUT_TAG} processed {min(k + BATCH, len(times_dt))}/{len(times_dt)}")

        # release
        del x_bt, y_bt, y_np

    # Combine (mm/6h)
    da_all = xr.concat(preds, dim="time").sortby("time")
    da_mm6 = (da_all * 1000.0).assign_attrs(units="mm/6h")

    # 6h -> 1h
    weights_tbl = load_climo_weights("outputs/weights_hourly_carlisle_2005-2014.csv")
    da_hourly = split_6h_to_hourly(da_mm6, weights_tbl)  # (time,lat,lon), mm/h
    da_hourly.to_netcdf(f"outputs/precip_hourly_{OUT_TAG}_climo.nc")      # gridded: nc

    # Daily accumulation (00–24)
    daily_00 = da_mm6.resample(time="1D").sum(skipna=True).assign_attrs(units="mm/day")
    daily_00.to_dataset(name="precip_mm_day").to_netcdf(f"outputs/precip_daily_{OUT_TAG}_afnov2.nc")  # gridded: nc

    # AOI area mean export NC+CSV ----
    mask2d = rgn.mask(da_hourly.lon, da_hourly.lat).notnull()
    w_lat = np.cos(np.deg2rad(da_hourly["lat"]))
    w2d = w_lat.broadcast_like(da_hourly.isel(time=0))

    num_h = (da_hourly.where(mask2d) * w2d).sum(("lat","lon"))
    den_h = (w2d.where(mask2d)).sum(("lat","lon"))
    aoi_hourly_mean = (num_h / den_h).rename("precip_mm_h")
    save_series_nc_csv(
        aoi_hourly_mean,
        f"outputs/series/aoi_hourly_{OUT_TAG}_mm_h.nc",
        f"outputs/series/aoi_hourly_{OUT_TAG}_mm_h.csv",
        round_cols={"precip_mm_h": 4},
    )

    num_d = (daily_00.where(mask2d) * w2d).sum(("lat","lon"))
    den_d = (w2d.where(mask2d)).sum(("lat","lon"))
    aoi_daily_00 = (num_d / den_d).rename("precip_mm_day")
    df_d = aoi_daily_00.to_dataframe().reset_index()
    df_d["date"] = pd.to_datetime(df_d["time"]).dt.date
    df_d[["date","precip_mm_day"]].to_csv(
        f"outputs/precip_daily_{OUT_TAG}_AOI_mean_00UTC.csv", index=False)  # for quick lookup
    save_series_nc_csv(
        aoi_daily_00,
        f"outputs/series/aoi_daily_{OUT_TAG}_00UTC_mm_day.nc",
        f"outputs/series/aoi_daily_{OUT_TAG}_00UTC_mm_day.csv",
        round_cols={"precip_mm_day": 3},
    )

    # Station time series  export NC+CSV
    pt_hourly_nn = da_hourly.sel(lat=station_lat, lon=station_lon, method="nearest").rename("precip_mm_h")
    pt_hourly_bi = da_hourly.interp(lat=station_lat, lon=station_lon).rename("precip_mm_h")
    save_series_nc_csv(
        pt_hourly_nn,
        f"outputs/series/station_{station_id}_hourly_{OUT_TAG}_mm_h_nn.nc",
        f"outputs/series/station_{station_id}_hourly_{OUT_TAG}_mm_h_nn.csv",
        round_cols={"precip_mm_h": 4},
    )
    save_series_nc_csv(
        pt_hourly_bi,
        f"outputs/series/station_{station_id}_hourly_{OUT_TAG}_mm_h_bilin.nc",
        f"outputs/series/station_{station_id}_hourly_{OUT_TAG}_mm_h_bilin.csv",
        round_cols={"precip_mm_h": 4},
    )

    pt_daily_nn = daily_00.sel(lat=station_lat, lon=station_lon, method="nearest").rename("precip_mm_day")
    pt_daily_bi = daily_00.interp(lat=station_lat, lon=station_lon).rename("precip_mm_day")
    save_series_nc_csv(
        pt_daily_nn,
        f"outputs/series/station_{station_id}_daily_{OUT_TAG}_00UTC_mm_day_nn.nc",
        f"outputs/series/station_{station_id}_daily_{OUT_TAG}_00UTC_mm_day_nn.csv",
        round_cols={"precip_mm_day": 3},
    )
    save_series_nc_csv(
        pt_daily_bi,
        f"outputs/series/station_{station_id}_daily_{OUT_TAG}_00UTC_mm_day_bilin.nc",
        f"outputs/series/station_{station_id}_daily_{OUT_TAG}_00UTC_mm_day_bilin.csv",
        round_cols={"precip_mm_day": 3},
    )

    # regional cropped grid (UK) — nc only
    daily_mm = to_lon180(daily_00).sortby(["lat","lon"])
    lat_min, lat_max = 48.0, 60.0; lon_min, lon_max = -12.0, 5.0
    lat_slice = slice(lat_max, lat_min) if (daily_mm.lat[0] > daily_mm.lat[-1]) else slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)
    daily_mm.sel(lat=lat_slice, lon=lon_slice).to_netcdf(
        f"outputs/region/precip_daily_{OUT_TAG}_region.nc")

    # map plot
    def plot_map(da2d: xr.DataArray, title: str, fname: str, units: str = "mm", vmax=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7.2, 5.6))
        if vmax is None: pcm = plt.pcolormesh(da2d["lon"], da2d["lat"], da2d, shading="auto")
        else:            pcm = plt.pcolormesh(da2d["lon"], da2d["lat"], da2d, shading="auto", vmin=0, vmax=vmax)
        cb = plt.colorbar(pcm, pad=0.02); cb.set_label(units)
        plt.xlabel("Longitude (°)"); plt.ylabel("Latitude (°)")
        plt.title(title); plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

    monthly_total = daily_00.sum("time")
    plot_map(monthly_total, f"AFNOv2 precip — {OUT_TAG} TOTAL (mm)",
             f"outputs/maps/precip_{OUT_TAG}_monthly_total.png", units="mm", vmax=300)

    print(f"[DONE] {OUT_TAG} — monthly grids and AOI/station time series written to outputs/ and outputs/series/")

if __name__ == "__main__":
    main()
