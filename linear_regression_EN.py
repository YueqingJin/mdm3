import matplotlib.pyplot as plt

"""
ERA5-driven linear regression baseline for basin-mean daily rainfall (2015 only)

- Use ERA5Local to read atmospheric variables required by AFNO (year 2015)
- Compute area-weighted mean over the Carlisle basin (NRFA 76007)
- Aggregate 6-hourly values into daily mean features
- Align with E-OBS basin-mean daily precipitation
- Train from 2015-01-01 to 2015-06-30, test from 2015-07-01 to 2015-12-31, linear regression
"""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import regionmask

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from earth2studio.models.dx import PrecipitationAFNOv2
from earth2studio.utils.time import to_time_array
from era5_local import ERA5Local


# ----------------- AOI / Basin-mean utilities -----------------

AOI_SHP = "data/era5/nrfa_76007/NRFA_catchments.shp"
STATION_ID = "76007"   # Eden at Sheepmount
ERA5_ROOT = "data/era5"
EOBS_CSV  = "EOBS_areal_0024.csv"


def standardize_latlon(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Standardize coordinates to lat/lon and shift longitude to [-180, 180)."""
    ds = obj.to_dataset(name="__tmp__") if isinstance(obj, xr.DataArray) else obj

    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if "y" in ds.coords and "lat" not in ds.coords:
        ren["y"] = "lat"
    if "x" in ds.coords and "lon" not in ds.coords:
        ren["x"] = "lon"
    if ren:
        ds = ds.rename(ren)

    if "lat" not in ds.coords or "lon" not in ds.coords:
        raise RuntimeError("Latitude/longitude coordinates not found.")

    lon = ds["lon"].values
    if np.nanmax(lon) > 180:
        lon2 = ((lon + 180) % 360) - 180
        order = np.argsort(lon2)
        ds = ds.assign_coords(lon=("lon", lon2)).isel(lon=order)

    return ds["__tmp__"] if isinstance(obj, xr.DataArray) else ds


def build_aoi_region() -> regionmask.Regions:
    """Create regionmask Regions object from shapefile."""
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(AOI_SHP)
    gdf = gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)

    if "ID_STRING" in gdf.columns:
        sel = gdf[gdf["ID_STRING"].astype(str).str.fullmatch(rf"0*{STATION_ID}")]
    elif "ID" in gdf.columns:
        sel = gdf[gdf["ID"].astype(str).str.fullmatch(rf"0*{STATION_ID}")]
    else:
        mask = np.zeros(len(gdf), dtype=bool)
        for c in gdf.columns:
            if c == "geometry":
                continue
            mask |= gdf[c].astype(str).str.contains(
                rf"\b0*{STATION_ID}\b", case=False, na=False
            )
        sel = gdf[mask]
    if sel.empty:
        raise RuntimeError(f"Station ID {STATION_ID} not found in {AOI_SHP}")

    from shapely.ops import unary_union
    geom = unary_union(sel.geometry)
    rgn = regionmask.Regions(outlines=[geom])
    return rgn


def area_weighted_mean_4d(da4: xr.DataArray, rgn: regionmask.Regions) -> xr.DataArray:
    """
    Compute basin area-weighted mean for (time, variable, lat, lon) DataArray,
    and return (time, variable).
    """
    da4 = standardize_latlon(da4)

    mask2d = rgn.mask(da4["lon"], da4["lat"]).notnull()

    w_lat = np.cos(np.deg2rad(da4["lat"]))
    w2d = w_lat.broadcast_like(da4.isel(time=0, variable=0))
    w4 = w2d.broadcast_like(da4.isel(time=0))   # (variable, lat, lon)
    w4 = w4.broadcast_like(da4)                 # (time, variable, lat, lon)

    num = (da4.where(mask2d) * w4).sum(("lat", "lon"))
    den = (w4.where(mask2d)).sum(("lat", "lon"))
    return num / den


# ----------------- Main workflow -----------------


def main():
    # 1. AFNO package, only to get input variable names
    package = PrecipitationAFNOv2.load_default_package()
    model = PrecipitationAFNOv2.load_model(package)
    required_vars = [str(x) for x in model.input_coords()["variable"]]
    print("[INFO] AFNOv2 input variables:", required_vars)

    data = ERA5Local(root=ERA5_ROOT)
    rgn = build_aoi_region()

    # 2. 6-hourly time axis for 2015
    START = "2015-01-01 00:00"
    END   = "2015-12-31 18:00"
    times_dt = pd.date_range(START, END, freq="6H").to_pydatetime().tolist()
    print(f"[INFO] 2015 6-hourly: {times_dt[0]} .. {times_dt[-1]} "
          f"({len(times_dt)} steps)")

    feats_list = []
    BATCH = 32

    for k in range(0, len(times_dt), BATCH):
        batch_times_dt = times_dt[k:k + BATCH]
        print(f"[INFO] batch {k} – {k + len(batch_times_dt) - 1}")

        # Use ERA5Local to get raw fields (time, variable, lat, lon)
        x_raw_bt = data(to_time_array(batch_times_dt), required_vars)

        da_bt = xr.DataArray(
            x_raw_bt.values,
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": x_raw_bt["time"],
                "variable": [str(v) for v in x_raw_bt["variable"].values],
                "lat": x_raw_bt["lat"],
                "lon": x_raw_bt["lon"],
            },
            name="era5_vars",
        )

        aoi_bt = area_weighted_mean_4d(da_bt, rgn)  # (time, variable)
        feats_list.append(aoi_bt)

    da_all = xr.concat(feats_list, dim="time").sortby("time")
    print("[INFO] AOI-mean ERA5 shape:", da_all.shape)

    # 3. Aggregate to daily scale: daily mean
    daily = da_all.resample(time="1D").mean(skipna=True)

    df_feat = daily.to_dataframe(name="value").reset_index()
    df_feat["date"] = pd.to_datetime(df_feat["time"]).dt.date
    df_feat = df_feat.drop(columns=["time"])

    df_feat_wide = df_feat.pivot_table(
        index="date", columns="variable", values="value"
    ).reset_index()

    print("[INFO] feature columns:", list(df_feat_wide.columns))

    # 4. Read E-OBS basin-mean daily precipitation, use 2015 only
    obs = pd.read_csv(EOBS_CSV)
    obs["date"] = pd.to_datetime(obs["date"]).dt.date
    obs_2015 = obs[(obs["date"] >= pd.to_datetime("2015-01-01").date()) &
                   (obs["date"] <= pd.to_datetime("2015-12-31").date())]

    df_all = df_feat_wide.merge(obs_2015, on="date", how="inner")
    df_all = df_all.sort_values("date").reset_index(drop=True)
    print("[INFO] merged rows (2015):", len(df_all))

    df_all["date_dt"] = pd.to_datetime(df_all["date"])

    feature_cols = [c for c in df_all.columns
                    if c not in ("date", "date_dt", "OBS")]

    # Training set: Jan–Jun; test set: Jul–Dec
    df_all["month"] = df_all["date_dt"].dt.month
    train_mask = df_all["month"] <= 6
    test_mask  = df_all["month"] >= 7

    X_train = df_all.loc[train_mask, feature_cols].values
    y_train = df_all.loc[train_mask, "OBS"].values
    X_test  = df_all.loc[test_mask,  feature_cols].values
    y_test  = df_all.loc[test_mask,  "OBS"].values

    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    # 5. Linear regression
    reg = LinearRegression().fit(X_train, y_train)
    y_pred_raw = reg.predict(X_test)  # raw linear regression output
    y_pred = np.clip(y_pred_raw, 0.0, None)  # clip all negative predictions to 0

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    # ===== Export only test-set values =====
    df_test_out = df_all.loc[test_mask, ["date_dt", "OBS"]].copy()
    df_test_out.rename(columns={"date_dt": "date"}, inplace=True)
    df_test_out["LIN_baseline"] = y_pred  # linear regression prediction

    os.makedirs("outputs_baseline", exist_ok=True)
    out_path = "outputs_baseline/era5_linear_baseline_2015_H2_values.csv"
    df_test_out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")

    print("\n=== 2015 ERA5-based linear regression baseline ===")
    print("Features:", feature_cols)
    print("Intercept:", reg.intercept_)
    print("Coefficients:")
    for name, coef in zip(feature_cols, reg.coef_):
        print(f"  {name:15s}: {coef:.4f}")
    print(f"R²  = {r2:.3f}")
    print(f"RMSE = {rmse:.3f} mm/day")
    print(f"MAE  = {mae:.3f} mm/day")
    # ===== 6. Plot: test-set time series & scatter =====
    df_test = df_all.loc[test_mask, ["date_dt", "OBS"]].copy()
    df_test["LIN_baseline"] = y_pred

    # (1) Time series comparison
    plt.figure(figsize=(9, 4))
    plt.plot(df_test["date_dt"], df_test["OBS"],
             label="Observed", linewidth=1.5)
    plt.plot(df_test["date_dt"], df_test["LIN_baseline"],
             label="Linear baseline", linewidth=1.5)
    plt.xlabel("Date")
    plt.ylabel("Basin-mean rainfall (mm/day)")
    plt.title("Observed vs ERA5 linear baseline (2015 Jul–Dec)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # (2) Observed vs predicted scatter plot
    plt.figure(figsize=(5, 5))
    plt.scatter(df_test["OBS"], df_test["LIN_baseline"], alpha=0.7)
    mn, mx = df_test["OBS"].min(), df_test["OBS"].max()
    plt.plot([mn, mx], [mn, mx], "r--", label="1:1 line")
    plt.xlabel("Observed (mm/day)")
    plt.ylabel("Linear baseline prediction (mm/day)")
    plt.title("Observed vs predicted (2015 Jul–Dec)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 6. Save test-set predictions
    out = df_all.loc[test_mask, ["date"]].copy()
    out["OBS"] = y_test
    out["LIN_baseline"] = y_pred
    os.makedirs("outputs_baseline", exist_ok=True)
    out.to_csv("outputs_baseline/era5_linear_baseline_2015_H2.csv",
               index=False)
    print("[OK] wrote outputs_baseline/era5_linear_baseline_2015_H2.csv")


if __name__ == "__main__":
    main()
