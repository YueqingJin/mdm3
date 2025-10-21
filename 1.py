#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from earth2studio.data import ARCO

BOX = (-3.10, 54.80, -2.80, 55.05)   # W, S, E, N
START, END = "2015-11-01 00:00", "2015-11-30 23:00"
OUT = Path("outputs_rain")

def dim_names(da):
    lat = [d for d in da.dims if "lat" in d.lower()][0]
    lon = [d for d in da.dims if "lon" in d.lower()][0]
    return lat, lon

def clip_box(da, box):
    w, s, e, n = box
    lat, lon = dim_names(da)
    if da[lon].min() >= 0:  # 0..360 longitudes
        if w < 0: w += 360
        if e < 0: e += 360
    sel_lat = slice(n, s) if da[lat][0] > da[lat][-1] else slice(s, n)
    sel_lon = slice(w, e)
    return da.sel({lat: sel_lat, lon: sel_lon})

def area_mean_mm_per_hour(tp_da):
    lat, _ = dim_names(tp_da)
    w = np.cos(np.deg2rad(tp_da[lat]))
    w = w / w.mean()
    mm = tp_da * 1000.0
    return (mm * w).mean(dim=[d for d in mm.dims if d not in ["time"]])

def make_plot(series, title, ylabel, outpath, kind="line"):
    plt.figure(figsize=(12, 4))
    if kind == "bar":
        series.plot(kind="bar")
    else:
        series.plot()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Time (UTC)" if kind == "line" else "Date (UTC)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(START, END, freq="1H").to_pydatetime().tolist()

    data = ARCO()
    tp = data(times, "tp")              # ERA5 total precip (m per hour)
    tp = clip_box(tp, BOX)

    hourly = area_mean_mm_per_hour(tp).to_series()
    hourly.name = "rain_mm_hourly"

    daily = hourly.resample("1D").sum()
    daily.name = "rain_mm_daily"

    hourly.to_csv(OUT / "rain_hourly_mm.csv", header=True)
    daily.to_csv(OUT / "rain_daily_mm.csv", header=True)

    make_plot(hourly, "Carlisle — Hourly Rainfall (mm), Nov 2015", "mm",
              OUT / "rain_hourly_mm.png", kind="line")
    make_plot(daily, "Carlisle — Daily Rainfall (mm), Nov 2015", "mm",
              OUT / "rain_daily_mm.png", kind="bar")

    print(f"Done → {OUT.resolve()}")

if __name__ == "__main__":
    main()
