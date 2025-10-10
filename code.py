import os, json, time, zipfile, glob, shutil
import numpy as np
import xarray as xr

INIT_TIME = "2025-10-07T00:00:00Z"
LEAD_HOURS = 6
BBOX = {"lon_min": -2.9, "lat_min": 51.2, "lon_max": -2.1, "lat_max": 51.7}
VARIABLES = ["tp", "t2m", "w10m"]
THRESH_MM = 50.0
API = "https://climate.api.nvidia.com/v1/nvidia/fourcastnet"

key = os.environ.get("NGC_API_KEY")
if not key:
    raise SystemExit("Set NGC_API_KEY before running")

def fetch_zip():
    import requests
    payload = {
        "model": "fourcastnet",
        "init_time": INIT_TIME,
        "lead_time_hours": LEAD_HOURS,
        "bbox": BBOX,
        "variables": VARIABLES,
        "ensemble_size": 1,
        "noise_amplitude": 0.0
    }
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {key}",
               "NVCF-POLL-SECONDS": "5"}
    r = requests.post(API, headers=headers, data=json.dumps(payload), allow_redirects=False)
    if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("application/zip"):
        return r.content
    if r.status_code in (301, 302, 303):
        loc = r.headers.get("Location")
        fr = requests.get(loc, headers={"Authorization": f"Bearer {key}"})
        if fr.status_code == 200 and fr.headers.get("Content-Type", "").startswith("application/zip"):
            return fr.content
    if r.status_code == 202:
        req_id = r.headers.get("Nvcf-Reqid") or r.headers.get("nvcf-reqid")
        poll_url = f"{API}/{req_id}"
        import requests as rq
        t0 = time.time()
        while time.time() - t0 < 300:
            pr = rq.get(poll_url, headers={"Authorization": f"Bearer {key}"}, allow_redirects=False)
            if pr.status_code in (301, 302, 303):
                loc = pr.headers.get("Location")
                fr = rq.get(loc, headers={"Authorization": f"Bearer {key}"})
                if fr.status_code == 200 and fr.headers.get("Content-Type", "").startswith("application/zip"):
                    return fr.content
            if pr.status_code == 200 and pr.headers.get("Content-Type", "").startswith("application/zip"):
                return pr.content
            time.sleep(5)
    raise RuntimeError(f"Unexpected response: {r.status_code} {r.text[:150]}")

def unzip_and_load(zbytes):
    shutil.rmtree("unzipped", ignore_errors=True)
    os.makedirs("unzipped", exist_ok=True)
    with open("output.zip", "wb") as f:
        f.write(zbytes)
    with zipfile.ZipFile("output.zip", "r") as z:
        z.extractall("unzipped")
    files = sorted(glob.glob("unzipped/*.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds

def save_tif(da, name):
    import rioxarray
    if "time" in da.dims:
        da = da.sum("time") if da.name == "tp" else da.mean("time")
    if da.dims[-2:] != ("lat", "lon"):
        da = da.rename({da.dims[-2]: "lat", da.dims[-1]: "lon"})
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")
    da.rio.to_raster(name)

def main():
    z = fetch_zip()
    ds = unzip_and_load(z)
    if "tp" in ds:
        rain = (ds["tp"] * 1000).rename("tp")
        save_tif(rain, "rain_mm.tif")
        import rioxarray
        r = rain.sum("time") if "time" in rain.dims else rain
        r = r.rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")
        (r >= THRESH_MM).astype(np.uint8).rio.to_raster("flood_mask.tif")
    if "t2m" in ds:
        temp = (ds["t2m"] - 273.15).rename("t2m")
        save_tif(temp, "t2m_c.tif")
    if "w10m" in ds:
        wind = ds["w10m"].rename("w10m")
        save_tif(wind, "w10m_ms.tif")
    print("Generated rain_mm.tif, flood_mask.tif, t2m_c.tif, w10m_ms.tif")

if __name__ == "__main__":
    main()


#then i rain this on terminal in order to get output otherwise wont really work
#so on powershell put this .\.venv\Scripts\Activate.ps1
# python code.py

#and then yh you get four .tif files let me know if there is any problem in it but have run it working
#quite well might have edited something after tho so pls let me know have a look yh

# $env:NGC_API_KEY = "YOUR_KEY_HERE"  the your key here paste ur ngc api key
