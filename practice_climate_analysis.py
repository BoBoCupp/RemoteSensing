"""
Science Olympiad 2026 Remote Sensing - Climate Data Analysis
=============================================================
Section 3d (interpretation) + Section 3e (climate processes)

Time-series and comparative analysis of climate change indicators
using the specific satellites listed in the rules.

Topics covered:
  - CO2 trends (OCO-2 context)
  - ENSO / El Nino detection via SST anomalies
  - Sea level and ice mass change (GRACE)
  - Greenhouse gas distributions
  - Precipitation pattern changes (GPM)
  - CERES energy imbalance

Generates study charts saved to ./output/
"""

import ee
import requests
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from PIL import Image

PROJECT_ID = "remote-sensing-research-492504"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

ee.Initialize(project=PROJECT_ID)
print("Connected to Earth Engine.\n")


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")




def extract_time_series(collection, band, region, scale, reducer=None):
    """Extract a time series of mean values from an ImageCollection."""
    if reducer is None:
        reducer = ee.Reducer.mean()

    def extract(image):
        val = image.select(band).reduceRegion(
            reducer=reducer, geometry=region, scale=scale, maxPixels=1e8
        )
        return image.set("value", val.get(band))

    results = collection.map(extract)
    dates = results.aggregate_array("system:time_start").getInfo()
    values = results.aggregate_array("value").getInfo()
    return dates, values


# ================================================================
# 1. MODIS SST Time Series - ENSO Detection
# ================================================================
print("[1/6] ENSO Detection via MODIS Sea Surface Temperature...")

# Nino 3.4 region - the standard ENSO monitoring area
nino34 = ee.Geometry.Rectangle([-170, -5, -120, 5])

modis_sst = (
    ee.ImageCollection("MODIS/061/MYD11A1")
    .filterBounds(nino34)
    .filterDate("2018-01-01", "2025-12-31")
    .select("LST_Day_1km")
)

# Monthly composites
months = []
sst_values = []
for year in range(2018, 2026):
    for month in range(1, 13):
        if year == 2025 and month > 9:
            break
        start = f"{year}-{month:02d}-01"
        end_month = month + 1 if month < 12 else 1
        end_year = year if month < 12 else year + 1
        end = f"{end_year}-{end_month:02d}-01"

        monthly = modis_sst.filterDate(start, end).mean()
        val = monthly.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=nino34,
            scale=5000, maxPixels=1e9
        ).get("LST_Day_1km").getInfo()

        if val is not None:
            sst_values.append(val * 0.02 - 273.15)  # Scale + K to C
            months.append(f"{year}-{month:02d}")

    print(f"    {year} done...")

# Calculate anomaly (deviation from monthly mean)
sst_arr = np.array(sst_values)
monthly_means = np.array([
    np.mean(sst_arr[i::12]) for i in range(12)
])
anomalies = []
for i, val in enumerate(sst_arr):
    anomalies.append(val - monthly_means[i % 12])

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Raw SST
axes[0].plot(range(len(months)), sst_values, "b-", linewidth=0.8)
axes[0].set_title("MODIS Aqua - Nino 3.4 Region Sea Surface Temperature", fontsize=13)
axes[0].set_ylabel("SST (°C)")
tick_positions = list(range(0, len(months), 12))
axes[0].set_xticks(tick_positions)
axes[0].set_xticklabels([months[i] for i in tick_positions], rotation=45)
axes[0].grid(True, alpha=0.3)

# SST Anomaly (ENSO index)
colors = ["red" if a > 0 else "blue" for a in anomalies]
axes[1].bar(range(len(anomalies)), anomalies, color=colors, width=1.0, alpha=0.7)
axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="El Niño threshold (+0.5°C)")
axes[1].axhline(y=-0.5, color="blue", linestyle="--", alpha=0.5, label="La Niña threshold (-0.5°C)")
axes[1].axhline(y=0, color="black", linewidth=0.5)
axes[1].set_title("SST Anomaly (ENSO Index) - Nino 3.4 Region", fontsize=13)
axes[1].set_ylabel("Anomaly (°C)")
axes[1].set_xlabel("Red = El Niño conditions | Blue = La Niña conditions")
axes[1].set_xticks(tick_positions)
axes[1].set_xticklabels([months[i] for i in tick_positions], rotation=45)
axes[1].legend(loc="upper right")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
save_figure(fig, "09_enso_sst_timeseries.png")


# ================================================================
# 2. GRACE - Ice Mass Loss (Greenland & Antarctica)
# ================================================================
print("[2/6] GRACE - Ice Mass Change Time Series...")

# GRACE LAND product: values in meters of LWE. Convert to cm.
# Antarctica is NOT covered by the LAND product — use separate regions.
grace = ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/LAND").select("lwe_thickness_csr")

greenland = ee.Geometry.Rectangle([-55, 60, -20, 84])
india = ee.Geometry.Rectangle([70, 20, 82, 32])  # NW India groundwater depletion

regions = {"Greenland (ice loss)": greenland, "NW India (groundwater depletion)": india}
grace_data = {}

for name, region in regions.items():
    dates_ms, values = extract_time_series(grace, "lwe_thickness_csr", region, 50000)
    # Filter out None values and convert m to cm
    clean_values = [v * 100 for v in values if v is not None]
    grace_data[name] = {"values": clean_values}
    print(f"    {name}: {len(clean_values)} data points")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ax, (name, data) in zip(axes, grace_data.items()):
    vals = data["values"]
    x = range(len(vals))
    ax.plot(x, vals, "b-", linewidth=1.5)
    ax.fill_between(x, vals, alpha=0.3)
    ax.set_title(f"GRACE - {name}", fontsize=13)
    ax.set_ylabel("LWE Thickness Anomaly (cm)")
    ax.grid(True, alpha=0.3)
    if len(vals) > 10:
        z = np.polyfit(range(len(vals)), vals, 1)
        trend = np.poly1d(z)
        ax.plot(x, trend(range(len(vals))), "r--",
                label=f"Trend: {z[0]:.2f} cm/month")
        ax.legend()

axes[1].set_xlabel("Months (from GRACE mission start, 2002-2017)")
fig.suptitle("GRACE Satellite - Mass Change from Gravity Measurements\n"
             "Key evidence for ice loss and groundwater depletion",
             fontsize=14, y=1.02)
fig.tight_layout()
save_figure(fig, "10_grace_mass_loss.png")


# ================================================================
# 3. GPM Precipitation - Extreme Events & ENSO Effects
# ================================================================
print("[3/6] GPM - Precipitation Patterns & Extremes...")

# Compare El Nino year vs La Nina year precipitation
gpm_collection = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07").select("precipitation")

# Tropical Pacific cross-section (shows ENSO precipitation shift)
# Sample along the equator
lons = list(range(-180, 180, 5))
precip_profiles = {}

for year, label in [(2023, "2023 (El Niño)"), (2022, "2022 (La Niña)")]:
    gpm_year = gpm_collection.filterDate(f"{year}-01-01", f"{year}-12-31").mean()
    values = []
    for lon in lons:
        point = ee.Geometry.Point([lon, 0])  # Along equator
        val = gpm_year.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=point,
            scale=10000, maxPixels=1e6
        ).get("precipitation").getInfo()
        values.append(val if val else 0)
    precip_profiles[label] = values
    print(f"    {label} profile done")

fig, ax = plt.subplots(figsize=(14, 6))
for label, values in precip_profiles.items():
    ax.plot(lons, values, linewidth=2, label=label)
ax.set_title("GPM IMERG - Equatorial Precipitation Profile\nEl Niño vs La Niña Year", fontsize=14)
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Precipitation Rate (mm/hr)")
ax.axvline(x=-170, color="gray", linestyle=":", alpha=0.5)
ax.axvline(x=-120, color="gray", linestyle=":", alpha=0.5)
ax.annotate("Niño 3.4\nRegion", xy=(-145, max(max(v) for v in precip_profiles.values()) * 0.9),
            ha="center", fontsize=10, style="italic")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
save_figure(fig, "11_gpm_enso_precipitation.png")


# ================================================================
# 4. Earth Radiation Budget (CERES concept via MODIS thermal data)
# ================================================================
print("[4/6] Radiation Budget - Latitudinal OLR proxy via MODIS LST...")

# CERES is no longer in GEE, so we model the radiation budget concept using
# MODIS land surface temperature as an OLR proxy (Stefan-Boltzmann: OLR ~ σT⁴)
# and known solar geometry for incoming SW radiation.
# This teaches the same energy balance concept tested on the exam.

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant
S0 = 1361  # Solar constant W/m²

modis_lst = (
    ee.ImageCollection("MODIS/061/MYD11A1")
    .filterDate("2024-01-01", "2024-12-31")
    .select("LST_Day_1km")
    .mean()
    .multiply(0.02)  # scale factor -> Kelvin
)

lats = list(range(-85, 90, 5))
olr_proxy = []
sw_absorbed = []

for lat in lats:
    band = ee.Geometry.Rectangle([-180, lat, 180, lat + 5])
    temp_k = modis_lst.reduceRegion(
        ee.Reducer.mean(), band, 10000, maxPixels=1e9
    ).get("LST_Day_1km").getInfo()

    if temp_k and temp_k > 100:
        olr = SIGMA * temp_k**4  # W/m²
    else:
        # For ocean-dominated bands without LST, use approximate temp
        approx_t = 300 - 1.5 * abs(lat)  # rough latitudinal gradient
        olr = SIGMA * approx_t**4

    olr_proxy.append(olr)

    # Incoming solar absorbed: S0/4 * (1-albedo) * cos(lat) distribution
    # Approximate: more solar input at equator, less at poles
    cos_lat = np.cos(np.radians(lat + 2.5))
    # Average albedo varies: ~0.1 ocean tropics, ~0.6 ice poles
    albedo = 0.15 + 0.45 * max(0, (abs(lat) - 30) / 60)
    sw_abs = S0 * cos_lat * (1 - albedo) / np.pi  # annual average geometry
    sw_absorbed.append(max(0, sw_abs))

print("    Latitudinal profiles done")

net_radiation = [sw - olr for sw, olr in zip(sw_absorbed, olr_proxy)]

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(lats, sw_absorbed, "orange", linewidth=2, label="Absorbed SW (solar)")
axes[0].plot(lats, olr_proxy, "red", linewidth=2, label="Outgoing LW (σT⁴ from MODIS LST)")
axes[0].set_title("Earth Radiation Budget by Latitude\n"
                  "CERES concept: incoming solar vs outgoing thermal", fontsize=13)
axes[0].set_ylabel("Flux (W/m²)")
axes[0].set_xlabel("Latitude (°)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=0, color="gray", linestyle=":", alpha=0.3)

colors = ["red" if n > 0 else "blue" for n in net_radiation]
axes[1].bar(lats, net_radiation, width=4.5, color=colors, alpha=0.7)
axes[1].axhline(y=0, color="black", linewidth=1)
axes[1].set_title("Net Radiation by Latitude (Energy Balance)", fontsize=13)
axes[1].set_ylabel("Net Flux (W/m²)")
axes[1].set_xlabel("Latitude (°) | Red = energy surplus (tropics) | Blue = energy deficit (poles)\n"
                   "This imbalance drives atmospheric and ocean circulation (heat transport)")
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
save_figure(fig, "12_radiation_budget.png")


# ================================================================
# 5. Greenhouse Gas Context - CO2 & Methane (OCO-2 / Aqua)
# ================================================================
print("[5/6] Greenhouse Gas Distributions (Sentinel-5P as proxy for OCO-2/Aura)...")

# Sentinel-5P gives similar atmospheric composition data to what OCO-2 and Aura provide
# Using it as a study proxy since OCO-2 data isn't in GEE

# Methane (CH4) - relevant to greenhouse gas topic
ch4 = (
    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_CH4")
    .filterDate("2025-01-01", "2025-06-30")
    .select("CH4_column_volume_mixing_ratio_dry_air")
    .mean()
)

# NO2 - trace gas distribution (Aura measures this too)
no2 = (
    ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")
    .filterDate("2025-01-01", "2025-06-30")
    .select("tropospheric_NO2_column_number_density")
    .mean()
)

def fetch_thumbnail(image, vis_params, region, dimensions="800x600"):
    """Download a GEE thumbnail as a PIL Image."""
    params = {
        "region": region.getInfo() if hasattr(region, "getInfo") else region,
        "dimensions": dimensions,
        "format": "png",
        **vis_params,
    }
    url = image.getThumbURL(params)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

global_region = ee.Geometry.Rectangle([-180, -60, 180, 75])

img_ch4 = fetch_thumbnail(
    ch4,
    {"min": 1750, "max": 1950,
     "palette": ["blue", "cyan", "green", "yellow", "orange", "red"]},
    global_region,
    dimensions="1000x500",
)

img_no2 = fetch_thumbnail(
    no2,
    {"min": 0, "max": 0.00015,
     "palette": ["black", "purple", "blue", "cyan", "green", "yellow", "red"]},
    global_region,
    dimensions="1000x500",
)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].imshow(img_ch4)
axes[0].set_title("Atmospheric Methane (CH₄) Column Concentration\n"
                  "Sentinel-5P, Jan-Jun 2025 (similar to OCO-2/Aura measurements)", fontsize=12)
axes[0].set_xlabel("Methane is 80x more potent than CO₂ over 20 years | Note: wetlands, "
                   "agriculture, fossil fuel sources")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(img_no2)
axes[1].set_title("Tropospheric NO₂ Column Density\n"
                  "Sentinel-5P, Jan-Jun 2025 (similar to Aura/OMI measurements)", fontsize=12)
axes[1].set_xlabel("NO₂ from combustion: cities, shipping lanes, power plants clearly visible")
axes[1].set_xticks([])
axes[1].set_yticks([])

fig.tight_layout()
save_figure(fig, "13_greenhouse_gases.png")


# ================================================================
# 6. Carbon Cycle - Vegetation Productivity (MODIS NDVI)
# ================================================================
print("[6/6] Carbon Cycle - Vegetation Seasonal Cycle (MODIS NDVI)...")

# NDVI shows photosynthesis = carbon uptake, directly relevant to carbon cycle topic
ndvi = ee.ImageCollection("MODIS/061/MYD13A1").select("NDVI")

# Northern hemisphere growing season vs dormant season
ndvi_summer = (
    ndvi.filterDate("2025-06-01", "2025-08-31").mean().multiply(0.0001)
)
ndvi_winter = (
    ndvi.filterDate("2025-01-01", "2025-02-28").mean().multiply(0.0001)
)
ndvi_diff = ndvi_summer.subtract(ndvi_winter)

nh_region = ee.Geometry.Rectangle([-180, 0, 180, 75])

img_summer = fetch_thumbnail(
    ndvi_summer,
    {"min": 0, "max": 0.8, "palette": ["brown", "yellow", "green", "darkgreen"]},
    nh_region, dimensions="1000x400",
)
img_winter = fetch_thumbnail(
    ndvi_winter,
    {"min": 0, "max": 0.8, "palette": ["brown", "yellow", "green", "darkgreen"]},
    nh_region, dimensions="1000x400",
)
img_diff = fetch_thumbnail(
    ndvi_diff,
    {"min": -0.3, "max": 0.3, "palette": ["brown", "orange", "white", "lightgreen", "darkgreen"]},
    nh_region, dimensions="1000x400",
)

fig, axes = plt.subplots(3, 1, figsize=(12, 12))

axes[0].imshow(img_summer)
axes[0].set_title("MODIS NDVI - Northern Hemisphere Summer (Jun-Aug 2025)", fontsize=12)
axes[0].set_xlabel("Peak photosynthesis = peak carbon uptake from atmosphere")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(img_winter)
axes[1].set_title("MODIS NDVI - Northern Hemisphere Winter (Jan-Feb 2025)", fontsize=12)
axes[1].set_xlabel("Dormant vegetation = reduced carbon uptake")
axes[1].set_xticks([])
axes[1].set_yticks([])

axes[2].imshow(img_diff)
axes[2].set_title("NDVI Difference (Summer - Winter)", fontsize=12)
axes[2].set_xlabel("Green = strong seasonal cycle (deciduous forests, agriculture) | "
                   "Brown = winter-greener (rare)")
axes[2].set_xticks([])
axes[2].set_yticks([])

fig.suptitle("Carbon Cycle: Seasonal Vegetation Productivity\n"
             "The 'breathing' of the planet drives annual CO₂ oscillation",
             fontsize=14, y=1.02)
fig.tight_layout()
save_figure(fig, "14_carbon_cycle_ndvi.png")


print("\n" + "=" * 60)
print("All climate analysis charts saved to ./output/")
print("Study these for Sections 3d and 3e of the test!")
print("=" * 60)
