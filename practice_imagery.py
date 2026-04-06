"""
Science Olympiad 2026 Remote Sensing - Practice Imagery
========================================================
Section 3d (35% of test): Image & data interpretation from specific satellites.

Pulls and displays imagery from every satellite listed in the rules:
  - MODIS (Aqua): clouds, SST, aerosol optical depth
  - GOES: weather/cloud imagery
  - GPM: precipitation
  - ECOSTRESS: land surface temperature
  - GRACE: gravity/mass anomalies (ice, groundwater)
  - MODIS LST/Emissivity as OLR proxy (CERES radiation budget concept)
  - Suomi NPP / VIIRS: atmosphere, nighttime lights

Generates a study sheet of satellite images saved as PNG files in ./output/
"""

import ee
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from io import BytesIO
from PIL import Image

# --- Config ---
PROJECT_ID = "remote-sensing-research-492504"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

ee.Initialize(project=PROJECT_ID)
print("Connected to Earth Engine.\n")


def fetch_thumbnail(image, vis_params, region, dimensions="800x600", retries=2):
    """Download a GEE thumbnail as a PIL Image with retry logic."""
    params = {
        "region": region.getInfo() if hasattr(region, "getInfo") else region,
        "dimensions": dimensions,
        "format": "png",
        **vis_params,
    }
    url = image.getThumbURL(params)
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=180)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except requests.exceptions.ReadTimeout:
            if attempt < retries:
                print(f"    Timeout, retrying ({attempt + 1}/{retries})...")
            else:
                raise


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")




# ================================================================
# 1. MODIS (Aqua) - True Color Cloud Imagery
# ================================================================
print("[1/8] MODIS Aqua - True Color (clouds & weather)...")
modis_tc = (
    ee.ImageCollection("MODIS/061/MYD09GA")  # Aqua daily surface reflectance
    .filterDate("2025-08-15", "2025-08-20")
    .select(["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"])  # R, G, B
    .median()
)
# Hurricane season view of Gulf of Mexico
gulf_region = ee.Geometry.Rectangle([-100, 18, -75, 35])
img1 = fetch_thumbnail(
    modis_tc,
    {"min": 0, "max": 3500, "bands": ["sur_refl_b01", "sur_refl_b04", "sur_refl_b03"]},
    gulf_region,
)

fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(img1)
ax.set_title("MODIS Aqua - True Color Composite\nGulf of Mexico, Aug 2025", fontsize=14)
ax.set_xlabel("Bands: B1 (Red), B4 (Green), B3 (Blue) | 500m resolution")
ax.set_xticks([])
ax.set_yticks([])
save_figure(fig, "01_modis_aqua_truecolor.png")


# ================================================================
# 2. MODIS (Aqua) - Sea Surface Temperature
# ================================================================
print("[2/8] Sea Surface Temperature (NOAA OISST)...")
# MYD11A1 is Land Surface Temperature only — use NOAA OISST for actual ocean SST.
# OISST blends satellite (AVHRR) and in-situ observations for global ocean coverage.
# MODIS Aqua contributes to SST products but its L3 SST isn't in GEE directly.
sst = (
    ee.ImageCollection("NOAA/CDR/OISST/V2_1")
    .filterDate("2025-07-01", "2025-07-31")
    .select("sst")
    .mean()
    .multiply(0.01)  # scale factor -> °C
)
pacific_region = ee.Geometry.Rectangle([-180, -40, -70, 40])
img2 = fetch_thumbnail(
    sst,
    {"min": -2, "max": 32, "palette": ["blue", "cyan", "lime", "yellow", "red"]},
    pacific_region,
    dimensions="900x600",
)

fig, ax = plt.subplots(figsize=(10, 7))
ax.imshow(img2)
ax.set_title("Sea Surface Temperature (°C)\nPacific Ocean, July 2025 Average", fontsize=14)
ax.set_xlabel("Dataset: NOAA OISST V2.1 | 0.25° resolution\n"
              "Note: warm pool in western Pacific, cold tongue along equatorial east Pacific (La Niña pattern)")
ax.set_xticks([])
ax.set_yticks([])
sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list("", ["blue", "cyan", "lime", "yellow", "red"]),
    norm=plt.Normalize(-2, 32),
)
plt.colorbar(sm, ax=ax, label="Temperature (°C)", shrink=0.8)
save_figure(fig, "02_sst.png")


# ================================================================
# 3. MODIS (Aqua) - Aerosol Optical Depth
# ================================================================
print("[3/8] MODIS - Aerosol Optical Depth...")
# MOD08_M3 is the monthly 1-degree global product — lightweight and fast to render
modis_aod = (
    ee.ImageCollection("MODIS/061/MOD08_M3")
    .filterDate("2025-07-01", "2025-09-01")
    .select("Aerosol_Optical_Depth_Land_Ocean_Mean_Mean")
    .mean()
    .multiply(0.001)  # scale factor
)
global_region = ee.Geometry.Rectangle([-180, -60, 180, 75])
img3 = fetch_thumbnail(
    modis_aod,
    {"min": 0, "max": 0.6, "palette": ["white", "yellow", "orange", "red", "purple"]},
    global_region,
    dimensions="1000x500",
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(img3)
ax.set_title("MODIS - Aerosol Optical Depth (Land+Ocean)\nGlobal, Summer 2025 Average", fontsize=14)
ax.set_xlabel("Dataset: MOD08_M3 (monthly 1° grid) | High AOD = smoke, dust, pollution\n"
              "Note: Saharan dust over Atlantic, biomass burning in Africa/Amazon, Asian haze")
ax.set_xticks([])
ax.set_yticks([])
sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list("", ["white", "yellow", "orange", "red", "purple"]),
    norm=plt.Normalize(0, 0.8),
)
plt.colorbar(sm, ax=ax, label="AOD (unitless)", shrink=0.8)
save_figure(fig, "03_modis_aerosol_depth.png")


# ================================================================
# 4. GOES-16 - Cloud & Moisture Imagery
# ================================================================
print("[4/8] GOES-16 - Cloud & Moisture Imagery...")
# Filter to a daytime image (18 UTC = ~1pm Eastern)
goes = ee.Image(
    ee.ImageCollection("NOAA/GOES/16/MCMIPF")
    .filterDate("2024-09-01", "2024-09-02")
    .filter(ee.Filter.calendarRange(17, 19, "hour"))
    .first()
)
# GEE stores GOES CMI as raw DN values (not physical units).
# Band 2 (visible): DN ~0-4000, higher = brighter/more reflective
# Band 13 (IR 10.3μm): DN ~1500-3600, higher DN = warmer surface, lower = cold cloud tops
conus = ee.Geometry.Rectangle([-130, 15, -60, 55])

# Visible band (daytime clouds)
img4a = fetch_thumbnail(
    goes.select("CMI_C02"),
    {"min": 0, "max": 3000, "palette": ["black", "white"]},
    conus,
    dimensions="900x600",
)

# IR band - invert palette so cold cloud tops appear white (convention)
img4b = fetch_thumbnail(
    goes.select("CMI_C13"),
    {"min": 1500, "max": 3600, "palette": ["white", "gray", "blue", "cyan", "yellow", "red"]},
    conus,
    dimensions="900x600",
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(img4a)
axes[0].set_title("GOES-16 Band 2 (Visible 0.64μm)\nDaytime Cloud Imagery", fontsize=12)
axes[0].set_xlabel("Bright = thick clouds / high albedo")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(img4b)
axes[1].set_title("GOES-16 Band 13 (IR 10.3μm)\nBrightness Temperature", fontsize=12)
axes[1].set_xlabel("Warm (red) = surface | Cold (white) = high cloud tops = deep convection")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.suptitle("GOES-16 Geostationary Weather Satellite - Sep 1, 2024 ~17:00 UTC", fontsize=14, y=1.02)
fig.tight_layout()
save_figure(fig, "04_goes16_clouds.png")


# ================================================================
# 5. GPM - Global Precipitation
# ================================================================
print("[5/8] GPM - Global Precipitation...")
gpm = (
    ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07")
    .filterDate("2025-01-01", "2025-12-31")
    .select("precipitation")
)

# Annual mean
gpm_annual = gpm.mean()
img5a = fetch_thumbnail(
    gpm_annual,
    {"min": 0, "max": 0.7, "palette": ["white", "lightblue", "blue", "darkblue", "purple"]},
    global_region,
    dimensions="1000x500",
)

# Show seasonal difference (Jul vs Jan) - monsoon/ITCZ shift
gpm_jan = gpm.filterDate("2025-01-01", "2025-02-01").first()
gpm_jul = gpm.filterDate("2025-07-01", "2025-08-01").first()

img5b = fetch_thumbnail(
    gpm_jul.subtract(gpm_jan),
    {"min": -0.5, "max": 0.5, "palette": ["brown", "orange", "white", "blue", "darkblue"]},
    global_region,
    dimensions="1000x500",
)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].imshow(img5a)
axes[0].set_title("GPM IMERG - Mean Precipitation Rate\n2025 Annual Average", fontsize=13)
axes[0].set_xlabel("Note ITCZ (tropical rain band), monsoon regions, rain shadows")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(img5b)
axes[1].set_title("GPM IMERG - Seasonal Precipitation Difference\nJuly minus January 2025",
                  fontsize=13)
axes[1].set_xlabel("Blue = wetter in July | Brown = wetter in January (follows ITCZ migration)")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.tight_layout()
save_figure(fig, "05_gpm_precipitation.png")


# ================================================================
# 6. GRACE - Mass Anomaly (Ice Sheet / Groundwater)
# ================================================================
print("[6/8] GRACE - Gravity / Mass Anomalies...")
# GRACE land data is in meters of liquid water equivalent (LWE).
# Values range roughly -1.4 to +0.25 m. Latest images may have nulls,
# so use a late-mission image (2016) that shows accumulated ice loss clearly.
# Note: GRACE-FO (2018+) is not in GEE; this shows the original GRACE mission.
grace_land = (
    ee.ImageCollection("NASA/GRACE/MASS_GRIDS_V04/LAND")
    .filterDate("2016-06-01", "2016-12-31")
    .select("lwe_thickness_csr")
    .mean()
    .multiply(100)  # convert meters to cm for readability
)
# Land-only dataset, so use a land-focused region
land_region = ee.Geometry.Rectangle([-170, -55, 180, 75])
img6 = fetch_thumbnail(
    grace_land,
    {"min": -80, "max": 30, "palette": ["darkred", "red", "orange", "white", "cyan", "blue"]},
    land_region,
    dimensions="1000x500",
)

fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(img6)
ax.set_title("GRACE - Liquid Water Equivalent Thickness Anomaly (cm)\n"
             "Late 2016 (accumulated change from baseline)",
             fontsize=14)
ax.set_xlabel("Red = mass loss (Greenland ice sheet, groundwater depletion) | Blue = mass gain\n"
              "Note: strong signal over Greenland, also visible in India/California groundwater")
ax.set_xticks([])
ax.set_yticks([])
sm = plt.cm.ScalarMappable(
    cmap=LinearSegmentedColormap.from_list("", ["darkred", "red", "orange", "white", "cyan", "blue"]),
    norm=plt.Normalize(-80, 30),
)
plt.colorbar(sm, ax=ax, label="LWE Thickness Anomaly (cm)", shrink=0.8)
save_figure(fig, "06_grace_mass_anomaly.png")


# ================================================================
# 7. OLR Proxy via MODIS LST (CERES concept - radiation budget)
# ================================================================
print("[7/8] Radiation Budget - MODIS LST as OLR proxy (CERES concept)...")
# CERES data is not currently in GEE, but the test covers the concept.
# OLR is proportional to σT⁴ (Stefan-Boltzmann), so MODIS LST shows the pattern.

modis_lst = (
    ee.ImageCollection("MODIS/061/MYD11A1")
    .filterDate("2024-01-01", "2024-12-31")
    .select("LST_Day_1km")
    .mean()
    .multiply(0.02)  # scale to Kelvin
)

# Land surface temperature (proxy for thermal emission / OLR pattern)
img7a = fetch_thumbnail(
    modis_lst,
    {"min": 230, "max": 320, "palette": ["blue", "cyan", "yellow", "orange", "red"]},
    global_region,
    dimensions="1000x500",
)

# Emissivity map - shows surface type effects on radiation
modis_emis = (
    ee.ImageCollection("MODIS/061/MYD11A1")
    .filterDate("2024-01-01", "2024-12-31")
    .select("Emis_31")  # Band 31 emissivity (11 μm thermal IR)
    .mean()
    .multiply(0.002)  # scale factor
    .add(0.49)
)
img7b = fetch_thumbnail(
    modis_emis,
    {"min": 0.9, "max": 1.0, "palette": ["yellow", "orange", "red", "darkred"]},
    global_region,
    dimensions="1000x500",
)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))
axes[0].imshow(img7a)
axes[0].set_title("MODIS Aqua - Land Surface Temperature (K)\n"
                  "OLR proxy: hotter surfaces emit more (Stefan-Boltzmann: σT⁴)\n"
                  "2024 Annual Mean", fontsize=12)
axes[0].set_xlabel("Hot (red) = high OLR (deserts) | Cold (blue) = low emission (ice/high altitude)")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(img7b)
axes[1].set_title("MODIS Band 31 Emissivity (11μm thermal IR)\n"
                  "CERES measures this at TOA to compute Earth's energy budget", fontsize=12)
axes[1].set_xlabel("Emissivity varies by surface type: water ~0.99, sand ~0.92, vegetation ~0.97")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.tight_layout()
save_figure(fig, "07_radiation_budget.png")


# ================================================================
# 8. Suomi NPP / VIIRS - Nighttime Lights & Day-Night Band
# ================================================================
print("[8/8] Suomi NPP / VIIRS - Nighttime Lights...")
viirs = (
    ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    .filterDate("2025-01-01", "2025-12-31")
    .select("avg_rad")
    .mean()
)

# Global view
img8a = fetch_thumbnail(
    viirs,
    {"min": 0, "max": 30, "palette": ["black", "darkblue", "blue", "yellow", "white"]},
    global_region,
    dimensions="1000x500",
)

# Zoomed: India (rapid urbanization)
india = ee.Geometry.Rectangle([68, 6, 90, 36])
img8b = fetch_thumbnail(
    viirs,
    {"min": 0, "max": 50, "palette": ["black", "darkblue", "blue", "yellow", "white"]},
    india,
    dimensions="600x600",
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(img8a)
axes[0].set_title("VIIRS Day-Night Band - Nighttime Lights\n2025 Annual Average", fontsize=12)
axes[0].set_xlabel("Suomi NPP satellite | Urbanization & energy use proxy")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(img8b)
axes[1].set_title("VIIRS Nighttime Lights\nIndia Detail", fontsize=12)
axes[1].set_xlabel("Urban corridors, coastal cities, rural electrification visible")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.tight_layout()
save_figure(fig, "08_viirs_nighttime_lights.png")

print("\n" + "=" * 60)
print("All imagery saved to ./output/")
print("Study these images for Section 3d of the test!")
print("=" * 60)
