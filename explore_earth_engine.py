"""
Explore Google Earth Engine - Sample data retrieval demo.

First run will open a browser for authentication.
You'll need a Google Cloud project with the Earth Engine API enabled.
"""

import ee

# --- Authentication & Initialization ---
# First run: opens browser for OAuth. Pass your GCP project ID.
ee.Authenticate()  # Uncomment on first run
ee.Initialize(project="remote-sensing-research-492504" \
"")  # Replace with your project ID

print("=" * 60)
print("Google Earth Engine - Data Exploration Demo")
print("=" * 60)


# --- 1. Elevation: SRTM Digital Elevation Model ---
print("\n[1] SRTM Elevation at Mount Rainier, WA")
srtm = ee.Image("USGS/SRTMGL1_003")
point = ee.Geometry.Point([-121.7603, 46.8523])  # Mt. Rainier summit
elevation = srtm.sample(point, scale=30).first().get("elevation").getInfo()
print(f"    Elevation: {elevation} meters")


# --- 2. Sentinel-2: Recent cloud-free imagery metadata ---
print("\n[2] Sentinel-2 imagery over Seattle (last 3 months)")
seattle = ee.Geometry.Point([-122.3321, 47.6062])
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(seattle)
    .filterDate("2026-01-01", "2026-04-01")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)
count = s2.size().getInfo()
print(f"    Cloud-free scenes found: {count}")

if count > 0:
    first = s2.first()
    bands = first.bandNames().getInfo()
    date = first.date().format("YYYY-MM-dd").getInfo()
    print(f"    First scene date: {date}")
    print(f"    Bands: {', '.join(bands[:10])}...")


# --- 3. Landsat 9: Band statistics ---
print("\n[3] Landsat 9 - Mean reflectance over a farm field")
field = ee.Geometry.Rectangle([-119.52, 46.25, -119.50, 46.27])  # WA farmland
landsat = (
    ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    .filterBounds(field)
    .filterDate("2025-06-01", "2025-09-01")
    .median()
)
stats = landsat.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=field,
    scale=30,
    maxPixels=1e6,
)
sr_b4 = stats.get("SR_B4")  # Red band
sr_b5 = stats.get("SR_B5")  # NIR band
if sr_b4.getInfo() and sr_b5.getInfo():
    red = sr_b4.getInfo()
    nir = sr_b5.getInfo()
    ndvi = (nir - red) / (nir + red)
    print(f"    Red (B4):  {red:.1f}")
    print(f"    NIR (B5):  {nir:.1f}")
    print(f"    NDVI:      {ndvi:.3f}  (vegetation index)")


# --- 4. MODIS Land Surface Temperature ---
print("\n[4] MODIS Land Surface Temperature - Phoenix, AZ")
phoenix = ee.Geometry.Point([-112.074, 33.4484])
lst = (
    ee.ImageCollection("MODIS/061/MOD11A1")
    .filterDate("2025-07-01", "2025-07-31")
    .select("LST_Day_1km")
    .mean()
)
temp_val = lst.sample(phoenix, scale=1000).first().get("LST_Day_1km").getInfo()
temp_celsius = temp_val * 0.02 - 273.15  # Scale factor and K to C
temp_fahrenheit = temp_celsius * 9 / 5 + 32
print(f"    July 2025 avg daytime temp: {temp_celsius:.1f} C / {temp_fahrenheit:.1f} F")


# --- 5. Global Precipitation (GPM) ---
print("\n[5] GPM Precipitation - Monthly total, Amazon Basin")
amazon = ee.Geometry.Point([-60.0, -3.0])
precip = (
    ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07")
    .filterDate("2025-01-01", "2025-12-31")
    .select("precipitation")
)
monthly = precip.toList(12)
print("    Monthly avg precipitation (mm/hr):")
for i in range(min(6, precip.size().getInfo())):
    img = ee.Image(monthly.get(i))
    date = img.date().format("YYYY-MM").getInfo()
    val = img.sample(amazon, scale=10000).first().get("precipitation").getInfo()
    print(f"      {date}: {val:.2f} mm/hr")


# --- 6. Nighttime Lights (VIIRS) ---
print("\n[6] VIIRS Nighttime Lights - Comparing cities")
cities = {
    "New York":  ee.Geometry.Point([-74.006, 40.7128]),
    "Denver":    ee.Geometry.Point([-104.9903, 39.7392]),
    "Rural WY":  ee.Geometry.Point([-107.5, 43.0]),
}
viirs = (
    ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG")
    .filterDate("2025-01-01", "2025-12-31")
    .select("avg_rad")
    .mean()
)
for name, geom in cities.items():
    val = viirs.sample(geom, scale=500).first().get("avg_rad").getInfo()
    print(f"    {name:12s}: {val:.2f} nW/cm2/sr")


# --- 7. Browse the Data Catalog ---
print("\n[7] Data Catalog - Example dataset collections available:")
datasets = [
    ("COPERNICUS/S1_GRD",              "Sentinel-1 SAR"),
    ("COPERNICUS/S2_SR_HARMONIZED",    "Sentinel-2 Surface Reflectance"),
    ("COPERNICUS/S5P/OFFL/L3_NO2",     "Sentinel-5P Nitrogen Dioxide"),
    ("LANDSAT/LC09/C02/T1_L2",         "Landsat 9 Level-2"),
    ("MODIS/061/MOD13A1",              "MODIS Vegetation Indices (NDVI)"),
    ("MODIS/061/MOD11A1",              "MODIS Land Surface Temperature"),
    ("NASA/GPM_L3/IMERG_MONTHLY_V07",  "GPM Monthly Precipitation"),
    ("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG", "VIIRS Nighttime Lights"),
    ("NASA/NASADEM_HGT/001",           "NASADEM Elevation (30m)"),
    ("ECMWF/ERA5_LAND/MONTHLY_AGGR",   "ERA5-Land Climate Reanalysis"),
    ("JAXA/ALOS/AW3D30/V3_2",          "ALOS World 3D Elevation (30m)"),
    ("NASA/GRACE/MASS_GRIDS/LAND",      "GRACE Groundwater/Ice Mass"),
]
for dataset_id, description in datasets:
    print(f"    {dataset_id:<45s} {description}")

print("\n" + "=" * 60)
print("Full catalog: https://developers.google.com/earth-engine/datasets")
print("=" * 60)
