#!/usr/bin/env python
# coding: utf-8

# **10. Precipitation Downscaling Using Machine Learning**

# In[ ]:


import ee
import geemap

try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize()

cor1 = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]
geometry = ee.Geometry.Polygon(cor1)

time_start = "2023-01-01"
time_end = "2024-01-01"

pr = ee.ImageCollection("NOAA/PERSIANN-CDR").filterDate(time_start, time_end)

month_list = ee.List.sequence(1, 12)

def compute_monthly_precipitation(month):
    month = ee.Number(month)
    month_img = pr.filter(ee.Filter.calendarRange(month, month, "month")).sum()
    date = ee.Date.fromYMD(2023, month, 1)
    return month_img.toInt().set("system:time_start", date.millis()).set("system:index", date.format("YYYY-MM-dd"))

pr_monthly = ee.ImageCollection(month_list.map(compute_monthly_precipitation))

lc = ee.ImageCollection("MODIS/061/MCD12Q1").mode().select("LC_Type1")
dem = ee.Image("USGS/GTOPO30")
ndvi = ee.ImageCollection("MODIS/061/MOD13A1").select("NDVI").filterDate(time_start, time_end)
temp = ee.ImageCollection("MODIS/061/MOD11A2").select("LST_Day_1km").filterDate(time_start, time_end)

def compute_monthly_ndvi(month):
    month = ee.Number(month)
    month_img = ndvi.filter(ee.Filter.calendarRange(month, month, "month")).median()


# In[ ]:


date = ee.Date.fromYMD(2023, month, 1)
return month_img.multiply(0.0001).set("system:time_start", date.millis()).set("system:index", date.format("YYYY-MM-dd"))

ndvi_monthly = ee.ImageCollection(month_list.map(compute_monthly_ndvi))

def compute_monthly_temperature(month):
month = ee.Number(month)
month_img = temp.filter(ee.Filter.calendarRange(month, month, "month")).median()
date = ee.Date.fromYMD(2023, month, 1)
return month_img.multiply(0.02).set("system:time_start", date.millis()).set("system:index", date.format("YYYY-MM-dd"))

temp_monthly = ee.ImageCollection(month_list.map(compute_monthly_temperature))

def combine_variables(img):
return img.addBands(dem).addBands(lc).copyProperties(img, img.propertyNames())

collection = ndvi_monthly.combine(temp_monthly).combine(pr_monthly).map(combine_variables)

def train_model(img):
band_names = img.bandNames()
band_names_indep = band_names.remove("precipitation")

img_int = img.addBands(img.select("precipitation").max(0).toInt(), overwrite=True)

training = img_int.stratifiedSample(
    numPoints=100, classBand="precipitation", region=geometry, scale=1000
)

model = ee.Classifier.smileGradientTreeBoost(100).train(
    features=training, classProperty="precipitation", inputProperties=band_names
).setOutputMode("REGRESSION")

pr1000 = img_int.select(band_names_indep).classify(model)
return pr1000.rename("pr1000").addBands(img_int.select(["precipitation"], ["pr27000"])).copyProperties(img, img.propertyNames())

pr_model = collection.map(train_model)

cor = [70.38293457, 28.39841461]
geometry2 = ee.Geometry.Point(cor)

roi = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_5").filterBounds(geometry2).map(lambda f: f.simplify(1000))


# In[1]:


pr_mean = pr_model.select("pr1000").mean()

def correct_pr(img):
    return img.select("pr1000").add(img.select("pr27000").subtract(pr_mean)).copyProperties(img, img.propertyNames())

pr_1km_cor = pr_model.map(correct_pr)

pr_initial = pr_model.select(["pr27000"]).mean().clip(roi)
pr_sharpened = pr_model.select(["pr1000"]).mean().clip(roi)
pr_sharpened_cor = pr_1km_cor.select(["pr1000"]).mean().clip(roi)

Map = geemap.Map()
Map.centerObject(roi)
Map.addLayer(roi, {}, "ROI")
Map.addLayer(pr_initial, {"min": 0, "max": 300, "palette": ["blue", "green", "yellow", "red"]}, "pr_initial")
Map.addLayer(pr_sharpened, {"min": 0, "max": 300, "palette": ["blue", "green", "yellow", "red"]}, "pr_sharpened")
Map.addLayer(pr_sharpened_cor, {"min": 0, "max": 300, "palette": ["blue", "green", "yellow", "red"]}, "pr_sharpened_cor")


Map


# **11.Precipitation Anomaly Analysis (2000-2020) Using Google Earth Engine & CHIRPS Data**

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt


cor = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]
roi = ee.Geometry.Polygon(cor)

time_start = "2000-01-01"
time_end = "2020-12-31"

chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(time_start, time_end)

def temporal_collection(collection, start, count, interval, unit):
    seq = ee.List.sequence(0, ee.Number(count).subtract(1))
    origin_date = ee.Date(start)

    def compute_time_series(i):
        i = ee.Number(i) 
        start_date = origin_date.advance(i.multiply(interval), unit)
        end_date = origin_date.advance(i.add(1).multiply(interval), unit)

        return (
            collection.filterDate(start_date, end_date).sum()
            .set("system:time_start", start_date.millis())
            .set("system:time_end", end_date.millis())
        )

    return ee.ImageCollection(seq.map(compute_time_series))

monthly = temporal_collection(chirps, time_start, 240, 1, "month")


# In[3]:


pr_mean = monthly.mean()

def compute_anomaly(img):
    return img.subtract(pr_mean).copyProperties(img, img.propertyNames())

anomaly = monthly.map(compute_anomaly)

def extract_time_series(img):
    reduced = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=5000,
        bestEffort=True
    )
    return ee.Feature(None, {
        "date": img.date().format("YYYY-MM-dd"),
        "precip_anomaly": reduced.get("precipitation")
    })

time_series_fc = anomaly.map(extract_time_series)

time_series_list = time_series_fc.aggregate_array("properties").getInfo()

if time_series_list:
    df = pd.DataFrame(time_series_list)
    print(df.columns)  
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  
    df = df.dropna(subset=["date"])  

    plt.figure(figsize=(12, 5))
    plt.bar(df["date"], df["precip_anomaly"], color="blue", alpha=0.7)
    plt.xlabel("Year")
    plt.ylabel("Precipitation Anomaly (mm)")
    plt.title("Precipitation Anomaly (2000-2020)")
    plt.grid()
    plt.xticks(rotation=45)
    plt.show()
else:
    print("⚠️ No valid time series data extracted.")

Map = geemap.Map()
Map.centerObject(roi, zoom=6)
Map.addLayer(pr_mean.clip(roi), {}, "Mean Precipitation", False)
Map.addLayer(anomaly.filterDate("2010-01-01", "2011-01-01").toBands().clip(roi), {}, "Precipitation Anomaly 2010", False)

Map


# **12. Precipitation Analysis and Standardized Precipitation Index (SPI) Computation**

# In[ ]:


import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

time_start = ee.Date('2000-01-01')
time_end = ee.Date('2024-01-01')
time_dif = time_end.difference(time_start, 'month').round()

time_list = ee.List.sequence(0, ee.Number(time_dif).subtract(1), 1).map(
    lambda x: time_start.advance(x, 'month')
)

col = ee.ImageCollection("NOAA/PERSIANN-CDR").filterDate(time_start, time_end)

def monthly(date):
    start_date = ee.Date(date)
    end_date = start_date.advance(1, 'month')
    img = col.filterDate(start_date, end_date).sum()
    return img.set('system:time_start', start_date.millis())


monthly_col = ee.ImageCollection(time_list.map(monthly))

loc = [67, 27, 72, 32]  # [lon_min, lat_min, lon_max, lat_max]

ds = xr.open_dataset(monthly_col, engine="ee", crs="EPSG:4326", scale=0.27, geometry=loc)

annual = ds.resample(time="Y").sum("time")

annual.precipitation.plot(
    x="lon", y="lat", cmap="jet_r", col="time", robust=True, col_wrap=5
)

point = ds.sel(lon=70.38293457, lat=28.39841461, method="nearest")

date = point.time.values
pr = point.precipitation.values

plt.figure(figsize=(10, 5))
plt.plot(point.time, point.precipitation, marker="o", linestyle="-", color="b")


# In[5]:


plt.xlabel("Time")
plt.ylabel("Precipitation (mm)")
plt.title("Monthly Precipitation at Selected Point")
plt.xticks(rotation=45)
plt.grid()
plt.show()

get_ipython().system('pip install standard_precip')

from standard_precip import spi

df = pd.DataFrame({"date": date, "pr": pr})

spi_fun = spi.SPI()
spi_12months = spi_fun.calculate(
    df, "date", "pr", freq="M", scale=12, fit_type="lmom", dist_type="gam"
)


spi_12months.to_csv("spi_12month.csv")

from standard_precip.utils import plot_index
fig = plot_index(spi_12months, "date", "pr_scale_12_calculated_index")
plt.show()


# **13. Drought Monitoring and Vegetation Health Assessment using MODIS Data: VCI, TCI, and VHI Computation using Python**

# In[6]:


import geemap.foliumap as geemap


cor = [
    [68.6535, 27.5032],  
    [69.5983, 27.5032],  
    [69.5983, 28.2764],  
    [68.6535, 28.2764],  
    [68.6535, 27.5032]
]
roi = ee.Geometry.Polygon(cor)

time_start = '2001-01-01'
time_end = '2024-01-01'

ndvi = ee.ImageCollection("MODIS/061/MOD13A2")     .select('NDVI')     .filterDate(time_start, time_end)

temp = ee.ImageCollection("MODIS/061/MOD11A2")     .select('LST_Day_1km')     .filterDate(time_start, time_end)

ndvi_min = ndvi.min().multiply(0.0001)
ndvi_max = ndvi.max().multiply(0.0001)

def temporal_collection(collection, start, count, interval, unit):
    seq = ee.List.sequence(0, ee.Number(count).subtract(1))
    origin_date = ee.Date(start)

    def map_function(i):
        start_date = origin_date.advance(ee.Number(interval).multiply(i), unit)
        end_date = origin_date.advance(ee.Number(interval).multiply(ee.Number(i).add(1)), unit)
        return collection.filterDate(start_date, end_date).mean()             .set('system:time_start', start_date.millis())             .set('system:time_end', end_date.millis())

    return ee.ImageCollection(seq.map(map_function))

ndvi_monthly = temporal_collection(ndvi, time_start, 276, 1, 'month')

def compute_vci(img):
    index = img.expression(
        '(ndvi - min) / (max - min)',
        {'ndvi': img.select('NDVI').multiply(0.0001), 'min': ndvi_min, 'max': ndvi_max}
    )
    return index.rename('VCI').copyProperties(img, img.propertyNames())

vci = ndvi_monthly.map(compute_vci)

temp_max = temp.max().multiply(0.02)
temp_min = temp.min().multiply(0.02)
temp_monthly = temporal_collection(temp, time_start, 276, 1, 'month')

def compute_tci(img):
    index = img.expression(
        '(max - lst) / (max - min)',
        {'max': temp_max, 'min': temp_min, 'lst': img.multiply(0.02)}
    )
    return index.rename('TCI').copyProperties(img, img.propertyNames())

tci = temp_monthly.map(compute_tci)

modis_indices = vci.combine(tci)

def compute_vhi(img):
    vhi = img.expression(
        '0.5 * vci + (1 - 0.5) * tci',
        {'vci': img.select('VCI'), 'tci': img.select('TCI')}
    ).rename('VHI')
    return img.addBands(vhi).copyProperties(img, img.propertyNames())

drought = modis_indices.map(compute_vhi)

Map = geemap.Map(center=[28.39841461, 70.38293457], zoom=7)

ndvi_vis = {'min': 0, 'max': 1, 'palette': ['red', 'yellow', 'green']}
tci_vis = {'min': 0, 'max': 1, 'palette': ['blue', 'white', 'red']}
vhi_vis = {'min': 0, 'max': 1, 'palette': ['brown', 'yellow', 'green']}

Map.addLayer(ndvi_max, ndvi_vis, 'NDVI Max')
Map.addLayer(vci.first(), ndvi_vis, 'VCI (Vegetation Condition Index)')
Map.addLayer(tci.first(), tci_vis, 'TCI (Thermal Condition Index)')
Map.addLayer(drought.first().select('VHI'), vhi_vis, 'VHI (Vegetation Health Index)')

Map


# **14. Precipitation Intensity Analysis using NOAA/PERSIANN-CDR**

# In[8]:



cor = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]
geometry = ee.Geometry.Polygon(cor)

pr = ee.ImageCollection("NOAA/PERSIANN-CDR").filterDate('2023', '2024')

wet_days = pr.map(lambda img: img.gte(1)).sum()

total_precip = pr.map(lambda img: img.updateMask(img.gte(1))).sum()

pr_intensity = total_precip.divide(wet_days).rename("pr_intensity")

def compute_pr_intensity(collection, year):
    year = ee.Date(f"{year}-01-01")
    filtered = collection.filterDate(year, year.advance(1, "year"))


    num_wet = filtered.map(lambda img: img.gte(1)).sum()
    pr_total = filtered.map(lambda img: img.updateMask(img.gte(1))).sum()

    intensity = pr_total.divide(num_wet).rename(f"pr_intensity_{year.get('year').getInfo()}")


sdii2010 = compute_pr_intensity(ee.ImageCollection("NOAA/PERSIANN-CDR"), 2010)
sdii2015 = compute_pr_intensity(ee.ImageCollection("NOAA/PERSIANN-CDR"), 2015)

import geemap
Map = geemap.Map()
Map.centerObject(geometry)
Map.addLayer(pr_intensity, {"palette": ["red", "white", "blue"]}, "Precipitation Intensity", False, 0.5)
Map.addLayer(sdii2010, {"palette": ["red", "white", "blue"]}, "Precipitation Intensity 2010", False, 0.5)
Map.addLayer(sdii2015, {"palette": ["red", "white", "blue"]}, "Precipitation Intensity 2015", False, 0.5)

Map

