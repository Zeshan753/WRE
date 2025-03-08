#!/usr/bin/env python
# coding: utf-8

# **1. Soil Moisture Estimation Using Landsat-8 and TOTRAM Algorithm in Indus River Downstream, Pakistan**

# In[1]:


import ee
import geemap
import matplotlib.pyplot as plt
import pandas as pd

try:
    ee.Initialize()
except ee.EEException:
    ee.Authenticate()
    ee.Initialize()

cor = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]
roi = ee.Geometry.Polygon(cor)

landsat = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterBounds(roi)
    .filterDate('2020-01-01', '2022-12-31')
    .filter(ee.Filter.lt('CLOUD_COVER', 10))
)

def calculate_indices(img):
    ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('ndvi')
    lst = img.select('ST_B10').multiply(0.00341802).add(149).rename('lst')
    return img.addBands([ndvi, lst])

parameters = landsat.map(calculate_indices)

def mask_vegetation(img):
    ndvi_full = img.select('ndvi').gt(0.3)
    lst_full = img.select('lst').updateMask(ndvi_full)
    return lst_full

lst_full_cover = parameters.map(mask_vegetation)

def mask_bareland(img):
    ndvi_bareland = img.select('ndvi').gte(0).And(img.select('ndvi').lt(0.2))
    lst_bareland = img.select('lst').updateMask(ndvi_bareland)
    return lst_bareland

lst_bareland = parameters.map(mask_bareland)

vw = ee.Number(lst_full_cover.min().reduceRegion(
    reducer=ee.Reducer.min(), geometry=roi, scale=100, bestEffort=True
).values().get(0))

vd = ee.Number(lst_full_cover.max().reduceRegion(
    reducer=ee.Reducer.max(), geometry=roi, scale=100, bestEffort=True
).values().get(0))

iw = ee.Number(lst_bareland.min().reduceRegion(
    reducer=ee.Reducer.min(), geometry=roi, scale=100, bestEffort=True
).values().get(0))

id = ee.Number(lst_bareland.max().reduceRegion(
    reducer=ee.Reducer.max(), geometry=roi, scale=100, bestEffort=True
).values().get(0))

sd = id.subtract(vd)
sw = iw.subtract(vw)

def compute_soil_moisture(img):
    soil_moisture = img.expression(
        '(id + sd * ndvi - lst) / (id - iw + (sd - sw) * ndvi)',
        {
            'id': id, 'sd': sd, 'ndvi': img.select('ndvi'),
            'lst': img.select('lst'), 'iw': iw, 'sw': sw
        }
    ).rename('soil_moisture')
    return img.addBands(soil_moisture)

sm = parameters.map(compute_soil_moisture)

def extract_time_series(img):
    reduced = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=100,
        bestEffort=True
    )
    return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), 'soil_moisture': reduced.get('soil_moisture')})

time_series_fc = sm.map(extract_time_series)

time_series_list = time_series_fc.aggregate_array('properties').getInfo()
df = pd.DataFrame(time_series_list)

if not df.empty and 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['soil_moisture'], marker='o', linestyle='-', color='b')
    plt.xlabel("Date")
    plt.ylabel("Soil Moisture")
    plt.title("Soil Moisture Time Series using Landsat-8 (TOTRAM Algorithm)")
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
else:
    print("⚠️ No valid soil moisture data found for the selected ROI and time range.")

lc = (
    ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
    .select('label')
    .filterBounds(roi)
    .mode()
)
mask = lc.neq(0).And(lc.neq(6))

sm_mean = sm.mean() 
sm_masked = sm_mean.updateMask(mask)

Map = geemap.Map()
Map.centerObject(roi, zoom=8)

Map.addLayer(mask.clip(roi), {}, "Mask")
Map.addLayer(sm_masked.clip(roi), {}, "Soil Moisture Masked")

Map


# **2. Evapotranspiration and Crop Water Stress Index (CWSI) Analysis Using NASA MODIS and HydroSHEDS Data**

# In[2]:



cor = [70.38293457, 28.39841461]
loc = ee.Geometry.Point(cor)

roi = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_5").filterBounds(loc)

time_start = '2001-01-01'
time_end = '2024-01-01'

modis = ee.ImageCollection("MODIS/061/MOD16A2GF").select(['ET', 'PET']).filterDate(time_start, time_end)

et_mean = modis.select('ET').mean().multiply(0.1).clip(roi)

et_mean_summer = modis.select('ET').filter(ee.Filter.calendarRange(6, 8, 'month')).mean().multiply(0.1).clip(roi)

def compute_cwsi(img):
    cwsi = img.expression(
        '1 - (et / pet)', {
            'et': img.select('ET').multiply(0.1),
            'pet': img.select('PET').multiply(0.1)
        }
    ).rename('CWSI')
    return cwsi.copyProperties(img, ['system:time_start', 'system:time_end'])

cwsi = modis.map(compute_cwsi)
cwsi_mean = cwsi.mean().clip(roi)

lc = ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1').mode()
crop_mask = lc.eq(12) 
cwsi_crop = cwsi_mean.updateMask(crop_mask)

region_coordinates = roi.geometry().bounds().coordinates().getInfo()[0]

Map = geemap.Map()
Map.centerObject(roi, zoom=6)

Map.addLayer(roi, {}, "Basin ROI")
Map.addLayer(et_mean, {'min': 0, 'max': 100, 'palette': ['blue', 'green', 'yellow', 'red']}, "Mean ET")
Map.addLayer(et_mean_summer, {'min': 0, 'max': 100, 'palette': ['blue', 'green', 'yellow', 'red']}, "Summer ET")
Map.addLayer(cwsi_mean, {'min': 0, 'max': 1, 'palette': ['green', 'yellow', 'red']}, "CWSI Mean")
Map.addLayer(crop_mask, {'palette': ['gray', 'green']}, "Crop Mask")
Map.addLayer(cwsi_crop, {'min': 0, 'max': 1, 'palette': ['green', 'yellow', 'red']}, "CWSI Mean (Crops)")

Map 


# **3. Groundwater Storage Analysis in Punjab Using Python and NASA GLDAS Data**

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

geometry = ee.Geometry.Polygon([
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
])

time_start = ee.Date('2003-01-01')
time_end = ee.Date('2024-01-01')

gldas = (
    ee.ImageCollection("NASA/GLDAS/V022/CLSM/G025/DA1D")
    .select('GWS_tavg')
    .filterDate(time_start, time_end)
)

def extract_gw(image):
    """Extracts mean groundwater storage for each timestamp."""
    mean_gw = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=27000,
        bestEffort=True
    )
    return ee.Feature(None, {
        'date': image.date().format(),
        'GW_Storage': mean_gw.get('GWS_tavg')
    })

gw_time_series = gldas.map(extract_gw).filter(ee.Filter.notNull(['GW_Storage']))

data_list = gw_time_series.aggregate_array('GW_Storage').getInfo()
date_list = gw_time_series.aggregate_array('date').getInfo()

df = pd.DataFrame({'Date': pd.to_datetime(date_list), 'GW_Storage': data_list})

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['GW_Storage'], marker='o', linestyle='-', color='b', label='Groundwater Storage')
plt.xlabel('Year')
plt.ylabel('GW Storage (mm)')
plt.title('Groundwater Storage Time Series (GLDAS)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()


print("Task Completed")


# **4. Water Body Detection and Turbidity Assessment Using Sentinel-2 in Python**

# In[4]:


cor = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]
roi = ee.Geometry.Polygon(cor)

sen = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
       .select(['B2', 'B3', 'B4', 'B8'])
       .filterDate('2023-01-01', '2024-01-01')
       .filterBounds(roi)
       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
       .median()
       .multiply(0.0001))

ndwi = sen.normalizedDifference(['B3', 'B8']).rename('ndwi')

thr = ndwi.gt(0.1)
sen_mask = sen.updateMask(thr)

ndti = sen_mask.normalizedDifference(['B4', 'B3']).rename('ndti')


Map = geemap.Map()
Map.centerObject(roi, zoom=6)

Map.addLayer(sen.clip(roi), {'bands': ['B8', 'B4', 'B3']}, "False Color Sentinel-2")
Map.addLayer(ndwi.clip(roi), {}, "NDWI")
Map.addLayer(thr.clip(roi), {}, "Thresholded NDWI")
Map.addLayer(sen_mask.clip(roi), {}, "Masked Sentinel-2")
Map.addLayer(ndti.clip(roi), {'palette': ['blue', 'green', 'yellow', 'orange', 'red']}, "NDTI")

Map


# **5. Water Body Detection Using MODIS NDWI in Python**

# In[6]:


import numpy as np
cor = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]
roi = ee.Geometry.Polygon(cor)

modis = (ee.ImageCollection("MODIS/061/MOD09A1")
         .filterDate('2001-01-01', '2023-12-31')
         .map(lambda img: img.select('sur_refl.*')
              .multiply(0.0001)
              .normalizedDifference(['sur_refl_b04', 'sur_refl_b02'])
              .rename('ndwi'))
         )

count = modis.size().getInfo()
if count == 0:
    raise ValueError("No images found in the specified date range!")

ndwi_median = modis.median().clip(roi)
band_names = ndwi_median.bandNames().getInfo()
if not band_names:
    ndwi_median = modis.mosaic().clip(roi)

otsu_thr = 0.2  

ndwi_thr = ndwi_median.gt(otsu_thr)
band_names_thr = ndwi_thr.bandNames().getInfo()
if not band_names_thr:
    raise ValueError("Thresholded NDWI image has no bands! Check data filtering.")

Map = geemap.Map()
Map.centerObject(roi, zoom=8)
Map.addLayer(roi, {}, "ROI")
Map.addLayer(ndwi_median, {'palette': ['blue', 'green', 'yellow', 'red']}, "NDWI Median")
Map.addLayer(ndwi_thr, {}, "NDWI Thresholded Water Bodies")

Map


# **6. Flood Depth Mapping Using JRC Flood Hazard Data**

# In[7]:


import matplotlib.pyplot as plt

cor = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]

geometry = ee.Geometry.Polygon(cor)

jrc = ee.ImageCollection("JRC/CEMS_GLOFAS/FloodHazard/v1").filterBounds(geometry)

return_periods = jrc.aggregate_array('return_period').getInfo()
print("Available Return Periods:", return_periods)

def get_flood_image(return_period):
    return jrc.filter(ee.Filter.eq('return_period', return_period)).mosaic().clip(geometry)

flood_10 = get_flood_image(10)
flood_20 = get_flood_image(20)
flood_100 = get_flood_image(100)

band_names = flood_10.bandNames().getInfo()
print("Available Bands in Flood Data:", band_names)

flood_band = band_names[0] if band_names else None
if flood_band is None:
    raise ValueError("No bands available in the dataset!")

Map = geemap.Map()
Map.centerObject(geometry, zoom=8)

Map.addLayer(flood_10, {'palette': ['skyblue', 'blue', 'darkblue']}, "Flood Hazard (10-year)")
Map.addLayer(flood_20, {'palette': ['skyblue', 'blue', 'darkblue']}, "Flood Hazard (20-year)")
Map.addLayer(flood_100, {'palette': ['skyblue', 'blue', 'darkblue']}, "Flood Hazard (100-year)")

histogram = flood_10.reduceRegion(
    reducer=ee.Reducer.histogram(),
    geometry=geometry,
    scale=1000,
    bestEffort=True
).get(flood_band)  
hist_data = histogram.getInfo()
plt.figure(figsize=(8, 5))
plt.bar(hist_data['bucketMeans'], hist_data['histogram'], width=0.02, color='blue', alpha=0.7)
plt.xlabel("Flood Depth")
plt.ylabel("Frequency")
plt.title("Flood Depth Distribution (10-year Return Period)")
plt.grid()
plt.show()

Map


# **7. Surface Water and Ocean Topography (SWORD)**

# In[8]:


cor = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]
geometry = ee.Geometry.Polygon(cor)

nodes_merged = ee.FeatureCollection("projects/sat-io/open-datasets/SWORD/nodes_merged")
reaches_merged = ee.FeatureCollection("projects/sat-io/open-datasets/SWORD/reaches_merged")

nodes = nodes_merged.filterBounds(geometry)
reach = reaches_merged.filterBounds(geometry)

Map = geemap.Map()
Map.centerObject(geometry, zoom=8)

Map.addLayer(nodes, {"color": "blue"}, "SWORD Nodes")
Map.addLayer(reach, {"color": "red"}, "SWORD Reaches")

Map


# **8. Lake Area Time Series Analysis Using MODIS NDWI**

# In[10]:


cor = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]

roi = ee.Geometry.Polygon(cor)

time_start = '2001-01-01'
time_end = '2023-12-31'

modis = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(time_start, time_end)

ndwi = modis.map(lambda img: img.select('sur.*')
                 .multiply(0.0001)
                 .normalizedDifference(['sur_refl_b04', 'sur_refl_b02'])
                 .rename('ndwi')
                 .copyProperties(img, img.propertyNames()))

ndwi_filtered = ndwi.filterDate('2010-01-01', '2011-12-31')
ndwi_median = ndwi_filtered.median().clip(roi)

def compute_lake_area(img):
    thr = img.gt(0.1)
    mask = thr.updateMask(thr)
    area = mask.multiply(ee.Image.pixelArea().divide(1e6))
    
    area_value = area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=500,
        bestEffort=True,
        maxPixels=1e6
    )

    return ee.Feature(None, {'date': img.get('system:time_start'), 'lake_area': area_value.get('ndwi')})

lake_area_fc = ee.FeatureCollection(ndwi.map(compute_lake_area))

lake_area_list = lake_area_fc.aggregate_array('lake_area').getInfo()
dates_list = lake_area_fc.aggregate_array('date').getInfo()

dates = pd.to_datetime([ee.Date(d).format('YYYY-MM-dd').getInfo() for d in dates_list])

df = pd.DataFrame({'Date': dates, 'Lake Area (sq km)': lake_area_list})

plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Lake Area (sq km)'], marker='o', linestyle='-', color='b')
plt.xlabel("Year")
plt.ylabel("Lake Area (sq km)")
plt.title("Lake Area Time Series (NDWI > 0.1)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

Map = geemap.Map()
Map.centerObject(roi, zoom=8)

Map.addLayer(roi, {}, "ROI")
Map.addLayer(ndwi_median, {'palette': ['blue', 'green', 'yellow', 'red']}, "NDWI Median")
Map


# **9. Surface Water Area and Turbidity Analysis Using Landsat and Sentinel-2 Data in Python**

# In[11]:



cor = [
    [69.44, 27.67],
    [75.23, 27.67],
    [75.23, 34.02],
    [69.44, 34.02],
    [69.44, 27.67]
]
roi = ee.Geometry.Polygon(cor)


sen = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
       .select(['B2', 'B3', 'B4', 'B8'])
       .filterDate('2023-01-01', '2024-01-01')
       .filterBounds(roi)
       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
       .median()
       .multiply(0.0001))


ndwi = sen.normalizedDifference(['B3', 'B8']).rename('ndwi')


thr = ndwi.gt(0.1)
sen_mask = sen.updateMask(thr)


ndti = sen_mask.normalizedDifference(['B4', 'B3']).rename('ndti')

Map = geemap.Map()
Map.centerObject(roi, zoom=6)

Map.addLayer(sen.clip(roi), {'bands': ['B8', 'B4', 'B3']}, "False Color Sentinel-2")
Map.addLayer(ndwi.clip(roi), {}, "NDWI")
Map.addLayer(thr.clip(roi), {}, "Thresholded NDWI")
Map.addLayer(sen_mask.clip(roi), {}, "Masked Sentinel-2")
Map.addLayer(ndti.clip(roi), {'palette': ['blue', 'green', 'yellow', 'orange', 'red']}, "NDTI")

Map


# In[ ]:




