!pip install cdsapi
!pip install xarray pandas openpyxl
import os
import pandas as pd
import xarray as xr
import zipfile
import shutil
import cdsapi
from google.colab import files
# Replace 'your_key' with the actual key from the CDS API page
api_key = """
url: https://cds.climate.copernicus.eu/api
key: f*Insert your api key*f
"""

# Write the API key to the .cdsapirc file
with open('/root/.cdsapirc', 'w') as file:
    file.write(api_key)

print("CDS API key configured.")
os.environ['CDSAPI_CONFIG_FILE'] = '/root/cdsapirc'

# Define parameters
gcm_list = ["ACCESS-CM2", "AWI-CM-1-1-MR", "BCC-CSM2-MR", "CESM2", "CMCC-ESM2", "CNRM-CM6-1", "CNRM-ESM2-1", "INM-CM5-0", "MIROC6", "MPI-ESM1-2-LR", "MRI-ESM2-0", "NorESM2-LM", "NorESM2-MM"]
Gcm_var = ["daily_maximum_near_surface_air_temperature", "daily_minimum_near_surface_air_temperature", "precipitation"]
coordinates = [37, 74, 35, 76]  # [North, West, South, East]

desired_coords = [
    ("Khunjerab", 36.85, 75.4),
    ("Naltar", 36.13, 74.18),
    ("Ziarat", 36.83, 74.42),
]

variable_key_mapping = {
    "precipitation": "pr",
    "daily_maximum_near_surface_air_temperature": "tasmax",
    "daily_minimum_near_surface_air_temperature": "tasmin"
}

# Historical period: 2013-2014
Historical_Period = [
    str(year) for year in range(2013, 2014)
]

# Directories
output_dir = "/climate_data"
temp_dir = "/temp"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# CDS API setup
client = cdsapi.Client()

# Main loop
for gcm in gcm_list:
    extracted_data = pd.DataFrame()
    for variable in Gcm_var:
        # Define request
        request = {
        "temporal_resolution": "daily",
        "experiment": "historical",
        "variable": variable,
        "model": gcm,
        "area": coordinates,
        "year": Historical_Period,
         "month": [
         "01", "02", "03",
         "04", "05", "06",
         "07", "08", "09",
         "10", "11", "12"
         ],
         "day": [
         "01", "02", "03",
         "04", "05", "06",
         "07", "08", "09",
         "10", "11", "12",
         "13", "14", "15",
         "16", "17", "18",
         "19", "20", "21",
         "22", "23", "24",
         "25", "26", "27",
         "28", "29", "30",
         "31"
          ]
          }

            # Download file
        zip_file_path = os.path.join(output_dir, f"{gcm}_historical_{variable}_.zip")
        try:
            client.retrieve("projections-cmip6", request).download(zip_file_path)
            print(f"Downloaded: {zip_file_path}")
        except Exception as e:
            print(f"Failed to download for {gcm} - {variable} - : {e}")
            continue

        # Extract the zip file to temporary folder
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Process the NetCDF file
        nc_files = [f for f in os.listdir(temp_dir) if f.endswith('.nc')]
        if not nc_files:
            print(f"No NetCDF file found for {gcm} - {variable} -.")
            continue

        # Extract data from NetCDF
        nc_file_path = os.path.join(temp_dir, nc_files[0])
        ds = xr.open_dataset(nc_file_path, decode_times=False)

        # Align dimensions between time values and the DataFrame index
        time_values = ds['time'].values
        extracted_data = extracted_data.reindex(range(len(time_values)))
        extracted_data['Date'] = pd.to_datetime(time_values, errors='coerce')

        # Map variable to NetCDF key
        netcdf_key = variable_key_mapping.get(variable, None)
        if not netcdf_key:
            print(f"No NetCDF key found for variable '{variable}'. Skipping...")
            continue

        # Extract data for specified coordinates
        for nam, lat, lon in desired_coords:
            try:
                grid_data = ds.sel(lat=lat, lon=lon, method='nearest')[netcdf_key].values
                column_name = f"{gcm}_{netcdf_key}_{nam}"
                extracted_data[column_name] = grid_data
            except KeyError:
                print(f"Variable '{netcdf_key}' not found in {nc_file_path}. Skipping...")
                continue

        # Clean up temporary folder
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

    # Save extracted data to CSV
    csv_file_path = os.path.join(output_dir, f"{gcm}_historical_data.csv")
    extracted_data.to_csv(csv_file_path, index=False)
    files.download(csv_file_path)
    print(f"Saved data to: {csv_file_path}")

print("Processing complete.")
