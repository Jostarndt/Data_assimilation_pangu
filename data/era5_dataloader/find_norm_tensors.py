import xarray as xr
import numpy as np
import os
import glob
import re
import pdb

def calculate_variable_stats(files, variable_name):
    axis = (1,-2,-1)
    running_sum = None
    running_sum_sq = None
    total_samples = None
    
    for fid, f in enumerate(files):
        with xr.open_dataset(f) as ds:
            data = np.array([ds[var].values for var in variable_name])
            data_summed = data.sum(axis=axis)

            if running_sum is None:#during first iteration
                running_sum = np.zeros_like(data_summed)
                running_sum_sq = np.zeros_like(data_summed)
                total_samples = np.zeros_like(data_summed)

            running_sum += data_summed# first dim is variables
            running_sum_sq += data.sum(axis=axis) ** 2
            total_samples += data.size / data.shape[0]
    
    mean = running_sum / total_samples
    std = np.sqrt(running_sum_sq / total_samples - mean ** 2)

    mean_shape = mean.shape + (1,) * (4 - mean.ndim)
    std_shape = std.shape + (1,) * (4 - std.ndim)
    return mean.reshape(mean_shape), std.reshape(std_shape)



def filter_files_before_year(files, year_threshold=2018):
    """Filter files to only include those before the specified year"""
    filtered_files = []
    for f in files:
        # Extract year from filename (assumes format: era5_240_YYYY_*.nc)
        year_match = re.search(r'era5_240_(\d{4})_', f)
        if year_match:
            year = int(year_match.group(1))
            if year < year_threshold:
                filtered_files.append(f)
    return filtered_files

def calculate_variable_stats_direct(variable_name):
    data_path = "/srv/data/era_arches/era5_240/full"# Adjust this path
    
    # Get list of files (adjust pattern as needed)
    file_pattern = os.path.join(data_path, "era5_240_*.nc")
    files = sorted(glob.glob(file_pattern))
    
    print(f"Found {len(files)} files")
    print(f"Calculating stats for variable: {variable_name}")

    files = filter_files_before_year(files, year_threshold=2018)
    
    # Calculate statistics
    mean, std = calculate_variable_stats(files, variable_name)

    return mean, std

if __name__ == "__main__":
    # Set your data path and variable name
    data_path = "/srv/data/era_arches/era5_240/full"# Adjust this path
    #variable_name = "your_variable_name"
    #variable_name = ["u_component_of_wind", "v_component_of_wind"]
    variable_name = ["total_precipitation_24hr","2m_temperature"]
    #variable_name = "2m_temperature"
    
    
    # Get list of files (adjust pattern as needed)
    file_pattern = os.path.join(data_path, "era5_240_*.nc")
    files = sorted(glob.glob(file_pattern))
    
    print(f"Found {len(files)} files")
    print(f"Calculating stats for variable: {variable_name}")

    files = filter_files_before_year(files, year_threshold=2018)
    
    # Calculate statistics
    mean, std = calculate_variable_stats(files, variable_name)
    
    print(f"\nResults for {variable_name}:")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
