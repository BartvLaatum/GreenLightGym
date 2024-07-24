import argparse
from os.path import join

import numpy as np
import pandas as pd

from greenlight_gym.common.utils import compute_sky_temp

def load_raw_weather_data(path, filename):
    raw_weather_data = pd.read_csv(join(path, filename), skiprows=31,)
    return raw_weather_data

# seperate the data into years
def split_data(raw_weather_data, years):
    """
    Split the raw weather data into separate DataFrames for each year
    Args:
        raw_weather_data: DataFrame containing the raw weather data
        years: np.array, years for which to split the data
    Returns:
        dfs: list of DataFrames, each containing the weather data for a single year
    """
    dfs = []
    for year in years:
        dfs.append(raw_weather_data[raw_weather_data["YYYYMMDD"] // 10000 == year].reset_index(drop=True))
    return dfs

# fill nan values with median
def fill_cloud_nan_values(dfs):
    """
    Fill NaN values in the cloud cover column with the median value for each year
    Args:
        dfs: list of DataFrames, each containing the weather data for a single year
    Returns:
        dfs: list of DataFrames, each containing the weather data for a single year
    """
    for df in dfs:
        df["    N"] = df["    N"].replace("     ", np.nan)  # Replace empty strings with NaN
        median_value = df["    N"].median()
        df["    N"].fillna(median_value, inplace=True)
    return dfs

# create new columns from the existing data
def preprocess_weather_data(dfs):
    """
    Preprocess the weather data
    Args:
        dfs: list of DataFrames, each containing the weather data for a single year
    Returns:
        dfs: list of DataFrames, each containing the preprocessed weather data for a single year
    """
    for df in dfs:
        # convert J.cm-2.h-1 to W.m-2 (1m2  = 1e4 cm2, 1h = 3600s)
        df["global radiation"] = df["    Q"] * 1e4/3600

        # convert variables 0.1 units to SI units
        df["wind speed"] = df["   FH"] / 10
        df["air temperature"] = df["    T"] / 10

        df['sky temperature'] = compute_sky_temp(df['air temperature'], df['    N'].astype(int)/9)
        
        df['CO2 concentration'] = 400.
        # extract day of the year
        df['YYYYMMDD'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')

        # Extract the day of the year and create a new column with it
        df['day number'] = df['YYYYMMDD'].dt.dayofyear-1
        # convert hours to seconds since start of the year
        df["time"] = (df["   HH"]- 1) * 3600 + df["day number"] * 86400

        df["??"] = 0

        df['RH'] = df['    U']*1.
    return dfs

# interpolate the data
def interpolate_weather_data(dfs):
    """
    Interpolate the weather data
    Args:
        dfs: list of DataFrames, each containing the preprocessed weather data for a single year
    Returns:
        dfs: list of DataFrames, each containing the interpolated weather data for a single year
    """
    interpolated_dfs = []
    for df in dfs:
        df = df[["time", "global radiation", "wind speed", "air temperature", "sky temperature", "CO2 concentration", "??", "RH"]]
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Set the 'time' column as the index of the DataFrame
        df = df.set_index('time')

        # Resample the data to a five-minute frequency, creating NaNs for missing data points
        resampled_df = df.resample('5min').mean()

        # Interpolate the NaN values in the resampled DataFrame
        interpolated_df = resampled_df.interpolate(method='linear')

        # Calculate the time delta in seconds from the start of the year for each index
        start_of_year = interpolated_df.index[0].replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        interpolated_df['time'] = (interpolated_df.index - start_of_year).total_seconds()
        # drop the index
        interpolated_df = interpolated_df.reset_index(drop=True)

        interpolated_df["day number"] = np.floor(interpolated_df["time"] / 86400)

        # reorder the columns
        interpolated_df = interpolated_df[["time", "global radiation", "wind speed", "air temperature", "sky temperature","??", "CO2 concentration", "day number", "RH"]]
        interpolated_dfs.append(interpolated_df)
    return interpolated_dfs

# create function to save the columns
def save_weather_data(dfs, out_path, save_prefix, years):
    """
    Save the preprocessed and interpolated weather data to a file
    Args:
        dfs: list of DataFrames, each containing the interpolated weather data for a single year
        out_path: str, path to the output directory
        save_prefix: str, prefix for the output file name
        years: np.array, years for which to save the data
    """
    for i, df in enumerate(dfs):
        df.to_csv(join(out_path, f"{save_prefix}{years[i]}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the KNMI weather data")
    parser.add_argument("--in_path", type=str, default="envs/data/", help="Path to the raw weather data")
    parser.add_argument("--out_path", type=str, default="envs/data/Amsterdam/", help="Path to the output directory")
    parser.add_argument("--filename", type=str, default="raw_weather_data_adam", help="Path to the output directory")
    parser.add_argument("--start_year", type=int, default=2011, help="Start year of the data")
    parser.add_argument("--end_year", type=int, default=2020, help="End year of the data")
    parser.add_argument("--save_prefix", type=str, default="KNMI", help="Prefix for the output file name")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    assert args.start_year in [2001,2011], "Start year must be 2001 or 2011"
    assert args.end_year == args.start_year + 9, "End year must at end of decade after start year"
    years = np.arange(args.start_year, args.end_year+1)

    filename = f"{args.filename}_{args.start_year}{args.end_year}.txt"
    raw_weather_data = load_raw_weather_data(args.in_path, filename)

    dfs = split_data(raw_weather_data, years)
    dfs = fill_cloud_nan_values(dfs)
    dfs = preprocess_weather_data(dfs)
    dfs = interpolate_weather_data(dfs)
    save_weather_data(dfs, args.out_path, args.save_prefix, years)
