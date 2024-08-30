import pandas as pd
import numpy as np
from os import path


def load_data(data_path, data_file):
    # load data from csv
    df = pd.read_csv(path.join(data_path, data_file))
    return df

def compute_profit_eps(df):
    # compute profit per episode
    N = (df[df['episode'] == 0]).shape[0]
    profits_per_episode = df[['Profits', 'episode']].groupby('episode').sum().reset_index()
    return profits_per_episode


def aggregate_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    '''
    Function that computes statistics for violations and profits per episode.
    This function takes in a DataFrame and the name of the column to be used for violations.

    Args:
        - df: the DataFrame
        - column: the name of the column to be used for violations

    Returns:
        - episode: the episode number
        - Profits: the total profits for the episode
        - CO2 Violation Time (%): the percentage of time with CO2 violations
        - CO2 Violation (ppm): the average magnitude of CO2 violations
    '''
    # print(df)
    N = (df[df['episode'] == 0]).shape[0]
    profits_per_episode = df[['Profits', 'episode']].groupby('episode').sum().reset_index()
    # CO2 violation time per episode, considering each row as 5 minutes
    co2_violation_time_updated = df[df[column] > 0].groupby('episode').size()/N*100 # % of time with violation
    # print(co2_violation_time_updated)
    # co2_violation_time_updated = 100-co2_violation_time_updated
    # print(co2_violation_time_updated)
    # Average magnitude of CO2 violations per episode, for positive violations only
    # avg_co2_violation_magnitude_updated = df[[column, 'episode']].groupby('episode')[column].sum()
    avg_co2_violation_magnitude_updated = df[df[column] > 0].groupby('episode')[column].mean()
    # Combine the updated results into a summary DataFrame
    summary_df_updated = pd.DataFrame({
        f'Time within boundary (%)': co2_violation_time_updated,
        f'{column} (abs)': avg_co2_violation_magnitude_updated,
    }).reset_index()

    # add coefficient to resulting 

    # Create a DataFrame of all unique episodes to ensure all are represented
    all_episodes_df = pd.DataFrame(df['episode'].unique(), columns=['episode'])

    # Merge the summary of violations with the complete list of episodes
    # This ensures episodes with no violations are included, filling missing values appropriately
    full_summary_df = pd.merge(all_episodes_df, summary_df_updated, on='episode', how='left').fillna(0)
    # print(full_summary_df['coefficients'])
    full_summary_df = pd.merge(profits_per_episode, full_summary_df, on='episode', how='left').fillna(0)
    full_summary_df['Time within boundary (%)'] = 100- full_summary_df['Time within boundary (%)']
    return full_summary_df

def ci(std, n, z=1.96):
    return z*std/np.sqrt(n)


def calculate_twb(dataframes, labels):
    '''
    Function that calculates the average time within boundary (TWB) for different violation types.

    Args:
        - dataframes: a list of DataFrames containing violation data
        - labels: a list of labels for each DataFrame

    Returns:
        - twb_df: a DataFrame containing the average TWB for each violation type and label
        - twb_df_ci: a DataFrame containing the confidence intervals for the average TWB
    '''
    twb_df = pd.DataFrame()
    twb_df_ci = pd.DataFrame()
    N = dataframes[0]['episode'].unique().shape[0]
    vars = ['CO2 violation', 'Temperature violation', 'Humidity violation']
    for j, df in enumerate(dataframes):
        violations = [aggregate_data(df, var) for var in vars]

        twb = np.array([violations[i]['Time within boundary (%)'].mean() for i in range(len(vars))])
        df_twb = pd.DataFrame({labels[j]: twb,}, index=vars)
        twb_df = pd.concat([twb_df, df_twb], axis=1)
        
        cis = [ci(violations[i]['Time within boundary (%)'].std(), N) for i in range(len(vars))]
        df_twb_ci = pd.DataFrame({labels[j]: cis,}, index=vars)
        twb_df_ci = pd.concat([twb_df_ci, df_twb_ci], axis=1)
    return twb_df.T, twb_df_ci.T

