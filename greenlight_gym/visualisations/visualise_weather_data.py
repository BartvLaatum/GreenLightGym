import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from greenlight_gym.common.utils import loadWeatherData
import seaborn as sns
import matplotlib.pyplot as plt

train_data_path = 'envs/data/'


location = 'Amsterdam'
train_start_day = 59
train_end_day = 274	
train_growth_years = np.arange(2011, 2020)
source = 'KNMI'
n_days = train_end_day - train_start_day
pred_horizon = 0
h = 1
nd = 10

weather_columns = ['Global radiation', 'Outdoor temperature', 'Vapour density', 'CO2 concentration', 'Wind speed', 'Sky temperature', 'Soil temperature', 'DLI', 'is Day', 'is Day smooth']

weather_columns = weather_columns[:-4]
train_data = np.array([loadWeatherData(train_data_path, location, source, growth_year, train_start_day, n_days, pred_horizon, h, nd)[:,:-4] for growth_year in train_growth_years[:3]]).reshape(-1, len(weather_columns))

test_growth_years = np.arange(2001, 2011)
test_start_days = [59, 90, 120, 151, 181, 212, 243]
test_n_days = 10
test_data = np.array([loadWeatherData(train_data_path, location, source, growth_year, test_start_day, test_n_days, pred_horizon, h, nd)[:,:-4] for growth_year in test_growth_years[:] for test_start_day in test_start_days]).reshape(-1, len(weather_columns))

train_data = pd.DataFrame(train_data, columns=weather_columns)
test_data = pd.DataFrame(test_data, columns=weather_columns)

# Extract the air temperature column from the train_data and test_data DataFrames
train_air_temperature = train_data['Outdoor temperature']
test_air_temperature = test_data['Outdoor temperature']

# Plot the probability distribution of air temperature for train and test set
sns.kdeplot(train_air_temperature, label='Train Set')
sns.kdeplot(test_air_temperature, label='Test Set')
plt.xlabel('Air Temperature')
plt.ylabel('Probability')
plt.title('Probability Distribution of Outdoor Temperature')
plt.legend()
plt.show()

# Plot the distribution of air temperature for train and test set
plt.hist(train_air_temperature, bins=30, alpha=0.5, label='Train Set')
plt.hist(test_air_temperature, bins=30, alpha=0.5, label='Test Set')
plt.xlabel('Air Temperature')
plt.ylabel('Frequency')
plt.title('Distribution of Outdoor Temperature')
plt.legend()
plt.show()