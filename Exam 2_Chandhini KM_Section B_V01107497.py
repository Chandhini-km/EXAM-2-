#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# # SECTION B Part B

# In[32]:


# using data from pinksheet 


# In[3]:


# Set working directory
import os
os.chdir('C:\\Users\\Chand\\Downloads\\Test 2')
print(os.getcwd())


# In[5]:


# Load the data
df = pd.read_excel('CMO-Historical-Data-Monthly (2).xlsx', sheet_name="Monthly Prices", skiprows=6)


# In[6]:


# Rename the first column to "Date"
df.rename(columns={df.columns[0]: 'Date'}, inplace=True)


# In[7]:


# Convert the Date column to datetime format
def parse_date(date_str):
    # Split the string by 'M' to separate year and month
    year, month = date_str.split('M')
    # Create a date string in the format 'YYYY-MM-01'
    return f"{year}-{month}-01"

df['Date'] = pd.to_datetime(df['Date'].apply(parse_date))

# Print the data types to confirm the conversion
print(df.dtypes)


# In[8]:


# Select specific columns (Date and selected commodities)
commodity = df.iloc[:, [0, 2, 24, 69, 71, 60, 30]]
commodity.columns = [col.lower().replace(' ', '_') for col in commodity.columns]  # Clean column names

print(commodity.dtypes)


# In[9]:


# Remove the Date column for analysis
commodity_data = commodity.drop(columns=['date'])

# Column names to test
columns_to_test = commodity_data.columns


# In[10]:


# Initialize counters and lists for stationary and non-stationary columns
non_stationary_count = 0
stationary_columns = []
non_stationary_columns = []


# In[11]:


# Loop through each column and perform the ADF test
for col in columns_to_test:
    adf_result = adfuller(commodity_data[col])
    p_value = adf_result[1]  # Extract p-value for the test
    print(f"\nADF test result for column: {col}")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {p_value}")
    
    # Check if the p-value is greater than 0.05 (commonly used threshold)
    if p_value > 0.05:
        non_stationary_count += 1
        non_stationary_columns.append(col)
    else:
        stationary_columns.append(col)


# In[12]:


# Print the number of non-stationary columns and the lists of stationary and non-stationary columns
print(f"\nNumber of non-stationary columns: {non_stationary_count}")
print(f"Non-stationary columns: {non_stationary_columns}")
print(f"Stationary columns: {stationary_columns}")


# In[13]:


# Co-Integration Test (Johansen's Test)
def johansen_test(df, alpha=0.05):
    out = coint_johansen(df, det_order=0, k_ar_diff=1)
    d = {'0.90': 0, '0.95': 1, '0.99': 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]
    print(f"Trace statistic: {traces}")
    print(f"Critical values: {cvts}")
    print(f"Eigenvalues: {out.eig}")
    for col, trace, cvt in zip(df.columns, traces, cvts):
        if trace > cvt:
            print(f"{col} is cointegrated.")
        else:
            print(f"{col} is not cointegrated.")
    return out


# In[14]:


# Perform Johansen cointegration test
coint_test = johansen_test(commodity_data)


# In[15]:


# Number of cointegrating relationships (assuming r = 1 if there's at least one significant eigenvalue)
r = sum(coint_test.lr1 > coint_test.cvt[:, 1])  # Replace with the actual number from the test results


# In[16]:


df.columns


# In[17]:


# Extract 'GOLD' column
gold_data = df['GOLD']

# Plot gold prices
plt.figure(figsize=(12, 6))
plt.plot(gold_data.index, gold_data, label='Gold Price')
plt.title('Gold Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($/troy oz)')
plt.legend()
plt.show()


# In[18]:


# Split data into train and test sets
train_size = int(len(gold_data) * 0.8)
train, test = gold_data[:train_size], gold_data[train_size:]

print("Training data size:", len(train))
print("Test data size:", len(test))


# # Fitting SARIMA Model

# In[19]:


# Fit SARIMA model
sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()


# In[20]:


# Forecast
sarima_forecast = sarima_result.get_forecast(steps=len(test))
sarima_forecast_index = test.index
sarima_forecast_mean = sarima_forecast.predicted_mean
sarima_forecast_conf_int = sarima_forecast.conf_int()


# In[21]:


# Plot SARIMA Forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(sarima_forecast_index, sarima_forecast_mean, label='SARIMA Forecast', color='orange')
plt.fill_between(sarima_forecast_index, sarima_forecast_conf_int.iloc[:, 0], sarima_forecast_conf_int.iloc[:, 1], color='orange', alpha=0.3)
plt.title('SARIMA Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price ($/troy oz)')
plt.legend()
plt.show()


# ## Interpretation 
# a line chart comparing the actual prices of an asset  gold, given the "Price ($/troy oz)" label) against a SARIMA (Seasonal AutoRegressive Integrated Moving Average) forecast. The chart is divided into two sections: "Train" and "Test.
# 
# Key Observations
# 
# Data Range: The x-axis spans approximately 800 data points, representing a timeframe of around 800 days (assuming daily data).
# Price Range: The y-axis covers prices from 0 to 2500, with the majority of data points clustering between 0 and 500 for a significant portion of the time series.
# Train Data: The blue line, representing the "Train" data, exhibits a relatively stable pattern with fluctuations throughout the initial period.
# Test Data: The orange line, representing the "Test" data, shows a more volatile pattern with an upward trend towards the end of the timeframe.
# SARIMA Forecast: The yellow shaded area represents the SARIMA forecast. It appears to follow the general trend of the "Train" data but deviates significantly from the "Test" data, particularly in the latter part of the timeframe.
# 
# Interpretation
# 
# The SARIMA model was trained on the "Train" data to predict future prices.
# The model's performance seems reasonable during the initial period ("Train" data) but becomes less accurate in predicting the sharp price increase observed in the "Test" data.
# This discrepancy suggests that the SARIMA model might not be capturing all the underlying factors driving the price movements, especially during periods of high volatility.

# In[22]:


# Evaluation metrics
sarima_rmse = np.sqrt(mean_squared_error(test, sarima_forecast_mean))
sarima_mape = np.mean(np.abs((test - sarima_forecast_mean) / test)) * 100
sarima_mae = mean_absolute_error(test, sarima_forecast_mean)

print(f'SARIMA RMSE: {sarima_rmse}')
print(f'SARIMA MAPE: {sarima_mape}')
print(f'SARIMA MAE: {sarima_mae}')


# In[23]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


# # Fitting LSTM Model

# In[24]:


# Prepare data for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

look_back = 12
train_values = train.values
test_values = test.values


# In[25]:


X_train, Y_train = create_dataset(train_values, look_back)
X_test, Y_test = create_dataset(test_values, look_back)


# In[26]:


# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# In[27]:


# LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')


# In[28]:


# Train model
lstm_model.fit(X_train, Y_train, epochs=50, verbose=0)


# In[29]:


# Forecast
lstm_forecast = lstm_model.predict(X_test)
lstm_forecast = np.reshape(lstm_forecast, (lstm_forecast.shape[0]))


# In[30]:


# Plot LSTM Forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index[look_back:], test[look_back:], label='Test')
plt.plot(test.index[look_back:], lstm_forecast, label='LSTM Forecast', color='green')
plt.title('LSTM Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price ($/troy oz)')
plt.legend()
plt.show()


# ## Interpretation
# 
# The image presents a line chart comparing the actual prices of an asset  gold, given the "Price ($/troy oz)" label) against an LSTM (Long Short-Term Memory) forecast. The chart is divided into two sections: "Train" and "Test."
# 
# Key Observations:
# 
# Data Range: The x-axis spans approximately 800 data points, representing a timeframe of around 800 days (assuming daily data).
# Price Range: The y-axis covers prices from 0 to 2500, with the majority of data points clustering between 0 and 500 for a significant portion of the time series.
# Train Data: The blue line, representing the "Train" data, exhibits a relatively stable pattern with fluctuations throughout the initial period.
# Test Data: The orange line, representing the "Test" data, shows a more volatile pattern with an upward trend towards the end of the timeframe.
# LSTM Forecast: The green line, representing the LSTM forecast, generally follows the trend of the "Train" data but deviates significantly from the "Test" data, especially in the latter part of the timeframe.
# Interpretation:
# 
# The LSTM model was trained on the "Train" data to predict future prices.
# The model's performance seems reasonable during the initial period ("Train" data) but becomes less accurate in predicting the sharp price increase observed in the "Test" data.
# This discrepancy suggests that the LSTM model might not be capturing all the underlying factors driving the price movements, especially during periods of high volatility.

# In[31]:


# Evaluation metrics
lstm_rmse = np.sqrt(mean_squared_error(test[look_back:], lstm_forecast))
lstm_mape = np.mean(np.abs((test[look_back:] - lstm_forecast) / test[look_back:])) * 100
lstm_mae = mean_absolute_error(test[look_back:], lstm_forecast)

print(f'LSTM RMSE: {lstm_rmse}')
print(f'LSTM MAPE: {lstm_mape}')
print(f'LSTM MAE: {lstm_mae}')


# In[ ]:




