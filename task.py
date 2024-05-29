import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


#завантаження даних
file_path = 'AirPassengers.csv'
data = pd.read_csv(file_path)

df = pd.DataFrame(data)
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(df.index, df['#Passengers'], label='Кількість пасажирів')
plt.title('Кількість авіапасажирів (1949-1960)')
plt.xlabel('Дата')
plt.ylabel('Кількість пасажирів')
plt.legend()
plt.show()


#перевірка стаціонарності ряду за допомогою тесту Дікі-Фуллера
result = adfuller(df['#Passengers'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])


#автокореляція та часткова автокореляція
plt.figure(figsize=(10, 6))
plot_acf(df['#Passengers'], lags=40, ax=plt.gca())
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(df['#Passengers'], lags=40, ax=plt.gca())
plt.show()


#розбиття на тренд, сезонність та залишок
decomposition = seasonal_decompose(df['#Passengers'], model='multiplicative', period=12)
decomposition.plot()
plt.show()


#застосуємо диференціювання, аби зробити ряд стаціонарним
df['Passengers'] = df['#Passengers'].diff().dropna()

result_diff = adfuller(df['Passengers'].dropna())
print('ADF Statistic (диференційований ряд):', result_diff[0])
print('p-value (диференційований ряд):', result_diff[1])


#побудова моделі ARIMA
model = ARIMA(df['Passengers'], order=(5, 1, 0))
fit = model.fit()
print(fit.summary())

forecast = fit.forecast(steps=24)
plt.figure(figsize=(10, 6))
plt.plot(df['Passengers'], label='Історичні дані')
plt.plot(forecast, label='Прогноз')
plt.title('Прогноз кількості авіапасажирів')
plt.xlabel('Дата')
plt.ylabel('Кількість пасажирів')
plt.legend()
plt.show()


# Прогнозування на основі моделі
future_dates = pd.date_range(start='1961-01-01', periods=24, freq='MS')
forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Passengers'])

plt.figure(figsize=(10, 6))
plt.plot(df['Passengers'], label='Історичні дані')
plt.plot(forecast_df, label='Прогноз')
plt.title('Прогноз кількості авіапасажирів (1961-1962)')
plt.xlabel('Дата')
plt.ylabel('Кількість пасажирів')
plt.legend()
plt.show()
