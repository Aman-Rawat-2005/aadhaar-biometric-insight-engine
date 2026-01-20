from statsmodels.tsa.arima.model import ARIMA

def forecast_series(series, steps=6):
    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast
