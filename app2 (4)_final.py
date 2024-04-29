import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet



START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Gold Price Prediction')

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365
# Fetch historical gold price data

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data('GC=F')
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()


# Prepare the data
features = ['Open', 'High', 'Low', 'Volume']  # You can customize the features based on your needs
X = data [features]
y = data ['Close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Show and plot forecast
st.subheader('Prediction data')




# Initialize the plot
fig, ax = plt.subplots()
x_vals = np.arange(len(y_test))
y_vals_actual = y_test.values
line_actual, = ax.plot(x_vals, y_vals_actual, label='Actual', color='blue')
y_vals_pred = model.predict(X_test)
line_pred, = ax.plot(x_vals, y_vals_pred, label='Predicted', color='green')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

st.write(f'Prediction plot for {n_years} years')
#fig1 = plot_plotly(model, y_pred)
# Create a Matplotlib plot
plt.figure(figsize=(10, 6))

plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted', color='red')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Predicted plot')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Convert the Matplotlib plot to Plotly
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=data['Date'], y=y_test, name='Actual', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=data['Date'], y=y_pred, name='Predicted', line=dict(color='red')))
fig1.update_layout(title='Predicted plot', xaxis_title='Date', yaxis_title='Value',xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)



# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})


m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")