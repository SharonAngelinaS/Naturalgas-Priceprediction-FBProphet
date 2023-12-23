import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.markdown(f"""
<style>
.reportview-container {{
    background-color: white;
}}
</style>
""", unsafe_allow_html=True)

def set_background_color(color):
    hex_color = f"#{color}"
    css = f"""
    <style>
    .reportview-container {{
        background-color: {hex_color};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set the background color to white
set_background_color("FFFFFF")

START = "2000-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Natural Gas Price Forecast App using FBProphet ')

# Ticker symbol for natural gas futures on Yahoo Finance
ticker_symbol = "NG=F"

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(ticker_symbol)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="gas_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="gas_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
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
