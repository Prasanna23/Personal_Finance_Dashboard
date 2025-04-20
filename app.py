import streamlit as st
from utils.portfolio import load_portfolio, fetch_prices, calculate_value

st.set_page_config(page_title="DCA Visualizer", layout="wide")

st.title("DCA Visualizer: Portfolio Tracker")

# Load CSV
uploaded_file = st.file_uploader("Upload your portfolio CSV", type="csv")

if uploaded_file:
    df = load_portfolio(uploaded_file)
    tickers = df['ticker'].tolist()
    prices = fetch_prices(tickers)
    df = calculate_value(df, prices)
    
    st.subheader("Current Portfolio")
    st.dataframe(df)

    st.metric("Total Value", f"${df['current_value'].sum():,.2f}")
    st.metric("Total Gain/Loss", f"${df['gain_loss'].sum():,.2f}")
else:
    st.info("Upload a CSV with columns: ticker, shares, cost_basis")
