# black scholes model pricer

import numpy as np
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return call

def main():
    st.title('Black-Scholes Option Pricing Model')
    st.write('This app calculates the price of a European call option using the Black-Scholes model.')

    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        S = st.slider('Stock Price (S)', min_value=1, max_value=200, value=100)
        K = st.slider('Strike Price (K)', min_value=1, max_value=200, value=100)
        T = st.slider('Time to Maturity (T) in years', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
    with col2:
        r = st.slider('Risk-free Rate (r)', min_value=0.0, max_value=0.20, value=0.05, step=0.01)
        sigma = st.slider('Volatility (σ)', min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    # Calculate option price
    option_price = black_scholes_call(S, K, T, r, sigma)
    st.write(f'## Call Option Price: ${option_price:.2f}')

    # Create price visualization
    stock_prices = np.linspace(max(1, S-50), S+50, 100)
    option_prices = [black_scholes_call(s, K, T, r, sigma) for s in stock_prices]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_prices,
        y=option_prices,
        mode='lines',
        name='Option Price'
    ))

    fig.update_layout(
        title='Option Price vs Stock Price',
        xaxis_title='Stock Price',
        yaxis_title='Option Price',
        hovermode='x'
    )

    st.plotly_chart(fig)

    # explanation of parameters
    st.markdown("""
    ### Parameters Explanation:
    - **S**: Current stock price
    - **K**: Strike price of the option
    - **T**: Time to maturity in years
    - **r**: Risk-free interest rate (annual)
    - **σ**: Stock price volatility (annual)
    """)

if __name__ == '__main__':
    main()