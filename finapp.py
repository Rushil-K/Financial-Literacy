import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Initialize Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ----------------- UI Layout -----------------
st.title("📊 Open-Source AI Financial Advisor")
st.sidebar.header("Select a Section")

option = st.sidebar.selectbox(
    "Choose Module", 
    ["Financial Literacy", "Investment Analysis", "Market News & AI Insights", "Hyperpersonalized Investment Prediction"]
)

# ----------------- Financial Literacy Module -----------------
if option == "Financial Literacy":
    st.subheader("📖 Financial Literacy")
    tab1, tab2, tab3, tab4 = st.tabs(["Budgeting", "Loans & Credit Score", "Savings", "Investment Basics"])
    
    with tab1:
        st.markdown("### Budget Planner")
        income = st.number_input("Enter your Monthly Income", min_value=0.0)
        expenses = st.number_input("Enter your Monthly Expenses", min_value=0.0)
        savings = income - expenses
        st.write(f"💰 Your Monthly Savings: **₹{savings}**")
        st.progress(min(1, savings / max(1, income)))  # Show savings ratio

    with tab2:
        st.markdown("### Loan & Credit Score Basics")
        st.write("- Keep credit utilization low")
        st.write("- Pay EMIs on time")
        st.write("- Avoid frequent credit inquiries")

    with tab3:
        st.markdown("### Best Saving Practices")
        st.write("1. Follow the **50-30-20 rule**: 50% Needs, 30% Wants, 20% Savings.")
        st.write("2. Automate savings.")
        st.write("3. Invest in **FDs, RDs, or SIPs**.")

    with tab4:
        st.markdown("### Investment Basics")
        st.write("💡 **Stocks**: High risk, high return")
        st.write("💡 **Mutual Funds**: Diversified, moderate risk")
        st.write("💡 **ETFs**: Passive investing, lower fees")
        st.write("💡 **Gold**: Hedge against inflation")

# ----------------- Investment Analysis Module -----------------
elif option == "Investment Analysis":
    st.subheader("📊 Stock & Mutual Fund Analysis")
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS):", "AAPL")
    
    # Fetch Stock Data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    # Display Stock Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name=ticker))
    st.plotly_chart(fig)

    # Moving Average & RSI Calculation
    hist['MA50'] = hist['Close'].rolling(50).mean()
    hist['RSI'] = 100 - (100 / (1 + (hist['Close'].diff().rolling(14).mean() / hist['Close'].diff().rolling(14).std())))

    st.write(f"**Current Price:** ₹{round(hist['Close'].iloc[-1], 2)}")
    st.write(f"**50-Day Moving Average:** ₹{round(hist['MA50'].iloc[-1], 2)}")

    # AI-Powered Stock Prediction
    st.subheader("📊 AI Stock Prediction")
    data = hist[['Close']].dropna()
    data['Day'] = np.arange(len(data))
    model = LinearRegression()
    model.fit(data[['Day']], data['Close'])
    future_price = model.predict([[len(data) + 5]])[0]

    st.write(f"📌 **Predicted Price in 5 Days: ₹{round(future_price, 2)}**")
    if future_price > hist['Close'].iloc[-1]:
        st.success("🚀 **AI Recommendation: BUY**")
    else:
        st.warning("📉 **AI Recommendation: SELL**")

# ----------------- Market News & AI Insights -----------------
elif option == "Market News & AI Insights":
    st.subheader("📰 Market News & AI Insights")

    # Scrape Financial News
    url = "https://www.moneycontrol.com/news/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = [h.text.strip() for h in soup.find_all('h2')][:5]

    for headline in headlines:
        score = sia.polarity_scores(headline)['compound']
        sentiment = "📈 Positive" if score > 0.05 else "📉 Negative" if score < -0.05 else "⚖️ Neutral"
        st.write(f"**{headline}** - {sentiment}")

# ----------------- Hyperpersonalized Investment Prediction -----------------
elif option == "Hyperpersonalized Investment Prediction":
    st.subheader("🔍 Personalized Investment Strategy")

    # User Persona Inputs
    st.markdown("### 🏦 Provide Your Financial Information")
    age = st.slider("Your Age", 18, 65, 30)
    income = st.number_input("Enter Your Annual Income (₹)", min_value=50000, step=10000)
    risk_profile = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"])
    investment_goal = st.selectbox("Investment Goal", ["Wealth Growth", "Retirement", "Education", "Emergency Fund"])

    # Personalized Investment Strategy
    st.markdown("### 📈 Recommended Portfolio")
    
    if risk_profile == "Low":
        st.write("✅ **60% Debt Instruments (FDs, Bonds, PPFs)**")
        st.write("✅ **30% Mutual Funds (Low-risk Index Funds)**")
        st.write("✅ **10% Gold**")
    elif risk_profile == "Moderate":
        st.write("✅ **40% Mutual Funds (Balanced Funds, Blue-chip Stocks)**")
        st.write("✅ **30% Debt Instruments**")
        st.write("✅ **20% Direct Stocks (Large-cap, ETFs)**")
        st.write("✅ **10% Gold**")
    else:
        st.write("✅ **60% Direct Stocks (Mid & Small-cap, Growth Stocks)**")
        st.write("✅ **30% Mutual Funds (Equity-heavy, High-growth Funds)**")
        st.write("✅ **10% Crypto/Gold**")

    st.success("📊 Your investment strategy is optimized based on your financial profile.")

