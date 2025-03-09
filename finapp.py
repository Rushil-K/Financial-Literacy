import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Download Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load Open-Source AI Model for Financial News Analysis
news_analyzer = pipeline("sentiment-analysis")

# API Key for Open-Source Financial News API (Replace with free API key)
FINANCIAL_NEWS_API_KEY = "your_newsapi_key"

# ----------------- UI Layout -----------------
st.title("📊 AI-Powered Open-Source Financial Advisor")
st.sidebar.header("Select a Module")

option = st.sidebar.selectbox(
    "Choose Module", 
    ["Financial Literacy", "AI-Powered Global Stock Market Analysis", "AI-Driven Market News", "Hyperpersonalized AI Investment Prediction"]
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
        st.progress(min(1, savings / max(1, income)))

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

# ----------------- AI-Powered Global Stock Market Analysis -----------------
elif option == "AI-Powered Global Stock Market Analysis":
    st.subheader("📊 AI-Powered Global Stock Market Analysis")

    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS, BTC-USD, EURUSD=X):", "AAPL")
    
    if ticker:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty:
            st.error("❌ Invalid stock ticker. Please enter a valid symbol.")
        else:
            st.markdown(f"### 📈 {ticker} Stock Price Movement (Last 1 Year)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name=ticker))
            st.plotly_chart(fig)

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

# ----------------- AI-Driven Market News -----------------
elif option == "AI-Driven Market News":
    st.subheader("📰 AI-Driven Market News")

    news_url = f"https://newsapi.org/v2/top-headlines?category=business&apiKey={FINANCIAL_NEWS_API_KEY}"
    response = requests.get(news_url)

    if response.status_code == 200:
        articles = response.json()["articles"]
        for article in articles[:5]:  
            sentiment = news_analyzer(article['title'])[0]['label']
            st.write(f"### 📰 {article['title']} ({sentiment})")
            st.write(f"📅 {article['publishedAt']}")
            st.write(f"🔗 [Read More]({article['url']})")
            st.write("---")
    else:
        st.error("❌ Unable to fetch news. Please check your API key.")

# ----------------- Hyperpersonalized AI Investment Prediction -----------------
elif option == "Hyperpersonalized AI Investment Prediction":
    st.subheader("🔍 AI-Powered Personalized Investment Strategy")

    age = st.slider("Your Age", 18, 65, 30)
    income = st.number_input("Enter Your Annual Income (₹)", min_value=50000, step=10000)
    risk_profile = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"])
    investment_goal = st.selectbox("Investment Goal", ["Wealth Growth", "Retirement", "Education", "Emergency Fund"])

    st.markdown("### 📈 AI-Optimized Recommended Portfolio")
    
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

    st.success("📊 AI has optimized your investment strategy based on your financial profile.")
