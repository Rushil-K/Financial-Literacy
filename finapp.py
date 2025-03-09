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
import openai
from streamlit_chat import message

# Download Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# ----------------- UI Layout -----------------
st.title("📊 Open-Source AI Financial Advisor")
st.sidebar.header("Select a Section")

option = st.sidebar.selectbox(
    "Choose Module", 
    ["Financial Literacy", "Investment Analysis", "Market News & AI Insights", "AI Budgeting & Chatbot"]
)

# ----------------- AI-Driven Budgeting & Expense Tracking -----------------
if option == "AI Budgeting & Chatbot":
    st.subheader("💰 AI-Driven Budgeting & Financial Assistant")

    # User Inputs
    income = st.number_input("Enter Monthly Income (₹)", min_value=0.0)
    rent = st.number_input("Rent Expenses (₹)", min_value=0.0)
    food = st.number_input("Food & Groceries (₹)", min_value=0.0)
    transport = st.number_input("Transport (₹)", min_value=0.0)
    other_expenses = st.number_input("Other Expenses (₹)", min_value=0.0)

    total_expense = rent + food + transport + other_expenses
    savings = income - total_expense

    st.write(f"**Total Monthly Expenses:** ₹{total_expense}")
    st.write(f"💰 **Estimated Savings:** ₹{savings}")

    if savings < 0:
        st.error("⚠️ Your expenses exceed your income! Consider reducing unnecessary spending.")
    elif savings < (0.2 * income):
        st.warning("🔸 Your savings are below the recommended 20% of income.")
    else:
        st.success("✅ Your savings are on track!")

    # AI-Driven Savings Recommendation
    goal = st.selectbox("Choose Your Savings Goal", ["Emergency Fund", "Retirement", "Buying a House", "Wealth Growth"])
    if goal == "Emergency Fund":
        st.write("💡 **Tip:** Aim to save at least 6 months' worth of expenses in a liquid fund.")
    elif goal == "Retirement":
        st.write("💡 **Tip:** Start investing in **mutual funds, ETFs, or PPF** for long-term wealth creation.")
    elif goal == "Buying a House":
        st.write("💡 **Tip:** Maintain a **50-30-20 rule**, where 20% goes to your home purchase savings.")
    else:
        st.write("💡 **Tip:** Diversify across **stocks, gold, and high-yield mutual funds** for optimal returns.")

    # ----------------- NLP-Based Financial Chatbot -----------------
    st.subheader("💬 AI Financial Chatbot")
    openai.api_key = "your_openai_api_key"  # Replace with actual OpenAI API key

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Ask me anything about personal finance:")
    
    if user_input:
        st.session_state.messages.append({"user": user_input})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": user_input}]
        )
        bot_response = response["choices"][0]["message"]["content"]
        st.session_state.messages.append({"bot": bot_response})

    for msg in st.session_state.messages:
        if "user" in msg:
            message(msg["user"], is_user=True)
        else:
            message(msg["bot"])

# ----------------- AI-Driven Fraud Alerts & Risk Assessment -----------------
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

    # Fraud Alerts & Risk Assessment
    st.subheader("🚨 AI-Driven Fraud Alerts")
    fraud_data = {
        "Ponzi Schemes": "Avoid investments that guarantee 'high returns with no risk'.",
        "Fake Loan Offers": "Banks never ask for prepayment to process a loan.",
        "Crypto Scams": "If it sounds too good to be true, it's likely a scam."
    }
    for fraud, tip in fraud_data.items():
        st.warning(f"⚠️ **{fraud}**: {tip}")

# ----------------- Personalized Investment Recommendations -----------------
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

    hist['MA50'] = hist['Close'].rolling(50).mean()
    hist['RSI'] = 100 - (100 / (1 + (hist['Close'].diff().rolling(14).mean() / hist['Close'].diff().rolling(14).std())))

    st.write(f"**Current Price:** ₹{round(hist['Close'].iloc[-1], 2)}")
    st.write(f"**50-Day Moving Average:** ₹{round(hist['MA50'].iloc[-1], 2)}")

    if hist['RSI'].iloc[-1] < 30:
        st.write("✅ **Stock is Oversold. Good Buying Opportunity!**")
    elif hist['RSI'].iloc[-1] > 70:
        st.write("⚠️ **Stock is Overbought. Consider Selling!**")
    else:
        st.write("📈 **Stock is in a Neutral Zone. Hold or Wait.**")

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


