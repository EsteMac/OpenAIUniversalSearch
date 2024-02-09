import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv
import json  
import yfinance as yf  # Import the yfinance library to get stock prices
import matplotlib.pyplot as plt  # Import the matplotlib.pyplot library to plot the stock price data
import matplotlib.dates as mdates  # Import for date formatting in the stock price chart
from datetime import datetime, timedelta  # Import for date formatting in the weather results

load_dotenv()  # take environment variables from .env.

# Azure OpenAI API Endpoints
API_ENDPOINTS = {
    "GPT-3.5": os.getenv("AZURE_OPENAI_API_ENDPOINT_GPT35"),
    "GPT-3.5-turbo": os.getenv("AZURE_OPENAI_API_ENDPOINT_GPT35_16"),
    "GPT-4": os.getenv("AZURE_OPENAI_API_ENDPOINT_GPT4"),
    "GPT-4-32k": os.getenv("AZURE_OPENAI_API_ENDPOINT_GPT4_32")
}

# Azure OpenAI API Key
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Bing API
BING_SUBSCRIPTION_KEY = os.getenv("BING_SUBSCRIPTION_KEY")
BING_SEARCH_URL = os.getenv("BING_SEARCH_URL")

# ScholarAI API information
# See https://gptstore.ai/plugins/scholar-ai-net for more information
SCHOLAR_AI_URL = os.getenv("SCHOLAR_AI_URL")

# OpenWeatherMap API
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
OPENWEATHERMAP_URL = os.getenv("OPENWEATHERMAP_URL")

# News API
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = os.getenv("NEWS_API_URL")

st.set_page_config(page_title="IntelliPrompt Search with OpenAI")
st.title("IntelliPrompt Search with OpenAI")

# Adding a subtitle using subheader
st.subheader("Search web, news, research papers, stock prices, and weather with automated source routing, based on your natural language query.")

# Sidebar for model selection and configuration
# Set default values for model_selection, temperature, and top_p
default_model = "GPT-4-32k"
default_temperature = 1.0
default_top_p = 0.9

# If the session state does not have set values, set them to defaults
if 'model_selection' not in st.session_state:
    st.session_state['model_selection'] = default_model

if 'temperature' not in st.session_state:
    st.session_state['temperature'] = default_temperature

if 'top_p' not in st.session_state:
    st.session_state['top_p'] = default_top_p

# Sidebar for model selection and configuration
model_selection = st.sidebar.selectbox('Select the AI model:', 
                                       list(API_ENDPOINTS.keys()), 
                                       index=list(API_ENDPOINTS.keys()).index(st.session_state['model_selection']))
# Temperature setting with a tooltip
temperature = st.sidebar.slider(
    'Select the temperature:', 
    min_value=0.0, max_value=2.0, value=st.session_state['temperature'], step=0.1,
    help="Controls randomness. Lower = more predictable, higher = more surprising."
)

# top_p setting with a tooltip
top_p = st.sidebar.slider(
    'Select top_p:', 
    min_value=0.0, max_value=1.0, value=st.session_state['top_p'], step=0.01,
    help="Controls diversity. Lower = more focused, higher = more varied."
)

clear_button = st.sidebar.button("New Chat", key="clear")
with open("sidebar.md", "r") as sidebar_file:
    sidebar_content = sidebar_file.read()
st.sidebar.markdown(sidebar_content)
if clear_button:
    st.session_state["messages"] = []
    st.session_state['model_selection'] = default_model
    st.session_state['temperature'] = default_temperature
    st.session_state['top_p'] = default_top_p

# Initializing the 'messages' list in the session state if it doesn't exist.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Displaying the chat history on the webpage by iterating over each message in the 'messages' list.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])  # Each message is displayed according to its 'role'.

# The azure_openai_complete function takes temperature and top_p as parameters
def azure_openai_complete(messages, model_name, temperature, top_p):
    endpoint = API_ENDPOINTS[model_name]
    headers = {
        'Content-Type': 'application/json',
        'api-key': AZURE_OPENAI_API_KEY
    }
    data = {
        'messages': messages,
        'temperature': temperature,
        'top_p': top_p
    }
    
    response = requests.post(endpoint, headers=headers, json=data)
    
    if response.status_code != 200:
        st.write(f"Error {response.status_code}: {response.text}")
        return "Sorry, I couldn't fetch a response at the moment."

    response_data = response.json()

    if 'choices' not in response_data:
        st.write(f"Unexpected API response: {response_data}")
        return "Sorry, I couldn't process the information."

    return response_data['choices'][0]['message']['content']

# Fetches Bing web search results for the query
def fetch_bing_results(query):
    headers = {
        "Ocp-Apim-Subscription-Key": BING_SUBSCRIPTION_KEY
    }
    params = {
        "q": query
    }
    response = requests.get(BING_SEARCH_URL, headers=headers, params=params)
    results = response.json()

    # Format the results, e.g., take top 5 results and their descriptions.
    formatted_results = "\n\n".join([
        f"Title: {item['name']}\nURL: {item['url']}\nDescription: {item['snippet']}"
        for item in results.get("webPages", {}).get("value", [])[:5]
    ])
    
    return formatted_results if formatted_results else "No results found for your query."

# Fetches ScholarAI results for the query
def search_scholar_ai(query):
    params = {
        'keywords': query,
        'query': query,
        'num_results_to_show': 5  # Number of results to show can be adjusted
    }
    response = requests.get(SCHOLAR_AI_URL, params=params)
    return response.json()

# Extracts the weather location from the prompt using regex
def extract_location(prompt):
    # Define patterns for extracting location
    patterns = [
        r"weather in ([\w\s]+)",  # Matches "weather in <location>"
        r"weather at ([\w\s]+)",  # Matches "weather at <location>"
        r"temperature in ([\w\s]+)",  # Matches "temperature in <location>"
        r"temperature at ([\w\s]+)",  # Matches "temperature at <location>"
        r"humidity in ([\w\s]+)",  # Matches "humidity in <location>"
        r"humidity at ([\w\s]+)",  # Matches "humidity at <location>"
        r"wind in ([\w\s]+)",  # Matches "wind in <location>"
        r"wind at ([\w\s]+)",  # Matches "wind at <location>"
        r"sunset in ([\w\s]+)",  # Matches "sunset in <location>"
        r"sunrise at ([\w\s]+)",  # Matches "sunrise at <location>"
        r"first light in ([\w\s]+)",  # Matches "first light in <location>"
        r"last light at ([\w\s]+)",  # Matches "last light at <location>"
        # Add more patterns as needed
    ]

    # Check each pattern to see if there's a match in the prompt
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            # Return the first captured group, which is the location
            return match.group(1).strip()

    # If no patterns match, return None
    return None

# Function to convert weather Unix timestamp to local time
def convert_unix_to_local_time(unix_timestamp, timezone_offset):
    local_time = datetime.utcfromtimestamp(unix_timestamp) + timedelta(seconds=timezone_offset)
    return local_time.strftime('%I:%M %p')  # Format: HH:MM am/pm

# Function to fetch weather data and convert Unix timestamp to local time
def fetch_weather_data(location):
    if not location:  # If location is not provided, return None
        return None
    params = {
        'q': location,
        'appid': OPENWEATHERMAP_API_KEY,
        'units': 'imperial'  # or 'imperial' for Fahrenheit
    }
    response = requests.get(OPENWEATHERMAP_URL, params=params)
    if response.status_code != 200:
        return None
    weather_data = response.json()

    # Convert Unix timestamp to local time
    weather_data['sys']['sunrise'] = convert_unix_to_local_time(weather_data['sys']['sunrise'], weather_data['timezone'])
    weather_data['sys']['sunset'] = convert_unix_to_local_time(weather_data['sys']['sunset'], weather_data['timezone'])

    return json.dumps(weather_data, indent=2)

# Extract the stock ticker from the prompt using regex
def extract_stock_ticker(prompt):
    match = re.search(r"stock price for ([\w\.-]+)", prompt, re.IGNORECASE)
    return match.group(1).strip() if match else None

# Function to fetch stock price data
def fetch_stock_price(ticker):
    stock = yf.Ticker(ticker)
    try:
        hist = stock.history(period="1d")
        if not hist.empty and 'Close' in hist:
            current_price = hist['Close'].iloc[-1]
        else:
            current_price = None
        print(f"Last closing price for {ticker} is {current_price}")
        return current_price
    except Exception as e:
        print(f"Exception occurred: {e}")
        return f"Error fetching stock price: {e}"
    
# Function to plot stock price chart
def plot_stock_chart(ticker):
    # Display the Yahoo Finance message first
    yahoo_finance_link = f"[Yahoo Finance](https://finance.yahoo.com/quote/{ticker})"
    st.markdown(f"<div style='padding-left: 57px; padding-bottom: 10px;'>Here is the recent stock price history for {ticker} sourced from <a href='{yahoo_finance_link}' target='_blank'>Yahoo Finance</a>.</div>", unsafe_allow_html=True)
    
    # Use a modern and clean style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Fetching stock data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")  # Adjust the period as needed

    # Creating the plot
    plt.figure(figsize=(10, 5))
    plt.plot(hist.index, hist['Close'], color='#F504FD', linewidth=2, label='Closing Price')

    # Formatting the date on the x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.gcf().autofmt_xdate()

    # Setting titles and labels
    plt.title(f"{ticker} Stock Price - Last 1 Month", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price (USD)", fontsize=12)

    # Adding legend and grid
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Displaying the plot in Streamlit
    st.pyplot(plt)

# Function to fetch news using the News API
def fetch_news(query):
    # Calculate the date for 7 days ago
    seven_days_ago = datetime.now() - timedelta(days=7)
    formatted_date = seven_days_ago.strftime("%Y-%m-%d")
    print(f"Formatted date: {formatted_date}")  # Debugging

    # API parameters
    params = {
        "q": query,  
        "from": formatted_date, # Use the calculated date
        "sortBy": "popularity",
        "apiKey": NEWS_API_KEY
    }

    print(f"Request params: {params}")  # Debugging

    # API request
    response = requests.get(NEWS_API_URL, params=params)
    print(f"Request URL: {response.url}")  # Debugging

    print(f"Response status code: {response.status_code}")  # Debugging
    print(f"Response text: {response.text}")  # Debugging

    if response.status_code != 200:
        return f"Error fetching news: {response.text}"
    
    # Processing the response
    news_data = response.json()
    print(f"News data: {news_data}")  # Debugging

    if news_data.get("status") != "ok":
        return "Error in news data response"

    articles = news_data.get("articles", [])
    formatted_articles = "\n\n".join([
        f"Title: {article['title']}\nURL: {article['url']}\nDescription: {article['description']}"
        for article in articles
    ])

    return formatted_articles if formatted_articles else "No news articles found."

# Function to determine the type of query
def determine_query_type(prompt):
    # Keywords to determine query type
    scholar_keywords = ['article', 'articles', 'journal', 'journals' 'study', 'studies' 'conference paper', 'research']
    weather_keywords = ['weather', 'current temperature', 'temperature outside', 'current humidity', 'humidity outside', 'current wind direction', 'sunrise', 'sunset', 'first light', 'last light']
    stock_price_keywords = ['stock price', 'current stock']
    news_keywords = ['news', 'latest news', 'headlines', 'current news']

    # Logic to determine query type
    if any(word in prompt.lower() for word in scholar_keywords):
        return 'scholar'
    elif any(word in prompt.lower() for word in weather_keywords):
        return 'weather'
    elif any(word in prompt.lower() for word in stock_price_keywords):
        return 'stock_price'
    elif any(word in prompt.lower() for word in news_keywords):
        return 'news'
    else:
        return 'web'

# Chat input and processing
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Determine the type of query
    query_type = determine_query_type(prompt)
    message_content = None  # Initialize the variable to hold the message content
    response = None  # Initialize response to None

    with st.spinner("Thinking..."):
        if query_type == 'web':
            # Process for web search
            bing_results = fetch_bing_results(prompt)
            # Formulate the Bing web search results to send to the AI model
            message_content = f"Below are SEARCH RESULTS which include the latest news in real-time, stock prices in real-time, and stock trading activity in real-time, in response to a user question. The SEARCH RESULTS also represent the top most relevant real-time updates of global information in response to the user question. These SEARCH RESULTS help you help the user browse the internet in real-time. You now have real-time internet browsing capabilities. When responding to the question please think step-by-step. Answer the question being as informative as possible. Use information from the SEARCH RESULTS to answer the user question. Make sure to add web links from the URLs in the below SEARCH RESULTS to the response. SEARCH RESULTS:\n{bing_results}"
            # Debugging print statement after defining bing_message_content
            print("Before sending Bing search results to AI: ", message_content)
        elif query_type == 'scholar':
            # Process for scholar search
            scholar_ai_results = search_scholar_ai(prompt)
            # Formulate the Scholar AI content to send to the AI model
            message_content = f"Below are the SCHOLAR AI RESULTS which give you open access to scientific literature from peer-reviewed journals. The abstract endpoint finds relevant papers based on 2 to 6 keywords. After getting abstracts, ALWAYS prompt the user offering to go into more detail. Use the fulltext endpoint to retrieve the entire paper's text and access specific details using the provided pdf_url, if available. ALWAYS hyperlink the pdf_url from the responses if available. Offer to retrieve chunks of fulltext incrementally only upon request. SCHOLAR AI RESULTS:\n{scholar_ai_results}"
            # Debugging print statement after defining scholar_ai_message_content
            print("Before sending Scholar AI results to AI: ", message_content)
        elif query_type == 'weather': 
            location = extract_location(prompt)
            if location:
                weather_results = fetch_weather_data(location)
                if weather_results:
                    # Send the entire JSON response to the AI
                    message_content = f"Below is the OPENWEATHERMAP WEATHER RESULTS for {location}. OPENWEATHERMAP WEATHER RESULTS give you access to live weather data, including current weather information. When responding to the user prompt please think step-by-step. Answer the prompt being as informative as possible. Use information from the OPENWEATHERMAP WEATHER RESULTS to respond to the user prompt. Here are the OPENWEATHERMAP WEATHER RESULTS:\n{weather_results}"
                    # Debugging print statement after defining weather_message_content
                    print("Before sending weather results to AI: ", message_content)
                else:
                    response = "Sorry, I couldn't fetch the weather data at the moment."
            else:
                response = "Please provide a location for the weather information."
        elif query_type == 'stock_price':
            ticker = extract_stock_ticker(prompt)
            if ticker:
                stock_price = fetch_stock_price(ticker)
                if stock_price:
                    message_content = f"Below is the STOCK PRICE RESULTS for {ticker}. STOCK PRICE RESULTS give you access to live stock market data, including current stock market prices. When responding the user prompt please thing step-by-step. Answer the prompt being as informative as possible. Use information from the STOCK PRICE RESULTS to respond to the user prompt. Here are the STOCK PRICE RESULTS:\n{stock_price}"
                else:
                    response = "Sorry, I couldn't fetch the stock price at the moment."
            else:
                response = "Please provide the stock ticker symbol for the stock price information."
        elif query_type == 'news':
            # Process for news search
            news_results = fetch_news(prompt)
            message_content = f"Below are the NEWS RESULTS which give you access to the latest news articles and headlines in response to a user prompt. These NEWS RESULTS help you help the user locate articles and breaking news headlines from news sources and blogs across the web. You now have real-time internet browsing capabilities. When responding to the question please think step-by-step. Answer the question being as informative as possible. Use information from the NEWS RESULTS to answer the user question. Make sure to add web links from the URLs in the below NEWS RESULTS to the response. NEWS RESULTS:\n{news_results}"
            # Debugging print statement after defining news_message_content
            print("Before sending news results to AI: ", message_content)

        # Check if a message content was generated; if so, create azure_messages payload
        if message_content:
            azure_messages = [
                {"role": "user", "content": prompt},
                {"role": "system", "content": message_content}
            ]
            # Fetch the response from the AI model
            response = azure_openai_complete(azure_messages, model_selection, temperature, top_p)
            print("AI Response: ", response)

        # If response wasn't set, it means there was an error or a prompt for more info
        if not response:
            # If message content is empty, it means we need more info for weather
            response = message_content if message_content else "Could you please provide more information?"

        # Modify the AI response by adding the source
        if query_type == 'web':
            response = "**From Bing search with OpenAI:**\n\n" + response
        elif query_type == 'scholar':
            response = "**From Scholar AI search with OpenAI:**\n\n" + response
        elif query_type == 'weather':
            response = "**From OpenWeatherMap with OpenAI:**\n\n" + response
        elif query_type == 'stock_price':
            response = "**From Yahoo! Finance with OpenAI:**\n\n" + response
        elif query_type == 'news':
            response = "**From Newsapi.org with OpenAI:**\n\n" + response

        # Add the modified AI response to the chat
        with st.chat_message("assistant"):
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

        # Plot the stock chart after the AI response for stock price queries
        if query_type == 'stock_price' and ticker:
            plot_stock_chart(ticker)