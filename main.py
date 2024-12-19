import json
import os
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from typing import Dict, Any

# Configure OpenAI client
try:
    client = OpenAI(
        api_key=open('API_KEY', 'r').read().strip()
    )
except FileNotFoundError:
    st.error("API_KEY file not found.")
    st.stop()

def get_stock_price(ticker: str) -> str:
    """Get the most recent stock price."""
    try:
        price = yf.Ticker(ticker).history(period='1d').iloc[-1].Close
        return f"${price:.2f}"
    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"

def calculate_SMA(ticker: str, window: int) -> str:
    """Calculate Simple Moving Average."""
    try:
        data = yf.Ticker(ticker).history(period='1y').Close
        sma = data.rolling(window=window).mean().iloc[-1]
        return f"${sma:.2f}"
    except Exception as e:
        return f"Error calculating SMA for {ticker}: {str(e)}"

def calculate_EMA(ticker: str, window: int) -> str:
    """Calculate Exponential Moving Average."""
    try:
        data = yf.Ticker(ticker).history(period='1y').Close
        ema = data.ewm(span=window, adjust=False).mean().iloc[-1]
        return f"${ema:.2f}"
    except Exception as e:
        return f"Error calculating EMA for {ticker}: {str(e)}"

def calculate_RSI(ticker: str) -> str:
    """Calculate Relative Strength Index."""
    try:
        data = yf.Ticker(ticker).history(period='1y').Close
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        return f"{rsi:.2f}"
    except Exception as e:
        return f"Error calculating RSI for {ticker}: {str(e)}"

def calculate_MACD(ticker: str) -> str:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    try:
        data = yf.Ticker(ticker).history(period='1y').Close
        short_EMA = data.ewm(span=12, adjust=False).mean()
        long_EMA = data.ewm(span=26, adjust=False).mean()
        
        MACD = short_EMA - long_EMA
        signal = MACD.ewm(span=9, adjust=False).mean()
        MACD_histogram = MACD - signal
        
        return f"MACD: {MACD.iloc[-1]:.2f}, Signal: {signal.iloc[-1]:.2f}, Histogram: {MACD_histogram.iloc[-1]:.2f}"
    except Exception as e:
        return f"Error calculating MACD for {ticker}: {str(e)}"

def plot_stock_price(ticker: str) -> None:
    """Plot stock price history."""
    try:
        data = yf.Ticker(ticker).history(period='1y')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data.Close)
        ax.set_title(f'{ticker} Stock Price Over Last Year')
        ax.set_xlabel('Date')
        ax.set_ylabel('Stock Price ($USD)')
        ax.grid(True)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error plotting data for {ticker}: {str(e)}")

# Define available functions as tools for the API
tools = [
    {
        'type': 'function',
        'function': {
            'name': 'get_stock_price',
            'description': 'Gets the recent stock price of the given company.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The ticker symbol of the company stock (example: AAPL for Apple)'
                    }
                },
                'required': ['ticker']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate_SMA',
            'description': 'Calculates the Simple Moving Average.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The ticker symbol of the company stock'
                    },
                    'window': {
                        'type': 'integer',
                        'description': 'Timeframe to consider when calculating the SMA'
                    }
                },
                'required': ['ticker', 'window']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate_EMA',
            'description': 'Calculates the Exponential Moving Average.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The ticker symbol of the company stock'
                    },
                    'window': {
                        'type': 'integer',
                        'description': 'Timeframe to consider when calculating the EMA'
                    }
                },
                'required': ['ticker', 'window']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate_RSI',
            'description': 'Calculate the Relative Strength Index.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The ticker symbol of the company stock'
                    }
                },
                'required': ['ticker']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate_MACD',
            'description': 'Calculate the MACD indicator.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The ticker symbol of the company stock'
                    }
                },
                'required': ['ticker']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'plot_stock_price',
            'description': 'Plot the stock price history for the last year.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'ticker': {
                        'type': 'string',
                        'description': 'The ticker symbol of the company stock'
                    }
                },
                'required': ['ticker']
            }
        }
    }
]

# Map function names to their implementations
available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_EMA': calculate_EMA,
    'calculate_RSI': calculate_RSI,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price
}

# Streamlit UI
st.title('Stock Analysis Chatbot Assistant')

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.write(message['content'])

# User input
user_input = st.chat_input('Ask me about stocks...')

if user_input:
    # Add user message to chat
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.write(user_input)

    try:
        # Call OpenAI API using the new client
        response = client.chat.completions.create(
            model="gpt-4",
            messages=st.session_state['messages'],
            tools=tools,
            tool_choice='auto'
        )

        response_message = response.choices[0].message

        # Handle tool/function calls
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.type == 'function':
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    with st.chat_message('assistant'):
                        if function_name in available_functions:
                            result = available_functions[function_name](**function_args)
                            st.write(f"Analysis for {function_args['ticker']}:")
                            if function_name == 'plot_stock_price':
                                plot_stock_price(**function_args)
                            else:
                                st.write(result)
                        else:
                            st.error(f"Function {function_name} not found")
        else:
            # Display direct response
            with st.chat_message('assistant'):
                st.write(response_message.content)

        # Add assistant's response to chat history
        st.session_state['messages'].append({
            'role': 'assistant',
            'content': response_message.content or 'Function executed successfully'
        })

    except Exception as e:
        st.error(f"Error: {str(e)}")