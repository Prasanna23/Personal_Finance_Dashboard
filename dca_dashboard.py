# Must be the first import and command
import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="InvestWise - Investment Strategy Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Other imports
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Title and description
st.title("🎯 InvestWise")
st.header("Smart DCA vs. Lump Sum Analysis")
st.markdown("""
    Compare Dollar-Cost Averaging (DCA) with Lump Sum investment strategies across multiple stocks.
    Track your investment growth, analyze portfolio allocation, and make data-driven decisions.
    
    💡 *Add stocks using the sidebar to get started!*
""")

def calculate_lump_sum_investment(symbol, start_date, total_amount):
    try:
        # Convert start_date to timezone-aware datetime
        if isinstance(start_date, date):
            start_date = pd.Timestamp.combine(start_date, pd.Timestamp.min.time())
            start_date = pd.Timestamp(start_date).tz_localize('America/New_York')
        else:
            start_date = pd.Timestamp(start_date)
            if start_date.tz is None:
                start_date = start_date.tz_localize('America/New_York')
            else:
                start_date = start_date.tz_convert('America/New_York')
        
        # Get end date (midnight)
        end_date = pd.Timestamp.now(tz='America/New_York').normalize()
        
        # Get historical data
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date, interval='1d')
        
        if hist.empty:
            st.error(f'No historical data found for {symbol}. Please check the symbol and try again.')
            return None
        
        # Get initial price and calculate shares
        initial_price = hist.iloc[0]['Close']
        shares = total_amount / initial_price
        
        # Calculate value over time
        df = pd.DataFrame({
            'Date': hist.index,
            'Price': hist['Close'],
            'Shares': [shares] * len(hist),
            'Investment': [total_amount] * len(hist)
        })
        
        current_price = hist.iloc[-1]['Close']
        current_value = shares * current_price
        gain_loss = current_value - total_amount
        gain_loss_percentage = (gain_loss / total_amount * 100) if total_amount > 0 else 0
        
        return {
            'total_shares': shares,
            'total_invested': total_amount,
            'current_value': current_value,
            'gain_loss': gain_loss,
            'gain_loss_percentage': gain_loss_percentage,
            'current_price': current_price,
            'history': df
        }
    except Exception as e:
        st.error(f'Error calculating lump sum investment for {symbol}: {str(e)}')
        return None

def calculate_dca_investment(symbol, start_date, frequency_days, amount_per_investment):
    # Convert start_date to timezone-aware datetime
    if isinstance(start_date, date):
        start_date = pd.Timestamp.combine(start_date, pd.Timestamp.min.time())
        start_date = pd.Timestamp(start_date).tz_localize('America/New_York')
    else:
        start_date = pd.Timestamp(start_date)
        if start_date.tz is None:
            start_date = start_date.tz_localize('America/New_York')
        else:
            start_date = start_date.tz_convert('America/New_York')
    
    # Get end date (midnight)
    end_date = pd.Timestamp.now(tz='America/New_York').normalize()
    
    # Get historical data
    stock = yf.Ticker(symbol)
    hist = stock.history(start=start_date, end=end_date, interval='1d')
    
    if hist.empty:
        raise ValueError(f'No historical data found for {symbol}')
    
    # Create a mapping of normalized dates to actual trading data
    trading_data = {}
    for idx, row in hist.iterrows():
        norm_date = idx.normalize()
        if norm_date not in trading_data:
            trading_data[norm_date] = {'date': idx, 'price': row['Close']}
    
    # Generate investment dates
    investment_dates = pd.date_range(
        start=start_date.normalize(),
        end=end_date,
        freq=f'{frequency_days}D',
        tz='America/New_York'
    )
    
    # Initialize tracking
    actual_investments = []
    processed_dates = set()  # Keep track of dates we've already processed
    
    # For each investment date, find the next available trading day
    for inv_date in investment_dates:
        inv_date = inv_date.normalize()
        
        # Find the next trading day
        next_trade_date = None
        current_date = inv_date
        while current_date <= end_date:
            if current_date in trading_data:
                next_trade_date = current_date
                break
            current_date += pd.Timedelta(days=1)
        
        # If we found a valid trading day and haven't processed it yet
        if next_trade_date is not None and next_trade_date not in processed_dates:
            trade_info = trading_data[next_trade_date]
            actual_investments.append({
                'Date': trade_info['date'],
                'Price': trade_info['price'],
                'Investment': amount_per_investment
            })
            processed_dates.add(next_trade_date)
    
    # Convert to DataFrame and calculate shares
    if not actual_investments:
        raise ValueError(f'No valid trading days found for {symbol} in the given date range')
        
    df = pd.DataFrame(actual_investments)
    df['Shares'] = df['Investment'] / df['Price']
    
    # Calculate totals
    total_shares = df['Shares'].sum()
    total_invested = df['Investment'].sum()
    current_price = hist.iloc[-1]['Close']
    current_value = total_shares * current_price
    
    total_gain_loss = current_value - total_invested
    gain_loss_percentage = (total_gain_loss / total_invested) * 100 if total_invested > 0 else 0
    
    return {
        'total_shares': total_shares,
        'total_invested': total_invested,
        'current_value': current_value,
        'gain_loss': total_gain_loss,
        'gain_loss_percentage': gain_loss_percentage,
        'current_price': current_price,
        'history': df
    }

def main():
    # Initialize session state for stocks if not exists
    if 'stocks' not in st.session_state:
        st.session_state.stocks = []
    
    # Sidebar for adding stocks
    with st.sidebar:
        st.subheader("Add New Stock")
        with st.form("add_stock"):
            symbol = st.text_input('Stock Symbol (e.g., AAPL)')
            start_date = st.date_input('Start Date',
                                     value=datetime.now() - timedelta(days=365),
                                     max_value=datetime.now())
            frequency = st.selectbox('Investment Frequency',
                                   ['Daily', 'Weekly', 'Monthly'],
                                   index=1)
            amount = st.number_input('Amount per Investment ($)',
                                   min_value=1.0,
                                   value=100.0)
            add_stock = st.form_submit_button("Add Stock")
    
    # Initialize session state for storing stocks
    if 'stocks' not in st.session_state:
        st.session_state.stocks = []
    
    # Add new stock to the list
    if add_stock and symbol:
        new_stock = {
            'symbol': symbol.upper(),
            'start_date': start_date,
            'frequency': frequency,
            'amount': amount
        }
        if not any(s['symbol'] == new_stock['symbol'] for s in st.session_state.stocks):
            st.session_state.stocks.append(new_stock)
            st.sidebar.success(f"Added {symbol.upper()} to portfolio")
        else:
            st.sidebar.error(f"{symbol.upper()} is already in your portfolio")
    
    # Display and manage current stocks
    if st.session_state.stocks:
        st.subheader("Your Portfolio")
        
        # Convert frequency to days
        frequency_map = {'Daily': 1, 'Weekly': 7, 'Monthly': 30}
        
        # Calculate results for all stocks
        total_portfolio_value = 0
        total_portfolio_cost = 0
        
        for idx, stock in enumerate(st.session_state.stocks):
            try:
                results = calculate_dca_investment(
                    stock['symbol'],
                    stock['start_date'],
                    frequency_map[stock['frequency']],
                    stock['amount']
                )
                
                # Create expandable section for each stock
                with st.expander(f"{stock['symbol']} - Investing ${stock['amount']} {stock['frequency'].lower()}", expanded=True):
                        # Add lump sum comparison section
                    st.markdown("### Compare with Lump Sum")
                    lump_sum_col1, lump_sum_col2 = st.columns(2)
                    
                    with lump_sum_col1:
                        show_lump_sum = st.checkbox(
                            "Enable Lump Sum Comparison",
                            key=f"lump_sum_{idx}",
                            help="Compare DCA with a one-time lump sum investment"
                        )
                    
                    if show_lump_sum:
                        with lump_sum_col1:
                            lump_sum_amount = st.number_input(
                                "Lump Sum Amount ($)",
                                min_value=1.0,
                                value=float(stock['amount']) * 12,  # Default to 1 year of DCA
                                key=f"lump_amount_{idx}"
                            )
                        
                        with lump_sum_col2:
                            lump_sum_date = st.date_input(
                                "Investment Date",
                                value=stock['start_date'],
                                min_value=stock['start_date'],
                                max_value=datetime.now().date(),
                                key=f"lump_date_{idx}"
                            )
                    # Add edit controls in columns
                    edit_col1, edit_col2, edit_col3, edit_col4 = st.columns(4)
                    
                    with edit_col1:
                        new_amount = st.number_input(
                            'Investment Amount ($)',
                            min_value=1.0,
                            value=float(stock['amount']),
                            key=f"amount_{idx}"
                        )
                    
                    with edit_col2:
                        new_start_date = st.date_input(
                            'Start Date',
                            value=stock['start_date'],
                            max_value=datetime.now(),
                            key=f"date_{idx}"
                        )
                    
                    with edit_col3:
                        new_frequency = st.selectbox(
                            'Frequency',
                            ['Daily', 'Weekly', 'Monthly'],
                            index=['Daily', 'Weekly', 'Monthly'].index(stock['frequency']),
                            key=f"freq_{idx}"
                        )
                    
                    with edit_col4:
                        update_col1, update_col2 = st.columns(2)
                        with update_col1:
                            if st.button("Update", key=f"update_{idx}"):
                                stock['amount'] = new_amount
                                stock['start_date'] = new_start_date
                                stock['frequency'] = new_frequency
                                st.rerun()
                        with update_col2:
                            if st.button("Remove", key=f"remove_{idx}"):
                                st.session_state.stocks.remove(stock)
                                st.rerun()
                    
                    # Show metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric('Total Invested', f"${results['total_invested']:,.2f}")
                    with metric_col2:
                        st.metric('Current Value', f"${results['current_value']:,.2f}")
                    with metric_col3:
                        st.metric('Total Gain/Loss',
                                 f"${results['gain_loss']:,.2f}",
                                 f"{results['gain_loss_percentage']:,.2f}%")
                    with metric_col4:
                        st.metric('Shares Owned', f"{results['total_shares']:.4f}")
                    
                    # Show investment history in a table
                    st.dataframe(results['history'], use_container_width=True)
                
                total_portfolio_value += results['current_value']
                total_portfolio_cost += results['total_invested']
                
            except Exception as e:
                st.error(f"Error calculating for {stock['symbol']}: {str(e)}")
        
        # Display portfolio summary
        st.subheader("Portfolio Summary")
        total_gain_loss = total_portfolio_value - total_portfolio_cost
        gain_loss_percentage = (total_gain_loss / total_portfolio_cost * 100) if total_portfolio_cost > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Portfolio Cost", f"${total_portfolio_cost:,.2f}")
        with col2:
            st.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
        with col3:
            st.metric("Total Portfolio Gain/Loss",
                     f"${total_gain_loss:,.2f}",
                     f"{gain_loss_percentage:,.2f}%")
        
        # Create portfolio visualization
        st.subheader("Portfolio Visualization")
        
        # Function to get stock history with cache busting based on parameters
        def get_stock_history(symbol, start_date, amount, frequency):
            # Create a cache key that includes all parameters
            cache_key = f"{symbol}_{start_date}_{amount}_{frequency}"
            
            if 'stock_history_cache' not in st.session_state:
                st.session_state.stock_history_cache = {}
            
            # If parameters changed, invalidate cache
            if cache_key not in st.session_state.stock_history_cache:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, interval='1d')
                st.session_state.stock_history_cache[cache_key] = hist
            
            return st.session_state.stock_history_cache[cache_key]
        
        # Combine all stock histories for visualization
        all_stock_data = {}
        dca_results = {}
        portfolio_values = {}
        earliest_date = pd.Timestamp.now(tz='America/New_York')
        
        for stock in st.session_state.stocks:
            try:
                symbol = stock['symbol']
                hist = get_stock_history(
                    symbol,
                    stock['start_date'],
                    stock['amount'],
                    stock['frequency']
                )
                
                if not hist.empty:
                    all_stock_data[symbol] = hist
                    earliest_date = min(earliest_date, hist.index[0])
            except Exception as e:
                st.warning(f"Couldn't fetch historical data for {symbol}: {str(e)}")
        
        if all_stock_data:
            # Process each stock and calculate DCA results first
            for idx, stock in enumerate(st.session_state.stocks):
                symbol = stock['symbol']
                
                # Get stock data
                results = get_stock_history(
                    symbol,
                    stock['start_date'],
                    stock['amount'],
                    stock['frequency']
                )
                
                if results is None:
                    continue
                
                all_stock_data[symbol] = results
                
                # Calculate DCA results
                dca_result = calculate_dca_investment(
                    symbol,
                    stock['start_date'],
                    frequency_map[stock['frequency']],
                    stock['amount']
                )
                dca_results[symbol] = dca_result
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{'type': 'scatter'}, {'type': 'pie'}],
                       [{'type': 'scatter', 'colspan': 2}, None]],
                subplot_titles=(
                    "Normalized Stock Prices",
                    "Portfolio Allocation",
                    "Investment Values"
                ),
                vertical_spacing=0.15,
                horizontal_spacing=0.1,
                row_heights=[0.4, 0.6]
            )
            
            # Add traces for each stock
            for idx, stock in enumerate(st.session_state.stocks):
                symbol = stock['symbol']
                hist = all_stock_data.get(symbol)
                
                if hist is None:
                    continue
                
                # Calculate normalized price (percentage change from start)
                normalized_price = hist['Close'] / hist['Close'].iloc[0] * 100
                
                # Add normalized price line
                fig.add_trace(
                    go.Scatter(x=hist.index, y=normalized_price,
                              name=f"{symbol} (Normalized)",
                              mode='lines'),
                    row=1, col=1
                )
            
            # Plot 2: Investment value over time
            for symbol, hist in all_stock_data.items():
                stock_info = next(s for s in st.session_state.stocks if s['symbol'] == symbol)
                
                # Calculate DCA results if not already calculated
                if symbol not in dca_results:
                    dca_results[symbol] = calculate_dca_investment(
                        symbol,
                        stock_info['start_date'],
                        frequency_map[stock_info['frequency']],
                        stock_info['amount']
                    )
                    portfolio_values[symbol] = dca_results[symbol]['current_value']
                
                results = dca_results[symbol]
                df = results['history'].copy()
                
                # Set Date as index if it's not already
                if 'Date' in df.columns:
                    df.set_index('Date', inplace=True)
                
                df = df.sort_index()  # Sort by date
                
                # Calculate daily value for DCA
                df['Cumulative_Shares'] = df['Shares'].cumsum()
                df['Value'] = df['Cumulative_Shares'] * df['Price']
                
                # Get all dates from historical data for smooth line
                all_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                
                # Create a continuous series
                continuous_df = pd.DataFrame(index=all_dates)
                continuous_df['Value'] = None
                
                # Fill in known values
                for idx, row in df.iterrows():
                    continuous_df.loc[idx, 'Value'] = row['Value']
                
                # Forward fill for smooth line
                continuous_df['Value'] = continuous_df['Value'].ffill()
                
                # Add DCA line
                fig.add_trace(
                    go.Scatter(x=continuous_df.index, y=continuous_df['Value'],
                              name=f"{symbol} DCA",
                              mode='lines'),
                    row=2, col=1
                )
                
                # Store the results for comparison
                if 'comparison_results' not in st.session_state:
                    st.session_state.comparison_results = {}
                
                comparison_key = f"{symbol}_{idx}"
                
                # Calculate lump sum results if needed
                if show_lump_sum:
                    # Create a unique key that includes all parameters
                    cache_key = f"{symbol}_{idx}_{lump_sum_amount}_{lump_sum_date}"
                    
                    if (comparison_key not in st.session_state.comparison_results or
                        st.session_state.comparison_results[comparison_key].get('cache_key') != cache_key):
                        # Calculate lump sum investment with user-specified parameters
                        lump_sum_results = calculate_lump_sum_investment(
                            symbol,
                            lump_sum_date,
                            lump_sum_amount
                        )
                        
                        if lump_sum_results is not None:
                            lump_sum_results['cache_key'] = cache_key
                            st.session_state.comparison_results[comparison_key] = lump_sum_results
                        else:
                            st.error(f"Unable to calculate lump sum comparison for {symbol}. Please check the symbol and try again.")
                            show_lump_sum = False
                
                # Add lump sum comparison if requested
                if show_lump_sum and comparison_key in st.session_state.comparison_results:
                    lump_sum_results = st.session_state.comparison_results[comparison_key]
                    stock_dca_results = dca_results[symbol]
                    
                    # Add lump sum line
                    lump_sum_df = lump_sum_results['history']
                    lump_sum_df['Value'] = lump_sum_df['Shares'] * lump_sum_df['Price']
                    
                    fig.add_trace(
                        go.Scatter(x=lump_sum_df['Date'], y=lump_sum_df['Value'],
                                  name=f"{symbol} Lump Sum",
                                  mode='lines',
                                  line=dict(dash='dash')),
                        row=2, col=1
                    )
                    
                    # Add comparison metrics
                    st.markdown(f"### {symbol} - DCA vs Lump Sum Comparison")
                    st.markdown(f"""💰 **Investment Details:**
                        - DCA: ${stock['amount']} {stock['frequency'].lower()}
                        - Lump Sum: ${lump_sum_amount:,.2f} on {lump_sum_date.strftime('%Y-%m-%d')}
                    """)
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    with comp_col1:
                        st.metric(
                            f"{symbol} DCA Value",
                            f"${stock_dca_results['current_value']:,.2f}",
                            f"{stock_dca_results['gain_loss_percentage']:,.2f}%"
                        )
                    with comp_col2:
                        st.metric(
                            f"{symbol} Lump Sum Value",
                            f"${lump_sum_results['current_value']:,.2f}",
                            f"{lump_sum_results['gain_loss_percentage']:,.2f}%"
                        )
                    with comp_col3:
                        value_diff = stock_dca_results['current_value'] - lump_sum_results['current_value']
                        diff_pct = (value_diff / lump_sum_results['current_value'] * 100)
                        better_strategy = "DCA" if value_diff > 0 else "Lump Sum"
            portfolio_values = {}
            for symbol in all_stock_data:
                portfolio_values[symbol] = dca_results[symbol]['current_value']

            # Add pie chart
            fig.add_trace(
                go.Pie(
                    labels=list(portfolio_values.keys()),
                    values=list(portfolio_values.values()),
                    textinfo='label+percent',
                    hovertemplate='%{label}<br>Value: $%{value:,.2f}<br>%{percent}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=900,
                showlegend=True,
                title_text="Portfolio Performance",
                hovermode='x unified'
            )
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price (%)", row=1, col=1)
            fig.update_yaxes(title_text="Value ($)", row=2, col=1)
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Add stocks to your portfolio using the sidebar to get started!")
    
    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: grey; padding: 20px;'>
            <p>Built with ❤️ using Streamlit and yfinance</p>
            <p>📊 Compare investment strategies | 📈 Track performance | 🎯 Make informed decisions</p>
            <p style='font-size: 0.8em;'>Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
