import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import altair as alt
import datetime

# Page Configuration
st.set_page_config(
    page_title="Advanced VaR Calculator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

# Cache data fetching for better performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(tickers, start, end):
    try:
        data = yf.download(tickers, start, end, auto_adjust=False)
        
        # Check if data was successfully retrieved
        if data.empty:
            st.error(f"No data found for the selected tickers: {tickers}")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# VaR Class Definition
class VaR:
    def __init__(self, ticker, start_date, end_date, rolling_window, confidence_level, portfolio_val, weights=None):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.rolling = rolling_window
        self.conf_level = confidence_level
        self.portf_val = portfolio_val
        self.historical_var = None
        self.parametric_var = None
        self.component_vars = None
        self.var_contributions = None
        
        # Use provided weights or default to equal weights
        if weights is None:
            self.weights = np.array([1 / len(self.ticker)] * len(self.ticker))
        else:
            self.weights = weights
            
        self.data_loaded = self.data()
        
        if self.data_loaded:
            self.historical_method()
            self.parametric_method()
            if len(self.ticker) > 1:
                self.calculate_component_var()
        
    def data(self):
        # Use cached function to fetch data
        df = fetch_stock_data(self.ticker, self.start, self.end)
        
        if df is None or df.empty:
            return False
            
        # Check for Adj Close column
        if "Adj Close" not in df.columns:
            st.error("Data doesn't contain Adjusted Close prices.")
            return False
            
        self.adj_close_df = df["Adj Close"]
        
        # Check for sufficient data
        if len(self.adj_close_df) < self.rolling + 5:  # Need at least rolling window + a few extra days
            st.error(f"Not enough data for the selected date range and {self.rolling}-day rolling window.")
            return False
            
        # Calculate returns
        self.log_returns_df = np.log(self.adj_close_df / self.adj_close_df.shift(1))
        self.log_returns_df = self.log_returns_df.dropna()
        
        # Check if returns calculation succeeded
        if self.log_returns_df.empty:
            st.error("Failed to calculate returns.")
            return False
            
        # Calculate portfolio returns using weights
        try:
            # Handle single ticker case
            if len(self.ticker) == 1:
                self.portfolio_returns = self.log_returns_df
            else:
                self.portfolio_returns = (self.log_returns_df * self.weights).sum(axis=1)
                
            self.rolling_returns = self.portfolio_returns.rolling(window=self.rolling).sum()
            self.rolling_returns = self.rolling_returns.dropna()
            
            # Check if we have any data after rolling window
            if len(self.rolling_returns) == 0:
                st.error(f"No data available after applying {self.rolling}-day rolling window. Try a shorter window or longer date range.")
                return False
                
            return True
        except Exception as e:
            st.error(f"Error in portfolio calculations: {e}")
            return False

    def historical_method(self):
        try:
            if len(self.rolling_returns) > 0:
                historical_VaR = -np.percentile(self.rolling_returns, 100 - (self.conf_level * 100)) * self.portf_val
                self.historical_var = historical_VaR
            else:
                st.warning("Not enough data for historical VaR calculation.")
                self.historical_var = 0
        except Exception as e:
            st.error(f"Error calculating historical VaR: {e}")
            self.historical_var = 0

    def parametric_method(self):
        try:
            self.cov_matrix = self.log_returns_df.cov() * 252
            self.portfolio_std = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))
            parametric_VaR = self.portfolio_std * norm.ppf(self.conf_level) * np.sqrt(self.rolling / 252) * self.portf_val
            self.parametric_var = parametric_VaR
        except Exception as e:
            st.error(f"Error calculating parametric VaR: {e}")
            self.parametric_var = 0
    
    def calculate_component_var(self):
        """Calculate component VaR and contribution percentages based on Professor Fusai's methodology"""
        try:
            # Ensure multiple assets and valid parametric VaR
            if len(self.ticker) > 1 and self.parametric_var > 0:
                # Portfolio variance
                portfolio_variance = np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # Initialize containers for results
                self.component_vars = {}
                self.var_contributions = {}
                
                # For each asset, calculate its component VaR
                for i, ticker in enumerate(self.ticker):
                    # Calculate covariance between asset and portfolio
                    # We need to ensure we're accessing the DataFrame correctly
                    cov_with_portfolio = 0
                    for j, other_ticker in enumerate(self.ticker):
                        # Access covariance matrix by ticker names, not by indices
                        cov_with_portfolio += self.weights[j] * self.cov_matrix.loc[ticker, other_ticker]
                    
                    # Marginal VaR calculation
                    z_value = norm.ppf(self.conf_level)
                    marginal_var = z_value * (cov_with_portfolio / portfolio_volatility) * np.sqrt(self.rolling / 252)
                    
                    # Component VaR = weight * Marginal VaR
                    component_var = self.weights[i] * marginal_var * self.portf_val
                    self.component_vars[ticker] = component_var
                    
                    # Calculate contribution percentage
                    self.var_contributions[ticker] = (component_var / self.parametric_var) * 100
            else:
                # For a single asset, component VaR equals parametric VaR
                if len(self.ticker) == 1:
                    self.component_vars = {self.ticker[0]: self.parametric_var}
                    self.var_contributions = {self.ticker[0]: 100.0}
                else:
                    # Fallback for invalid VaR
                    self.component_vars = {ticker: 0 for ticker in self.ticker}
                    self.var_contributions = {ticker: 0 for ticker in self.ticker}
        
        except Exception as e:
            st.error(f"Error calculating component VaR: {e}")
            import traceback
            st.warning(traceback.format_exc())
            
            # Reset to zero values as fallback
            self.component_vars = {ticker: 0 for ticker in self.ticker}
            self.var_contributions = {ticker: 0 for ticker in self.ticker}

    def plot_var_results(self, title, var_value, returns_dollar, conf_level):
        try:
            # Adjust the figure size to make the chart fit half page
            plt.figure(figsize=(12, 6))
            plt.hist(returns_dollar, bins=50, density=True)
            plt.xlabel(f'\n {title} VaR = ${var_value:.2f}')
            plt.ylabel('Frequency')
            plt.title(f"Distribution of Portfolio's {self.rolling}-Day Returns ({title} VaR)")
            plt.axvline(-var_value, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {conf_level:.0%} confidence level')
            plt.legend()
            plt.tight_layout()
            return plt
        except Exception as e:
            st.error(f"Error creating {title} chart: {e}")
            # Return an empty figure as fallback
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f"Could not create chart: {e}", 
                     horizontalalignment='center', verticalalignment='center')
            return plt

# Stress scenario dates
STRESS_SCENARIOS = {
    "None": (None, None),
    "2008 Financial Crisis": ("2008-09-01", "2009-03-31"),
    "2020 COVID Crash": ("2020-02-15", "2020-04-15"),
    "2022 Rate Hikes": ("2022-01-01", "2022-06-30"),
}

# Initialize session state variables
if 'recent_outputs' not in st.session_state:
    st.session_state['recent_outputs'] = []
    
if 'first_run' not in st.session_state:
    st.session_state['first_run'] = True

# Sidebar for User Inputs
with st.sidebar:
    st.title('📈 VaR Calculator')
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/desai-prathmesh/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Prathmesh Desai`</a>', unsafe_allow_html=True)

    tickers_input = st.text_input('Enter tickers separated by space', 'AAPL MSFT GOOG')
    tickers = tickers_input.split()
    
    # Portfolio Rebalancing Section
    st.subheader("Portfolio Weights")
    use_equal_weights = st.checkbox("Use equal weights", value=True)
    
    weights = {}
    if use_equal_weights:
        for ticker in tickers:
            weights[ticker] = 1.0 / len(tickers)
    else:
        total_weight = 0
        for ticker in tickers:
            weight = st.slider(f"{ticker} weight (%)", 0, 100, int(100/len(tickers)))
            weights[ticker] = weight / 100
            total_weight += weights[ticker]
        
        # Normalize weights if they don't sum to 1
        if total_weight != 1.0:
            st.warning(f"Weights sum to {total_weight*100:.1f}%. Normalizing to 100%.")
            weights = {t: w/total_weight for t, w in weights.items()}
    
    # Display the final weights
    weights_df = pd.DataFrame({"Weight (%)": [f"{w*100:.1f}%" for w in weights.values()]}, index=weights.keys())
    st.dataframe(weights_df)
    
    # Standard VaR inputs
    min_date = datetime.date(2010, 1, 1)  # Setting a reasonable minimum date
    max_date = datetime.date.today()
    
    start_date = st.date_input('Start date', value=datetime.date(2020, 1, 1), min_value=min_date, max_value=max_date)
    end_date = st.date_input('End date', value=max_date, min_value=start_date, max_value=max_date)
    
    if start_date >= end_date:
        st.error("End date must be after start date.")
    
    rolling_window = st.slider('Rolling window (days)', min_value=5, max_value=252, value=20)
    confidence_level = st.slider('Confidence level', min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    portfolio_val = st.number_input('Portfolio value ($)', value=100000, min_value=1000)
    
    # Market Stress Scenario Section
    st.subheader("Stress Scenario Testing")
    selected_scenario = st.selectbox("Select market stress scenario", list(STRESS_SCENARIOS.keys()))
    
    calculate_btn = st.button('Calculate VaR')

def calculate_and_display_var(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_val, weights_dict, scenario=None):
    try:
        # Display "Calculating..." status
        status_placeholder = st.empty()
        status_placeholder.info("Calculating VaR... Please wait.")
        
        # Check if tickers are provided
        if not tickers:
            st.error("Please enter at least one ticker symbol.")
            status_placeholder.empty()
            return
        
        # Convert weights dict to array in the right order
        weights_array = np.array([weights_dict[ticker] for ticker in tickers])
        
        # Calculate VaR with the given parameters
        var_instance = VaR(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_val, weights_array)
        
        # Clear the status message
        status_placeholder.empty()
        
        # Check if data was loaded successfully
        if not var_instance.data_loaded:
            return
            
        # Layout for charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.info("Historical VaR Chart")
            historical_chart = var_instance.plot_var_results("Historical", var_instance.historical_var, var_instance.rolling_returns * var_instance.portf_val, confidence_level)
            st.pyplot(historical_chart)

        with chart_col2:
            st.info("Parametric VaR Chart")
            parametric_chart = var_instance.plot_var_results("Parametric", var_instance.parametric_var, var_instance.rolling_returns * var_instance.portf_val, confidence_level)
            st.pyplot(parametric_chart)

        # Layout for input summary and VaR results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info("Input Summary")
            st.write(f"Tickers: {tickers}")
            st.write(f"Weights: {', '.join([f'{t}: {w*100:.1f}%' for t, w in weights_dict.items()])}")
            st.write(f"Start Date: {start_date}")
            st.write(f"End Date: {end_date}")
            st.write(f"Rolling Window: {rolling_window} days")
            st.write(f"Confidence Level: {confidence_level:.2%}")
            st.write(f"Portfolio Value: ${portfolio_val:,.2f}")

        with col2:
            st.info("VaR Calculation Output")
            data = {
                "Method": ["Historical", "Parametric"],
                "VaR Value": [f"${var_instance.historical_var:,.2f}", f"${var_instance.parametric_var:,.2f}"]
            }
            df = pd.DataFrame(data)
            st.table(df)
        
        # Risk Attribution (Component VaR)
        if len(tickers) > 1 and var_instance.component_vars:  # Only show for multi-asset portfolios with calculated components
            st.info("Risk Attribution Analysis")
            
            # Convert component VaR and contributions to DataFrames for display
            attribution_data = {
                "Asset": list(var_instance.component_vars.keys()),
                "Component VaR ($)": [f"${v:,.2f}" for v in var_instance.component_vars.values()],
                "Contribution (%)": [f"{v:.2f}%" for v in var_instance.var_contributions.values()]
            }
            
            attribution_df = pd.DataFrame(attribution_data)
            
            # Add numeric columns for sorting
            attribution_df["VaR_numeric"] = list(var_instance.component_vars.values())
            attribution_df["Contribution_numeric"] = list(var_instance.var_contributions.values())
            
            # Sort by contribution percentage
            attribution_df = attribution_df.sort_values("Contribution_numeric", ascending=False)
            
            # Display the table (without numeric columns)
            st.table(attribution_df[["Asset", "Component VaR ($)", "Contribution (%)"]])
            
            # Create bar chart of contributions
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(attribution_df["Asset"], attribution_df["Contribution_numeric"])
            ax.set_ylabel("Contribution to VaR (%)")
            ax.set_title("Risk Attribution by Asset")
            ax.set_ylim(0, max(attribution_df["Contribution_numeric"]) * 1.1)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Verify Component VaR sum equals Parametric VaR
            total_component_var = sum(var_instance.component_vars.values())
            st.write(f"Sum of Component VaRs: ${total_component_var:,.2f}")
            st.write(f"Parametric VaR: ${var_instance.parametric_var:,.2f}")
            st.write(f"Difference: ${total_component_var - var_instance.parametric_var:,.2f}")
            st.write(f"Sum of Contributions: {sum(var_instance.var_contributions.values()):.2f}%")
        
        # Calculate stress scenario VaR if requested
        if scenario and scenario != "None":
            st.info(f"Stress Test: {scenario}")
            
            # Get scenario dates
            scenario_start, scenario_end = STRESS_SCENARIOS[scenario]
            
            # Create a new VaR instance with the scenario data
            scenario_var = VaR(tickers, scenario_start, scenario_end, rolling_window, 
                             confidence_level, portfolio_val, weights_array)
            
            # Only display if stress data was loaded successfully
            if scenario_var.data_loaded:
                # Show comparison
                stress_data = {
                    "Scenario": [scenario, scenario],
                    "Method": ["Historical", "Parametric"],
                    "Normal VaR": [f"${var_instance.historical_var:,.2f}", f"${var_instance.parametric_var:,.2f}"],
                    "Stress VaR": [f"${scenario_var.historical_var:,.2f}", f"${scenario_var.parametric_var:,.2f}"],
                    "Change (%)": [f"{((scenario_var.historical_var/var_instance.historical_var)-1)*100:+.2f}%" if var_instance.historical_var > 0 else "N/A",
                                  f"{((scenario_var.parametric_var/var_instance.parametric_var)-1)*100:+.2f}%" if var_instance.parametric_var > 0 else "N/A"]
                }
                
                stress_df = pd.DataFrame(stress_data)
                st.table(stress_df)
                
                # Add a simple chart comparing normal vs stress VaR
                fig, ax = plt.subplots(figsize=(10, 5))
                x = ["Historical", "Parametric"]
                normal_var = [var_instance.historical_var, var_instance.parametric_var]
                stress_var = [scenario_var.historical_var, scenario_var.parametric_var]
                
                bar_width = 0.35
                normal_bars = ax.bar(np.arange(len(x)), normal_var, bar_width, label="Normal VaR")
                stress_bars = ax.bar(np.arange(len(x)) + bar_width, stress_var, bar_width, label="Stress VaR")
                
                ax.set_ylabel("VaR Value ($)")
                ax.set_title(f"VaR Comparison: Normal vs {scenario}")
                ax.set_xticks(np.arange(len(x)) + bar_width / 2)
                ax.set_xticklabels(x)
                ax.legend()
                
                # Add value labels
                for bars in [normal_bars, stress_bars]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                                f'${height:,.0f}', ha='center', va='bottom')
                
                st.pyplot(fig)
        
        # Store in recent outputs (limit to 5 entries)
        st.session_state['recent_outputs'].append({
            "Historical": f"${var_instance.historical_var:,.2f}",
            "Parametric": f"${var_instance.parametric_var:,.2f}"
        })
        
        # Keep only the 5 most recent outputs
        if len(st.session_state['recent_outputs']) > 5:
            st.session_state['recent_outputs'] = st.session_state['recent_outputs'][-5:]
        
        # Display Recent VaR Output table
        st.info("Previous VaR Calculation Outputs")
        recent_df = pd.DataFrame(st.session_state['recent_outputs'])
        st.table(recent_df)
        
    except Exception as e:
        st.error(f"An error occurred during calculation: {e}")
        import traceback
        st.error(traceback.format_exc())

# Run default calculation on first load
if st.session_state['first_run']:
    st.session_state['first_run'] = False
    # Default values for first run
    default_tickers = 'AAPL MSFT GOOG'.split()
    default_weights = {ticker: 1/len(default_tickers) for ticker in default_tickers}
    default_start_date = datetime.date(2020, 1, 1)
    default_end_date = datetime.date.today()
    default_rolling_window = 20
    default_confidence_level = 0.95
    default_portfolio_val = 100000

    # Perform the default calculation
    calculate_and_display_var(
        default_tickers, 
        default_start_date, 
        default_end_date, 
        default_rolling_window, 
        default_confidence_level, 
        default_portfolio_val,
        default_weights
    )

# Display Results on Button Click
if calculate_btn:
    calculate_and_display_var(
        tickers, 
        start_date, 
        end_date, 
        rolling_window, 
        confidence_level, 
        portfolio_val,
        weights,
        selected_scenario
    )
