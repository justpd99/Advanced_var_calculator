# Advanced VaR (Value-at-Risk) Calculator

![VaR Calculator](https://img.shields.io/badge/Risk%20Analysis-Advanced%20VaR%20Calculator-blue)
![Python](https://img.shields.io/badge/Python-3.7%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)

An interactive Value at Risk (VaR) calculator built with Streamlit that helps investors and risk managers estimate potential portfolio losses.

## 🔗 Live Demo

**Try it out:** [VaR Calculator Web App](https://advanced-var-calculator.streamlit.app)

Use the hosted application to calculate VaR on your portfolio without installing anything!

## 📊 Features

- **Multiple VaR Methodologies**: Calculate VaR using both Historical and Parametric approaches
- **Portfolio Customization**: Input multiple tickers with custom weighting
- **Advanced Risk Attribution**: Analyze which assets contribute most to your portfolio's risk using component VaR methodology
- **Stress Testing**: Test your portfolio against historical stress scenarios like the 2008 Financial Crisis, 2020 COVID Crash, and 2022 Rate Hikes
- **Interactive Visualizations**: View your risk profile through intuitive histograms and bar charts
- **History Tracking**: Keep track of your recent VaR calculations for comparison

## 📋 Prerequisites

- Python 3.7+
- Required packages:
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - yfinance
  - scipy
  - altair

## 🔧 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/var-calculator.git
   cd var-calculator
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run var_calculator.py
   ```

2. Open your browser and go to the URL shown in the terminal (typically http://localhost:8501)

3. Using the sidebar controls:
   - Enter ticker symbols for the assets in your portfolio (e.g., "NVDA MSFT PG")
   - Choose between equal weights or set custom weights for each asset
   - Select date range for historical data
   - Adjust rolling window period, confidence level, and portfolio value
   - Choose a stress scenario (optional)

4. Click "Calculate VaR" to generate results including:
   - Historical and Parametric VaR calculations
   - Distribution charts
   - Risk attribution analysis
   - Stress test comparisons (if selected)

## 📚 Understanding VaR

Value at Risk (VaR) represents the maximum potential loss an investment portfolio might experience over a specified time period at a given confidence level.

For example, a 1-day 95% VaR of $10,000 means there's a 95% probability that your portfolio won't lose more than $10,000 in a single day.

### VaR Methodologies

This calculator implements two common VaR methodologies:

1. **Historical Method**: Uses actual historical returns to estimate potential losses
2. **Parametric Method**: Assumes returns follow a normal distribution and uses portfolio standard deviation

## 🔄 Stress Testing

The application allows you to test your portfolio against historical stress scenarios:

- **2008 Financial Crisis**: September 2008 - March 2009
- **2020 COVID Crash**: February 2020 - April 2020
- **2022 Rate Hikes**: January 2022 - June 2022

This provides insight into how your current portfolio might perform under similar market conditions.

## 💡 Risk Attribution

For multi-asset portfolios, the tool breaks down each asset's contribution to the overall VaR:

- **Component VaR**: Shows how much each asset contributes to the total portfolio VaR
- **VaR Contribution Percentage**: Displays each asset's relative contribution to total risk

This helps you identify which positions carry the most risk and provides a more accurate picture of risk distribution within your portfolio.

## ⚠️ Limitations

- Relies on historical data and assumes past performance can indicate future risk
- May not fully capture extreme market events or "black swan" scenarios
- Returns are assumed to follow normal distribution in the parametric method
- Does not account for liquidity risk or other non-market risk factors

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📞 Contact

Created by: Prathmesh S Desai (https://www.linkedin.com/in/desai-prathmesh/)
