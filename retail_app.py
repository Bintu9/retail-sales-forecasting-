# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Retail Sales Forecast Dashboard",
    layout="wide"
)

st.title("Retail Sales Forecast Dashboard")

# -------------------------------
# Load data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\Bintu\OneDrive\teks\Walmart.csv")  # Update path
    # Convert IsHoliday to numeric if exists
    if 'IsHoliday' in df.columns:
        df['IsHoliday'] = df['IsHoliday'].astype(int)
    # Parse dates with dayfirst=True
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    return df

df = load_data()

# -------------------------------
# Sidebar filters
# -------------------------------
st.sidebar.header("Filters")
store_options = df['Store'].unique()
selected_stores = st.sidebar.multiselect("Select Store(s)", store_options, default=store_options[:1])

forecast_weeks = st.sidebar.slider("Forecast Horizon (Weeks)", min_value=4, max_value=20, value=12, step=1)

# Numeric filters
fuel_range = (df['Fuel_Price'].min(), df['Fuel_Price'].max()) if 'Fuel_Price' in df.columns else (0,1000)
unemp_range = (df['Unemployment'].min(), df['Unemployment'].max()) if 'Unemployment' in df.columns else (0,100)

if 'Fuel_Price' in df.columns:
    fuel_range = st.sidebar.slider("Fuel Price Range", float(df['Fuel_Price'].min()), float(df['Fuel_Price'].max()), fuel_range)
if 'Unemployment' in df.columns:
    unemp_range = st.sidebar.slider("Unemployment Rate Range", float(df['Unemployment'].min()), float(df['Unemployment'].max()), unemp_range)

# -------------------------------
# Filter data
# -------------------------------
filtered_data = df[df['Store'].isin(selected_stores)]
if 'Fuel_Price' in filtered_data.columns:
    filtered_data = filtered_data[(filtered_data['Fuel_Price'] >= fuel_range[0]) & (filtered_data['Fuel_Price'] <= fuel_range[1])]
if 'Unemployment' in filtered_data.columns:
    filtered_data = filtered_data[(filtered_data['Unemployment'] >= unemp_range[0]) & (filtered_data['Unemployment'] <= unemp_range[1])]

# -------------------------------
# Weekly Sales Trend
# -------------------------------
st.subheader("Weekly Sales Trend")
fig = px.line(filtered_data, x='Date', y='Weekly_Sales', color='Store', markers=True,
              title="Weekly Sales Trend by Store")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Top Products or Departments
# -------------------------------
if 'Product' in filtered_data.columns:
    st.subheader("Top 10 Products by Sales")
    top_items = (filtered_data.groupby(['Store','Product'])['Weekly_Sales']
                 .sum().reset_index())
    top_items = top_items.groupby('Store').apply(lambda x: x.nlargest(10,'Weekly_Sales')).reset_index(drop=True)
    y_col = 'Product'
elif 'Dept' in filtered_data.columns:
    st.subheader("Top 10 Departments by Sales")
    top_items = (filtered_data.groupby(['Store','Dept'])['Weekly_Sales']
                 .sum().reset_index())
    top_items = top_items.groupby('Store').apply(lambda x: x.nlargest(10,'Weekly_Sales')).reset_index(drop=True)
    y_col = 'Dept'
else:
    top_items = None
    st.info("No 'Product' or 'Dept' column found. Skipping Top Items chart.")

if top_items is not None:
    fig2 = px.bar(top_items, x='Weekly_Sales', y=y_col, color='Store', orientation='h',
                  title="Top 10 Items by Store")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# SARIMAX Forecast with Exogenous Variables
# -------------------------------
st.subheader(f"{forecast_weeks}-Week Sales Forecast")
forecast_dfs = []
comparison_dfs = []

for store in selected_stores:
    store_data = filtered_data[filtered_data['Store']==store].sort_values('Date')
    sales_series = store_data.set_index('Date')['Weekly_Sales']

    # Exogenous variables
    exog_cols = [col for col in ['Fuel_Price','Unemployment','IsHoliday'] if col in store_data.columns]
    exog = store_data[exog_cols].copy() if exog_cols else None
    if exog is not None:
        exog.index = store_data['Date']  # Align indices

    if len(sales_series) < 10:
        st.warning(f"Not enough data to forecast for Store {store}")
        continue

    # Fit SARIMAX
    model = SARIMAX(sales_series, order=(1,1,1), seasonal_order=(1,1,1,52), exog=exog)
    model_fit = model.fit(disp=False)

    # Prepare future exog for forecast
    if exog is not None:
        last_exog = exog.iloc[[-1]].copy()
        future_exog = pd.concat([last_exog]*forecast_weeks, ignore_index=True)
        future_exog.index = pd.date_range(
            start=sales_series.index.max() + pd.Timedelta(weeks=1),
            periods=forecast_weeks,
            freq='W'
        )
    else:
        future_exog = None

    # Forecast
    forecast_index = pd.date_range(start=sales_series.index.max()+pd.Timedelta(weeks=1),
                                   periods=forecast_weeks, freq='W')
    forecast = model_fit.forecast(steps=forecast_weeks, exog=future_exog)

    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted_Sales': forecast.values,
        'Store': store
    })
    forecast_dfs.append(forecast_df)

    # Combine actual + forecast for plotting
    actual_df = store_data[['Date','Weekly_Sales']].copy()
    actual_df['Store'] = store
    actual_df.rename(columns={'Weekly_Sales':'Sales'}, inplace=True)
    actual_df['Type'] = 'Actual'

    forecast_plot_df = forecast_df.rename(columns={'Forecasted_Sales':'Sales'})
    forecast_plot_df['Type'] = 'Forecast'

    comparison_dfs.append(pd.concat([actual_df, forecast_plot_df], ignore_index=True))

forecast_all = pd.concat(forecast_dfs)
comparison_all = pd.concat(comparison_dfs)

# Plot Actual vs Forecast
fig3 = px.line(comparison_all, x='Date', y='Sales', color='Store', line_dash='Type',
               markers=True, title="Actual vs Forecast Sales by Store")
st.plotly_chart(fig3, use_container_width=True)

# Forecast table
st.subheader("Forecast Table")
st.dataframe(forecast_all)

# Download forecast CSV
csv = forecast_all.to_csv(index=False)
st.download_button("Download Forecast as CSV", data=csv, file_name="forecast_exog.csv", mime="text/csv")

# -------------------------------
# Store Metrics
# -------------------------------
st.subheader("Store Metrics")
metrics_data = (filtered_data.groupby('Store')['Weekly_Sales']
                .agg(['sum','mean']).reset_index()
                .rename(columns={'sum':'Total Sales','mean':'Average Weekly Sales'}))
metrics_data['Total Sales'] = metrics_data['Total Sales'].apply(lambda x: f"${x:,.2f}")
metrics_data['Average Weekly Sales'] = metrics_data['Average Weekly Sales'].apply(lambda x: f"${x:,.2f}")
st.dataframe(metrics_data)
