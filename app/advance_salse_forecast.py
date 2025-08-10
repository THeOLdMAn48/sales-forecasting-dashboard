# sales_forecast_advanced.py
import pandas as pd
import numpy as np
from prophet import Prophet
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from scipy import stats
import io

st.set_page_config(page_title="Advanced Sales Forecast Dashboard", layout="wide")

@st.cache_data
def load_data(path="train.csv"):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

data = load_data()

# Sidebar
st.sidebar.header("Filter & Settings")
stores = sorted(data['store'].unique())
items = sorted(data['item'].unique())

selected_stores = st.sidebar.multiselect("Select store(s)", stores, default=[stores[0]])
selected_items = st.sidebar.multiselect("Select item(s)", items, default=[items[0]])
periods = st.sidebar.number_input("Forecast days", min_value=30, max_value=365, value=180, step=30)
holdout_days = st.sidebar.number_input("Holdout days for MAE (for accuracy)", min_value=30, max_value=365, value=90, step=30)
detect_anomalies = st.sidebar.checkbox("Enable anomaly detection (residual z-score)", value=True)

# Helper: prepare df for a store-item pair
def prepare_df(store, item):
    df = data[(data['store']==store) & (data['item']==item)].sort_values('date')
    df = df[['date', 'sales']].rename(columns={'date':'ds', 'sales':'y'})
    return df

# Cache model training per key (store,item,periods) to avoid refit repeatedly
@st.cache_resource
def train_prophet(df_tuple_key, df_records):
    # df_tuple_key is just a hashable key (store,item) ; df_records is authoritative for fitting
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(df_records)
    return model

# Function to compute forecast + metrics for one pair
def forecast_pair(store, item):
    df_pair = prepare_df(store, item)
    if df_pair.shape[0] < 60:
        st.warning(f"Not enough data for store {store}, item {item} (need >= 60 rows). Skipping.")
        return None

    # Train-test split for accuracy (holdout last n days)
    df_train = df_pair.iloc[:-holdout_days]
    df_holdout = df_pair.iloc[-holdout_days:]

    model = train_prophet((store,item), df_train)
    future = model.make_future_dataframe(periods=periods + holdout_days)
    forecast = model.predict(future)

    # compute MAE on holdout
    # Align forecast with holdout
    fc_holdout = forecast[['ds','yhat']].set_index('ds').reindex(df_holdout['ds']).reset_index()
    mae = mean_absolute_error(df_holdout['y'].values, fc_holdout['yhat'].values)

    # For anomalies: residuals on training set
    pred_train = forecast.set_index('ds').reindex(df_train['ds']).reset_index()
    residuals = df_train['y'].values - pred_train['yhat'].values
    # z-score
    z = stats.zscore(residuals)
    anomaly_dates = df_train['ds'].iloc[np.where(np.abs(z) > 2)[0]].tolist() if len(residuals)>0 else []

    return {
        'store': store,
        'item': item,
        'df_train': df_train,
        'df_holdout': df_holdout,
        'forecast': forecast,    # contains ds, yhat, yhat_lower, yhat_upper
        'mae': mae,
        'anomaly_dates': anomaly_dates
    }

# Run forecasting for all selected combos
results = []
progress = st.sidebar.progress(0)
total = len(selected_stores) * len(selected_items)
i = 0
for s in selected_stores:
    for it in selected_items:
        i += 1
        progress.progress(int(i/total * 100))
        res = forecast_pair(s, it)
        if res:
            results.append(res)
progress.empty()

if not results:
    st.warning("No forecasts produced. Check selections or data availability.")
    st.stop()

# UI layout: show summary metrics
st.title("📈 Advanced Sales Forecasting Dashboard")
col1, col2 = st.columns([3,1])

with col2:
    st.header("Summary")
    for r in results:
        st.markdown(f"**Store {r['store']} - Item {r['item']}**  — MAE (holdout {holdout_days} days): **{r['mae']:.2f}**")
    st.markdown("---")
    st.markdown("Download combined forecasts below")

# Combine forecasts for download: keep ds, store, item, yhat, yhat_lower, yhat_upper
combined = []
for r in results:
    f = r['forecast'][['ds','yhat','yhat_lower','yhat_upper']].copy()
    f['store'] = r['store']
    f['item'] = r['item']
    combined.append(f)
combined_df = pd.concat(combined)
csv_bytes = combined_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download all forecasts (CSV)", data=csv_bytes, file_name="combined_forecasts.csv", mime="text/csv")

# Plot interactive plotly graph: actual + forecast + confidence + anomalies
fig = go.Figure()
for r in results:
    # historical actuals
    fig.add_trace(go.Scatter(x=r['df_train']['ds'], y=r['df_train']['y'],
                             mode='markers', marker=dict(size=4),
                             name=f"Actual S{r['store']}-I{r['item']} (train)"))
    # holdout actuals
    fig.add_trace(go.Scatter(x=r['df_holdout']['ds'], y=r['df_holdout']['y'],
                             mode='markers', marker=dict(size=6),
                             name=f"Actual S{r['store']}-I{r['item']} (holdout)"))
    # forecast line
    fig.add_trace(go.Scatter(x=r['forecast']['ds'], y=r['forecast']['yhat'],
                             mode='lines', line=dict(width=2),
                             name=f"Forecast S{r['store']}-I{r['item']}"))
    # confidence band
    fig.add_traces([
        go.Scatter(x=r['forecast']['ds'], y=r['forecast']['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False),
        go.Scatter(x=r['forecast']['ds'], y=r['forecast']['yhat_lower'], mode='lines', fill='tonexty',
                   fillcolor='rgba(0,100,80,0.1)', line=dict(width=0), showlegend=False)
    ])
    # anomalies on training if enabled
    if detect_anomalies and r['anomaly_dates']:
        # find y values for those dates
        df_anom = r['df_train'][r['df_train']['ds'].isin(r['anomaly_dates'])]
        fig.add_trace(go.Scatter(x=df_anom['ds'], y=df_anom['y'], mode='markers',
                                 marker=dict(size=10, color='red', symbol='x'),
                                 name=f"Anomalies S{r['store']}-I{r['item']}"))

fig.update_layout(title="Actual vs Forecast (multiple store-item)", xaxis_title="Date", yaxis_title="Sales", height=700)
st.plotly_chart(fig, use_container_width=True)

# Show component plots for first result (trend, yearly)
st.subheader("Model Components (first selection)")
first = results[0]
m = Prophet(yearly_seasonality=True, daily_seasonality=False)
m.fit(first['df_train'])
comp = m.predict(m.make_future_dataframe(periods=0))
fig_comp = m.plot_components(m.predict(m.make_future_dataframe(periods=periods)))
st.pyplot(fig_comp)

st.success("Done. Use the download button above to export forecasts.")

