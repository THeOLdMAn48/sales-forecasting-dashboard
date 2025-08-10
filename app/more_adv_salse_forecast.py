# sales_forecast_multimodel.py
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy import stats
import plotly.graph_objects as go
import pickle
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Multi-Model Sales Forecast", layout="wide")

# --------- Helpers ----------
@st.cache_data
def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_series(df, store, item):
    sub = df[(df['store']==store) & (df['item']==item)].sort_values('date')
    series = sub[['date','sales']].set_index('date').asfreq('D').fillna(0)  # daily freq, fill 0
    series = series.rename(columns={'sales':'y'})
    series = series.reset_index().rename(columns={'date':'ds'})
    return series

# Simple anomaly detection on residuals
def detect_anomalies(series, preds_train):
    residuals = series['y'].values - preds_train
    if len(residuals) < 2:
        return []
    z = stats.zscore(residuals, nan_policy='omit')
    anom_idx = np.where(np.abs(z) > 2)[0]
    return series['ds'].iloc[anom_idx].tolist()

# Create lag features for XGBoost
def create_lag_features(df_series, lags=14):
    df = df_series.copy()
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df = df.dropna().reset_index(drop=True)
    return df

# --------- Models ----------
@st.cache_resource
def train_prophet_model(train_df):
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m.fit(train_df)
    return m

def forecast_prophet(model, periods, last_date):
    future = model.make_future_dataframe(periods=periods)
    fc = model.predict(future)
    return fc[['ds','yhat','yhat_lower','yhat_upper']]

def train_arima_model(train_series, order=(1,1,1)):
    # train_series is DataFrame with ds & y indexed by ds
    model = SARIMAX(train_series['y'], order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def forecast_arima(res, steps):
    fc = res.get_forecast(steps=steps)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)
    idx = mean.index
    df = pd.DataFrame({'ds': idx, 'yhat': mean.values, 'yhat_lower': conf.iloc[:,0].values, 'yhat_upper': conf.iloc[:,1].values})
    df = df.reset_index(drop=True)
    return df

def train_xgboost(train_df, lags=14):
    df_lag = create_lag_features(train_df, lags=lags)
    X = df_lag.drop(columns=['ds','y'])
    y = df_lag['y']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    return model, df_lag

def forecast_xgboost(model, hist_df, periods, lags=14):
    # hist_df: original series with ds & y
    df_hist = hist_df.copy().set_index('ds')
    preds = []
    last = df_hist['y'].copy()
    for i in range(periods):
        # create feature row
        cur_idx = last.index[-1] + pd.Timedelta(days=1)
        lag_vals = [last.iloc[-lag] if lag<=len(last) else 0 for lag in range(1,lags+1)]
        feat = pd.DataFrame([lag_vals + [cur_idx.dayofweek, cur_idx.month]],
                            columns=[f'lag_{i}' for i in range(1,lags+1)] + ['dayofweek','month'])
        p = model.predict(feat)[0]
        preds.append((cur_idx, p))
        last.loc[cur_idx] = p  # append predicted to last for recursive forecasting
    df = pd.DataFrame(preds, columns=['ds','yhat'])
    df['yhat_lower'] = df['yhat']*0.9
    df['yhat_upper'] = df['yhat']*1.1
    return df

# --------- UI ----------
st.title("🔮 Advanced Multi-Model Sales Forecasting (Prophet | ARIMA | XGBoost)")

df_all = load_data()

stores = sorted(df_all['store'].unique())
items = sorted(df_all['item'].unique())

col1, col2, col3 = st.columns([2,2,1])
with col1:
    store = st.selectbox("Store", stores, index=0)
with col2:
    item = st.selectbox("Item", items, index=0)
with col3:
    periods = st.number_input("Forecast days", min_value=30, max_value=365, value=180, step=30)

# optional settings
st.write("⚙️ Advanced Settings")
arima_order = st.text_input("ARIMA order (p,d,q)", value="1,1,1")
use_xgb = st.checkbox("Include XGBoost (slower)", value=True)
detect_anom = st.checkbox("Detect anomalies (z-score)", value=True)

# prepare series
series = prepare_series(df_all, store, item)
if series.shape[0] < 60:
    st.warning("Not enough data for this store-item (need >=60 days). Choose different selection.")
    st.stop()

# split train/holdout for metrics
holdout_days = min(90, int(series.shape[0]*0.2))
train_df = series.iloc[:-holdout_days].rename(columns={'ds':'ds','y':'y'})
holdout_df = series.iloc[-holdout_days:].reset_index(drop=True)

# Train Prophet
with st.spinner("Training Prophet..."):
    prophet_model = train_prophet_model(train_df)
    fc_prophet = forecast_prophet(prophet_model, periods+holdout_days, train_df['ds'].iloc[-1])

# Evaluate Prophet MAE on holdout
fc_hold_prophet = fc_prophet.set_index('ds').reindex(holdout_df['ds']).reset_index()
mae_prophet = mean_absolute_error(holdout_df['y'].values, fc_hold_prophet['yhat'].values)

# Train ARIMA
p,d,q = [int(x) for x in arima_order.split(",")]
with st.spinner("Training ARIMA..."):
    arima_res = train_arima_model(train_df.set_index('ds'), order=(p,d,q))
    fc_arima = forecast_arima(arima_res, steps=periods+holdout_days)

# Align and compute ARIMA MAE
fc_hold_arima = fc_arima.set_index('ds').reindex(holdout_df['ds']).reset_index()
mae_arima = mean_absolute_error(holdout_df['y'].values, fc_hold_arima['yhat'].values)

# Train XGBoost if selected
mae_xgb = None
fc_xgb = None
if use_xgb:
    with st.spinner("Training XGBoost (may take time)..."):
        xgb_model, df_lag = train_xgboost(train_df, lags=14)
        fc_xgb = forecast_xgboost(xgb_model, train_df, periods+holdout_days, lags=14)
        fc_hold_xgb = fc_xgb.set_index('ds').reindex(holdout_df['ds']).reset_index()
        mae_xgb = mean_absolute_error(holdout_df['y'].values, fc_hold_xgb['yhat'].values)

# Combine forecasts into a single DataFrame for plotting
def combine_forecasts(fc, label):
    tmp = fc.copy()
    tmp = tmp.rename(columns={'yhat':'yhat_'+label, 'yhat_lower':'low_'+label, 'yhat_upper':'high_'+label})
    return tmp[['ds', f'yhat_{label}', f'low_{label}', f'high_{label}']]

df_plot = series.copy().rename(columns={'ds':'ds','y':'y_actual'})

# Prophet df
prophet_trim = fc_prophet[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'yhat':'yhat_prophet','yhat_lower':'low_prophet','yhat_upper':'high_prophet'})
combined = prophet_trim

# ARIMA
arima_trim = fc_arima[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'yhat':'yhat_arima','yhat_lower':'low_arima','yhat_upper':'high_arima'})
combined = combined.merge(arima_trim, on='ds', how='left')

# XGBoost
if fc_xgb is not None:
    xgb_trim = fc_xgb[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={'yhat':'yhat_xgb','yhat_lower':'low_xgb','yhat_upper':'high_xgb'})
    combined = combined.merge(xgb_trim, on='ds', how='left')

# Merge actuals
combined = combined.merge(df_plot[['ds','y_actual']], on='ds', how='left')

# Calculate summary metrics for next 'periods' forecast only (exclude holdout part)
future_mask = combined['ds'] > train_df['ds'].max()
future_df = combined[future_mask].copy()

sum_future = future_df['yhat_prophet'].sum() if 'yhat_prophet' in future_df else 0
sum_past = series.tail(180)['y'].sum() if series.shape[0] >= 180 else series['y'].sum()
growth_pct = ((sum_future - sum_past) / (sum_past+1e-9)) * 100

# Anomalies on training
anomaly_dates = []
if detect_anom:
    # prophet preds on train
    prophet_train_preds = fc_prophet.set_index('ds').reindex(train_df['ds']).reset_index()['yhat'].values
    anomaly_dates = detect_anomalies(train_df.reset_index(drop=True), prophet_train_preds)

# --------- UI OUTPUT ----------
left, right = st.columns([3,1])
with left:
    st.subheader(f"Store {store} - Item {item} — Forecast next {periods} days")
    # Plotly interactive plot
    fig = go.Figure()
    # actual
    fig.add_trace(go.Scatter(x=combined['ds'], y=combined['y_actual'], name='Actual', mode='markers+lines', marker=dict(size=4)))
    # prophet
    fig.add_trace(go.Scatter(x=combined['ds'], y=combined['yhat_prophet'], name='Prophet', mode='lines'))
    fig.add_trace(go.Scatter(x=combined['ds'], y=combined['low_prophet'], name='Prophet_upper', mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=combined['ds'], y=combined['high_prophet'], name='Prophet_lower', mode='lines', fill='tonexty', fillcolor='rgba(0,100,80,0.1)', line=dict(width=0), showlegend=False))
    # arima
    fig.add_trace(go.Scatter(x=combined['ds'], y=combined['yhat_arima'], name='ARIMA', mode='lines'))
    # xgb
    if 'yhat_xgb' in combined.columns:
        fig.add_trace(go.Scatter(x=combined['ds'], y=combined['yhat_xgb'], name='XGBoost', mode='lines'))
    # anomalies
    if anomaly_dates:
        anom_vals = train_df[train_df['ds'].isin(anomaly_dates)]
        fig.add_trace(go.Scatter(x=anom_vals['ds'], y=anom_vals['y'], mode='markers', marker=dict(color='red', size=8), name='Anomaly'))

    fig.update_layout(height=600, xaxis_title='Date', yaxis_title='Sales', legend=dict(orientation='h'))
    st.plotly_chart(fig, use_container_width=True)

    # Components: show Prophet components for train + future
    st.subheader("Prophet components (trend & yearly seasonality)")
    comp_fig = prophet_model.plot_components(prophet_model.predict(prophet_model.make_future_dataframe(periods=0)))
    st.pyplot(comp_fig)

with right:
    st.metric("Prophet MAE (holdout)", f"{mae_prophet:.2f}")
    st.metric("ARIMA MAE (holdout)", f"{mae_arima:.2f}")
    if mae_xgb is not None:
        st.metric("XGBoost MAE (holdout)", f"{mae_xgb:.2f}")
    st.markdown("---")
    st.markdown("### Auto Insights")
    st.write(f"- Expected next {periods} days total (Prophet): **{sum_future:.0f}** units")
    st.write(f"- Recent {min(180, series.shape[0])} days total sales: **{sum_past:.0f}** units")
    st.write(f"- Growth vs past period: **{growth_pct:.2f}%**")
    if anomaly_dates:
        st.write(f"- {len(anomaly_dates)} anomalies detected in training data (marked in chart)")

# Download combined forecast
buffer = io.BytesIO()
combined.to_excel(buffer, index=False, engine='openpyxl')
buffer.seek(0)
st.download_button("📥 Download Combined Forecasts (XLSX)", data=buffer, file_name=f"combined_forecast_S{store}_I{item}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.success("Forecast complete. Tip: For faster runs select only one store and one item.")
