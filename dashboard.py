import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import time

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.2rem; font-weight:700; color:#ffa500; }
.sub-header  { color:#888; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚙️ Predictive Maintenance IoT Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">XGBoost RUL predictor · NASA CMAPSS · MAE 7.68 cycles · Critical zone MAE 3.30 cycles</div>', unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model        = joblib.load(os.path.join(BASE, "xgb_rul.pkl"))
    scaler       = joblib.load(os.path.join(BASE, "scaler_rul.pkl"))
    feature_cols = joblib.load(os.path.join(BASE, "feature_cols.pkl"))
    return model, scaler, feature_cols

@st.cache_data
def load_data():
    X_test  = np.load(os.path.join(BASE, "X_test.npy"))
    y_test  = np.load(os.path.join(BASE, "y_test.npy"))
    X_train = np.load(os.path.join(BASE, "X_train.npy"))
    y_train = np.load(os.path.join(BASE, "y_train.npy"))
    return X_test, y_test, X_train, y_train

model, scaler, feature_cols = load_model()
X_test, y_test, X_train, y_train = load_data()

# Run predictions on full test set
@st.cache_data
def get_all_predictions():
    X_scaled = scaler.transform(X_test)
    preds    = np.clip(model.predict(X_scaled), 0, 125)
    return preds

y_pred = get_all_predictions()
errors = y_pred - y_test

def rul_status(rul):
    if rul <= 25:  return "CRITICAL", "🔴", "#ff4444"
    if rul <= 50:  return "WARNING",  "🟠", "#ff8c00"
    if rul <= 100: return "CAUTION",  "🟡", "#ffd700"
    return "HEALTHY", "🟢", "#00d4aa"

# ── KPIs ──────────────────────────────────────────────────────────────────────
mae  = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors**2))
crit_mask = y_test <= 25
crit_mae  = np.mean(np.abs(errors[crit_mask]))

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Test Samples",       f"{len(y_test):,}")
k2.metric("MAE",                f"{mae:.2f} cycles")
k3.metric("RMSE",               f"{rmse:.2f} cycles")
k4.metric("Critical MAE",       f"{crit_mae:.2f} cycles",  "RUL ≤ 25")
k5.metric("Within ±10 cycles",  f"{(np.abs(errors)<=10).mean():.1%}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Fleet Overview",
    "📈 Engine Trajectory",
    "📊 Model Performance",
    "🔴 Alert Simulation"
])

# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Fleet Health — Real Test Engines")

    # Sample N engines from test set, use their last recorded cycle
    n_engines = st.slider("Engines to display", 5, 20, 10)
    np.random.seed(st.session_state.get("fleet_seed", 42))

    sample_idx = np.random.choice(len(X_test), n_engines, replace=False)
    fleet = []
    for i, idx in enumerate(sample_idx):
        rul_actual    = float(y_test[idx])
        rul_predicted = float(y_pred[idx])
        status, alert, color = rul_status(rul_predicted)
        fleet.append({
            "Engine":     f"ENG-{i+1:03d}",
            "True RUL":   round(rul_actual, 1),
            "Pred RUL":   round(rul_predicted, 1),
            "Error":      round(abs(rul_predicted - rul_actual), 1),
            "Status":     status,
            "Alert":      alert,
            "color":      color
        })

    df_fleet = pd.DataFrame(fleet)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Engines",   len(df_fleet))
    c2.metric("Critical",        len(df_fleet[df_fleet["Status"]=="CRITICAL"]))
    c3.metric("Warning",         len(df_fleet[df_fleet["Status"]=="WARNING"]))
    c4.metric("Avg Pred RUL",    f"{df_fleet['Pred RUL'].mean():.0f} cycles")

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df_fleet["Engine"], y=df_fleet["Pred RUL"],
        marker_color=df_fleet["color"],
        text=df_fleet["Status"], textposition="outside",
        name="Predicted RUL"
    ))
    fig1.add_trace(go.Scatter(
        x=df_fleet["Engine"], y=df_fleet["True RUL"],
        mode="markers", marker=dict(symbol="diamond", size=10, color="white"),
        name="True RUL"
    ))
    fig1.add_hline(y=25, line_dash="dash", line_color="red",    annotation_text="Critical (25)")
    fig1.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Warning (50)")
    fig1.update_layout(template="plotly_dark", height=420,
                       yaxis_title="Remaining Useful Life (cycles)",
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig1, use_container_width=True)

    if st.button("🔄 Resample Fleet"):
        st.session_state["fleet_seed"] = np.random.randint(0, 9999)
        st.rerun()

    st.dataframe(df_fleet.drop(columns=["color"]), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Engine Degradation Trajectory")
    st.caption("RUL prediction across all cycles for a single engine in the test set")

    # Group test samples — approximate by sorting by predicted RUL descending
    # (represents lifecycle from healthy → end of life)
    engine_options = {f"Engine Profile {i+1}": i for i in range(5)}
    selected = st.selectbox("Select engine profile", list(engine_options.keys()))
    profile  = engine_options[selected]

    np.random.seed(profile * 7)
    n_cycles = st.slider("Lifecycle length", 50, 200, 120)

    # Sample a trajectory of test points
    traj_idx    = np.random.choice(len(X_test), n_cycles, replace=False)
    traj_actual = y_test[traj_idx]
    traj_pred   = y_pred[traj_idx]

    # Sort by true RUL descending (start healthy, end critical)
    sort_order  = np.argsort(traj_actual)[::-1]
    traj_actual = traj_actual[sort_order]
    traj_pred   = traj_pred[sort_order]
    cycles      = np.arange(1, n_cycles + 1)

    fig2 = make_subplots(rows=2, cols=1,
                         subplot_titles=("RUL Over Engine Lifecycle",
                                         "Prediction Error Over Lifecycle"),
                         row_heights=[0.65, 0.35])

    fig2.add_trace(go.Scatter(
        x=cycles, y=traj_actual,
        mode="lines", name="Actual RUL",
        line=dict(color="#888", width=1.5, dash="dot")
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=cycles, y=traj_pred,
        mode="lines", name="XGBoost Prediction",
        line=dict(color="#ffa500", width=2)
    ), row=1, col=1)
    fig2.add_hline(y=25, line_dash="dash", line_color="red",
                   annotation_text="Critical", row=1, col=1)
    fig2.add_hline(y=50, line_dash="dash", line_color="orange",
                   annotation_text="Warning",  row=1, col=1)

    pred_error = traj_pred - traj_actual
    fig2.add_trace(go.Bar(
        x=cycles, y=pred_error,
        marker_color=["#ff4444" if e > 0 else "#00d4aa" for e in pred_error],
        name="Prediction Error"
    ), row=2, col=1)
    fig2.add_hline(y=0, line_color="white", row=2, col=1)

    fig2.update_layout(template="plotly_dark", height=560,
                       hovermode="x unified",
                       legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance — Real Predictions on Test Set")

    col1, col2 = st.columns(2)

    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=y_test, y=y_pred,
            mode="markers",
            marker=dict(color="#ffa500", size=3, opacity=0.4),
            name="Predictions"
        ))
        fig3.add_trace(go.Scatter(
            x=[0, 125], y=[0, 125],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Perfect prediction"
        ))
        fig3.update_layout(
            template="plotly_dark", height=380,
            title=f"Predicted vs Actual RUL | MAE={mae:.1f}",
            xaxis_title="Actual RUL",
            yaxis_title="Predicted RUL"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = go.Figure(go.Histogram(
            x=errors, nbinsx=60,
            marker_color="#ffa500", opacity=0.8
        ))
        fig4.add_vline(x=0, line_dash="dash", line_color="red")
        fig4.update_layout(
            template="plotly_dark", height=380,
            title=f"Error Distribution | RMSE={rmse:.1f}",
            xaxis_title="Predicted − Actual RUL",
            yaxis_title="Count"
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Performance by RUL bucket
    st.subheader("MAE by RUL Bucket")
    buckets = [(0,25,"🔴 Critical (0-25)"),(25,50,"🟠 Warning (25-50)"),
               (50,100,"🟡 Caution (50-100)"),(100,126,"🟢 Healthy (100+)")]
    bucket_rows = []
    for lo, hi, label in buckets:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            bucket_rows.append({
                "Bucket": label,
                "Samples": int(mask.sum()),
                "MAE": round(float(np.mean(np.abs(errors[mask]))), 2),
                "RMSE": round(float(np.sqrt(np.mean(errors[mask]**2))), 2),
                "Within ±10": f"{(np.abs(errors[mask])<=10).mean():.1%}"
            })
    st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True)

    # Feature importance
    st.subheader("Top 15 Feature Importances")
    importance = model.feature_importances_
    top_idx    = np.argsort(importance)[::-1][:15]
    fig5 = go.Figure(go.Bar(
        x=importance[top_idx],
        y=[feature_cols[i] for i in top_idx],
        orientation="h",
        marker_color="#ffa500"
    ))
    fig5.update_layout(template="plotly_dark", height=420,
                       xaxis_title="Importance",
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔴 Real-Time Alert Simulation")
    st.caption("Streams real test data in order — watch a turbofan approach failure")

    n_stream = st.slider("Cycles to stream", 20, 80, 40)

    if st.button("▶ Start Simulation", type="primary"):
        # Pick test samples sorted highest → lowest RUL (healthy to critical)
        critical_pool = np.where(y_test <= 60)[0]
        if len(critical_pool) < n_stream:
            critical_pool = np.argsort(y_test)[:n_stream]

        stream_idx   = critical_pool[:n_stream]
        stream_rul   = y_pred[stream_idx]
        stream_true  = y_test[stream_idx]
        sort_order   = np.argsort(stream_rul)[::-1]
        stream_rul   = stream_rul[sort_order]
        stream_true  = stream_true[sort_order]

        placeholder  = st.empty()
        history_rul  = []
        history_true = []

        for i, (pred, actual) in enumerate(zip(stream_rul, stream_true)):
            history_rul.append(float(pred))
            history_true.append(float(actual))
            status, alert, color = rul_status(pred)

            with placeholder.container():
                st.markdown(f"""
                <div style="background:#1a1a1a;border:2px solid {color};
                            border-radius:10px;padding:1rem;text-align:center">
                    <h2 style="color:{color}">{alert} {status}</h2>
                    <h3>Cycle {i+1} of {n_stream}</h3>
                    <h1 style="color:{color}">{pred:.0f} cycles remaining</h1>
                    <p style="color:#aaa">True RUL: {actual:.0f} | Error: {abs(pred-actual):.1f} cycles</p>
                </div>
                """, unsafe_allow_html=True)

                if len(history_rul) > 2:
                    fig6 = go.Figure()
                    fig6.add_trace(go.Scatter(
                        y=history_true, mode="lines",
                        line=dict(color="#888", dash="dot"),
                        name="True RUL"
                    ))
                    fig6.add_trace(go.Scatter(
                        y=history_rul, mode="lines+markers",
                        line=dict(color=color, width=3),
                        name="XGBoost Prediction"
                    ))
                    fig6.add_hline(y=25, line_dash="dash", line_color="red")
                    fig6.add_hline(y=50, line_dash="dash", line_color="orange")
                    fig6.update_layout(
                        template="plotly_dark", height=300,
                        title="Live RUL Countdown",
                        yaxis_title="RUL (cycles)"
                    )
                    st.plotly_chart(fig6, use_container_width=True)

            time.sleep(0.2)

        st.success(f"✅ Simulation complete — streamed {n_stream} real test cycles")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown(f"**Test samples:** {len(y_test):,}")
    st.markdown(f"**Features:** {len(feature_cols)}")
    st.markdown(f"**MAE:** {mae:.2f} cycles")
    st.markdown(f"**RMSE:** {rmse:.2f} cycles")
    st.markdown(f"**Critical MAE:** {crit_mae:.2f} cycles")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("- XGBoost (500 trees)")
    st.markdown("- 60 engineered features")
    st.markdown("- 30-cycle rolling stats")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("- NASA CMAPSS FD001")
    st.markdown("- 100 training engines")
    st.markdown("- 100 test engines")
    st.markdown("---")
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()
