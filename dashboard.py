import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, os, time

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
st.markdown('<h1 style="color:#ffa500">⚙️ Predictive Maintenance IoT Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#888">XGBoost RUL predictor · NASA CMAPSS · Real engine trajectories</p>', unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    model        = joblib.load(os.path.join(BASE, "xgb_rul.pkl"))
    scaler       = joblib.load(os.path.join(BASE, "scaler_rul.pkl"))
    feature_cols = joblib.load(os.path.join(BASE, "feature_cols.pkl"))
    return model, scaler, feature_cols

@st.cache_data
def load_and_predict():
    # X_test.npy is pre-scaled by features.py — do NOT apply scaler again
    X_test  = np.load(os.path.join(BASE, "X_test.npy"))
    y_test  = np.load(os.path.join(BASE, "y_test.npy"))
    model, scaler, feature_cols = load_model()
    y_pred  = np.clip(model.predict(X_test), 0, 125)
    return y_test, y_pred

@st.cache_data
def get_engine_trajectories():
    """Split test data into individual engine trajectories."""
    y_test, y_pred = load_and_predict()
    # Engine boundary = where RUL increases (new engine starts)
    boundaries = np.where(np.diff(y_test) > 10)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(y_test)]])
    engines = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if end - start < 10:
            continue
        true_rul = y_test[start:end]
        pred_rul = y_pred[start:end]
        # Only include engines that actually degrade (reach below 80)
        engines.append({
            "id":       f"ENG-{i+1:03d}",
            "true":     true_rul,
            "pred":     pred_rul,
            "length":   end - start,
            "min_true": true_rul.min(),
            "final_pred": pred_rul[-1],
        })
    return engines

model, scaler, feature_cols = load_model()
y_test, y_pred = load_and_predict()
engines = get_engine_trajectories()

# Sort engines: most degraded first
engines_sorted = sorted(engines, key=lambda e: e["final_pred"])

errors   = y_pred - y_test
mae      = np.mean(np.abs(errors))
rmse     = np.sqrt(np.mean(errors**2))
crit_mae = np.mean(np.abs(errors[y_test <= 25]))

def rul_status(rul):
    if rul <= 25:  return "CRITICAL", "🔴", "#ff4444"
    if rul <= 50:  return "WARNING",  "🟠", "#ff8c00"
    if rul <= 100: return "CAUTION",  "🟡", "#ffd700"
    return "HEALTHY", "🟢", "#00d4aa"

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Test Samples",      f"{len(y_test):,}")
k2.metric("MAE",               f"{mae:.2f} cycles")
k3.metric("RMSE",              f"{rmse:.2f} cycles")
k4.metric("Critical MAE",      f"{crit_mae:.2f} cycles", "RUL ≤ 25")
k5.metric("Within ±10 cycles", f"{(np.abs(errors)<=10).mean():.1%}")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏭 Fleet Overview", "📈 Engine Trajectory",
    "📊 Model Performance", "🔴 Alert Simulation"
])

# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Fleet Health — Final Predicted RUL per Engine")
    n_show = st.slider("Engines to display", 5, min(30, len(engines)), 15)
    fleet  = []
    for i, eng in enumerate(engines_sorted[:n_show]):
        status, alert, color = rul_status(eng["final_pred"])
        fleet.append({
            "Engine":     eng["id"],
            "Cycles":     eng["length"],
            "True RUL":   round(float(eng["true"][-1]), 1),
            "Pred RUL":   round(float(eng["final_pred"]), 1),
            "Min True":   round(float(eng["min_true"]), 1),
            "Status":     status,
            "color":      color
        })
    df_fleet = pd.DataFrame(fleet)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engines shown", len(df_fleet))
    c2.metric("Critical",      len(df_fleet[df_fleet["Status"]=="CRITICAL"]))
    c3.metric("Warning",       len(df_fleet[df_fleet["Status"]=="WARNING"]))
    c4.metric("Avg Pred RUL",  f"{df_fleet['Pred RUL'].mean():.0f}")

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df_fleet["Engine"], y=df_fleet["Pred RUL"],
        marker_color=df_fleet["color"],
        text=df_fleet["Status"], textposition="outside",
        name="Predicted RUL"
    ))
    fig1.add_trace(go.Scatter(
        x=df_fleet["Engine"], y=df_fleet["True RUL"],
        mode="markers",
        marker=dict(symbol="diamond", size=10, color="white"),
        name="True RUL"
    ))
    fig1.add_hline(y=25, line_dash="dash", line_color="red",    annotation_text="Critical (25)")
    fig1.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Warning (50)")
    fig1.update_layout(template="plotly_dark", height=420,
                       yaxis_title="Remaining Useful Life (cycles)",
                       legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig1, use_container_width=True)
    st.dataframe(df_fleet.drop(columns=["color"]), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Single Engine — Full Degradation Trajectory")

    eng_names = [e["id"] for e in engines]
    selected  = st.selectbox("Select engine", eng_names)
    eng_data  = next(e for e in engines if e["id"] == selected)

    true_rul = eng_data["true"]
    pred_rul = eng_data["pred"]
    cycles   = np.arange(1, len(true_rul) + 1)

    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=("RUL Over Engine Lifecycle", "Prediction Error"),
        row_heights=[0.65, 0.35]
    )
    fig2.add_trace(go.Scatter(
        x=cycles, y=true_rul, mode="lines",
        name="Actual RUL", line=dict(color="#888", width=2, dash="dot")
    ), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=cycles, y=pred_rul, mode="lines",
        name="XGBoost Prediction", line=dict(color="#ffa500", width=2)
    ), row=1, col=1)
    fig2.add_hline(y=25, line_dash="dash", line_color="red",    row=1, col=1,
                   annotation_text="Critical")
    fig2.add_hline(y=50, line_dash="dash", line_color="orange", row=1, col=1,
                   annotation_text="Warning")

    err = pred_rul - true_rul
    fig2.add_trace(go.Bar(
        x=cycles, y=err,
        marker_color=["#ff4444" if e > 0 else "#00d4aa" for e in err],
        name="Error"
    ), row=2, col=1)
    fig2.add_hline(y=0, line_color="white", row=2, col=1)

    fig2.update_layout(template="plotly_dark", height=560,
                       hovermode="x unified",
                       legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

    final_status, alert, color = rul_status(pred_rul[-1])
    st.markdown(f"""
    <div style="background:#1a1a1a;border:2px solid {color};
                border-radius:10px;padding:1rem;text-align:center">
        <h3 style="color:{color}">{alert} {final_status} — Final RUL: {pred_rul[-1]:.0f} cycles</h3>
        <p style="color:#aaa">True RUL: {true_rul[-1]:.0f} | Error: {abs(pred_rul[-1]-true_rul[-1]):.1f} cycles</p>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance — Real Test Predictions")

    col1, col2 = st.columns(2)
    with col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color="#ffa500", size=3, opacity=0.3)
        ))
        fig3.add_trace(go.Scatter(
            x=[0,125], y=[0,125], mode="lines",
            line=dict(color="red", dash="dash"), name="Perfect"
        ))
        fig3.update_layout(template="plotly_dark", height=380,
                           title=f"Predicted vs Actual | MAE={mae:.1f}",
                           xaxis_title="Actual RUL", yaxis_title="Predicted RUL")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = go.Figure(go.Histogram(
            x=errors, nbinsx=60,
            marker_color="#ffa500", opacity=0.8
        ))
        fig4.add_vline(x=0, line_dash="dash", line_color="red")
        fig4.update_layout(template="plotly_dark", height=380,
                           title=f"Error Distribution | RMSE={rmse:.1f}",
                           xaxis_title="Predicted − Actual", yaxis_title="Count")
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("MAE by RUL Bucket")
    buckets = [(0,25,"🔴 Critical"),(25,50,"🟠 Warning"),
               (50,100,"🟡 Caution"),(100,126,"🟢 Healthy")]
    rows = []
    for lo, hi, label in buckets:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            rows.append({
                "Bucket": label, "Samples": int(mask.sum()),
                "MAE": round(float(np.mean(np.abs(errors[mask]))), 2),
                "RMSE": round(float(np.sqrt(np.mean(errors[mask]**2))), 2),
                "Within ±10": f"{(np.abs(errors[mask])<=10).mean():.1%}"
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Top 15 Feature Importances")
    imp     = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:15]
    fig5 = go.Figure(go.Bar(
        x=imp[top_idx],
        y=[feature_cols[i] for i in top_idx],
        orientation="h", marker_color="#ffa500"
    ))
    fig5.update_layout(template="plotly_dark", height=420,
                       xaxis_title="Importance",
                       yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔴 Real-Time Alert Simulation — Single Engine Countdown")

    # Only show engines that go critical (reach below 30 cycles)
    critical_engines = [e for e in engines if e["min_true"] <= 30]
    if not critical_engines:
        critical_engines = engines

    eng_names_crit = [e["id"] for e in critical_engines]
    sel_eng = st.selectbox("Select engine to monitor", eng_names_crit, key="alert_eng")
    eng     = next(e for e in critical_engines if e["id"] == sel_eng)

    speed = st.slider("Playback speed", 1, 10, 5)
    st.caption(f"Engine length: {eng['length']} cycles | Final true RUL: {eng['true'][-1]:.0f}")

    if st.button("▶ Start Real-Time Simulation", type="primary"):
        true_rul = eng["true"]
        pred_rul = eng["pred"]

        placeholder  = st.empty()
        hist_cycles  = []
        hist_pred    = []
        hist_true    = []

        for i in range(len(pred_rul)):
            rul    = float(pred_rul[i])
            actual = float(true_rul[i])
            status, alert, color = rul_status(rul)

            hist_cycles.append(i + 1)
            hist_pred.append(rul)
            hist_true.append(actual)

            with placeholder.container():
                st.markdown(f"""
                <div style="background:#1a1a1a;border:2px solid {color};
                            border-radius:10px;padding:1rem;text-align:center">
                    <h2 style="color:{color}">{alert} {status}</h2>
                    <h3>{sel_eng} · Cycle {i+1} of {eng['length']}</h3>
                    <h1 style="color:{color}">{rul:.0f} cycles predicted remaining</h1>
                    <p style="color:#aaa">True RUL: {actual:.0f} · Error: {abs(rul-actual):.1f}</p>
                </div>
                """, unsafe_allow_html=True)

                if len(hist_cycles) > 3:
                    fig6 = go.Figure()
                    fig6.add_trace(go.Scatter(
                        y=hist_true, x=hist_cycles, mode="lines",
                        line=dict(color="#888", dash="dot"), name="True RUL"
                    ))
                    fig6.add_trace(go.Scatter(
                        y=hist_pred, x=hist_cycles, mode="lines",
                        line=dict(color=color, width=3), name="XGBoost"
                    ))
                    fig6.add_hline(y=25, line_dash="dash", line_color="red")
                    fig6.add_hline(y=50, line_dash="dash", line_color="orange")
                    fig6.update_layout(
                        template="plotly_dark", height=300,
                        title="Live RUL Countdown",
                        yaxis_title="RUL (cycles)", xaxis_title="Cycle"
                    )
                    st.plotly_chart(fig6, use_container_width=True)

            time.sleep(1.0 / speed)

        st.success(f"✅ Engine {sel_eng} simulation complete")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown(f"**Test samples:** {len(y_test):,}")
    st.markdown(f"**Engines tracked:** {len(engines)}")
    st.markdown(f"**MAE:** {mae:.2f} cycles")
    st.markdown(f"**RMSE:** {rmse:.2f} cycles")
    st.markdown(f"**Critical MAE:** {crit_mae:.2f} cycles")
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("- XGBoost (500 trees)")
    st.markdown("- Sample-weighted training")
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
