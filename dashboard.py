import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

API_URL = "http://localhost:8002"

st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:700; color:#ffa500; }
    .sub-header  { color:#888; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚙️ Predictive Maintenance IoT Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time RUL prediction from turbofan engine sensors | XGBoost + Rolling Window Features | NASA CMAPSS Dataset</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Fleet Configuration")
num_engines  = st.sidebar.slider("Number of engines to simulate", 3, 8, 5)
start_cycles = st.sidebar.slider("Starting cycle", 1, 300, 50)
deg_rate     = st.sidebar.slider("Degradation rate", 0.1, 2.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance**")
st.sidebar.markdown("- MAE: 7.68 cycles")
st.sidebar.markdown("- RMSE: 11.55 cycles")
st.sidebar.markdown("- Critical zone MAE: 3.30 cycles")
st.sidebar.markdown("- Dataset: NASA CMAPSS FD001")

# ── Engine base sensor values ─────────────────────────────────────────────────
BASE_SENSORS = {
    "s2":642.0,"s3":1590.0,"s4":1400.0,"s7":554.0,
    "s8":2388.0,"s9":9065.0,"s11":47.0,"s12":521.0,
    "s13":2388.0,"s14":8138.0,"s15":8.4,"s17":392.0,
    "s20":39.0,"s21":23.0
}

DEGRADATION_DIRECTION = {
    "s2":-1,"s3":+1,"s4":+1,"s7":-1,
    "s8":0,"s9":-1,"s11":+1,"s12":+1,
    "s13":0,"s14":-1,"s15":+1,"s17":-1,
    "s20":-1,"s21":+1
}

def get_sensor_values(cycle, engine_offset=0, deg=0.5):
    sensors = {}
    for s, base in BASE_SENSORS.items():
        direction = DEGRADATION_DIRECTION[s]
        noise     = np.random.normal(0, abs(base) * 0.001)
        drift     = direction * (cycle / 300) * abs(base) * 0.02 * deg
        sensors[s] = round(base + drift + noise + engine_offset, 2)
    return sensors

def predict_rul(engine_id, cycle, sensors):
    payload = {"engine_id": engine_id, "cycle": cycle}
    payload.update(sensors)
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        return r.json()
    except:
        return None

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏭 Fleet Overview", "📈 Engine Deep Dive", "🔴 Alert Simulation"])

# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Fleet Health Overview")

    if st.button("🔄 Refresh Fleet Status", type="primary"):
        fleet_data = []
        for i in range(1, num_engines + 1):
            eng_id  = f"ENG-{i:03d}"
            cycle   = start_cycles + (i * 30)
            offset  = np.random.uniform(-2, 2)
            sensors = get_sensor_values(cycle, offset, deg_rate)
            result  = predict_rul(eng_id, cycle, sensors)
            if result:
                fleet_data.append({
                    "Engine":     eng_id,
                    "Cycle":      cycle,
                    "RUL":        result["rul_predicted"],
                    "Status":     result["status"],
                    "Alert":      result["alert_level"],
                    "Action":     result["recommendation"]
                })

        df = pd.DataFrame(fleet_data)

        # KPI row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Engines",   len(df))
        col2.metric("Critical",        len(df[df["Status"]=="CRITICAL"]),
                    delta=None if len(df[df["Status"]=="CRITICAL"])==0 else "⚠️")
        col3.metric("Avg RUL",         f"{df['RUL'].mean():.0f} cycles")
        col4.metric("Min RUL",         f"{df['RUL'].min():.0f} cycles")

        st.markdown("---")

        # Fleet RUL bar chart
        colors = []
        for s in df["Status"]:
            if s == "CRITICAL": colors.append("#ff4444")
            elif s == "WARNING": colors.append("#ff8c00")
            elif s == "CAUTION": colors.append("#ffd700")
            else: colors.append("#00d4aa")

        fig = go.Figure(go.Bar(
            x=df["Engine"], y=df["RUL"],
            marker_color=colors,
            text=df["Status"],
            textposition="outside"
        ))
        fig.add_hline(y=25,  line_dash="dash", line_color="red",
                      annotation_text="Critical threshold (25)")
        fig.add_hline(y=50,  line_dash="dash", line_color="orange",
                      annotation_text="Warning threshold (50)")
        fig.update_layout(
            title="Fleet RUL Status",
            xaxis_title="Engine ID",
            yaxis_title="Remaining Useful Life (cycles)",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Single Engine — Degradation Trajectory")

    col1, col2 = st.columns(2)
    with col1:
        eng_id    = st.text_input("Engine ID", value="ENG-001")
        max_cycle = st.slider("Simulate up to cycle", 50, 350, 250)

    if st.button("📊 Plot Degradation Trajectory"):
        trajectory = []
        with st.spinner("Simulating engine lifecycle..."):
            for cycle in range(10, max_cycle, 5):
                sensors = get_sensor_values(cycle, deg=deg_rate)
                result  = predict_rul(eng_id, cycle, sensors)
                if result:
                    trajectory.append({
                        "cycle": cycle,
                        "rul":   result["rul_predicted"],
                        "status": result["status"]
                    })

        df_traj = pd.DataFrame(trajectory)

        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=("Predicted RUL Over Lifecycle",
                                            "Key Sensor Trends"))

        colors_map = {"HEALTHY":"#00d4aa","CAUTION":"#ffd700",
                      "WARNING":"#ff8c00","CRITICAL":"#ff4444"}
        for status, grp in df_traj.groupby("status"):
            fig.add_trace(go.Scatter(
                x=grp["cycle"], y=grp["rul"],
                mode="markers", name=status,
                marker=dict(color=colors_map.get(status,"gray"), size=6)
            ), row=1, col=1)

        fig.add_hline(y=25, line_dash="dash", line_color="red",   row=1, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="orange", row=1, col=1)

        # Sensor trends
        cycles = list(range(10, max_cycle, 5))
        s2_vals = [get_sensor_values(c, deg=deg_rate)["s2"] for c in cycles]
        s11_vals = [get_sensor_values(c, deg=deg_rate)["s11"] for c in cycles]

        fig.add_trace(go.Scatter(x=cycles, y=s2_vals, name="s2 (Fan Speed)",
                                 line=dict(color="#00d4aa")), row=2, col=1)
        fig.add_trace(go.Scatter(x=cycles, y=s11_vals, name="s11 (Pressure)",
                                 line=dict(color="#ffa500")), row=2, col=1)

        fig.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔴 Critical Failure Alert Simulation")
    st.caption("Simulates an engine approaching failure in real-time")

    eng_id = st.text_input("Engine to monitor", value="ENG-CRITICAL")

    if st.button("▶ Start Alert Simulation", type="primary"):
        placeholder = st.empty()
        alert_log   = []

        for cycle in range(280, 361, 5):
            sensors = get_sensor_values(cycle, deg=2.0)
            result  = predict_rul(eng_id, cycle, sensors)

            if result:
                alert_log.append({
                    "Cycle": cycle,
                    "RUL":   result["rul_predicted"],
                    "Status": result["status"]
                })

                with placeholder.container():
                    rul    = result["rul_predicted"]
                    status = result["status"]
                    color  = "#ff4444" if status=="CRITICAL" else \
                             "#ff8c00" if status=="WARNING" else "#ffd700"

                    st.markdown(f"""
                    <div style="background:#1a1a1a;border:2px solid {color};
                                border-radius:10px;padding:1rem;text-align:center">
                        <h2 style="color:{color}">{result['alert_level']} {status}</h2>
                        <h3>Engine: {eng_id} | Cycle: {cycle}</h3>
                        <h1 style="color:{color}">{rul:.0f} cycles remaining</h1>
                        <p>{result['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if len(alert_log) > 1:
                        df_log = pd.DataFrame(alert_log)
                        fig = go.Figure(go.Scatter(
                            x=df_log["Cycle"], y=df_log["RUL"],
                            mode="lines+markers",
                            line=dict(color=color, width=3)
                        ))
                        fig.add_hline(y=25, line_dash="dash",
                                      line_color="red",
                                      annotation_text="CRITICAL")
                        fig.update_layout(
                            template="plotly_dark", height=300,
                            title="Live RUL Countdown",
                            xaxis_title="Cycle",
                            yaxis_title="RUL"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            time.sleep(0.3)

        st.error("🚨 Simulation complete — engine reached end of life")
