
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import json, urllib.request, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CO₂ Digital Twin — India",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0f172a}
[data-testid="stToolbar"]{display:none!important}
#MainMenu{display:none!important}
footer{display:none!important}
[data-testid="stDecoration"]{display:none!important}
a[href*="github"]{display:none!important}
button[title="View app on GitHub"]{display:none!important}
[data-testid="stHeader"]{background:transparent}
.block-container{padding:1rem 1.5rem}
.metric-card{background:#1e293b;border-radius:12px;padding:14px 18px;
             border:1px solid #334155;text-align:center;margin-bottom:8px}
.metric-val{font-size:26px;font-weight:700;margin:0;line-height:1.2}
.metric-lbl{font-size:10px;color:#94a3b8;text-transform:uppercase;
            letter-spacing:.08em;margin:2px 0 0}
.pill{display:inline-block;padding:4px 16px;border-radius:20px;
      font-size:12px;font-weight:700;letter-spacing:.06em;margin-top:6px}
.pill-green{background:#064e3b;color:#10b981}
.pill-yellow{background:#451a03;color:#f59e0b}
.pill-red{background:#4c0519;color:#f43f5e}
.section-title{font-size:10px;font-weight:700;color:#475569;
               text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label{color:#cbd5e1!important;font-size:12px!important}
h1,h2,h3{color:#f1f5f9!important}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODELS (cached forever) ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    model  = joblib.load("models/best_model_catboost.pkl")
    le_st  = joblib.load("models/label_encoder_state.pkl")
    le_reg = joblib.load("models/label_encoder_region.pkl")
    return model, le_st, le_reg

@st.cache_data
def load_data():
    return pd.read_csv("data/master_dataset_final.csv")

@st.cache_data
def load_geojson():
    try:
        with open("data/india_states.geojson", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"GeoJSON load error: {e}")
        return None

model, le_st, le_reg = load_models()
df = load_data()
geojson = load_geojson()

FEATURES = ["State_enc","Region_enc","Year","Energy_Use_TWh","RE_Generation_TWh",
            "Non_RE_Energy_TWh","RE_Fraction","GDP_BillionINR","Urbanization_Percent",
            "CO2_per_TWh","RE_vs_NonRE_Ratio","Energy_per_GDP"]

STATE_REGION = df[["State","Region"]].drop_duplicates().set_index("State")["Region"].to_dict()

_df_enc = df.copy()
_df_enc["State_enc"]  = le_st.transform(_df_enc["State"])
_df_enc["Region_enc"] = le_reg.transform(_df_enc["Region"])
latest_enc = _df_enc[_df_enc["Year"] == 2024].set_index("State")
STATES = sorted(df["State"].unique().tolist())

GEOJSON_NAME_MAP = {
    "Odisha": "Orissa",
    "Uttarakhand": "Uttaranchal",
}

STATE_COORDS = {
    "Andhra Pradesh":(15.9,79.7),"Arunachal Pradesh":(28.2,94.7),
    "Assam":(26.2,92.9),"Bihar":(25.1,85.3),"Chhattisgarh":(21.3,81.9),
    "Goa":(15.3,74.0),"Gujarat":(22.3,71.2),"Haryana":(29.1,76.1),
    "Himachal Pradesh":(31.1,77.2),"Jharkhand":(23.6,85.3),
    "Karnataka":(15.3,75.7),"Kerala":(10.9,76.3),
    "Madhya Pradesh":(23.5,77.4),"Maharashtra":(19.7,75.7),
    "Manipur":(24.7,93.9),"Meghalaya":(25.5,91.4),"Mizoram":(23.2,92.8),
    "Nagaland":(26.2,94.6),"Odisha":(20.9,85.1),"Punjab":(31.1,75.3),
    "Rajasthan":(27.0,74.2),"Sikkim":(27.5,88.5),"Tamil Nadu":(11.1,78.7),
    "Telangana":(17.9,79.4),"Tripura":(23.9,91.9),
    "Uttar Pradesh":(26.8,80.9),"Uttarakhand":(30.1,79.3),
    "West Bengal":(22.9,87.9)
}

COLOR_MAP = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#f43f5e"}

# ── CORE PREDICT (single state, fast) ────────────────────────────────────────
def predict_co2(state, year, energy_use, re_fraction, gdp, urban, co2_per_twh_base):
    re_gen = energy_use * re_fraction
    non_re = max(energy_use - re_gen, 0.001)
    region = STATE_REGION.get(state, "WR")
    row = pd.DataFrame([[
        int(le_st.transform([state])[0]),
        int(le_reg.transform([region])[0]),
        year, energy_use, re_gen, non_re, re_fraction,
        gdp, urban, co2_per_twh_base,
        re_gen/(non_re+0.001), energy_use/max(gdp,0.001)
    ]], columns=FEATURES)
    return max(float(model.predict(row)[0]), 0)

def co2_color(val):
    if val < 100:  return "#10b981", "LOW",    "pill-green"
    if val < 200:  return "#f59e0b", "MEDIUM", "pill-yellow"
    return             "#f43f5e", "HIGH",   "pill-red"

# ── CACHE BASELINE MAP (computed once, never again) ──────────────────────────
@st.cache_data
def get_baseline_map():
    rows = []
    for state in STATES:
        if state not in latest_enc.index:
            continue
        r = latest_enc.loc[state]
        pred = predict_co2(state, 2024,
                           float(r["Energy_Use_TWh"]),
                           float(r["RE_Fraction"]),
                           float(r["GDP_BillionINR"]),
                           float(r["Urbanization_Percent"]),
                           float(r["CO2_per_TWh"]))
        c, status, _ = co2_color(pred)
        rows.append({
            "State": state,
            "GeoName": GEOJSON_NAME_MAP.get(state, state),
            "CO2": round(pred, 1),
            "Status": status,
            "color": c,
            "RE_pct": round(float(r["RE_Fraction"])*100, 1),
            "Energy": round(float(r["Energy_Use_TWh"]), 1),
            "lat": STATE_COORDS.get(state,(20,80))[0],
            "lon": STATE_COORDS.get(state,(20,80))[1],
        })
    return pd.DataFrame(rows)

# ── CACHE SENSITIVITY (per state, cached) ────────────────────────────────────
@st.cache_data
def get_sensitivity(state, energy_use, gdp, urban, co2_per_twh_base):
    re_range = list(range(5, 81, 5))
    vals = [predict_co2(state, 2024, energy_use, r/100, gdp, urban, co2_per_twh_base)
            for r in re_range]
    return re_range, vals

@st.cache_data
def get_scenarios(state, year, energy_use, gdp, urban, co2_per_twh_base):
    scen = {"Current": None, "20%": 20, "40%": 40, "60%": 60, "80%": 80}
    return scen

# ── CACHE MAP FIGURE (only rebuilds when selected state changes) ──────────────
@st.cache_data
def build_map_figure(selected_state, selected_co2, selected_re, selected_energy, _geojson):
    base_df = get_baseline_map().copy()
    # Update selected state row
    mask = base_df["State"] == selected_state
    base_df.loc[mask, "CO2"]    = round(selected_co2, 1)
    base_df.loc[mask, "RE_pct"] = selected_re
    base_df.loc[mask, "Energy"] = selected_energy
    _, status, _ = co2_color(selected_co2)
    base_df.loc[mask, "Status"] = status

    if _geojson:
        fig = px.choropleth(
            base_df,
            geojson=_geojson,
            locations="GeoName",
            featureidkey="properties.NAME_1",
            color="Status",
            color_discrete_map=COLOR_MAP,
            hover_name="State",
            hover_data={
                "CO2": True, "RE_pct": True, "Energy": True,
                "Status": False, "color": False,
                "GeoName": False, "lat": False, "lon": False
            },
            labels={"CO2":"CO₂ (MtCO₂)","RE_pct":"RE %","Energy":"Energy (TWh)"}
        )
        # White border on selected state
        sel_geo  = GEOJSON_NAME_MAP.get(selected_state, selected_state)
        sel_feat = [f for f in _geojson["features"]
                    if f["properties"]["NAME_1"] == sel_geo]
        if sel_feat:
            fig.add_trace(go.Choropleth(
                geojson={"type":"FeatureCollection","features":sel_feat},
                locations=[sel_geo],
                featureidkey="properties.NAME_1",
                z=[1],
                colorscale=[[0,"rgba(0,0,0,0)"],[1,"rgba(0,0,0,0)"]],
                showscale=False,
                marker=dict(line=dict(color="white", width=3)),
                hoverinfo="skip"
            ))
        fig.update_geos(fitbounds="locations", visible=False, bgcolor="#0f172a")
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            legend=dict(
                title="CO₂ Level", font=dict(color="#cbd5e1", size=11),
                bgcolor="#1e293b", bordercolor="#334155", borderwidth=1,
                orientation="h", yanchor="bottom", y=0.01,
                xanchor="center", x=0.5
            ),
            height=440
        )
    else:
        fig = go.Figure()
        for _, row in base_df.iterrows():
            is_sel = row["State"] == selected_state
            fig.add_trace(go.Scattergeo(
                lat=[row["lat"]], lon=[row["lon"]],
                mode="markers+text",
                marker=dict(size=22 if is_sel else 14, color=row["color"],
                            line=dict(width=3 if is_sel else 0, color="white")),
                text=row["State"][:3],
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                hovertemplate=(f"<b>{row['State']}</b><br>CO₂: {row['CO2']} MtCO₂<br>"
                               f"Status: {row['Status']}<br>RE: {row['RE_pct']}%<extra></extra>"),
                showlegend=False
            ))
        fig.update_layout(
            geo=dict(scope="asia", center=dict(lat=22, lon=80), projection_scale=4.8,
                     bgcolor="#0f172a", lakecolor="#0f172a", landcolor="#1e293b",
                     showland=True, showcountries=True, countrycolor="#475569",
                     showocean=True, oceancolor="#0f172a"),
            paper_bgcolor="#0f172a", margin={"r":0,"t":0,"l":0,"b":0}, height=440
        )
    return fig

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;
            padding:12px 20px;margin-bottom:14px;display:flex;
            align-items:center;justify-content:space-between">
  <div>
    <span style="font-size:18px;font-weight:700;color:#f1f5f9">
      🌍 CO₂ Emission Digital Twin — India
    </span>
    <span style="font-size:11px;color:#64748b;margin-left:10px">
      State-Level Forecasting · CatBoost Model · R² = 0.9973
    </span>
  </div>
  <span style="font-size:11px;color:#10b981;font-weight:600;background:#064e3b;
               padding:4px 12px;border-radius:20px">● LIVE PREDICTIONS</span>
</div>
""", unsafe_allow_html=True)

# ── LAYOUT ───────────────────────────────────────────────────────────────────
left, mid, right = st.columns([1.05, 1.9, 1.05])

with left:
    st.markdown('<div class="section-title">🎛️ Control Panel</div>', unsafe_allow_html=True)
    selected_state = st.selectbox("Select State", STATES, index=STATES.index("Maharashtra"))
    base = latest_enc.loc[selected_state]
    year       = st.slider("Forecast Year", 2024, 2035, 2024)
    re_pct     = st.slider("Renewable Energy %", 1, 80,
                            int(round(float(base["RE_Fraction"])*100)), step=1)
    energy_use = st.slider("Energy Use (TWh)", 20, 500,
                            int(round(float(base["Energy_Use_TWh"]))), step=5)

    re_fraction   = re_pct / 100.0
    gdp           = float(base["GDP_BillionINR"])
    urban         = float(base["Urbanization_Percent"])
    co2_per_twh_b = float(base["CO2_per_TWh"])

    predicted = predict_co2(selected_state, year, energy_use,
                             re_fraction, gdp, urban, co2_per_twh_b)

    baseline_pred = predict_co2(selected_state, 2024,
                                 float(base["Energy_Use_TWh"]),
                                 float(base["RE_Fraction"]),
                                 gdp, urban, co2_per_twh_b)
    pct_change  = ((predicted - baseline_pred) / max(baseline_pred, 0.001)) * 100
    color_hex, status, pill_cls = co2_color(predicted)
    non_re_pct  = 100 - re_pct

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Prediction Output</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="border-color:{color_hex}55">
      <p class="metric-lbl">Predicted CO₂ Emissions</p>
      <p class="metric-val" style="color:{color_hex}">{predicted:.1f}</p>
      <p class="metric-lbl">MtCO₂</p>
      <span class="pill {pill_cls}">{status} EMISSIONS</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        chg_col = "#10b981" if pct_change <= 0 else "#f43f5e"
        arrow   = "▼" if pct_change <= 0 else "▲"
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-lbl">Change</p>
          <p class="metric-val" style="color:{chg_col};font-size:20px">
            {arrow} {abs(pct_change):.1f}%</p>
          <p class="metric-lbl">vs 2024 base</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <p class="metric-lbl">RE Share</p>
          <p class="metric-val" style="color:#2563eb;font-size:20px">{re_pct}%</p>
          <p class="metric-lbl">renewable</p>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:10px">
      <div style="display:flex;justify-content:space-between;
                  font-size:10px;color:#94a3b8;margin-bottom:4px">
        <span>🟢 RE {re_pct}%</span><span>🔴 Non-RE {non_re_pct}%</span>
      </div>
      <div style="height:8px;border-radius:4px;background:#0f172a;overflow:hidden;display:flex">
        <div style="width:{re_pct}%;background:#10b981"></div>
        <div style="flex:1;background:#f43f5e"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">State Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#1e293b;border-radius:8px;padding:10px 12px;
                border:1px solid #334155;font-size:11px;color:#94a3b8;line-height:1.8">
      <b style="color:#cbd5e1">{selected_state}</b><br>
      Region: {STATE_REGION.get(selected_state,"—")}<br>
      GDP: ₹{gdp:,.0f}B INR<br>
      Urbanization: {urban:.1f}%<br>
      Actual CO₂ (2024): {float(base["Carbon_Emissions_MtCO2"]):.1f} MtCO₂
    </div>
    """, unsafe_allow_html=True)

    # ── DOWNLOAD BUTTON ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📥 Download</div>', unsafe_allow_html=True)
    base_map = get_baseline_map().copy()
    mask = base_map["State"] == selected_state
    base_map.loc[mask, "CO2"] = round(predicted, 1)
    _, st_lbl, _ = co2_color(predicted)
    base_map.loc[mask, "Status"] = st_lbl
    csv = base_map[["State","CO2","Status","RE_pct","Energy"]].to_csv(index=False)
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv,
        file_name="co2_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

with mid:
    st.markdown('<div class="section-title">🗺️ India CO₂ Map — All States</div>',
                unsafe_allow_html=True)
    fig_map = build_map_figure(
        selected_state, predicted, re_pct, energy_use, geojson
    )
    st.plotly_chart(fig_map, use_container_width=True)
    st.markdown("""
    <div style="display:flex;gap:20px;justify-content:center;margin-top:2px">
      <span style="font-size:11px;color:#10b981">● LOW &lt; 100 MtCO₂</span>
      <span style="font-size:11px;color:#f59e0b">● MEDIUM 100–200 MtCO₂</span>
      <span style="font-size:11px;color:#f43f5e">● HIGH &gt; 200 MtCO₂</span>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown(f'<div class="section-title">📉 RE Sensitivity — {selected_state}</div>',
                unsafe_allow_html=True)
    re_range, sens_vals = get_sensitivity(
        selected_state, energy_use, gdp, urban, co2_per_twh_b
    )
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(
        x=re_range, y=sens_vals, mode="lines+markers",
        line=dict(color="#2563eb", width=2), marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.12)"
    ))
    fig_s.add_vline(x=re_pct, line_color=color_hex, line_dash="dash",
                    annotation_text=f"{re_pct}%",
                    annotation_font_color=color_hex, annotation_font_size=10)
    fig_s.update_layout(
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=9),
        margin=dict(l=8,r=8,t=8,b=28), height=155,
        xaxis=dict(title="RE %", gridcolor="#334155"),
        yaxis=dict(title="CO₂", gridcolor="#334155"),
        showlegend=False
    )
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:6px">🎯 RE Scenarios</div>',
                unsafe_allow_html=True)
    s_lbls = ["Current", "20%", "40%", "60%", "80%"]
    s_re   = [re_pct, 20, 40, 60, 80]
    s_vals = [predict_co2(selected_state, year, energy_use,
                          r/100, gdp, urban, co2_per_twh_b) for r in s_re]
    s_cols = [co2_color(v)[0] for v in s_vals]
    fig_sc = go.Figure(go.Bar(
        x=s_lbls, y=s_vals, marker_color=s_cols,
        text=[f"{v:.0f}" for v in s_vals],
        textposition="outside", textfont=dict(color="#cbd5e1", size=9)
    ))
    fig_sc.update_layout(
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=9),
        margin=dict(l=8,r=8,t=8,b=28), height=155,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(title="CO₂", gridcolor="#334155"),
        showlegend=False
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:6px">📈 Historical Trend</div>',
                unsafe_allow_html=True)
    hist = df[df["State"]==selected_state].sort_values("Year")
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=hist["Year"], y=hist["Carbon_Emissions_MtCO2"],
        mode="lines", line=dict(color="#8b5cf6", width=2),
        fill="tozeroy", fillcolor="rgba(139,92,246,0.1)"
    ))
    fig_h.add_trace(go.Scatter(
        x=[year], y=[predicted], mode="markers",
        marker=dict(size=12, color=color_hex, symbol="star",
                    line=dict(width=1, color="white"))
    ))
    fig_h.update_layout(
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=9),
        margin=dict(l=8,r=8,t=8,b=28), height=155,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(title="CO₂", gridcolor="#334155"),
        showlegend=False
    )
    st.plotly_chart(fig_h, use_container_width=True)

# ── BOTTOM TABLE ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">📋 All States — Prediction Overview</div>',
            unsafe_allow_html=True)
tbl = get_baseline_map()[["State","CO2","Status","RE_pct","Energy"]].copy()
tbl.columns = ["State","CO₂ (MtCO₂)","Status","RE %","Energy (TWh)"]
tbl = tbl.sort_values("CO₂ (MtCO₂)", ascending=False).reset_index(drop=True)
st.dataframe(tbl, use_container_width=True, height=240, hide_index=True)
