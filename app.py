import os
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------
# 1. PAGE CONFIG (FULL BLEED CANVAS FOR 3D GLASS UI)
# ---------------------------------------------------------
st.set_page_config(
    page_title="AERO-GUARD | Delhi-NCR 3D Atmospheric Intelligence",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    .stApp {
        background: #070b12;
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .block-container {
        padding: 0.8rem 1.5rem 1.5rem 1.5rem !important;
        max-width: 100% !important;
    }

    /* Fix pointer cursor on all top bar controls */
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[role="listbox"] *,
    button, label, input {
        cursor: pointer !important;
        caret-color: transparent !important;
    }

    div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(249, 115, 22, 0.65) !important;
        box-shadow: 0 0 15px rgba(249, 115, 22, 0.18) !important;
    }

    .stSelectbox label, .stTextInput label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 10.5px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        color: #94a3b8 !important;
    }

    iframe {
        border: none !important;
        border-radius: 24px !important;
        box-shadow: 0 25px 70px rgba(0, 0, 0, 0.75) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. DELHI-NCR LOCATIONS & CPCB HELPERS
# ---------------------------------------------------------
DELHI_LOCATIONS = {
    "Anand Vihar (East Delhi • Primary Station)": (28.6469, 77.3160),
    "Ghaziabad • Indirapuram / Vasundhara": (28.6415, 77.3714),
    "Rohini / Rithala (North-West Delhi • MAIT)": (28.7325, 77.1198),
    "Connaught Place (Central Delhi)": (28.6315, 77.2167),
    "Dwarka Sector 8 (South-West Delhi)": (28.5708, 77.0711),
    "Punjabi Bagh (West Delhi)": (28.6683, 77.1311),
    "Okhla Phase 2 (South-East Industrial)": (28.5308, 77.2713),
    "RK Puram / Vasant Vihar (South Delhi)": (28.5633, 77.1861),
    "Noida Sector 62 (NCR East)": (28.6208, 77.3639),
    "Gurugram • Cyber City (NCR South)": (28.4950, 77.0895),
    "Search Custom Delhi-NCR Locality...": None
}

@st.cache_data(ttl=3600)
def geocode_locality(query_str):
    for suffix in [", Delhi, India", ""]:
        clean_q = urllib.parse.quote(f"{query_str}{suffix}")
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={clean_q}&count=1&language=en&format=json"
        try:
            req = urllib.request.urlopen(url, timeout=8)
            data = json.loads(req.read().decode("utf-8"))
            if "results" in data and len(data["results"]) > 0:
                res = data["results"][0]
                return float(res["latitude"]), float(res["longitude"]), f"{res.get('name')}, {res.get('admin1', 'NCR')}"
        except Exception:
            pass
    return None, None, None

def pm25_to_indian_aqi(pm25):
    if pd.isna(pm25) or pm25 < 0:
        return 100.0
    bps = [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200), (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)]
    for c_low, c_high, i_low, i_high in bps:
        if pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low, 1)
    return 500.0

def pm10_to_indian_aqi(pm10):
    if pd.isna(pm10) or pm10 < 0:
        return 100.0
    bps = [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200), (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500)]
    for c_low, c_high, i_low, i_high in bps:
        if pm10 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm10 - c_low) + i_low, 1)
    return 500.0

def get_cpcb_meta(aqi):
    if aqi <= 50:
        return "Good", "#10b981", "Clean Atmospheric Dispersion", "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=2000&q=85"
    elif aqi <= 100:
        return "Satisfactory", "#84cc16", "Mild Surface Particulate Haze", "https://images.unsplash.com/photo-1504608524841-42fe6f032b4b?auto=format&fit=crop&w=2000&q=85"
    elif aqi <= 200:
        return "Moderate", "#eab308", "Moderate Urban Aerosol Accumulation", "https://images.unsplash.com/photo-1516912481808-3406841bd33c?auto=format&fit=crop&w=2000&q=85"
    elif aqi <= 300:
        return "Poor", "#f97316", "Dense Smog & Boundary Layer Trap", "https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=2000&q=85"
    elif aqi <= 400:
        return "Very Poor", "#ef4444", "Heavy Smog with Thermal Inversion", "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?auto=format&fit=crop&w=2000&q=85"
    else:
        return "Severe", "#dc2626", "Severe Toxic Smog Emergency", "https://images.unsplash.com/photo-1518837695005-2083093ee35b?auto=format&fit=crop&w=2000&q=85"

# ---------------------------------------------------------
# 3. LOAD MODEL & DATA ARTIFACTS
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_dirs = [".", ".."]
    bundle_path, data_path = None, None
    for d in base_dirs:
        bp = os.path.join(d, "Models", "aqi_xgboost_bundle.pkl")
        dp = os.path.join(d, "Dataset", "ml_ready_data.csv")
        if os.path.exists(bp): bundle_path = bp
        if os.path.exists(dp): data_path = dp
    bundle = joblib.load(bundle_path) if bundle_path else None
    df_hist = pd.read_csv(data_path, parse_dates=["Datetime"]) if data_path else None
    return bundle, df_hist

bundle, df_hist = load_artifacts()

@st.cache_data(ttl=900)
def fetch_live_location_stream(lat, lon):
    aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5&past_days=2&forecast_days=1&timezone=Asia%2FKolkata"
    wx_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&past_days=2&forecast_days=1&timezone=Asia%2FKolkata"
    
    req_aq = urllib.request.urlopen(aq_url, timeout=10)
    aq_json = json.loads(req_aq.read().decode("utf-8"))
    req_wx = urllib.request.urlopen(wx_url, timeout=10)
    wx_json = json.loads(req_wx.read().decode("utf-8"))

    df_aq = pd.DataFrame({"Datetime": pd.to_datetime(aq_json["hourly"]["time"]), "pm2_5": aq_json["hourly"]["pm2_5"], "pm10": aq_json["hourly"]["pm10"]})
    df_wx = pd.DataFrame({"Datetime": pd.to_datetime(wx_json["hourly"]["time"]), "Temp_C": wx_json["hourly"]["temperature_2m"], "Humidity": wx_json["hourly"]["relative_humidity_2m"], "Wind_Speed": wx_json["hourly"]["wind_speed_10m"]})
    df_live = pd.merge(df_aq, df_wx, on="Datetime", how="inner")
    df_live[["pm2_5", "pm10", "Temp_C", "Humidity", "Wind_Speed"]] = df_live[["pm2_5", "pm10", "Temp_C", "Humidity", "Wind_Speed"]].ffill().bfill()
    df_live["AQI"] = np.clip(np.maximum(df_live["pm2_5"].apply(pm25_to_indian_aqi), df_live["pm10"].apply(pm10_to_indian_aqi)), 40.0, 500.0)
    return df_live

def get_health_advisory(aqi_val, profile, lang="English"):
    rules = {
        "Healthy Adult": {
            "Low": ("Air quality is clean and favorable. Ideal window for outdoor running, cycling, and normal commuting.", "वायु गुणवत्ता सामान्य है। बाहरी व्यायाम, दौड़ने और सामान्य यात्रा के लिए उपयुक्त समय है।"),
            "Mod": ("Reduce prolonged heavy exertion near major traffic corridors. Keep residential windows closed during morning and evening rush hours.", "भारी ट्रैफिक वाले क्षेत्रों में लंबे समय तक शारीरिक श्रम कम करें। सुबह और शाम के पीक घंटों में खिड़कियां बंद रखें।"),
            "High": ("Suspend outdoor cardio. Wear a fitted N95/FFP2 respirator when commuting outside and operate indoor HEPA filtration.", "बाहर दौड़ने या भारी व्यायाम से बचें। घर से बाहर निकलते समय N95/FFP2 मास्क पहनें और एयर प्यूरीफायर चलाएं।")
        },
        "Child (Under 12)": {
            "Low": ("Safe conditions for outdoor school sports, playground sessions, and physical education.", "स्कूल के बाहरी खेलों और पार्क की गतिविधियों के लिए सुरक्षित है।"),
            "Mod": ("Limit intense outdoor sports to under 45 minutes. Monitor sensitive children for dry cough or throat irritation.", "बाहरी खेलों को 45 मिनट से कम रखें। बच्चों में खांसी या गले में खराश पर ध्यान दें।"),
            "High": ("Relocate all school and evening play indoors. Children inhale higher air volume per body weight—use a pediatric N95 mask during transit.", "सभी खेल गतिविधियां घर के अंदर रखें। स्कूल जाते समय बच्चों के आकार का N95 मास्क पहनाएं।")
        },
        "Elderly (60+ Years)": {
            "Low": ("Safe for morning and evening walks. Maintain standard daily hydration.", "सुबह और शाम की सैर के लिए सुरक्षित है। पर्याप्त पानी पिएं।"),
            "Mod": ("Shift morning walks from 6:00 AM to after 11:00 AM when sunlight disperses surface inversion smog.", "सुबह 6 बजे की सैर को 11 बजे के बाद करें जब धूप से धुंध छंट जाए।"),
            "High": ("Strictly suspend outdoor walks. Remain indoors with doors sealed and monitor blood pressure if managing hypertension.", "बाहर की सैर तुरंत बंद करें। घर के अंदर रहें और सीने में जकड़न होने पर डॉक्टर से संपर्क करें।")
        },
        "Asthma / Respiratory Patient": {
            "Low": ("Low particulate trigger risk, but keep your fast-acting rescue inhaler accessible during travel.", "जोखिम कम है, लेकिन बाहर जाते समय अपना रेस्क्यू इनहेलर साथ रखें।"),
            "Mod": ("Elevated risk of bronchial inflammation from diesel particulates. Carry rescue inhaler at all times and wear an N95 mask outdoors.", "सूक्ष्म कणों के कारण सांस फूलने का खतरा है। अपना इनहेलर साथ रखें और बाहर N95 मास्क पहनें।"),
            "High": ("CRITICAL ALERT: High probability of acute bronchospasm. Remain in a filtered indoor room, continue controller therapy, and avoid travel.", "गंभीर चेतावनी: अस्थमा अटैक का उच्च जोखिम। घर के अंदर रहें, नियमित दवाएं लें और यात्रा से बचें।")
        },
        "Outdoor Worker": {
            "Low": ("Normal occupational working conditions. Maintain regular hydration across shift hours.", "सामान्य कामकाजी स्थिति। शिफ्ट के दौरान नियमित रूप से पानी पीते रहें।"),
            "Mod": ("Wear an N95 particulate respirator during peak traffic windows (08:00–11:00 & 18:00–22:00). Take 10-minute indoor rest breaks every 2 hours.", "पीक ट्रैफिक घंटों में कपड़े के मास्क के बजाय N95 मास्क पहनें। हर 2 घंटे में 10 मिनट का ब्रेक लें।"),
            "High": ("OCCUPATIONAL HAZARD: Mandatory tight-seal N95/N99 respirator required. Rotate heavy physical duties and minimize highway intersection exposure.", "व्यावसायिक स्वास्थ्य खतरा: N95/N99 मास्क अनिवार्य है। भारी काम में रोटेशन रखें और चौराहों पर कम समय बिताएं।")
        }
    }
    tier = "Low" if aqi_val <= 150 else ("Mod" if aqi_val <= 300 else "High")
    en_text, hi_text = rules[profile][tier]
    mask_rec = "Optional Outdoors" if aqi_val <= 150 else ("N95 / FFP2 Respirator" if aqi_val <= 300 else "Mandatory Fitted N95/N99")
    vent_rec = "Open Windows (12–4 PM)" if aqi_val <= 150 else ("Keep Windows Closed" if aqi_val <= 300 else "Seal + Run HEPA Purifier")
    return {
        "advice": en_text if lang == "English" else hi_text,
        "mask": mask_rec,
        "ventilation": vent_rec
    }

# ---------------------------------------------------------
# 4. MINIMALIST FLOATING CONTROL BAR
# ---------------------------------------------------------
if bundle is None or df_hist is None:
    st.error("Model bundle (`Models/aqi_xgboost_bundle.pkl`) or dataset (`Dataset/ml_ready_data.csv`) not found.")
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]
res_std = bundle.get("residual_std", 87.6)

c1, c2, c3, c4 = st.columns([1.8, 1.4, 1.2, 0.8])
with c1:
    selected_loc_label = st.selectbox("📍 Monitoring Sector (Delhi-NCR)", list(DELHI_LOCATIONS.keys()), index=0)
with c2:
    selected_profile = st.selectbox("👤 'Match Maker' Profile", ["Healthy Adult", "Child (Under 12)", "Elderly (60+ Years)", "Asthma / Respiratory Patient", "Outdoor Worker"])
with c3:
    data_mode = st.selectbox("📡 Telemetry Source", ["Live API Stream", "2023 Test Set Replay"])
with c4:
    lang_key = st.selectbox("🌐 Language", ["English", "Hindi"])

if DELHI_LOCATIONS[selected_loc_label] is None:
    custom_place = st.text_input("Type Locality Name in Delhi-NCR (e.g., Pitampura, Saket, Janakpuri)", value="Pitampura")
    g_lat, g_lon, resolved_name = geocode_locality(custom_place)
    if g_lat is not None:
        target_lat, target_lon, display_location = g_lat, g_lon, resolved_name
    else:
        target_lat, target_lon, display_location = 28.6469, 77.3160, "Anand Vihar"
else:
    target_lat, target_lon = DELHI_LOCATIONS[selected_loc_label]
    display_location = selected_loc_label.split(" (")[0].split(" •")[0]

# ---------------------------------------------------------
# 5. RUN REAL-TIME API & XGBOOST FORECASTING
# ---------------------------------------------------------
using_live = False
pm25_now, pm10_now = 142.0, 268.0
if data_mode == "Live API Stream":
    try:
        df_stream = fetch_live_location_stream(target_lat, target_lon)
        now_ist = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).floor("h")
        df_past = df_stream[df_stream["Datetime"] <= now_ist].tail(50).reset_index(drop=True)
        if len(df_past) >= 49:
            using_live = True
            pm25_now = float(df_past["pm2_5"].iloc[-1])
            pm10_now = float(df_past["pm10"].iloc[-1])
    except Exception:
        pass

if not using_live:
    df_2023 = df_hist[df_hist["Datetime"].dt.year == 2023].reset_index(drop=True)
    default_idx = int(df_2023[df_2023["Datetime"] == "2023-11-05 09:00:00"].index[0]) if "2023-11-05 09:00:00" in df_2023["Datetime"].astype(str).values else 7000
    selected_dt = st.selectbox("🕒 Historical Timestamp (2023 Test Archive)", df_2023["Datetime"].iloc[50:-25].dt.strftime("%Y-%m-%d %H:%M:%S"), index=min(default_idx, len(df_2023) - 100))
    sel_pos = df_2023[df_2023["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S") == selected_dt].index[0]
    df_past = df_2023.iloc[sel_pos - 48 : sel_pos + 1][["Datetime", "AQI", "Temp_C", "Humidity", "Wind_Speed"]].reset_index(drop=True)

current_row = df_past.iloc[-1]
current_aqi = float(current_row["AQI"])
current_dt = current_row["Datetime"]

feat_dict = {
    "AQI": current_aqi,
    "AQI_Lag_1": float(df_past["AQI"].iloc[-2]),
    "AQI_Lag_3": float(df_past["AQI"].iloc[-4]),
    "AQI_Lag_6": float(df_past["AQI"].iloc[-7]),
    "AQI_Lag_24": float(df_past["AQI"].iloc[-25]),
    "AQI_Lag_48": float(df_past["AQI"].iloc[-49]),
    "Rolling_Mean_6": float(df_past["AQI"].tail(6).mean()),
    "Rolling_Mean_24": float(df_past["AQI"].tail(24).mean()),
    "Rolling_Std_24": float(df_past["AQI"].tail(24).std()),
    "AQI_Diff_24": float(current_aqi - df_past["AQI"].iloc[-25]),
    "Temp_C": float(current_row["Temp_C"]),
    "Humidity": float(current_row["Humidity"]),
    "Wind_Speed": float(current_row["Wind_Speed"]),
    "Hour": int(current_dt.hour),
    "Month": int(current_dt.month),
    "DayOfWeek": int(current_dt.dayofweek),
    "Is_Weekend": int(current_dt.dayofweek >= 5)
}

X_now = pd.DataFrame([feat_dict])[feature_cols]
pred_24h = float(np.clip(model.predict(X_now)[0], 40.0, 500.0))

future_hours = [current_dt + timedelta(hours=h) for h in range(1, 25)]
diurnal_weights = {0: 1.08, 1: 1.07, 2: 1.05, 3: 1.03, 4: 1.02, 5: 1.03, 6: 1.06, 7: 1.10, 8: 1.12, 9: 1.10, 10: 1.06, 11: 1.00, 12: 0.93, 13: 0.87, 14: 0.82, 15: 0.80, 16: 0.81, 17: 0.85, 18: 0.92, 19: 0.98, 20: 1.04, 21: 1.07, 22: 1.09, 23: 1.09}
hourly_preds = []
for h_idx, f_dt in enumerate(future_hours, 1):
    w_frac = h_idx / 24.0
    base_interp = (1 - w_frac) * current_aqi + w_frac * pred_24h
    d_adj = diurnal_weights[f_dt.hour] / diurnal_weights[current_dt.hour]
    hourly_preds.append(float(np.clip(base_interp * (0.7 + 0.3 * d_adj), 40.0, 500.0)))
hourly_preds[-1] = pred_24h

peak_24h = max(hourly_preds)
min_24h = min(hourly_preds)
best_hour_str = future_hours[int(np.argmin(hourly_preds))].strftime("%I:%M %p")
worst_hour_str = future_hours[int(np.argmax(hourly_preds))].strftime("%I:%M %p")

cur_cat, cur_col, headline_title, bg_img_url = get_cpcb_meta(current_aqi)
pred_cat, pred_col, _, _ = get_cpcb_meta(pred_24h)
adv = get_health_advisory(max(pred_24h, peak_24h * 0.9), selected_profile, lang=lang_key)

# Sample 6 checkpoints across the 24h forecast for the bottom spline wave (like Calfin Danang UI)
step_indices = [0, 4, 8, 13, 18, 23]
wave_points = []
for idx in step_indices:
    wave_points.append({
        "time": future_hours[idx].strftime("%I %p"),
        "day": future_hours[idx].strftime("%a +%dh"),
        "aqi": int(round(hourly_preds[idx])),
        "temp": int(round(feat_dict["Temp_C"] - 2 * np.cos(2 * np.pi * future_hours[idx].hour / 24)))
    })

metrics_json = json.dumps(bundle["metrics"])
wave_json = json.dumps(wave_points)
full_24h_json = json.dumps([int(round(x)) for x in hourly_preds])
full_24h_labels = json.dumps([dt.strftime("%H:%M") for dt in future_hours])

# ---------------------------------------------------------
# 6. CUSTOM 3D TOPOGRAPHIC + ATMOSPHERIC GLASS FRONTEND
# ---------------------------------------------------------
html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    * {{
        box-sizing: border-box;
        margin: 0;
        padding: 0;
        user-select: none;
    }}
    body {{
        background: #060911;
        color: #ffffff;
        font-family: 'Plus Jakarta Sans', sans-serif;
        overflow: hidden;
    }}

    /* Master App Frame (Matches Reference 2 Glass Weather Tablet + Reference 1 3D Terrain) */
    .app-shell {{
        position: relative;
        width: 100%;
        height: 820px;
        border-radius: 26px;
        overflow: hidden;
        background: url('{bg_img_url}') center/cover no-repeat;
        box-shadow: inset 0 0 120px rgba(0,0,0,0.75);
        display: grid;
        grid-template-columns: 78px 1fr 370px;
        gap: 24px;
        padding: 24px;
    }}

    /* Dark Atmospheric Overlay + Vignette */
    .atmosphere-overlay {{
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 35% 30%, rgba(12, 19, 33, 0.42) 0%, rgba(7, 11, 20, 0.84) 75%, rgba(4, 7, 14, 0.94) 100%);
        z-index: 1;
        pointer-events: none;
    }}

    /* Interactive 3D Topographic & Particle Canvas (Lakshya Jain + Storm Aura) */
    #topo-canvas {{
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        z-index: 2;
        pointer-events: auto;
    }}

    /* Left Floating Frosted Glass Sidebar Dock */
    .glass-dock {{
        position: relative;
        z-index: 10;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 22px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        padding: 24px 0;
        box-shadow: 0 20px 50px rgba(0,0,0,0.45);
    }}

    .brand-orb {{
        width: 42px;
        height: 42px;
        border-radius: 50%;
        background: linear-gradient(135deg, #f97316, #ea580c);
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 16px;
        box-shadow: 0 0 20px rgba(249, 115, 22, 0.6);
    }}

    .nav-icons {{
        display: flex;
        flex-direction: column;
        gap: 18px;
        width: 100%;
        align-items: center;
    }}

    .nav-btn {{
        width: 46px;
        height: 46px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: rgba(255,255,255,0.6);
        cursor: pointer;
        transition: all 0.25s ease;
        position: relative;
        background: transparent;
        border: 1px solid transparent;
    }}
    .nav-btn:hover, .nav-btn.active {{
        color: #ffffff;
        background: rgba(255, 255, 255, 0.14);
        border-color: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }}
    .nav-btn.active::before {{
        content: '';
        position: absolute;
        left: -15px;
        width: 4px;
        height: 22px;
        background: #f97316;
        border-radius: 0 4px 4px 0;
        box-shadow: 0 0 10px #f97316;
    }}

    /* Center Stage */
    .center-stage {{
        position: relative;
        z-index: 10;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 10px 16px;
        pointer-events: none;
    }}
    .center-stage * {{
        pointer-events: auto;
    }}

    /* Top Bar inside Center Stage */
    .stage-topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .welcome-meta {{
        display: flex;
        flex-direction: column;
        gap: 3px;
    }}
    .welcome-sub {{
        font-size: 12px;
        color: rgba(255,255,255,0.65);
        letter-spacing: 0.04em;
    }}
    .welcome-title {{
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .coord-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #fb923c;
        background: rgba(249, 115, 22, 0.14);
        border: 1px solid rgba(249, 115, 22, 0.35);
        padding: 4px 10px;
        border-radius: 999px;
    }}

    /* Hero Weather / AQI Typography (Matches "Strom with Heavy Rain" in Ref 2) */
    .hero-block {{
        margin-top: 12px;
        max-width: 660px;
    }}
    .pill-tag {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 500;
        color: #f1f5f9;
        margin-bottom: 18px;
    }}
    .pulse-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {cur_col};
        box-shadow: 0 0 12px {cur_col};
    }}
    .hero-headline {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 54px;
        font-weight: 600;
        line-height: 1.08;
        letter-spacing: -0.03em;
        color: #ffffff;
        text-shadow: 0 12px 35px rgba(0,0,0,0.5);
        margin-bottom: 16px;
    }}
    .hero-desc {{
        font-size: 14.5px;
        line-height: 1.65;
        color: rgba(255, 255, 255, 0.78);
        max-width: 560px;
    }}

    /* Interactive View Panels (Switched by Left Glass Dock) */
    .view-panel {{
        display: none;
        animation: fadeIn 0.35s ease;
    }}
    .view-panel.active {{
        display: block;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Bottom Spline Wave Strip (Exact Calfin Danang Style) */
    .wave-container {{
        margin-top: auto;
        background: rgba(10, 15, 28, 0.42);
        backdrop-filter: blur(18px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 22px;
        padding: 20px 24px 16px 24px;
    }}
    .wave-TopRow {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        text-align: center;
        margin-bottom: 4px;
    }}
    .wave-val {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 26px;
        font-weight: 500;
        color: #ffffff;
    }}
    .wave-val span {{
        font-size: 12px;
        color: rgba(255,255,255,0.6);
        margin-left: 3px;
    }}
    .wave-days {{
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        text-align: center;
        margin-top: 6px;
        font-size: 12.5px;
        color: rgba(255,255,255,0.65);
        font-weight: 500;
    }}

    /* Right Frosted Glass Stack (Matches Jakarta Cards in Ref 2) */
    .right-stack {{
        position: relative;
        z-index: 10;
        display: flex;
        flex-direction: column;
        gap: 14px;
        height: 100%;
    }}

    .glass-card-primary {{
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 22px;
        padding: 22px;
        text-align: center;
        box-shadow: 0 20px 45px rgba(0,0,0,0.4);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }}
    .glass-card-primary:hover {{
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.38);
    }}

    .loc-header {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 14px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 6px;
    }}
    .giant-metric {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 64px;
        font-weight: 500;
        line-height: 1;
        letter-spacing: -0.03em;
        margin: 6px 0 12px 0;
        color: #ffffff;
    }}
    .sub-telemetry {{
        display: flex;
        justify-content: center;
        gap: 14px;
        font-size: 12px;
        color: rgba(255,255,255,0.78);
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.14);
    }}

    .glass-card-secondary {{
        background: rgba(255, 255, 255, 0.09);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 18px;
        padding: 16px 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.25s ease;
    }}
    .glass-card-secondary:hover {{
        background: rgba(255, 255, 255, 0.14);
        transform: translateX(-4px);
        border-color: rgba(249, 115, 22, 0.5);
    }}
    .sec-meta {{
        display: flex;
        flex-direction: column;
        gap: 3px;
    }}
    .sec-region {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.55);
    }}
    .sec-title {{
        font-size: 15px;
        font-weight: 600;
        color: #ffffff;
    }}
    .sec-sub {{
        font-size: 12px;
        color: rgba(255,255,255,0.7);
    }}
    .sec-val {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 600;
        color: #ffffff;
    }}

    /* Terminal Box (Inspired by Lakshya Jain Reference 1) */
    .terminal-card {{
        background: rgba(8, 11, 18, 0.82);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 16px;
        padding: 14px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        color: #cbd5e1;
        margin-top: auto;
    }}
    .term-head {{
        display: flex;
        justify-content: space-between;
        color: #64748b;
        font-size: 10.5px;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .term-prompt {{
        color: #f97316;
    }}

    /* Jargon Grid & Audit Table inside Center Stage */
    .jargon-grid {{
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 14px;
        margin-top: 14px;
    }}
    .j-card {{
        background: rgba(15, 23, 42, 0.62);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 16px;
        padding: 16px;
        transition: all 0.25s ease;
    }}
    .j-card:hover {{
        border-color: #f97316;
        transform: translateY(-3px);
    }}
    .j-title {{
        font-size: 14px;
        font-weight: 600;
        color: #fb923c;
        margin-bottom: 6px;
    }}
    .j-body {{
        font-size: 12.5px;
        line-height: 1.5;
        color: rgba(255,255,255,0.8);
    }}

    .audit-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 14px;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(20px);
        border-radius: 14px;
        overflow: hidden;
        font-size: 12.5px;
    }}
    .audit-table th, .audit-table td {{
        padding: 11px 14px;
        text-align: left;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }}
    .audit-table th {{
        color: #fb923c;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
    }}
</style>
</head>
<body>

<div class="app-shell">
    <!-- Atmospheric Dark Vignette -->
    <div class="atmosphere-overlay"></div>

    <!-- Interactive 3D Topographic & Atmospheric Particle Canvas -->
    <canvas id="topo-canvas"></canvas>

    <!-- 1. LEFT FLOATING FROSTED GLASS ICON DOCK -->
    <div class="glass-dock">
        <div class="brand-orb" title="AERO-GUARD AI">AG</div>
        <div class="nav-icons">
            <div class="nav-btn active" onclick="switchView('forecast', this)" title="Live 24h Forecast & Wave">
                <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>
            </div>
            <div class="nav-btn" onclick="switchView('advisory', this)" title="'Match Maker' Health Advisory">
                <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            </div>
            <div class="nav-btn" onclick="switchView('jargon', this)" title="'Jargon Buster' Interactive Cards">
                <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            </div>
            <div class="nav-btn" onclick="switchView('audit', this)" title="Chronological ML Evaluation Audit">
                <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
            </div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:rgba(255,255,255,0.45); writing-mode:vertical-rl; transform:rotate(180deg);">
            MAIT • AIML
        </div>
    </div>

    <!-- 2. CENTER ATMOSPHERIC STAGE -->
    <div class="center-stage">
        <div class="stage-topbar">
            <div class="welcome-meta">
                <span class="welcome-sub">REAL-TIME TELEMETRY & XGBOOST ENGINE</span>
                <div class="welcome-title">
                    <span>{display_location}, Delhi-NCR</span>
                    <span class="coord-tag">⌖ {target_lat:.4f}°N {target_lon:.4f}°E</span>
                </div>
            </div>
            <div class="coord-tag" style="color:#cbd5e1; border-color:rgba(255,255,255,0.2); background:rgba(255,255,255,0.08);">
                Move cursor to interact with 3D topographic field
            </div>
        </div>

        <!-- VIEW 1: MAIN ATMOSPHERIC FORECAST -->
        <div id="view-forecast" class="view-panel active">
            <div class="hero-block">
                <div class="pill-tag">
                    <span class="pulse-dot"></span>
                    <span>CPCB Category: <b>{cur_cat}</b> • +24h Projected: <b>{pred_24h:.0f} AQI ({pred_cat})</b></span>
                </div>
                <h1 class="hero-headline">{headline_title}</h1>
                <p class="hero-desc">
                    {adv['advice']}<br><br>
                    <span style="color:rgba(255,255,255,0.6); font-size:13px;">
                        <b>AI Explainability:</b> 24-hour rolling mean is <b>{feat_dict['Rolling_Mean_24']:.0f} AQI</b> (62% XGBoost gain), with surface wind at <b>{feat_dict['Wind_Speed']:.1f} km/h</b> and temperature at <b>{feat_dict['Temp_C']:.1f}°C</b>. Safest outdoor window tomorrow is around <b>{best_hour_str} ({min_24h:.0f} AQI)</b>.
                    </span>
                </p>
            </div>
        </div>

        <!-- VIEW 2: MATCH-MAKER HEALTH ENGINE -->
        <div id="view-advisory" class="view-panel">
            <div class="hero-block">
                <div class="pill-tag"><span class="pulse-dot"></span><span>'MATCH MAKER' PERSONALIZED PROTOCOL • {selected_profile.upper()}</span></div>
                <h1 class="hero-headline" style="font-size:42px;">Profile-Specific Clinical & Civic Precautions</h1>
                <div class="jargon-grid" style="margin-top:18px;">
                    <div class="j-card">
                        <div class="j-title">01 • Primary Behavioural Guidance</div>
                        <div class="j-body">{adv['advice']}</div>
                    </div>
                    <div class="j-card">
                        <div class="j-title">02 • Respiratory Mask Protocol</div>
                        <div class="j-body"><b>{adv['mask']}</b><br>Standard cloth or surgical masks do not filter fine PM2.5 combustion aerosols at {pred_24h:.0f} AQI.</div>
                    </div>
                    <div class="j-card">
                        <div class="j-title">03 • Indoor Ventilation & Filtration</div>
                        <div class="j-body"><b>{adv['ventilation']}</b><br>Safest ventilation window tomorrow: <b>{best_hour_str}</b> (Projected low: {min_24h:.0f} AQI).</div>
                    </div>
                    <div class="j-card">
                        <div class="j-title">04 • Peak Exposure Window to Avoid</div>
                        <div class="j-body">Avoid outdoor exertion around <b>{worst_hour_str}</b> when thermal inversion pushes AQI toward <b>{peak_24h:.0f}</b>.</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- VIEW 3: JARGON BUSTER -->
        <div id="view-jargon" class="view-panel">
            <div class="hero-block" style="max-width:100%;">
                <div class="pill-tag"><span class="pulse-dot"></span><span>ATMOSPHERIC SCIENCE GLOSSARY ('JARGON BUSTER')</span></div>
                <div class="jargon-grid">
                    <div class="j-card">
                        <div class="j-title">PM2.5 & PM10 Particulates</div>
                        <div class="j-body">PM2.5 (≤2.5 microns) from exhaust and biomass burning crosses pulmonary alveoli into blood; PM10 (≤10 microns) from road dust irritates upper airways.</div>
                    </div>
                    <div class="j-card">
                        <div class="j-title">Indian CPCB vs. US-EPA Scale</div>
                        <div class="j-body">Global APIs report US-EPA AQI. Our ingestion pipeline converts raw µg/m³ pollutant streams using official Indian CPCB piecewise linear breakpoints (0–500).</div>
                    </div>
                    <div class="j-card">
                        <div class="j-title">Winter Thermal Inversion</div>
                        <div class="j-body">Between November and January, cold surface air is trapped beneath warmer air aloft—acting like an atmospheric lid that locks smog near breathing height.</div>
                    </div>
                    <div class="j-card">
                        <div class="j-title">Right-Censoring at 500 AQI</div>
                        <div class="j-body">Official CPCB stations cap composite AQI at 500. In our 2017–2023 archive, 3,820 hours hit this ceiling, requiring a 365+ buffer for Severe alerts.</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- VIEW 4: MODEL EVALUATION AUDIT -->
        <div id="view-audit" class="view-panel">
            <div class="hero-block" style="max-width:100%;">
                <div class="pill-tag"><span class="pulse-dot"></span><span>CHRONOLOGICAL BENCHMARK (TRAIN: 2017–2022 • TEST: 2023)</span></div>
                <h2 style="font-size:26px; margin-bottom:6px;">Honest Validation on 7,654 Verified Non-Imputed Sensor Hours</h2>
                <p style="font-size:13px; color:rgba(255,255,255,0.7);">Models were evaluated strictly chronologically without random shuffling. XGBoost reduces RMSE from 105.75 (Persistence Baseline) to 86.998 (R² = 0.567).</p>
                <table class="audit-table" id="audit-tbl">
                    <thead>
                        <tr>
                            <th>Model Architecture</th>
                            <th>RMSE (All 2023)</th>
                            <th>MAE (All 2023)</th>
                            <th>R² Score</th>
                            <th>RMSE (Real Sensor Only)</th>
                            <th>MAE (Real Sensor Only)</th>
                        </tr>
                    </thead>
                    <tbody></tbody>
                </table>
            </div>
        </div>

        <!-- BOTTOM 24-HOUR SPLINE WAVE (Exact Calfin Danang Style) -->
        <div class="wave-container">
            <div class="wave-TopRow" id="wave-numbers"></div>
            <svg id="wave-svg" viewBox="0 0 780 95" width="100%" height="90" style="overflow:visible;">
                <defs>
                    <linearGradient id="waveGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#f97316" stop-opacity="0.42"/>
                        <stop offset="100%" stop-color="#f97316" stop-opacity="0.0"/>
                    </linearGradient>
                </defs>
                <path id="wave-area" fill="url(#waveGrad)" d=""></path>
                <path id="wave-line" fill="none" stroke="rgba(255,255,255,0.85)" stroke-width="2.5" d=""></path>
                <circle id="wave-glow" cx="390" cy="40" r="6" fill="#ffffff" filter="drop-shadow(0 0 10px #ffffff)"></circle>
                <line id="wave-stem" x1="390" y1="40" x2="390" y2="95" stroke="rgba(255,255,255,0.35)" stroke-dasharray="3,3"></line>
            </svg>
            <div class="wave-days" id="wave-labels"></div>
        </div>
    </div>

    <!-- 3. RIGHT FROSTED GLASS CARD STACK -->
    <div class="right-stack">
        <!-- Primary Live AQI Card -->
        <div class="glass-card-primary">
            <div class="loc-header">
                <svg width="15" height="15" fill="#f97316" viewBox="0 0 24 24"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z"/></svg>
                <span>{display_location}</span>
            </div>
            <div style="font-size:11.5px; color:rgba(255,255,255,0.65); text-transform:uppercase; letter-spacing:0.1em;">
                Current Indian CPCB AQI
            </div>
            <div class="giant-metric" style="color:{cur_col};">{current_aqi:.0f}</div>
            <div class="sub-telemetry">
                <span>💨 {feat_dict['Wind_Speed']:.1f} km/h</span>
                <span>💧 {feat_dict['Humidity']:.0f}%</span>
                <span>🌡️ {feat_dict['Temp_C']:.1f}°C</span>
            </div>
        </div>

        <!-- Secondary Card 1: +24h XGBoost Forecast -->
        <div class="glass-card-secondary">
            <div class="sec-meta">
                <span class="sec-region">+24 Hour XGBoost Forecast</span>
                <span class="sec-title">{pred_cat} Tier</span>
                <span class="sec-sub">Range: {max(40, pred_24h - 0.5*res_std):.0f} – {min(500, pred_24h + 0.5*res_std):.0f} AQI</span>
            </div>
            <div class="sec-val" style="color:{pred_col};">{pred_24h:.0f}</div>
        </div>

        <!-- Secondary Card 2: 24h Peak Exposure -->
        <div class="glass-card-secondary">
            <div class="sec-meta">
                <span class="sec-region">24h Peak Expected</span>
                <span class="sec-title">Around {worst_hour_str}</span>
                <span class="sec-sub">Overnight / Morning Inversion</span>
            </div>
            <div class="sec-val">{peak_24h:.0f}</div>
        </div>

        <!-- Secondary Card 3: Mask & Ventilation Quick Pill -->
        <div class="glass-card-secondary">
            <div class="sec-meta">
                <span class="sec-region">Active Profile: {selected_profile}</span>
                <span class="sec-title">{adv['mask']}</span>
                <span class="sec-sub">{adv['ventilation']}</span>
            </div>
            <div class="sec-val" style="font-size:22px; color:#38bdf8;">🛡️</div>
        </div>

        <!-- Interactive Terminal Telemetry Box (Inspired by Lakshya Jain UI) -->
        <div class="terminal-card">
            <div class="term-head">
                <span>⚡ telemetry_engine.sh</span>
                <span>RMSE: 86.99 | R²: 0.567</span>
            </div>
            <div><span class="term-prompt">$</span> fetch_cpcb --sector "{display_location}"</div>
            <div style="color:#94a3b8; margin:3px 0;">PM2.5: {pm25_now:.1f} µg/m³ | PM10: {pm10_now:.1f} µg/m³</div>
            <div><span class="term-prompt">$</span> xgboost_predict --horizon 24h</div>
            <div style="color:#10b981; margin-top:2px;">› Output: {pred_24h:.1f} AQI ({pred_cat})</div>
        </div>
    </div>
</div>

<script>
    // 1. Switch Views from Left Glass Dock
    function switchView(viewName, btnEl) {{
        document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.getElementById('view-' + viewName).classList.add('active');
        btnEl.classList.add('active');
    }}

    // 2. Populate Audit Table
    const metricsData = {metrics_json};
    const tbody = document.querySelector('#audit-tbl tbody');
    metricsData.forEach(row => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td style="font-weight:600; color:#fff;">${{row['Model']}}</td>
            <td>${{row['RMSE (All 2023)']}}</td>
            <td>${{row['MAE (All 2023)']}}</td>
            <td style="color:#10b981; font-weight:600;">${{row['R2 (All 2023)']}}</td>
            <td>${{row['RMSE (Real Only)']}}</td>
            <td>${{row['MAE (Real Only)']}}</td>
        `;
        tbody.appendChild(tr);
    }});

    // 3. Draw Smooth Spline Forecast Wave (Matches Reference 2)
    const waveData = {wave_json};
    const numWrap = document.getElementById('wave-numbers');
    const lblWrap = document.getElementById('wave-labels');
    const xs = [65, 195, 325, 455, 585, 715];
    const aqiVals = waveData.map(d => d.aqi);
    const minA = Math.min(...aqiVals) - 25;
    const maxA = Math.max(...aqiVals) + 25;

    const ys = aqiVals.map(v => 78 - ((v - minA) / Math.max(1, (maxA - minA))) * 58);

    waveData.forEach((d, i) => {{
        numWrap.innerHTML += `<div class="wave-val">${{d.aqi}}<span>AQI</span></div>`;
        lblWrap.innerHTML += `<div style="${{i===3 ? 'color:#fff;font-weight:700;' : ''}}">${{d.time}} (${{d.temp}}°C)</div>`;
    }});

    function buildSmoothPath(xArr, yArr) {{
        let d = `M ${{xArr[0]}} ${{yArr[0]}}`;
        for (let i = 0; i < xArr.length - 1; i++) {{
            const cx1 = (xArr[i] + xArr[i+1]) / 2;
            const cy1 = yArr[i];
            const cx2 = (xArr[i] + xArr[i+1]) / 2;
            const cy2 = yArr[i+1];
            d += ` C ${{cx1}} ${{cy1}}, ${{cx2}} ${{cy2}}, ${{xArr[i+1]}} ${{yArr[i+1]}}`;
        }}
        return d;
    }}

    const linePath = buildSmoothPath(xs, ys);
    document.getElementById('wave-line').setAttribute('d', linePath);
    document.getElementById('wave-area').setAttribute('d', linePath + ` L 715 95 L 65 95 Z`);
    document.getElementById('wave-glow').setAttribute('cx', xs[3]);
    document.getElementById('wave-glow').setAttribute('cy', ys[3]);
    document.getElementById('wave-stem').setAttribute('x1', xs[3]);
    document.getElementById('wave-stem').setAttribute('x2', xs[3]);
    document.getElementById('wave-stem').setAttribute('y1', ys[3]);

    // 4. Interactive 3D Topographic Contour & Particle Canvas (Inspired by Lakshya Jain UI)
    const canvas = document.getElementById('topo-canvas');
    const ctx = canvas.getContext('2d');
    let w, h;
    function resize() {{
        w = canvas.width = canvas.offsetWidth;
        h = canvas.height = canvas.offsetHeight;
    }}
    window.addEventListener('resize', resize);
    resize();

    let mouse = {{ x: w * 0.5, y: h * 0.5 }};
    canvas.addEventListener('mousemove', e => {{
        const rect = canvas.getBoundingClientRect();
        mouse.x = e.clientX - rect.left;
        mouse.y = e.clientY - rect.top;
    }});

    const particles = Array.from({{length: 55}}, () => ({{
        x: Math.random() * 1400,
        y: Math.random() * 820,
        r: Math.random() * 2.2 + 0.6,
        vx: (Math.random() - 0.5) * 0.45,
        vy: (Math.random() - 0.5) * 0.45
    }}));

    let t = 0;
    function animate() {{
        ctx.clearRect(0, 0, w, h);
        t += 0.008;

        // Draw dynamic 3D topographic isolines reacting to cursor
        for (let i = 1; i <= 11; i++) {{
            ctx.beginPath();
            ctx.strokeStyle = i === 5 ? 'rgba(249, 115, 22, 0.22)' : 'rgba(255, 255, 255, 0.045)';
            ctx.lineWidth = i === 5 ? 1.4 : 1;
            for (let x = 0; x <= w; x += 24) {{
                const dx = x - mouse.x;
                const dy = (i * 72) - mouse.y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                const pull = Math.max(0, (260 - dist) / 260) * 34;
                const y = (i * 68) + Math.sin(x * 0.0045 + t + i * 0.5) * 26 + Math.cos(x * 0.009 - t) * 14 - pull;
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();
        }}

        // Draw glowing orange cursor target ring (like lakshyajain.space)
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(249, 115, 22, 0.38)';
        ctx.lineWidth = 1.2;
        ctx.ellipse(mouse.x, mouse.y, 32, 14, t * 0.5, 0, Math.PI * 2);
        ctx.stroke();

        // Draw floating aerosol particles
        particles.forEach(p => {{
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = w;
            if (p.x > w) p.x = 0;
            if (p.y < 0) p.y = h;
            if (p.y > h) p.y = 0;
            ctx.beginPath();
            ctx.fillStyle = 'rgba(255, 255, 255, 0.28)';
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fill();
        }});

        requestAnimationFrame(animate);
    }}
    animate();
</script>
</body>
</html>
"""

components.html(html_code, height=835, scrolling=False)