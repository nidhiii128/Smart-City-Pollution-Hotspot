"""Smart City Pollution Hotspot Detection - interactive dashboard.

Run with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
from src.aqi import CATEGORIES, NAAQS_24H

st.set_page_config(page_title="Pollution Hotspots · Mumbai & Thane", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------- palette ---
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CITY_COLORS = {"Mumbai": SERIES[0], "Thane": SERIES[1]}
AQI_COLORS = {name: color for _, name, color in CATEGORIES}
AQI_ORDER = [name for _, name, _ in CATEGORIES]
SEVERITY_COLORS = {"Critical Hotspot": "#7b1f1f", "Hotspot": "#d9480f", "Elevated": "#f08c00",
                   "Moderate": "#f2c40f", "Low Pollution": "#9acd32", "Clean Zone": "#2e9d4e"}
SEQ = ["#fde7dc", "#f9b99b", "#f08a5d", "#d9480f", "#8b2a0a"]  # one-hue magnitude ramp
POLLUTANT_LABELS = {"pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂", "so2": "SO₂",
                    "co": "CO", "o3": "O₃", "aqi": "AQI"}
UNITS = {"aqi": "", "co": " µg/m³"} | {p: " µg/m³" for p in ["pm2_5", "pm10", "no2", "so2", "o3"]}
MAP_STYLE = "carto-positron"
MAP_CENTER = {"lat": 19.13, "lon": 72.97}


def label(p: str) -> str:
    return POLLUTANT_LABELS.get(p, p)


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None),
                      hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def aqi_badge(category: str) -> str:
    color = AQI_COLORS.get(category, "#888")
    return (f"<span style='background:{color};color:white;padding:2px 8px;border-radius:6px;"
            f"font-size:0.8rem'>{category}</span>")


# ------------------------------------------------------------------ data ----
with st.spinner("Loading analytics from the data store..."):
    hotspots = data.load("hotspots")
    daily = data.load("daily_stats")
    monthly = data.load("monthly_stats")
    diurnal = data.load("diurnal_profile")
    city_daily = data.load("city_daily")
    seasonal = data.load("seasonal_hotspots")
    clusters = data.load("cluster_profiles")
    forecasts = data.load("forecasts")
    metrics = data.load("model_metrics")
    importance = data.load("feature_importance")
    test_pred = data.load("test_predictions")

if daily.empty or hotspots.empty:
    st.error("No processed data found. Run the pipeline first:  `python -m src.pipeline`")
    st.stop()

# --------------------------------------------------------------- sidebar ----
info = data.run_info()
st.sidebar.title("🌫️ Pollution Hotspots")
st.sidebar.caption("Mumbai & Thane · Big Data analytics with Apache Spark + MongoDB")

cities = st.sidebar.multiselect("City", sorted(daily["city"].unique()),
                                default=sorted(daily["city"].unique()))
min_d, max_d = daily["date"].min().date(), daily["date"].max().date()
date_range = st.sidebar.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = map(pd.Timestamp, date_range)
else:
    d0, d1 = pd.Timestamp(min_d), pd.Timestamp(max_d)

src_label = {"openaq": "OpenAQ ground stations", "hybrid": "Open-Meteo CAMS + land-use downscaling",
             "openmeteo": "Open-Meteo CAMS model", "synthetic": "Offline simulator"}
st.sidebar.divider()
st.sidebar.markdown(f"**Data source:** {src_label.get(info.get('source'), info.get('source', '?'))}")
st.sidebar.markdown(f"**Serving layer:** {data.backend()}")
st.sidebar.markdown(f"**Coverage:** {info.get('start_date', '?')} → {info.get('end_date', '?')}")
st.sidebar.markdown(f"**Hourly records:** {info.get('rows', 0):,} · **Sites:** {info.get('sites', '?')}")
if st.sidebar.button("↻ Reload data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

d_f = daily[daily["city"].isin(cities) & daily["date"].between(d0, d1)]
cd_f = city_daily[city_daily["city"].isin(cities) & city_daily["date"].between(d0, d1)]
hs_f = hotspots[hotspots["city"].isin(cities)]
sites_f = hs_f.sort_values("site_name")

st.title("Smart City Pollution Hotspot Detection")
st.caption(f"Air-quality analytics for {', '.join(cities) or 'no city'} · "
           f"{d0:%d %b %Y} – {d1:%d %b %Y} · Indian NAQI (CPCB) scale")

tab_overview, tab_map, tab_history, tab_pred, tab_data = st.tabs(
    ["📊 Overview", "🗺️ Hotspot Map", "📈 Historical Patterns", "🔮 Predictions", "🗄️ Data Explorer"])

# ============================================================== OVERVIEW ====
with tab_overview:
    last7 = d_f[d_f["date"] > d_f["date"].max() - pd.Timedelta(days=7)]
    prev7 = d_f[(d_f["date"] <= d_f["date"].max() - pd.Timedelta(days=7))
                & (d_f["date"] > d_f["date"].max() - pd.Timedelta(days=14))]
    worst = last7.groupby("site_name")["aqi"].mean().sort_values(ascending=False)
    k1, k2, k3, k4 = st.columns(4)
    if len(last7):
        avg_now = last7["aqi"].mean()
        delta = avg_now - prev7["aqi"].mean() if len(prev7) else None
        k1.metric("Avg AQI · last 7 days", f"{avg_now:.0f}",
                  None if delta is None else f"{delta:+.0f} vs prior week", delta_color="inverse")
        k2.metric("Worst site · last 7 days", worst.index[0], f"AQI {worst.iloc[0]:.0f}",
                  delta_color="off")
    k3.metric("Critical hotspots", int((hs_f["severity_rank"] == 0).sum()),
              f"{int(hs_f['gi_class'].str.startswith('Hotspot').sum())} spatially significant (Gi*)",
              delta_color="off")
    exceed = 100 * d_f["pm2_5_exceeds"].mean() if len(d_f) else 0
    k4.metric("Site-days above PM2.5 limit", f"{exceed:.1f}%",
              f"NAAQS 24h limit {NAAQS_24H['pm2_5']:.0f} µg/m³", delta_color="off")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.line(cd_f.sort_values("date"), x="date", y="aqi", color="city",
                      color_discrete_map=CITY_COLORS, title="Daily mean AQI by city",
                      labels={"aqi": "AQI", "date": ""})
        for upper, name, color in CATEGORIES[:4]:
            fig.add_hline(y=upper, line_dash="dot", line_color=color, opacity=0.6,
                          annotation_text=name, annotation_position="top left",
                          annotation_font_size=10)
        fig.update_traces(line_width=2, hovertemplate="%{x|%d %b %Y}<br>AQI %{y:.0f}")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig), use_container_width=True)
    with c2:
        cat = (d_f.groupby(["city", "aqi_category"]).size().rename("days").reset_index())
        cat["share"] = 100 * cat["days"] / cat.groupby("city")["days"].transform("sum")
        fig = px.bar(cat, y="city", x="share", color="aqi_category", orientation="h",
                     category_orders={"aqi_category": AQI_ORDER}, color_discrete_map=AQI_COLORS,
                     title="Share of site-days by AQI category",
                     labels={"share": "% of site-days", "city": "", "aqi_category": ""},
                     custom_data=["aqi_category", "days"])
        fig.update_traces(hovertemplate="%{customdata[0]}: %{x:.1f}% (%{customdata[1]} site-days)"
                                        "<extra></extra>", marker_line_width=2,
                          marker_line_color="rgba(255,255,255,0.9)")
        st.plotly_chart(style(fig), use_container_width=True)

    st.subheader("Latest readings")
    latest = data.latest_readings()
    latest = latest[latest["city"].isin(cities)].sort_values("aqi", ascending=False)
    if len(latest):
        st.caption(f"As of {latest['timestamp'].max():%d %b %Y %H:%M} IST")
        fig = px.bar(latest, x="site_name", y="aqi", color="aqi_category",
                     color_discrete_map=AQI_COLORS, category_orders={"aqi_category": AQI_ORDER},
                     labels={"site_name": "", "aqi": "AQI", "aqi_category": ""},
                     custom_data=["pm2_5", "pm10", "dominant_pollutant"])
        fig.update_traces(hovertemplate="<b>%{x}</b><br>AQI %{y:.0f}<br>PM2.5 %{customdata[0]:.0f}"
                                        " · PM10 %{customdata[1]:.0f}<br>Dominant: %{customdata[2]}"
                                        "<extra></extra>")
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(style(fig, 360), use_container_width=True)

# ========================================================== HOTSPOT MAP =====
with tab_map:
    c1, c2, c3 = st.columns([1, 1, 2])
    metric = c1.selectbox("Heatmap metric", ["aqi", "pm2_5", "pm10", "no2", "so2", "o3"],
                          format_func=label)
    period = c2.selectbox("Period", ["Selected date range", "Winter", "Summer", "Monsoon",
                                     "Post-Monsoon"])
    if period == "Selected date range":
        agg = d_f.groupby(["site_id", "site_name", "city", "lat", "lon"])[metric].mean().reset_index()
        labels_df = hs_f[["site_id", "hotspot_label", "gi_class", "hotspot_score", "hotspot_rank"]]
    else:
        agg = (daily[daily["city"].isin(cities) & (daily["season"] == period)]
               .groupby(["site_id", "site_name", "city", "lat", "lon"])[metric].mean().reset_index())
        labels_df = seasonal[seasonal["season"] == period][["site_id", "hotspot_label"]]
        labels_df = labels_df.merge(hs_f[["site_id", "gi_class", "hotspot_score", "hotspot_rank"]],
                                    on="site_id")
    agg = agg.merge(labels_df, on="site_id", how="left")
    c3.caption("Heat intensity = mean concentration. Markers = K-Means hotspot class "
               "(clustered on all pollutants). Hover a marker for details.")

    if agg.empty:
        st.info("No data for this selection.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Densitymap(lat=agg["lat"], lon=agg["lon"], z=agg[metric], radius=45,
                                    colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
                                    opacity=0.75, hoverinfo="skip",
                                    colorbar=dict(title=label(metric), thickness=12)))
        for lbl, color in SEVERITY_COLORS.items():
            part = agg[agg["hotspot_label"] == lbl]
            if part.empty:
                continue
            fig.add_trace(go.Scattermap(
                lat=part["lat"], lon=part["lon"], mode="markers", name=lbl,
                marker=dict(size=14, color=color),
                customdata=part[["site_name", metric, "gi_class", "hotspot_score"]].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><br>" + label(metric)
                              + " %{customdata[1]:.1f}<br>Class: " + lbl
                              + "<br>Gi*: %{customdata[2]}<br>Score %{customdata[3]}<extra></extra>"))
        fig.update_layout(map=dict(style=MAP_STYLE, center=MAP_CENTER, zoom=9.6), height=620,
                          margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hotspot ranking")
    st.caption("Score (0–100) combines mean AQI, frequency of PM2.5 limit exceedance and the "
               "Getis-Ord Gi* spatial z-score. Gi* ≥ 1.96 means the site and its neighbours "
               "(within 8 km) are significantly more polluted than the region (95% confidence).")
    rank = hs_f.sort_values("hotspot_rank")[
        ["hotspot_rank", "site_name", "city", "zone", "hotspot_label", "hotspot_score", "mean_aqi",
         "mean_pm2_5", "mean_pm10", "mean_no2", "mean_so2", "pct_days_pm25_exceed", "gi_z",
         "gi_class", "dominant_pollutant", "peak_hour"]]
    st.dataframe(rank, hide_index=True, use_container_width=True, column_config={
        "hotspot_rank": "Rank", "site_name": "Site", "city": "City", "zone": "Land use",
        "hotspot_label": "Cluster class",
        "hotspot_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100,
                                                         format="%.0f"),
        "mean_aqi": st.column_config.NumberColumn("Mean AQI", format="%.0f"),
        "mean_pm2_5": st.column_config.NumberColumn("PM2.5", format="%.1f"),
        "mean_pm10": st.column_config.NumberColumn("PM10", format="%.1f"),
        "mean_no2": st.column_config.NumberColumn("NO₂", format="%.1f"),
        "mean_so2": st.column_config.NumberColumn("SO₂", format="%.1f"),
        "pct_days_pm25_exceed": st.column_config.NumberColumn("% days > PM2.5 limit", format="%.1f"),
        "gi_z": st.column_config.NumberColumn("Gi* z", format="%.2f"), "gi_class": "Gi* result",
        "dominant_pollutant": "Dominant", "peak_hour": "Peak hour"})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cluster profiles** (K-Means centroids, original units)")
        st.dataframe(clusters.sort_values("severity_rank").drop(columns=["cluster", "severity_rank"]),
                     hide_index=True, use_container_width=True)
    with c2:
        piv = seasonal[seasonal["city"].isin(cities)].pivot_table(
            index="site_name", columns="season", values="severity_rank", aggfunc="first")
        piv = piv[[s for s in ["Winter", "Summer", "Monsoon", "Post-Monsoon"] if s in piv]]
        piv = piv.loc[piv.mean(axis=1).sort_values().index]
        lbl = seasonal.pivot_table(index="site_name", columns="season", values="hotspot_label",
                                   aggfunc="first").reindex(index=piv.index, columns=piv.columns)
        fig = go.Figure(go.Heatmap(z=piv.to_numpy(), x=piv.columns, y=piv.index, text=lbl.to_numpy(),
                                   colorscale=[[0, SEQ[4]], [0.5, SEQ[2]], [1, SEQ[0]]],
                                   showscale=False, xgap=2, ygap=2,
                                   hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>"))
        fig.update_layout(title="How hotspots shift by season (darker = more severe)")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 620), use_container_width=True)

# ===================================================== HISTORICAL PATTERNS ==
with tab_history:
    c1, c2 = st.columns([1, 3])
    pol = c1.selectbox("Pollutant", ["pm2_5", "pm10", "no2", "so2", "co", "o3", "aqi"],
                       format_func=label, key="hist_pol")
    default_sites = hs_f.sort_values("hotspot_rank")["site_name"].head(2).tolist() + \
        hs_f.sort_values("hotspot_rank")["site_name"].tail(1).tolist()
    chosen = c2.multiselect("Sites to compare (max 8)", sites_f["site_name"].tolist(),
                            default=default_sites, max_selections=8)

    sel = d_f[d_f["site_name"].isin(chosen)].sort_values("date")
    if len(sel):
        smooth = st.toggle("7-day rolling mean", value=True)
        if smooth:
            sel = sel.assign(**{pol: sel.groupby("site_name")[pol]
                                .transform(lambda s: s.rolling(7, min_periods=1).mean())})
        fig = px.line(sel, x="date", y=pol, color="site_name",
                      color_discrete_sequence=SERIES, category_orders={"site_name": chosen},
                      title=f"Daily {label(pol)}" + (" (7-day mean)" if smooth else ""),
                      labels={pol: label(pol) + UNITS[pol], "date": "", "site_name": ""})
        if pol in NAAQS_24H and pol != "co":
            fig.add_hline(y=NAAQS_24H[pol], line_dash="dot", line_color="#d9480f",
                          annotation_text="NAAQS 24h limit", annotation_position="top left")
        fig.update_traces(line_width=2)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(style(fig, 420), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        m = monthly[monthly["city"].isin(cities)]
        piv = m.pivot_table(index="site_name", columns="month", values=pol)
        piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]
        fig = go.Figure(go.Heatmap(z=piv.to_numpy(), x=piv.columns, y=piv.index,
                                   colorscale=[[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)],
                                   xgap=2, ygap=2, colorbar=dict(title=label(pol), thickness=10),
                                   hovertemplate="%{y} · %{x}<br>" + label(pol)
                                                 + " %{z:.1f}<extra></extra>"))
        fig.update_layout(title=f"Monthly mean {label(pol)} by site")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style(fig, 640), use_container_width=True)
    with c2:
        dsel = diurnal[diurnal["site_name"].isin(chosen)] if pol in diurnal else pd.DataFrame()
        if len(dsel):
            fig = px.line(dsel.sort_values("hour"), x="hour", y=pol, color="site_name",
                          color_discrete_sequence=SERIES, category_orders={"site_name": chosen},
                          markers=True, title=f"Average daily cycle of {label(pol)}",
                          labels={"hour": "Hour of day (IST)", pol: label(pol), "site_name": ""})
            fig.update_traces(line_width=2, marker_size=8)
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(style(fig, 310), use_container_width=True)

        wk = d_f.assign(dow=d_f["date"].dt.day_name()).groupby(["city", "dow"])[pol].mean().reset_index()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fig = px.bar(wk, x="dow", y=pol, color="city", barmode="group",
                     color_discrete_map=CITY_COLORS, category_orders={"dow": order},
                     title=f"{label(pol)} by day of week",
                     labels={"dow": "", pol: label(pol), "city": ""})
        fig.update_traces(marker_line_width=2, marker_line_color="rgba(255,255,255,0.9)",
                          hovertemplate="%{x}: %{y:.1f}<extra></extra>")
        st.plotly_chart(style(fig, 310), use_container_width=True)

    st.subheader("Weather influence")
    c1, c2 = st.columns(2)
    wvar = c1.selectbox("Weather variable", ["wind_speed", "temperature", "humidity",
                                             "precipitation", "pressure"],
                        format_func=lambda s: s.replace("_", " ").title())
    sample = d_f.sample(min(4000, len(d_f)), random_state=1) if len(d_f) else d_f
    if len(sample):
        corr = d_f[[wvar, pol]].corr().iloc[0, 1]
        c2.metric(f"Correlation ({wvar.replace('_', ' ')} vs {label(pol)})", f"{corr:+.2f}")
        fig = px.scatter(sample, x=wvar, y=pol, color="city", color_discrete_map=CITY_COLORS,
                         opacity=0.45,
                         labels={wvar: wvar.replace("_", " ").title(), pol: label(pol), "city": ""},
                         hover_data={"site_name": True, "date": "|%d %b %Y"})
        fig.update_traces(marker_size=8)
        st.plotly_chart(style(fig, 380), use_container_width=True)

# ============================================================ PREDICTIONS ===
with tab_pred:
    if forecasts.empty:
        st.info("No forecasts yet - run the forecast stage.")
    else:
        fc = forecasts[forecasts["city"].isin(cities)]
        methods = (dict(zip(metrics["target"], metrics["selected_method"]))
                   if "selected_method" in metrics else {})
        method_txt = " · ".join(f"{label(t)}: {'Spark MLlib Gradient-Boosted Trees' if mth == 'GBTRegressor' else 'persistence (GBT did not beat it on the hold-out)'}"
                                for t, mth in methods.items()) or "Spark MLlib Gradient-Boosted Trees"
        st.caption(f"7-day forecast from {fc['date'].min():%d %b} to {fc['date'].max():%d %b %Y} · "
                   f"{method_txt} · weather input: "
                   f"{', '.join(fc['weather_source'].dropna().unique())}")
        days = sorted(fc["date"].dt.date.unique())
        day = st.select_slider("Forecast day", options=days, value=days[0],
                               format_func=lambda d: d.strftime("%a %d %b"))
        fday = fc[fc["date"].dt.date == day]

        c1, c2 = st.columns([3, 2])
        with c1:
            fig = go.Figure()
            for cat_name in AQI_ORDER:
                part = fday[fday["pred_aqi_category"] == cat_name]
                if part.empty:
                    continue
                fig.add_trace(go.Scattermap(
                    lat=part["lat"], lon=part["lon"], mode="markers", name=cat_name,
                    marker=dict(size=10 + part["pred_aqi"] / 12, color=AQI_COLORS[cat_name],
                                opacity=0.9),
                    customdata=part[["site_name", "pred_aqi", "pred_pm2_5", "lower_80_aqi",
                                     "upper_80_aqi"]].to_numpy(),
                    hovertemplate="<b>%{customdata[0]}</b><br>Predicted AQI %{customdata[1]:.0f} "
                                  "(80%: %{customdata[3]:.0f}–%{customdata[4]:.0f})<br>"
                                  "PM2.5 %{customdata[2]:.0f} µg/m³<extra></extra>"))
            fig.update_layout(map=dict(style=MAP_STYLE, center=MAP_CENTER, zoom=9.4), height=520,
                              margin=dict(l=0, r=0, t=0, b=0),
                              legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            top = fday.sort_values("pred_aqi", ascending=False)
            st.markdown(f"**Predicted AQI · {day:%A %d %B}**")
            st.dataframe(top[["site_name", "pred_aqi", "pred_aqi_category", "pred_pm2_5"]],
                         hide_index=True, use_container_width=True, height=470, column_config={
                             "site_name": "Site",
                             "pred_aqi": st.column_config.NumberColumn("AQI", format="%.0f"),
                             "pred_aqi_category": "Category",
                             "pred_pm2_5": st.column_config.NumberColumn("PM2.5", format="%.0f")})

        st.subheader("Site outlook")
        c1, c2 = st.columns([1, 3])
        site = c1.selectbox("Site", hs_f.sort_values("hotspot_rank")["site_name"].tolist())
        target = c1.radio("Target", ["aqi", "pm2_5"], format_func=label, horizontal=True)
        sid = hs_f.loc[hs_f["site_name"] == site, "site_id"].iloc[0]
        hist = daily[(daily["site_id"] == sid)].sort_values("date").tail(30)
        f1 = forecasts[forecasts["site_id"] == sid].sort_values("date")
        pred_col = f"pred_{target}"
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pd.concat([f1["date"], f1["date"][::-1]]),
                                 y=pd.concat([f1[f"upper_80_{target}"], f1[f"lower_80_{target}"][::-1]]),
                                 fill="toself", fillcolor="rgba(235,104,52,0.15)", line_width=0,
                                 name="80% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist["date"], y=hist[target], name="Observed",
                                 line=dict(color=SERIES[0], width=2),
                                 hovertemplate="%{x|%d %b}: %{y:.0f}<extra>Observed</extra>"))
        bridge = pd.concat([hist.tail(1).rename(columns={target: pred_col}), f1])
        fig.add_trace(go.Scatter(x=bridge["date"], y=bridge[pred_col], name="Forecast",
                                 line=dict(color=SERIES[1], width=2, dash="dash"),
                                 mode="lines+markers", marker_size=8,
                                 hovertemplate="%{x|%d %b}: %{y:.0f}<extra>Forecast</extra>"))
        fig.update_layout(title=f"{site}: last 30 days and 7-day {label(target)} forecast",
                          yaxis_title=label(target), hovermode="x unified")
        c2.plotly_chart(style(fig, 380), use_container_width=True)

        st.subheader("Model performance (chronological hold-out)")
        cols = st.columns(len(metrics))
        for col, (_, m) in zip(cols, metrics.iterrows()):
            col.markdown(f"**{label(m['target'])} model** · tested on data from {m['split_date']}")
            a, b, c = col.columns(3)
            a.metric("RMSE", f"{m['rmse']:.1f}",
                     f"{m['improvement_vs_baseline_pct']:+.1f}% vs persistence", delta_color="normal")
            b.metric("MAE", f"{m['mae']:.1f}")
            c.metric("R²", f"{m['r2']:.2f}")
            if m.get("selected_method") == "Persistence":
                col.warning(f"GBT lost to persistence (RMSE {m['baseline_rmse']:.1f}) on this hold-out, "
                            "so the forecast uses persistence. The training data covers too short a "
                            "period (e.g. only monsoon); ingest a longer history to fix this.")
            elif m.get("selected_method"):
                col.success("GBT beat persistence on the hold-out and is used for the forecast.")

        c1, c2 = st.columns(2)
        with c1:
            tp = test_pred[(test_pred["site_id"] == sid) & (test_pred["target"] == target)].sort_values("date")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tp["date"], y=tp["actual"], name="Actual",
                                     line=dict(color=SERIES[0], width=2)))
            fig.add_trace(go.Scatter(x=tp["date"], y=tp["predicted"], name="Predicted (next day)",
                                     line=dict(color=SERIES[1], width=2)))
            fig.update_layout(title=f"Back-test at {site}: actual vs predicted {label(target)}",
                              hovermode="x unified")
            st.plotly_chart(style(fig, 360), use_container_width=True)
        with c2:
            imp = importance[importance["target"] == target].sort_values("importance").tail(10)
            imp = imp.assign(feature=imp["feature"].str.replace(f"{target}_", "")
                             .str.replace("next_", "forecast ").str.replace("_", " "))
            fig = px.bar(imp, x="importance", y="feature", orientation="h",
                         title=f"Top features · {label(target)} model",
                         labels={"importance": "Relative importance", "feature": ""})
            fig.update_traces(marker_color=SERIES[0], hovertemplate="%{y}: %{x:.3f}<extra></extra>")
            st.plotly_chart(style(fig, 360), use_container_width=True)

# =========================================================== DATA EXPLORER ==
with tab_data:
    st.markdown("Query the hourly measurement store directly "
                f"(**{data.backend()}** · collection `hourly_measurements`).")
    c1, c2 = st.columns([2, 1])
    q_sites = c1.multiselect("Sites", sites_f["site_name"].tolist(),
                             default=sites_f.sort_values("hotspot_rank")["site_name"].head(1).tolist())
    q_end = d1
    q_range = c2.date_input("Hours between", (q_end - timedelta(days=6), q_end),
                            min_value=min_d, max_value=max_d, key="q_range")
    if q_sites and isinstance(q_range, tuple) and len(q_range) == 2:
        ids = tuple(hs_f.loc[hs_f["site_name"].isin(q_sites), "site_id"])
        hdf = data.hourly(ids, datetime.combine(q_range[0], time.min),
                          datetime.combine(q_range[1], time.max))
        st.caption(f"{len(hdf):,} hourly records")
        if len(hdf):
            fig = px.line(hdf, x="timestamp", y="aqi", color="site_name",
                          color_discrete_sequence=SERIES, title="Hourly AQI",
                          labels={"timestamp": "", "aqi": "AQI", "site_name": ""})
            fig.update_traces(line_width=2)
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(style(fig, 320), use_container_width=True)
            st.dataframe(hdf, hide_index=True, use_container_width=True, height=360)
            st.download_button("Download CSV", hdf.to_csv(index=False).encode(),
                               file_name="hourly_air_quality.csv", mime="text/csv")

st.divider()
st.caption("Pipeline: Open-Meteo / OpenAQ → Apache Spark (ETL, K-Means, Gi*, GBT forecasting) → "
           "MongoDB → Streamlit. AQI per CPCB National Air Quality Index.")
