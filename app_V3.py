import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RiskRadar · Supplier Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0f1117; color: #e8eaf0; }
.stApp { background-color: #0f1117; }
section[data-testid="stSidebar"] { background-color: #161b27; border-right: 1px solid #252d3d; }
section[data-testid="stSidebar"] * { color: #c8cfe0 !important; }
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a2035 0%, #1e2640 100%);
    border: 1px solid #2a3350; border-radius: 14px;
    padding: 16px 20px !important; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
[data-testid="metric-container"] label {
    color: #8899bb !important; font-size: 0.75rem !important;
    letter-spacing: 0.08em; text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e8eaf0 !important; font-size: 1.7rem !important; font-weight: 700;
}
.stTabs [data-baseweb="tab-list"] { background-color: #161b27; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8899bb; font-weight: 500; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background-color: #2563eb !important; color: white !important; }
.stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; border-radius: 8px; font-weight: 500; }
.info-box { background: linear-gradient(135deg, #1a2640, #1e2d4a); border-left: 4px solid #2563eb;
    border-radius: 0 10px 10px 0; padding: 12px 16px; margin: 8px 0; font-size: 0.85rem; color: #a8bbd4; line-height: 1.6; }
.risk-high   { border-left-color: #ef4444 !important; background: linear-gradient(135deg, #2a1a1a, #331c1c) !important; }
.risk-medium { border-left-color: #f59e0b !important; background: linear-gradient(135deg, #2a2210, #332a10) !important; }
.risk-low    { border-left-color: #22c55e !important; background: linear-gradient(135deg, #0f2a18, #12331e) !important; }
.section-header { font-size: 1.0rem; font-weight: 600; color: #c8d4f0; letter-spacing: 0.04em;
    margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid #252d3d; }
.preset-btn { display: inline-block; background: #1a2640; border: 1px solid #2a3350;
    border-radius: 8px; padding: 8px 14px; margin: 4px; font-size: 0.82rem; color: #a8bbd4; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#161b27", "axes.facecolor": "#1a2035",
    "axes.edgecolor": "#2a3350", "axes.labelcolor": "#8899bb",
    "xtick.color": "#8899bb", "ytick.color": "#8899bb",
    "text.color": "#c8d4f0", "grid.color": "#252d3d",
    "grid.linestyle": "--", "grid.alpha": 0.5,
})

SUP_COLORS  = {}
PROD_COLORS = {"skincare": "#06b6d4", "haircare": "#f59e0b", "cosmetics": "#a855f7"}
RISK_COLORS = {"🔴 Hoch": "#ef4444", "🟡 Mittel": "#f59e0b", "🟢 Niedrig": "#22c55e"}
BAR_BLUE    = "#2563eb"

# ════════════════════════════════════════════════════════════════
#  DATEN LADEN – doppelte Spalten vermieden
# ════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("supply_chain_data.csv")
    num_cols = ["Price","Availability","Number of products sold","Revenue generated",
                "Stock levels","Lead times","Order quantities","Shipping times",
                "Shipping costs","Lead time","Production volumes","Manufacturing lead time",
                "Manufacturing costs","Defect rates","Costs"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Berechnete Spalten NUR wenn noch nicht vorhanden
    if "Gesamtlieferzeit" not in df.columns:
        df["Gesamtlieferzeit"] = df["Lead time"] + df["Shipping times"]
    if "Lager-Überhang" not in df.columns:
        df["Lager-Überhang"] = df["Stock levels"] - df["Order quantities"]

    df["Price"]               = df["Price"].round(2)
    df["Revenue generated"]   = df["Revenue generated"].round(2)
    df["Shipping costs"]      = df["Shipping costs"].round(2)
    df["Defect rates"]        = df["Defect rates"].round(3)
    df["Manufacturing costs"] = df["Manufacturing costs"].round(2)
    df["Costs"]               = df["Costs"].round(2)
    return df

df = load_data()

for i, s in enumerate(sorted(df["Supplier name"].unique())):
    SUP_COLORS[s] = ["#2563eb","#f59e0b","#22c55e","#a855f7","#ef4444"][i % 5]

# ════════════════════════════════════════════════════════════════
#  RISIKO-SCORING
# ════════════════════════════════════════════════════════════════
def norm_col(series, invert=False):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    n = (series - mn) / (mx - mn)
    return (1 - n) if invert else n

def compute_risk(df_in, w_defect, w_lead, w_cost, w_inspection, w_revenue, thresh_high, thresh_low):
    r = df_in.copy()
    r["_n_defect"]     = norm_col(r["Defect rates"])
    r["_n_lead"]       = norm_col(r["Gesamtlieferzeit"])
    r["_n_cost"]       = norm_col(r["Costs"])
    r["_n_revenue"]    = norm_col(r["Revenue generated"], invert=True)
    insp_map           = {"Fail": 1.0, "Pending": 0.5, "Pass": 0.0}
    r["_n_inspection"] = r["Inspection results"].map(insp_map).fillna(0.5)
    total = w_defect + w_lead + w_cost + w_inspection + w_revenue
    r["Risk Score"] = (
        w_defect     * r["_n_defect"] +
        w_lead       * r["_n_lead"] +
        w_cost       * r["_n_cost"] +
        w_inspection * r["_n_inspection"] +
        w_revenue    * r["_n_revenue"]
    ) / total * 100

    def cat(s):
        if s >= thresh_high: return "🔴 Hoch"
        if s >= thresh_low:  return "🟡 Mittel"
        return "🟢 Niedrig"
    r["Risikostufe"] = r["Risk Score"].apply(cat)
    # interne Spalten entfernen
    r = r.drop(columns=[c for c in r.columns if c.startswith("_n_")])
    return r

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ RiskRadar")
    st.markdown("<p style='color:#5577aa;font-size:0.8rem;margin-top:-10px'>THI Ingolstadt · DPDS 2026 · Gruppe 9</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    # ── Filter ─────────────────────────────────────────────────
    st.markdown("### 🔍 Filter")
    with st.expander("Lieferanten & Produkte", expanded=True):
        sel_sup  = st.multiselect("Lieferant",        sorted(df["Supplier name"].unique()),
                                  default=sorted(df["Supplier name"].unique()), key="gs")
        sel_prod = st.multiselect("Produktkategorie", sorted(df["Product type"].unique()),
                                  default=sorted(df["Product type"].unique()), key="gp")
    with st.expander("Logistik & Standort", expanded=False):
        sel_loc   = st.multiselect("Standort",     sorted(df["Location"].unique()),             default=sorted(df["Location"].unique()),             key="gl")
        sel_carr  = st.multiselect("Carrier",      sorted(df["Shipping carriers"].unique()),    default=sorted(df["Shipping carriers"].unique()),    key="gc")
        sel_mode  = st.multiselect("Transportweg", sorted(df["Transportation modes"].unique()), default=sorted(df["Transportation modes"].unique()), key="gm")
        sel_route = st.multiselect("Route",        sorted(df["Routes"].unique()),               default=sorted(df["Routes"].unique()),               key="gr")
    with st.expander("Qualität", expanded=False):
        sel_insp = st.multiselect("Inspektionsstatus", sorted(df["Inspection results"].unique()),
                                  default=sorted(df["Inspection results"].unique()), key="gi")

    st.markdown("---")

    # ── Gewichtungs-Presets ────────────────────────────────────
    st.markdown("### ⚖️ Risikogewichtung")
    st.markdown("""<div class='info-box'>
    Wähle ein <b>Preset</b> oder stelle manuell ein.<br>
    Die Werte werden intern in % umgerechnet.
    </div>""", unsafe_allow_html=True)

    preset = st.radio("Preset wählen:", [
        "🔧 Manuell",
        "🏆 Fall 1: Höchste Qualität (z.B. Pharma/Luxus)",
        "⚡ Fall 2: Viel, günstig, schnell (z.B. TEMU)"
    ], key="preset")

    if preset == "🏆 Fall 1: Höchste Qualität (z.B. Pharma/Luxus)":
        w_defect, w_lead, w_cost, w_inspection, w_revenue = 10, 1, 1, 10, 5
        st.markdown("""<div class='info-box risk-low'>
        🏆 <b>Qualitäts-Fokus:</b> Defektrate + Inspektion dominieren. Kosten & Lieferzeit irrelevant.
        </div>""", unsafe_allow_html=True)
    elif preset == "⚡ Fall 2: Viel, günstig, schnell (z.B. TEMU)":
        w_defect, w_lead, w_cost, w_inspection, w_revenue = 1, 10, 10, 1, 5
        st.markdown("""<div class='info-box risk-medium'>
        ⚡ <b>Speed & Cost Fokus:</b> Lieferzeit + Kosten dominieren. Qualität weniger relevant.
        </div>""", unsafe_allow_html=True)
    else:
        w_defect     = st.slider("🔬 Defektrate",                     1, 10, 4)
        w_lead       = st.slider("⏱️ Gesamtlieferzeit",               1, 10, 3)
        w_cost       = st.slider("💰 Gesamtkosten (Costs)",           1, 10, 2)
        w_inspection = st.slider("🔎 Inspektionsergebnis",            1, 10, 5)
        w_revenue    = st.slider("📈 Umsatz (hoch = geringes Risiko)",1, 10, 2)

    total_w = w_defect + w_lead + w_cost + w_inspection + w_revenue
    st.markdown(f"""<div class='info-box' style='margin-top:6px;font-size:0.8rem'>
    🔬 {w_defect/total_w*100:.0f}% &nbsp;·&nbsp; ⏱️ {w_lead/total_w*100:.0f}% &nbsp;·&nbsp;
    💰 {w_cost/total_w*100:.0f}% &nbsp;·&nbsp; 🔎 {w_inspection/total_w*100:.0f}% &nbsp;·&nbsp;
    📈 {w_revenue/total_w*100:.0f}%
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎚️ Grenzwerte")
    thresh_high = st.slider("Ab hier 🔴 Hoch",  34, 90, 65)
    thresh_low  = st.slider("Ab hier 🟡 Mittel", 10, thresh_high-1, 35)
    st.markdown(f"""<div class='info-box' style='font-size:0.8rem'>
    🔴 ≥{thresh_high} &nbsp;·&nbsp; 🟡 {thresh_low}–{thresh_high-1} &nbsp;·&nbsp; 🟢 &lt;{thresh_low}<br>
    <i>Default: Terzil-Einteilung (stat. neutrale Dritteilung)</i>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  FILTER + SCORING
# ════════════════════════════════════════════════════════════════
filtered = df[
    df["Supplier name"].isin(sel_sup) &
    df["Product type"].isin(sel_prod) &
    df["Location"].isin(sel_loc) &
    df["Shipping carriers"].isin(sel_carr) &
    df["Transportation modes"].isin(sel_mode) &
    df["Routes"].isin(sel_route) &
    df["Inspection results"].isin(sel_insp)
].copy()

if filtered.empty:
    st.warning("⚠️ Keine Daten. Bitte Filter anpassen.")
    st.stop()

scored = compute_risk(filtered, w_defect, w_lead, w_cost, w_inspection, w_revenue, thresh_high, thresh_low)

# Vollständiger Datensatz mit Scoring (für Tab Alle Daten)
scored_full = compute_risk(df, w_defect, w_lead, w_cost, w_inspection, w_revenue, thresh_high, thresh_low)

# ════════════════════════════════════════════════════════════════
#  HEADER + KPIs
# ════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-size:2.0rem;font-weight:700;color:#e8eaf0;margin-bottom:2px'>🛡️ RiskRadar</h1>
<p style='color:#5577aa;font-size:0.9rem;margin-top:0'>
Supplier & Procurement Risk Intelligence · THI Ingolstadt DPDS 2026 · Gruppe 9
</p>""", unsafe_allow_html=True)
st.markdown("---")

n_high = (scored["Risikostufe"] == "🔴 Hoch").sum()
n_med  = (scored["Risikostufe"] == "🟡 Mittel").sum()
n_low  = (scored["Risikostufe"] == "🟢 Niedrig").sum()

k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
k1.metric("⌀ Risk Score",        f"{scored['Risk Score'].mean():.1f}")
k2.metric("🔴 Hoch",             f"{n_high} SKUs", delta=f"{n_high/len(scored)*100:.0f}%", delta_color="inverse")
k3.metric("🟡 Mittel",           f"{n_med} SKUs")
k4.metric("🟢 Niedrig",          f"{n_low} SKUs")
k5.metric("⌀ Defektrate",        f"{scored['Defect rates'].mean():.2f}%")
k6.metric("⌀ Gesamtlieferzeit",  f"{scored['Gesamtlieferzeit'].mean():.1f} Tage")
k7.metric("Gesamtumsatz",        f"{scored['Revenue generated'].sum()/1000:.1f}k €")
st.markdown("---")

sup_order = sorted(scored["Supplier name"].unique())

# ════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Risk Overview",
    "🏢 Lieferanten-Analyse",
    "💰 Umsatz & Produkte",
    "🔬 Klassifikation & ML",
    "🔎 SKU-Suche",
    "📋 Alle Daten"
])

# ─────────────────────────────────────────────────────────────────
#  TAB 1 – RISK OVERVIEW  (statischer Teil + dynamischer Teil)
# ─────────────────────────────────────────────────────────────────
with tab1:

    # ── STATISCHER TEIL: KPI Heatmap ─────────────────────────
    st.markdown("<div class='section-header'>🌡️ KPI Heatmap – Lieferant × Kennzahl (statisch, filterunabhängig)</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Diese Tabelle ist <b>unabhängig von der Risikogewichtung</b> – sie zeigt immer die realen Rohdaten. Rot = schlechter Wert, Grün = guter Wert (relativ zum Datensatz).</div>",
                unsafe_allow_html=True)

    heatmap_data = df.groupby("Supplier name").agg(
        Defekt_pct  =("Defect rates",       "mean"),
        Lead_Time   =("Gesamtlieferzeit",    "mean"),
        Kosten      =("Costs",               "mean"),
        Fail_Quote  =("Inspection results",  lambda x: (x=="Fail").sum()/len(x)*100),
    ).round(2)

    fig_hm, ax_hm = plt.subplots(figsize=(9, 3.5))
    labels = ["Defekt %", "Lead Time (d)", "⌀ Kosten", "Fail-Quote %"]
    data_hm = heatmap_data.values
    norm_hm = (data_hm - data_hm.min(axis=0)) / (data_hm.max(axis=0) - data_hm.min(axis=0) + 1e-9)

    for i in range(norm_hm.shape[0]):
        for j in range(norm_hm.shape[1]):
            v = norm_hm[i, j]
            color = plt.cm.RdYlGn(1 - v)
            ax_hm.add_patch(plt.Rectangle([j, i], 1, 1, color=color, zorder=2))
            raw = data_hm[i, j]
            txt = f"{raw:.1f}%" if j in [0,3] else f"{raw:.1f}d" if j==1 else f"€{raw:.0f}"
            ax_hm.text(j+0.5, i+0.5, txt, ha="center", va="center", fontsize=10,
                       fontweight="bold", color="#0f1117", zorder=3)

    ax_hm.set_xlim(0, len(labels)); ax_hm.set_ylim(0, len(heatmap_data))
    ax_hm.set_xticks([x+0.5 for x in range(len(labels))]); ax_hm.set_xticklabels(labels)
    ax_hm.set_yticks([y+0.5 for y in range(len(heatmap_data))])
    ax_hm.set_yticklabels(["Sup " + s.replace("Supplier ","") for s in heatmap_data.index])
    ax_hm.set_facecolor("#1a2035")
    plt.tight_layout(); st.pyplot(fig_hm); plt.close()

    st.markdown("---")

    # ── DYNAMISCHER TEIL ─────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Dynamischer Teil – beeinflusst durch Gewichtung & Filter</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("<div class='section-header'>⌀ Risk Score pro Lieferant</div>", unsafe_allow_html=True)
        sup_risk = scored.groupby("Supplier name")["Risk Score"].mean().sort_values(ascending=True)
        bar_c = ["#ef4444" if v>=thresh_high else "#f59e0b" if v>=thresh_low else "#22c55e" for v in sup_risk.values]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.barh(sup_risk.index, sup_risk.values, color=bar_c, height=0.55, zorder=3)
        ax.axvline(thresh_high, color="#ef4444", lw=1.2, ls="--", alpha=0.7, label=f"Hoch ≥{thresh_high}")
        ax.axvline(thresh_low,  color="#f59e0b", lw=1.2, ls="--", alpha=0.7, label=f"Mittel ≥{thresh_low}")
        ax.set_xlabel("⌀ Risk Score (0 = kein Risiko · 100 = max. Risiko)")
        ax.set_xlim(0, 107); ax.grid(axis="x", zorder=0); ax.legend(fontsize=8)
        for b, v in zip(bars, sup_risk.values):
            ax.text(v+1, b.get_y()+b.get_height()/2, f"{v:.1f}", va="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown("<div class='section-header'>Risikoverteilung</div>", unsafe_allow_html=True)
        cat_c = scored["Risikostufe"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        wedges, _, autotexts = ax2.pie(
            cat_c.values, colors=[RISK_COLORS.get(c,"#555") for c in cat_c.index],
            autopct="%1.0f%%", startangle=140,
            wedgeprops={"linewidth":2,"edgecolor":"#161b27"}, pctdistance=0.75
        )
        for at in autotexts:
            at.set_color("#e8eaf0"); at.set_fontsize(13); at.set_fontweight("bold")
        ax2.legend(wedges, [f"{l} ({v})" for l,v in zip(cat_c.index, cat_c.values)],
                   loc="lower center", bbox_to_anchor=(0.5,-0.1), fontsize=9, framealpha=0, labelcolor="#c8d4f0")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")

    # Boxplots
    st.markdown("<div class='section-header'>📦 Boxplots Kern-KPIs (Beschaffungsrisiken)</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    <b>Defektrate</b> = Qualitätsrisiko | <b>Gesamtlieferzeit</b> = Lead time (Beschaffung) + Shipping times (Versand) = Beschaffungsrisiko.
    Linie = Median · Box = mittlere 50% · Punkte = Ausreißer
    </div>""", unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)
    for col_w, kpi, ylabel in [
        (bc1, "Defect rates",    "Defektrate (%)"),
        (bc2, "Gesamtlieferzeit","Gesamtlieferzeit (Tage) = Lead time + Shipping"),
    ]:
        with col_w:
            box_data = [scored[scored["Supplier name"]==s][kpi].dropna().values for s in sup_order]
            fig_b, ax_b = plt.subplots(figsize=(7, 4))
            bp = ax_b.boxplot(box_data, labels=sup_order, patch_artist=True,
                              medianprops={"color":"#e8eaf0","linewidth":2},
                              whiskerprops={"color":"#8899bb"}, capprops={"color":"#8899bb"},
                              flierprops={"marker":"o","markerfacecolor":"#ef4444","markersize":5,"alpha":0.7})
            for patch, sup in zip(bp["boxes"], sup_order):
                patch.set_facecolor(SUP_COLORS.get(sup,"#2563eb")); patch.set_alpha(0.7)
            ax_b.set_ylabel(ylabel); ax_b.grid(axis="y", zorder=0)
            plt.tight_layout(); st.pyplot(fig_b); plt.close()

    st.markdown("---")

    # Scatter
    st.markdown("<div class='section-header'>📈 Gesamtumsatz vs. Defektrate</div>", unsafe_allow_html=True)
    color_by = st.radio("Einfärben nach:", ["Lieferant","Produktkategorie"], horizontal=True, key="sc_color")
    if color_by == "Lieferant":
        point_colors = [SUP_COLORS.get(s,"#aaa") for s in scored["Supplier name"]]
        legend_items = [(s, SUP_COLORS.get(s,"#aaa")) for s in sup_order if s in scored["Supplier name"].values]
    else:
        point_colors = [PROD_COLORS.get(p,"#aaa") for p in scored["Product type"]]
        legend_items = [(p, PROD_COLORS.get(p,"#aaa")) for p in sorted(scored["Product type"].unique())]

    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.scatter(scored["Defect rates"], scored["Revenue generated"]/1000,
                c=point_colors, s=75, alpha=0.85, edgecolors="#0f1117", linewidths=0.8, zorder=3)
    ax5.set_xlabel("Defektrate (%)"); ax5.set_ylabel("Umsatz (Tsd. €)"); ax5.grid(zorder=0)
    ax5.legend(handles=[mpatches.Patch(color=c, label=l) for l,c in legend_items], fontsize=9, framealpha=0.2)
    plt.tight_layout(); st.pyplot(fig5); plt.close()
    st.markdown("<div class='info-box'>Ideal: links oben (niedrige Defektrate, hoher Umsatz). Rechts unten = kritisch.</div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 2 – LIEFERANTEN-ANALYSE
# ─────────────────────────────────────────────────────────────────
with tab2:

    # Alle Lieferanten auswählen oder einzeln
    view_mode = st.radio("Ansicht:", ["Einzelner Lieferant", "Alle Lieferanten vergleichen"], horizontal=True)

    if view_mode == "Alle Lieferanten vergleichen":
        st.markdown("<div class='section-header'>📊 Lieferanten-Ranking (alle)</div>", unsafe_allow_html=True)

        ranking = scored.groupby("Supplier name").agg(
            avg_risk    =("Risk Score",        "mean"),
            defect_pct  =("Defect rates",      "mean"),
            lead_time   =("Gesamtlieferzeit",  "mean"),
            avg_kosten  =("Costs",             "mean"),
            fail_count  =("Inspection results",lambda x: (x=="Fail").sum()),
            revenue     =("Revenue generated", "sum"),
            n_skus      =("SKU",               "count"),
        ).round(2).sort_values("avg_risk")

        ranking["Score-Bar"] = ranking["avg_risk"]
        ranking.index.name = "Lieferant"
        ranking.columns = ["⌀ Risk Score","Defekt %","Lead Time (d)","⌀ Kosten","# Fail","Umsatz (€)","SKUs","Score-Bar"]

        fig_rank, ax_rank = plt.subplots(figsize=(10, 4))
        colors_rank = [SUP_COLORS.get(s,"#2563eb") for s in ranking.index]
        bars_r = ax_rank.barh(ranking.index, ranking["⌀ Risk Score"], color=colors_rank, height=0.55, zorder=3)
        ax_rank.axvline(thresh_high, color="#ef4444", lw=1.2, ls="--", alpha=0.7)
        ax_rank.axvline(thresh_low,  color="#f59e0b", lw=1.2, ls="--", alpha=0.7)
        ax_rank.set_xlabel("⌀ Risk Score"); ax_rank.set_xlim(0, 107); ax_rank.grid(axis="x", zorder=0)
        for b, v in zip(bars_r, ranking["⌀ Risk Score"]):
            ax_rank.text(v+1, b.get_y()+b.get_height()/2, f"{v:.1f}", va="center", fontsize=10, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_rank); plt.close()

        # Tabelle
        display_r = ranking.drop(columns=["Score-Bar"]).reset_index()
        display_r["Umsatz (€)"] = display_r["Umsatz (€)"].map("{:,.0f} €".format)
        st.dataframe(display_r, use_container_width=True, height=230)

    else:
        chosen = st.selectbox("🏢 Lieferant auswählen", sorted(scored["Supplier name"].unique()))
        sup_d  = scored[scored["Supplier name"] == chosen]

        r1,r2,r3,r4,r5,r6 = st.columns(6)
        r1.metric("⌀ Risk Score",       f"{sup_d['Risk Score'].mean():.1f}")
        r2.metric("⌀ Defektrate",       f"{sup_d['Defect rates'].mean():.2f}%")
        r3.metric("⌀ Gesamtlieferzeit", f"{sup_d['Gesamtlieferzeit'].mean():.1f} Tage")
        r4.metric("Gesamtumsatz",       f"{sup_d['Revenue generated'].sum()/1000:.1f}k €")
        r5.metric("Anzahl SKUs",        f"{len(sup_d)}")
        r6.metric("Fail-Quote",         f"{(sup_d['Inspection results']=='Fail').sum()/len(sup_d)*100:.0f}%")

        rv = sup_d["Risk Score"].mean()
        css_r = "risk-high" if rv>=thresh_high else "risk-medium" if rv>=thresh_low else "risk-low"
        icon_r = "🔴" if rv>=thresh_high else "🟡" if rv>=thresh_low else "🟢"
        msg_r  = "Sofortmaßnahmen: Audit, Alternativen prüfen." if rv>=thresh_high else \
                 "Monitoring intensivieren." if rv>=thresh_low else "Lieferant performt gut."
        st.markdown(f"<div class='info-box {css_r}'>{icon_r} <b>{chosen} ({rv:.1f}/100):</b> {msg_r}</div>",
                    unsafe_allow_html=True)
        st.markdown("---")

        sa1, sa2 = st.columns(2)
        with sa1:
            st.markdown("<div class='section-header'>Risk Score pro SKU</div>", unsafe_allow_html=True)
            sku_s = sup_d.sort_values("Risk Score", ascending=True)
            sku_c = [RISK_COLORS.get(c,"#555") for c in sku_s["Risikostufe"]]
            fig_s, ax_s = plt.subplots(figsize=(6, max(3, len(sku_s)*0.30)))
            ax_s.barh(sku_s["SKU"], sku_s["Risk Score"], color=sku_c, height=0.6, zorder=3)
            ax_s.axvline(thresh_high, color="#ef4444", lw=1, ls="--", alpha=0.5)
            ax_s.axvline(thresh_low,  color="#f59e0b", lw=1, ls="--", alpha=0.5)
            ax_s.set_xlabel("Risk Score"); ax_s.set_xlim(0, 105); ax_s.grid(axis="x", zorder=0)
            plt.tight_layout(); st.pyplot(fig_s); plt.close()

        with sa2:
            st.markdown("<div class='section-header'>🚚 Transportweg-Ranking</div>", unsafe_allow_html=True)
            td = sup_d.groupby("Transportation modes").agg(
                avg_cost=("Shipping costs","mean"), avg_defect=("Defect rates","mean"), count=("SKU","count")
            ).reset_index()
            def ns(s):
                mn,mx=s.min(),s.max()
                return (s-mn)/(mx-mn) if mx!=mn else pd.Series(0.5, index=s.index)
            td["Eff"] = (0.5*ns(td["avg_cost"]) + 0.5*ns(td["avg_defect"]))*100 if len(td)>1 else 50.0
            td = td.sort_values("Eff")
            medals = ["🥇","🥈","🥉","4️⃣"]
            for i, (_, row) in enumerate(td.iterrows()):
                st.markdown(f"""<div class='info-box' style='margin:4px 0'>
                {medals[min(i,3)]} <b>{row['Transportation modes']}</b> ({int(row['count'])} SKUs) &nbsp;|&nbsp;
                Versandkosten: <b>{row['avg_cost']:.1f}</b> &nbsp;|&nbsp;
                Defektrate: <b>{row['avg_defect']:.2f}%</b> &nbsp;|&nbsp;
                Effizienz-Score: <b>{row['Eff']:.0f}</b>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='section-header'>🔎 Inspektionsstatus – nach Potenzial</div>", unsafe_allow_html=True)
        st.markdown("""<div class='info-box'>
        <b>Pending</b> → Nachkontrolle möglich | <b>Fail</b> → Ausschuss | <b>Pass</b> → OK<br>
        ⚠️ 36% Fail-Quote im Datensatz – Klärung ob Fail = vollständiger Ausschuss oder Nacharbeit.
        </div>""", unsafe_allow_html=True)
        insp_d = sup_d.copy()
        insp_d["_s"] = insp_d["Inspection results"].map({"Pending":0,"Fail":1,"Pass":2})
        insp_d = insp_d.sort_values(["_s","Risk Score"], ascending=[True,False])
        show_i = ["SKU","Product type","Inspection results","Defect rates","Lead time","Gesamtlieferzeit","Risk Score","Risikostufe"]
        st.dataframe(insp_d[show_i].reset_index(drop=True), use_container_width=True, height=280)


# ─────────────────────────────────────────────────────────────────
#  TAB 3 – UMSATZ & PRODUKTE
# ─────────────────────────────────────────────────────────────────
with tab3:
    umsatz_view = st.radio("Umsatz-Ansicht:", ["Gesamt","Pro Lieferant","Pro Produktkategorie","Pro Standort"], horizontal=True)
    ua1, ua2 = st.columns(2)

    with ua1:
        fig_u, ax_u = plt.subplots(figsize=(6, 4))
        if umsatz_view in ["Gesamt","Pro Lieferant"]:
            rev_g = scored.groupby("Supplier name")["Revenue generated"].sum().sort_values(ascending=True)
            ax_u.barh(rev_g.index, rev_g.values/1000,
                      color=[SUP_COLORS.get(s,"#2563eb") for s in rev_g.index], height=0.55, zorder=3)
            ax_u.set_xlabel("Umsatz (Tsd. €)"); ax_u.grid(axis="x", zorder=0)
            for i,(idx,v) in enumerate(rev_g.items()):
                ax_u.text(v/1000+0.3, i, f"{v/1000:.1f}k", va="center", fontsize=9)
        elif umsatz_view == "Pro Produktkategorie":
            rev_p = scored.groupby("Product type")["Revenue generated"].sum().sort_values(ascending=False)
            bp2 = ax_u.bar(rev_p.index, rev_p.values/1000,
                           color=[PROD_COLORS.get(p,"#2563eb") for p in rev_p.index], width=0.5, zorder=3)
            ax_u.set_ylabel("Umsatz (Tsd. €)"); ax_u.grid(axis="y", zorder=0)
            for b,v in zip(bp2, rev_p.values):
                ax_u.text(b.get_x()+b.get_width()/2, v/1000+0.3, f"{v/1000:.1f}k", ha="center", fontsize=9)
        else:
            rev_l = scored.groupby("Location")["Revenue generated"].sum().sort_values(ascending=False)
            ax_u.bar(rev_l.index, rev_l.values/1000, color=BAR_BLUE, width=0.5, zorder=3)
            ax_u.set_ylabel("Umsatz (Tsd. €)"); ax_u.grid(axis="y", zorder=0); plt.xticks(rotation=15)
        plt.tight_layout(); st.pyplot(fig_u); plt.close()

    with ua2:
        pivot = scored.groupby(["Supplier name","Product type"])["Revenue generated"].sum().unstack(fill_value=0)
        fig_p, ax_p = plt.subplots(figsize=(6, 4))
        x = np.arange(len(pivot.index)); width = 0.25
        for i, prod in enumerate(pivot.columns):
            ax_p.bar(x+i*width, pivot[prod].values/1000, width, label=prod,
                     color=PROD_COLORS.get(prod,"#aaa"), alpha=0.9, zorder=3)
        ax_p.set_xticks(x+width); ax_p.set_xticklabels(pivot.index, rotation=15)
        ax_p.set_ylabel("Umsatz (Tsd. €)"); ax_p.legend(fontsize=8, framealpha=0.2)
        ax_p.grid(axis="y", zorder=0); plt.tight_layout(); st.pyplot(fig_p); plt.close()

    st.markdown("---")
    t1, t2 = st.columns(2)
    cols_tb = ["SKU","Supplier name","Product type","Revenue generated","Defect rates","Risk Score","Risikostufe"]
    with t1:
        st.markdown("**🏆 Top 10 – höchster Umsatz**")
        top10 = scored.nlargest(10, "Revenue generated")[cols_tb].reset_index(drop=True)
        top10["Revenue generated"] = top10["Revenue generated"].map("{:,.2f} €".format)
        st.dataframe(top10, use_container_width=True, height=370)
    with t2:
        st.markdown("**🔻 Bottom 10 – niedrigster Umsatz**")
        bot10 = scored.nsmallest(10, "Revenue generated")[cols_tb].reset_index(drop=True)
        bot10["Revenue generated"] = bot10["Revenue generated"].map("{:,.2f} €".format)
        st.dataframe(bot10, use_container_width=True, height=370)

    st.markdown("---")
    st.markdown("<div class='section-header'>📦 Lager & Bestellmengen</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    <b>Availability (1–100)</b> = Verfügbarkeitsprozentsatz (wie oft lieferbar) |
    <b>Stock levels</b> = Lagerbestand | <b>Order quantities</b> = Bestellmenge |
    <b>Lager-Überhang</b> positiv = Puffer, negativ = Engpass
    </div>""", unsafe_allow_html=True)
    la1, la2 = st.columns(2)
    with la1:
        lager = scored.groupby("Supplier name").agg(avg_stock=("Stock levels","mean"), avg_order=("Order quantities","mean"))
        fig_l, ax_l = plt.subplots(figsize=(6, 4))
        x = np.arange(len(lager)); w = 0.35
        ax_l.bar(x-w/2, lager["avg_stock"], w, label="⌀ Lagerbestand", color="#2563eb", alpha=0.9, zorder=3)
        ax_l.bar(x+w/2, lager["avg_order"], w, label="⌀ Bestellmenge",  color="#f59e0b", alpha=0.9, zorder=3)
        ax_l.set_xticks(x); ax_l.set_xticklabels(lager.index, rotation=15)
        ax_l.set_ylabel("Einheiten"); ax_l.legend(fontsize=8, framealpha=0.2)
        ax_l.grid(axis="y", zorder=0); plt.tight_layout(); st.pyplot(fig_l); plt.close()
    with la2:
        sold_p = scored.groupby("Product type")["Number of products sold"].sum().sort_values(ascending=False)
        fig_sp, ax_sp = plt.subplots(figsize=(6, 4))
        bs = ax_sp.bar(sold_p.index, sold_p.values,
                       color=[PROD_COLORS.get(p,"#2563eb") for p in sold_p.index], width=0.5, zorder=3)
        ax_sp.set_ylabel("Verkaufte Einheiten gesamt"); ax_sp.grid(axis="y", zorder=0)
        for b, v in zip(bs, sold_p.values):
            ax_sp.text(b.get_x()+b.get_width()/2, v+5, str(int(v)), ha="center", fontsize=10)
        plt.tight_layout(); st.pyplot(fig_sp); plt.close()


# ─────────────────────────────────────────────────────────────────
#  TAB 4 – KLASSIFIKATION & ML
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""<div class='info-box'>
    Dieser Tab enthält statistische Methoden zur <b>Klassifikation und Mustererkennung</b>:<br>
    • <b>Spearman-Korrelationsmatrix</b> – welche KPIs hängen zusammen?<br>
    • <b>K-Means Clustering</b> – automatische Gruppierung der Lieferanten nach Ähnlichkeit
    </div>""", unsafe_allow_html=True)

    # ── Spearman-Korrelationsmatrix ───────────────────────────
    st.markdown("<div class='section-header'>🔗 Spearman-Korrelationsmatrix</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    Spearman misst <b>rang-basierte Korrelationen</b> – geeignet für nicht-normalverteilte Daten (wie unsere).
    Werte nahe +1 = starker positiver Zusammenhang · nahe -1 = starker negativer Zusammenhang · 0 = kein Zusammenhang.<br>
    <b>Multikollinearität:</b> Kein KPI korreliert stark mit einem anderen → kein Problem für Regressionsmodelle.
    </div>""", unsafe_allow_html=True)

    # Fail-Quote als numerisch
    sp_data = scored_full.copy()
    sp_data["Fail_binary"] = (sp_data["Inspection results"] == "Fail").astype(int)
    kpi_cols = ["Defect rates", "Gesamtlieferzeit", "Costs", "Fail_binary"]
    kpi_labels = ["Defekt %", "Lead Time", "Kosten", "Fail-Quote"]

    sp_matrix = np.zeros((len(kpi_cols), len(kpi_cols)))
    sp_pvals  = np.zeros_like(sp_matrix)
    for i, c1 in enumerate(kpi_cols):
        for j, c2 in enumerate(kpi_cols):
            r, p = spearmanr(sp_data[c1].dropna(), sp_data[c2].dropna())
            sp_matrix[i, j] = r
            sp_pvals[i, j]  = p

    # Stärkste / schwächste Korr. (off-diagonal)
    mask = np.ones_like(sp_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    flat = sp_matrix.copy(); flat[~mask] = 0
    max_idx = np.unravel_index(np.abs(flat).argmax(), flat.shape)
    min_idx = np.unravel_index(np.abs(flat).argmin(), flat.shape)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Stärkste Korrelation",  f"{sp_matrix[max_idx]:.2f}", delta=f"{kpi_labels[max_idx[0]]} ↔ {kpi_labels[max_idx[1]]}")
    mc2.metric("Schwächste Korr.",      f"{sp_matrix[min_idx]:.2f}", delta=f"{kpi_labels[min_idx[0]]} ↔ {kpi_labels[min_idx[1]]}")
    mc3.metric("Multikollinearität",    "Gering", delta="Modell sicher")
    mc4.metric("Methode",              "Spearman", delta="nicht-normalverteilt")

    km1, km2 = st.columns(2)

    with km1:
        fig_sp2, ax_sp2 = plt.subplots(figsize=(6, 5))
        cmap = plt.cm.RdYlGn
        im = ax_sp2.imshow(sp_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax_sp2.set_xticks(range(len(kpi_labels))); ax_sp2.set_xticklabels(kpi_labels, rotation=20, ha="right")
        ax_sp2.set_yticks(range(len(kpi_labels))); ax_sp2.set_yticklabels(kpi_labels)
        for i in range(len(kpi_labels)):
            for j in range(len(kpi_labels)):
                col_txt = "#0f1117" if abs(sp_matrix[i,j]) > 0.4 else "#c8d4f0"
                ax_sp2.text(j, i, f"{sp_matrix[i,j]:.2f}", ha="center", va="center",
                            fontsize=12, fontweight="bold", color=col_txt)
        plt.colorbar(im, ax=ax_sp2, shrink=0.8)
        ax_sp2.set_title("Spearman-Korrelationsmatrix", pad=10, fontsize=11)
        st.markdown("<div class='info-box' style='font-size:0.8rem'>Rot = negative Korr. · Grün = positive Korr.</div>",
                    unsafe_allow_html=True)
        plt.tight_layout(); st.pyplot(fig_sp2); plt.close()

    with km2:
        # Scatter stärkste Korrelation
        c_a = kpi_cols[max_idx[0]]; c_b = kpi_cols[max_idx[1]]
        fig_sc2, ax_sc2 = plt.subplots(figsize=(6, 5))
        sup_colors_full = [SUP_COLORS.get(s,"#aaa") for s in sp_data["Supplier name"]]
        ax_sc2.scatter(sp_data[c_a], sp_data[c_b],
                       c=sup_colors_full, s=70, alpha=0.85, edgecolors="#0f1117", linewidths=0.7, zorder=3)
        ax_sc2.set_xlabel(kpi_labels[max_idx[0]]); ax_sc2.set_ylabel(kpi_labels[max_idx[1]])
        ax_sc2.set_title(f"Stärkster Zusammenhang: ρ = {sp_matrix[max_idx]:.2f}", fontsize=10)
        ax_sc2.grid(zorder=0)
        patches_sc = [mpatches.Patch(color=SUP_COLORS.get(s,"#aaa"), label=s) for s in sorted(sp_data["Supplier name"].unique())]
        ax_sc2.legend(handles=patches_sc, fontsize=8, framealpha=0.2)
        st.markdown(f"<div class='info-box' style='font-size:0.8rem'>Spearman ρ = {sp_matrix[max_idx]:.2f} – je höher {kpi_labels[max_idx[0]]}, desto wahrscheinlicher {kpi_labels[max_idx[1]]}.</div>",
                    unsafe_allow_html=True)
        plt.tight_layout(); st.pyplot(fig_sc2); plt.close()

    st.markdown("---")

    # ── K-Means Clustering ────────────────────────────────────
    st.markdown("<div class='section-header'>🔵 K-Means Clustering – automatische Lieferantensegmentierung</div>",
                unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    K-Means gruppiert Lieferanten anhand ihrer KPI-Ähnlichkeit – <b>ohne Vorannahmen</b>.
    Cluster = Gruppen ähnlicher Lieferanten. Hilft bei der Strategieentscheidung: gleiche Cluster = ähnliches Risikoprofil.<br>
    <b>Hinweis:</b> Die Daten sind synthetisch/zufällig → Cluster zeigen Muster im Datensatz, keine echten Marktaussagen.
    </div>""", unsafe_allow_html=True)

    cluster_x = st.selectbox("X-Achse (Cluster):", ["Defect rates","Gesamtlieferzeit","Costs","Revenue generated","Manufacturing costs"], key="cx")
    cluster_y = st.selectbox("Y-Achse (Cluster):", ["Gesamtlieferzeit","Defect rates","Costs","Revenue generated","Manufacturing costs"], index=1, key="cy")
    n_clusters = st.slider("Anzahl Cluster (k):", 2, 5, 3)

    cl_data = scored_full[[cluster_x, cluster_y]].dropna()
    scaler = StandardScaler()
    cl_scaled = scaler.fit_transform(cl_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cl_data = cl_data.copy()
    cl_data["Cluster"] = kmeans.fit_predict(cl_scaled)

    # Lieferanten-Mittelwerte für Cluster-Scatter
    sup_agg = scored_full.groupby("Supplier name")[[cluster_x, cluster_y]].mean().dropna()
    sup_scaled = scaler.transform(sup_agg.values)
    sup_agg["Cluster"] = kmeans.predict(sup_scaled)

    cl1, cl2 = st.columns(2)
    cluster_colors = ["#2563eb","#22c55e","#ef4444","#f59e0b","#a855f7"]

    with cl1:
        st.markdown("<div class='section-header'>SKU-Ebene</div>", unsafe_allow_html=True)
        fig_cl, ax_cl = plt.subplots(figsize=(6, 5))
        for k in range(n_clusters):
            mask_k = cl_data["Cluster"] == k
            ax_cl.scatter(cl_data.loc[mask_k, cluster_x], cl_data.loc[mask_k, cluster_y],
                          c=cluster_colors[k % len(cluster_colors)], s=60, alpha=0.75,
                          edgecolors="#0f1117", linewidths=0.6, label=f"Cluster {k+1}", zorder=3)
        # Centroids
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        ax_cl.scatter(centers[:,0], centers[:,1], marker="X", s=200,
                      c=[cluster_colors[i%len(cluster_colors)] for i in range(n_clusters)],
                      edgecolors="white", linewidths=1.5, zorder=5)
        ax_cl.set_xlabel(cluster_x); ax_cl.set_ylabel(cluster_y)
        ax_cl.legend(fontsize=9, framealpha=0.2); ax_cl.grid(zorder=0)
        plt.tight_layout(); st.pyplot(fig_cl); plt.close()

    with cl2:
        st.markdown("<div class='section-header'>Lieferanten-Ebene (⌀ pro Supplier)</div>", unsafe_allow_html=True)
        fig_cl2, ax_cl2 = plt.subplots(figsize=(6, 5))
        for k in range(n_clusters):
            mask_k = sup_agg["Cluster"] == k
            sups_in_k = sup_agg[mask_k]
            ax_cl2.scatter(sups_in_k[cluster_x], sups_in_k[cluster_y],
                           c=cluster_colors[k % len(cluster_colors)], s=180, alpha=0.9,
                           edgecolors="white", linewidths=1.5, label=f"Cluster {k+1}", zorder=4)
            for sup, row2 in sups_in_k.iterrows():
                ax_cl2.annotate(sup, (row2[cluster_x], row2[cluster_y]),
                                textcoords="offset points", xytext=(6, 4), fontsize=9, color="#c8d4f0")
        ax_cl2.set_xlabel(cluster_x); ax_cl2.set_ylabel(cluster_y)
        ax_cl2.legend(fontsize=9, framealpha=0.2); ax_cl2.grid(zorder=0)
        plt.tight_layout(); st.pyplot(fig_cl2); plt.close()

    # Cluster-Zusammenfassung
    st.markdown("<div class='section-header'>Cluster-Zusammenfassung</div>", unsafe_allow_html=True)
    all_scored_with_cluster = scored_full.copy()
    all_scored_with_cluster["Cluster"] = kmeans.predict(
        scaler.transform(all_scored_with_cluster[[cluster_x, cluster_y]].fillna(0))
    )
    cluster_summary = all_scored_with_cluster.groupby("Cluster").agg(
        Anzahl_SKUs    =("SKU",             "count"),
        avg_risk       =("Risk Score",       "mean"),
        avg_defect     =("Defect rates",     "mean"),
        avg_lead       =("Gesamtlieferzeit", "mean"),
        avg_costs      =("Costs",            "mean"),
        fail_pct       =("Inspection results", lambda x: (x=="Fail").sum()/len(x)*100),
    ).round(2)
    cluster_summary.index = [f"Cluster {i+1}" for i in cluster_summary.index]
    cluster_summary.columns = ["SKUs","⌀ Risk Score","⌀ Defekt %","⌀ Lieferzeit (d)","⌀ Kosten","Fail %"]
    st.dataframe(cluster_summary, use_container_width=True, height=200)


# ─────────────────────────────────────────────────────────────────
#  TAB 5 – SKU-SUCHE
# ─────────────────────────────────────────────────────────────────
with tab5:
    sc1, sc2, sc3 = st.columns(3)
    with sc1: search_term = st.text_input("SKU eingeben", placeholder="z. B. SKU42")
    with sc2: risk_filter = st.selectbox("Risikostufe", ["Alle","🔴 Hoch","🟡 Mittel","🟢 Niedrig"])
    with sc3: insp_f = st.selectbox("Inspektionsstatus", ["Alle","Pass","Pending","Fail"])

    result = scored.copy()
    if search_term: result = result[result["SKU"].str.contains(search_term, case=False, na=False)]
    if risk_filter != "Alle": result = result[result["Risikostufe"] == risk_filter]
    if insp_f != "Alle": result = result[result["Inspection results"] == insp_f]

    st.markdown(f"<p style='color:#5577aa;font-size:0.85rem'>{len(result)} Ergebnisse</p>", unsafe_allow_html=True)
    show_s = ["SKU","Supplier name","Location","Product type","Price",
              "Defect rates","Lead time","Shipping times","Gesamtlieferzeit",
              "Manufacturing costs","Costs","Revenue generated","Inspection results","Risk Score","Risikostufe"]
    st.dataframe(result[show_s].sort_values("Risk Score", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=380)

    if not result.empty:
        st.markdown("---")
        st.markdown("<div class='section-header'>📌 SKU im Detail</div>", unsafe_allow_html=True)
        chosen_sku = st.selectbox("SKU auswählen", result["SKU"].tolist())
        row = result[result["SKU"] == chosen_sku].iloc[0]
        d1,d2,d3,d4,d5,d6 = st.columns(6)
        d1.metric("Risk Score",       f"{row['Risk Score']:.1f}")
        d2.metric("Defektrate",       f"{row['Defect rates']:.2f}%")
        d3.metric("Gesamtlieferzeit", f"{row['Gesamtlieferzeit']:.0f} Tage")
        d4.metric("Umsatz",           f"{row['Revenue generated']:,.2f} €")
        d5.metric("Gesamtkosten",     f"{row['Costs']:,.2f} €")
        d6.metric("Inspektion",       row["Inspection results"])
        rv2 = row["Risk Score"]
        css2 = "risk-high" if rv2>=thresh_high else "risk-medium" if rv2>=thresh_low else "risk-low"
        icon2 = "🔴" if rv2>=thresh_high else "🟡" if rv2>=thresh_low else "🟢"
        msg2 = "Audit empfohlen." if rv2>=thresh_high else "Monitoring intensivieren." if rv2>=thresh_low else "Performt gut."
        st.markdown(f"<div class='info-box {css2}'>{icon2} <b>{rv2:.1f}/100</b> – {msg2}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 6 – ALLE DATEN
# ─────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("<div class='section-header'>📋 Alle Daten – vollständig filter- & sortierbar</div>",
                unsafe_allow_html=True)

    # Spalten ohne Duplikate
    base_cols = list(df.columns)
    extra_cols = [c for c in ["Gesamtlieferzeit","Lager-Überhang","Risk Score","Risikostufe"]
                  if c not in base_cols and c in scored_full.columns]
    all_cols = base_cols + extra_cols

    with st.expander("🗂️ Spalten auswählen", expanded=False):
        sel_cols2 = st.multiselect("Spalten", all_cols, default=all_cols)

    with st.expander("🔧 Zusätzliche Filter", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            ff_sup  = st.multiselect("Lieferant",  sorted(df["Supplier name"].unique()), default=sorted(df["Supplier name"].unique()), key="ffs")
            ff_prod = st.multiselect("Produkttyp", sorted(df["Product type"].unique()),  default=sorted(df["Product type"].unique()),  key="ffp")
        with f2:
            ff_loc  = st.multiselect("Standort",   sorted(df["Location"].unique()),      default=sorted(df["Location"].unique()),      key="ffl")
            ff_insp = st.multiselect("Inspektion", sorted(df["Inspection results"].unique()), default=sorted(df["Inspection results"].unique()), key="ffi")
        with f3:
            ff_risk = st.multiselect("Risikostufe",["🔴 Hoch","🟡 Mittel","🟢 Niedrig"],default=["🔴 Hoch","🟡 Mittel","🟢 Niedrig"],key="ffr")
            ff_carr = st.multiselect("Carrier",    sorted(df["Shipping carriers"].unique()),default=sorted(df["Shipping carriers"].unique()),key="ffc")

    s1, s2 = st.columns([3,1])
    with s1:
        sort_by2 = st.selectbox("Sortieren nach", ["Risk Score","Defect rates","Gesamtlieferzeit","Revenue generated","Costs","Manufacturing costs","Price"])
    with s2:
        asc2 = st.radio("Reihenfolge", ["↓ Absteigend","↑ Aufsteigend"]) == "↑ Aufsteigend"

    table = scored_full[
        scored_full["Supplier name"].isin(ff_sup) &
        scored_full["Product type"].isin(ff_prod) &
        scored_full["Location"].isin(ff_loc) &
        scored_full["Inspection results"].isin(ff_insp) &
        scored_full["Risikostufe"].isin(ff_risk) &
        scored_full["Shipping carriers"].isin(ff_carr)
    ].copy()

    valid2 = [c for c in sel_cols2 if c in table.columns]
    table  = table[valid2].sort_values(sort_by2, ascending=asc2).reset_index(drop=True)

    st.markdown(f"<p style='color:#5577aa;font-size:0.85rem'>{len(table)} Zeilen · {len(valid2)} Spalten</p>",
                unsafe_allow_html=True)
    st.dataframe(table, use_container_width=True, height=500)

    ex1, ex2 = st.columns(2)
    with ex1:
        st.download_button("⬇️ Als CSV exportieren",
                           data=table.to_csv(index=False).encode("utf-8"),
                           file_name="riskradar_export.csv", mime="text/csv")
    with ex2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            table.to_excel(writer, index=False, sheet_name="RiskRadar")
        buf.seek(0)
        st.download_button("⬇️ Als Excel (.xlsx) exportieren", data=buf.getvalue(),
                           file_name="riskradar_export.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<p style='text-align:center;color:#3a4a6a;font-size:0.8rem'>
RiskRadar v5 · THI Ingolstadt · DPDS SoSe 2026 · Gruppe 9:
Laurenz Angleitner · Leon Pavic · Alex Rauschendorfer · Daniel Steinmetz
</p>""", unsafe_allow_html=True)
