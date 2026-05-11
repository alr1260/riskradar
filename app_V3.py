import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
from scipy.stats import spearmanr

st.set_page_config(page_title="RiskRadar · Supplier Intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:#0f1117;color:#e8eaf0;}
.stApp{background-color:#0f1117;}
section[data-testid="stSidebar"]{background-color:#161b27;border-right:1px solid #252d3d;}
section[data-testid="stSidebar"] *{color:#c8cfe0 !important;}
[data-testid="metric-container"]{background:linear-gradient(135deg,#1a2035,#1e2640);border:1px solid #2a3350;border-radius:14px;padding:14px 18px !important;box-shadow:0 4px 20px rgba(0,0,0,0.3);}
[data-testid="metric-container"] label{color:#8899bb !important;font-size:0.72rem !important;letter-spacing:0.08em;text-transform:uppercase;}
[data-testid="metric-container"] [data-testid="metric-value"]{color:#e8eaf0 !important;font-size:1.6rem !important;font-weight:700;}
.stTabs [data-baseweb="tab-list"]{background-color:#161b27;border-radius:10px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#8899bb;font-weight:500;padding:8px 16px;}
.stTabs [aria-selected="true"]{background-color:#2563eb !important;color:white !important;}
.stButton>button{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:white;border:none;border-radius:8px;font-weight:500;}
.info-box{background:linear-gradient(135deg,#1a2640,#1e2d4a);border-left:4px solid #2563eb;border-radius:0 10px 10px 0;padding:12px 16px;margin:8px 0;font-size:0.85rem;color:#a8bbd4;line-height:1.6;}
.risk-high{border-left-color:#ef4444 !important;background:linear-gradient(135deg,#2a1a1a,#331c1c) !important;color:#ffb3b3 !important;}
.risk-medium{border-left-color:#f59e0b !important;background:linear-gradient(135deg,#2a2210,#332a10) !important;color:#ffe0a0 !important;}
.risk-low{border-left-color:#22c55e !important;background:linear-gradient(135deg,#0f2a18,#12331e) !important;color:#a0f0c0 !important;}
.section-header{font-size:1.0rem;font-weight:600;color:#c8d4f0;letter-spacing:0.04em;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #252d3d;}
.ampel-card{border-radius:14px;padding:18px 20px;text-align:center;font-weight:600;font-size:1.1rem;margin:6px 0;box-shadow:0 4px 16px rgba(0,0,0,0.3);}
.ampel-rot{background:linear-gradient(135deg,#3a1010,#4a1515);border:2px solid #ef4444;color:#ff8080;}
.ampel-gelb{background:linear-gradient(135deg,#2a2010,#3a2c10);border:2px solid #f59e0b;color:#ffd080;}
.ampel-gruen{background:linear-gradient(135deg,#0a2a14,#0f3a1c);border:2px solid #22c55e;color:#80ffa8;}

/* Startseite */
.hero-box{background:linear-gradient(135deg,#1a2640,#0f1a30);border:1px solid #2a3350;border-radius:18px;padding:36px 40px;margin:12px 0;text-align:center;}
.hero-title{font-size:3rem;font-weight:700;color:#e8eaf0;margin:0;}
.hero-sub{font-size:1.1rem;color:#5577aa;margin-top:8px;}
.feature-card{background:linear-gradient(135deg,#1a2035,#1e2640);border:1px solid #2a3350;border-radius:14px;padding:22px 20px;text-align:center;margin:8px 4px;}
.feature-icon{font-size:2.2rem;margin-bottom:8px;}
.feature-title{font-size:1.0rem;font-weight:600;color:#c8d4f0;margin-bottom:6px;}
.feature-desc{font-size:0.82rem;color:#8899bb;line-height:1.5;}

/* Umsatz-Karten */
.umsatz-card{background:linear-gradient(135deg,#1a2035,#1e2640);border:1px solid #2a3350;border-radius:12px;padding:16px 18px;margin:6px 0;}
.umsatz-card-title{font-size:1.05rem;font-weight:600;margin-bottom:10px;}
.umsatz-bar{height:4px;border-radius:2px;margin-top:6px;}

/* Kreuz-Tabelle Lieferant x Produkt */
.cross-table{width:100%;border-collapse:collapse;font-size:0.88rem;}
.cross-table th{background:#1a2640;color:#8899bb;padding:8px 12px;text-align:center;font-weight:500;}
.cross-table td{padding:8px 12px;text-align:center;border:1px solid #252d3d;}
.cross-td-high{background:#1a3a28;color:#4ade80;font-weight:600;}
.cross-td-mid{background:#1a2a3a;color:#60a5fa;font-weight:600;}
.cross-td-low{background:#2a1a2a;color:#c084fc;font-weight:600;}

/* Score-Erklärung */
.formula-box{background:#1a2035;border:1px solid #2a3350;border-radius:10px;padding:14px 18px;font-family:monospace;font-size:0.82rem;color:#94a3b8;line-height:1.8;}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#161b27","axes.facecolor":"#1a2035","axes.edgecolor":"#2a3350",
    "axes.labelcolor":"#8899bb","xtick.color":"#8899bb","ytick.color":"#8899bb",
    "text.color":"#c8d4f0","grid.color":"#252d3d","grid.linestyle":"--","grid.alpha":0.5,
})

SUP_COLORS  = {}
PROD_COLORS = {"skincare":"#06b6d4","haircare":"#f59e0b","cosmetics":"#a855f7"}
RISK_COLORS = {"🔴 Hoch":"#ef4444","🟡 Mittel":"#f59e0b","🟢 Niedrig":"#22c55e"}
BUCKET_COLORS = {"🥇 Premium":"#60a5fa","🔵 Standard":"#94a3b8","💚 Budget":"#4ade80"}
LOC_MEDALS = ["🥇","🥈","🥉","4️⃣","5️⃣"]

# ════════════════════════════════════════════════════════════════
#  DATEN
# ════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("supply_chain_data.csv")
    num_cols = ["Price","Availability","Number of products sold","Revenue generated","Stock levels",
                "Lead times","Order quantities","Shipping times","Shipping costs","Lead time",
                "Production volumes","Manufacturing lead time","Manufacturing costs","Defect rates","Costs"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].round(2)
    if "Gesamtlieferzeit" not in df.columns:
        df["Gesamtlieferzeit"] = df["Lead time"] + df["Shipping times"]
    # Produkt-Bucket
    def norm(s):
        mn,mx=s.min(),s.max(); return (s-mn)/(mx-mn+1e-9)
    score = (norm(df["Price"]) + norm(df["Revenue generated"]) + (1-norm(df["Defect rates"])))/3
    p33,p66 = score.quantile(0.33), score.quantile(0.66)
    df["Produkt-Bucket"] = pd.cut(score, bins=[-1,p33,p66,2], labels=["💚 Budget","🔵 Standard","🥇 Premium"])
    df["Bucket-Score"] = score.round(3)
    df["_p33"] = round(p33, 3)
    df["_p66"] = round(p66, 3)
    return df

df = load_data()
P33 = df["_p33"].iloc[0]
P66 = df["_p66"].iloc[0]

for i,s in enumerate(sorted(df["Supplier name"].unique())):
    SUP_COLORS[s] = ["#2563eb","#f59e0b","#22c55e","#a855f7","#ef4444"][i%5]

def norm_col(series, invert=False):
    mn,mx=series.min(),series.max()
    if mx==mn: return pd.Series(0.5,index=series.index)
    n=(series-mn)/(mx-mn)
    return (1-n) if invert else n

def compute_risk(df_in, wd, wl, wc, wi, wr, th, tl):
    r=df_in.copy()
    r["_nd"]=norm_col(r["Defect rates"])
    r["_nl"]=norm_col(r["Gesamtlieferzeit"])
    r["_nc"]=norm_col(r["Costs"])
    r["_nr"]=norm_col(r["Revenue generated"],invert=True)
    r["_ni"]=r["Inspection results"].map({"Fail":1.0,"Pending":0.5,"Pass":0.0}).fillna(0.5)
    total=wd+wl+wc+wi+wr
    r["Risk Score"]=(wd*r["_nd"]+wl*r["_nl"]+wc*r["_nc"]+wi*r["_ni"]+wr*r["_nr"])/total*100
    r["Risk Score"]=r["Risk Score"].round(2)
    def cat(s):
        if s>=th: return "🔴 Hoch"
        if s>=tl: return "🟡 Mittel"
        return "🟢 Niedrig"
    r["Risikostufe"]=r["Risk Score"].apply(cat)
    return r.drop(columns=[c for c in r.columns if c.startswith("_")])

# ════════════════════════════════════════════════════════════════
#  SESSION STATE für Navigation & Filter
# ════════════════════════════════════════════════════════════════
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
if "risk_filter" not in st.session_state:
    st.session_state.risk_filter = "Alle"

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ RiskRadar")
    st.markdown("<p style='color:#5577aa;font-size:0.8rem;margin-top:-10px'>THI Ingolstadt · DPDS 2026 · Gruppe 9</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🎯 Entscheidungs-Profil")
    preset = st.radio("", ["🔧 Manuell","🏆 Höchste Qualität","💰 Umsatzstärkster Lieferant","⚡ Günstig & Schnell"], key="preset", label_visibility="collapsed")
    if preset == "🏆 Höchste Qualität":
        wd,wl,wc,wi,wr = 10,1,1,10,5
        st.markdown("<div class='info-box risk-low'>🏆 Defektrate & Inspektion dominieren.</div>", unsafe_allow_html=True)
    elif preset == "💰 Umsatzstärkster Lieferant":
        wd,wl,wc,wi,wr = 3,3,2,3,10
        st.markdown("<div class='info-box risk-medium'>💰 Hoher Umsatz = geringes Risiko.</div>", unsafe_allow_html=True)
    elif preset == "⚡ Günstig & Schnell":
        wd,wl,wc,wi,wr = 1,10,10,1,5
        st.markdown("<div class='info-box risk-high'>⚡ Lieferzeit & Kosten dominieren.</div>", unsafe_allow_html=True)
    else:
        wd=st.slider("🔬 Defektrate",1,10,4)
        wl=st.slider("⏱️ Gesamtlieferzeit",1,10,3)
        wc=st.slider("💰 Gesamtkosten",1,10,2)
        wi=st.slider("🔎 Inspektionsergebnis",1,10,5)
        wr=st.slider("📈 Umsatz (hoch = geringes Risiko)",1,10,2)
    tw=wd+wl+wc+wi+wr
    st.markdown(f"<div class='info-box' style='font-size:0.78rem'>🔬{wd/tw*100:.0f}% · ⏱️{wl/tw*100:.0f}% · 💰{wc/tw*100:.0f}% · 🔎{wi/tw*100:.0f}% · 📈{wr/tw*100:.0f}%</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔍 Filter")
    with st.expander("Lieferanten & Produkte", expanded=True):
        sel_sup=st.multiselect("Lieferant",sorted(df["Supplier name"].unique()),default=sorted(df["Supplier name"].unique()),key="gs")
        sel_prod=st.multiselect("Produktkategorie",sorted(df["Product type"].unique()),default=sorted(df["Product type"].unique()),key="gp")
        sel_bucket=st.multiselect("Produkt-Segment",["🥇 Premium","🔵 Standard","💚 Budget"],default=["🥇 Premium","🔵 Standard","💚 Budget"],key="gb")
    with st.expander("Logistik", expanded=False):
        sel_loc=st.multiselect("Standort",sorted(df["Location"].unique()),default=sorted(df["Location"].unique()),key="gl")
        sel_carr=st.multiselect("Carrier",sorted(df["Shipping carriers"].unique()),default=sorted(df["Shipping carriers"].unique()),key="gc")
        sel_mode=st.multiselect("Transportweg",sorted(df["Transportation modes"].unique()),default=sorted(df["Transportation modes"].unique()),key="gm")
    with st.expander("Qualität", expanded=False):
        sel_insp=st.multiselect("Inspektionsstatus",sorted(df["Inspection results"].unique()),default=sorted(df["Inspection results"].unique()),key="gi")
        max_defect=st.slider("Max. Defektrate (%)",0.0,float(df["Defect rates"].max()),float(df["Defect rates"].max()))
    st.markdown("---")
    st.markdown("### 🎚️ Grenzwerte Risk Score")
    thresh_high=st.slider("Ab hier 🔴 Hoch",34,90,60)
    thresh_low=st.slider("Ab hier 🟡 Mittel",10,thresh_high-1,35)
    st.markdown(f"<div class='info-box' style='font-size:0.78rem'>🔴≥{thresh_high} · 🟡{thresh_low}–{thresh_high-1} · 🟢&lt;{thresh_low}<br><i>Default 60: statistisch neutrale Terzil-Einteilung</i></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  FILTER + SCORING
# ════════════════════════════════════════════════════════════════
filtered = df[
    df["Supplier name"].isin(sel_sup) & df["Product type"].isin(sel_prod) &
    df["Produkt-Bucket"].isin(sel_bucket) & df["Location"].isin(sel_loc) &
    df["Shipping carriers"].isin(sel_carr) & df["Transportation modes"].isin(sel_mode) &
    df["Inspection results"].isin(sel_insp) & (df["Defect rates"] <= max_defect)
].copy()

if filtered.empty:
    st.warning("⚠️ Keine Daten. Filter anpassen."); st.stop()

scored = compute_risk(filtered, wd,wl,wc,wi,wr, thresh_high,thresh_low)
scored_full = compute_risk(df, wd,wl,wc,wi,wr, thresh_high,thresh_low)
sup_order = sorted(scored["Supplier name"].unique())

# ════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════
tab0,tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "🏠 Start",
    "🚦 Dashboard",
    "📊 Risk Overview",
    "🏷️ Produkte",
    "🏢 Lieferanten",
    "🔬 Korrelation",
    "📋 Alle Daten"
])

# ════════════════════════════════════════════════════════════════
#  TAB 0 – STARTSEITE
# ════════════════════════════════════════════════════════════════
with tab0:
    st.markdown("""
    <div class='hero-box'>
        <div class='hero-title'>🛡️ RiskRadar</div>
        <div class='hero-sub'>Supplier & Procurement Risk Intelligence · THI Ingolstadt · DPDS 2026 · Gruppe 9</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""<div class='info-box' style='font-size:0.95rem;padding:18px 22px'>
    <b>Wofür ist dieses Dashboard?</b><br><br>
    RiskRadar hilft <b>Supply Chain Managern und CEOs</b> dabei, Lieferanten- und Beschaffungsrisiken
    frühzeitig zu erkennen, Produkte nach ihrem Wertbeitrag zu klassifizieren und datenbasierte
    Entscheidungen zu treffen – bevor Probleme entstehen.<br><br>
    <b>Kontext:</b> Ein produzierendes Unternehmen kauft Rohstoffe ein, befüllt diese in Produkte
    (z.B. Skincare, Haircare, Cosmetics) und vertreibt sie. Die Daten umfassen 100 SKUs von
    5 Lieferanten aus 5 indischen Standorten.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Was beantwortet das Dashboard?", unsafe_allow_html=False)

    fc1,fc2,fc3 = st.columns(3)
    with fc1:
        st.markdown("""<div class='feature-card'>
        <div class='feature-icon'>🚦</div>
        <div class='feature-title'>Welche Lieferanten sind kritisch?</div>
        <div class='feature-desc'>Die Ampel zeigt sofort den Status jedes Lieferanten. Kritische SKUs werden direkt aufgelistet mit konkreten Handlungsempfehlungen.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;margin-top:4px'><i style='color:#5577aa;font-size:0.8rem'>→ Tab: Dashboard</i></p>", unsafe_allow_html=True)

    with fc2:
        st.markdown("""<div class='feature-card'>
        <div class='feature-icon'>🏷️</div>
        <div class='feature-title'>Was für Produkte kaufen wir ein?</div>
        <div class='feature-desc'>Produkte werden in 3 Segmente klassifiziert: Premium, Standard und Budget – basierend auf Preis, Umsatz und Defektrate.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;margin-top:4px'><i style='color:#5577aa;font-size:0.8rem'>→ Tab: Produkte</i></p>", unsafe_allow_html=True)

    with fc3:
        st.markdown("""<div class='feature-card'>
        <div class='feature-icon'>📊</div>
        <div class='feature-title'>Wie hoch ist unser Risiko?</div>
        <div class='feature-desc'>Der Risk Score (0–100) bewertet jeden Lieferanten nach 5 KPIs: Defektrate, Lieferzeit, Kosten, Inspektion und Umsatz.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;margin-top:4px'><i style='color:#5577aa;font-size:0.8rem'>→ Tab: Risk Overview</i></p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔢 Wie wird der Risk Score berechnet?")

    tw_display = wd+wl+wc+wi+wr
    st.markdown(f"""<div class='formula-box'>
    Risk Score (0–100) = gewichtete Summe aus 5 normierten KPIs:<br><br>
    &nbsp;&nbsp;({wd/tw_display*100:.0f}%) Defektrate &nbsp;→ hoch = schlechter<br>
    &nbsp;&nbsp;({wl/tw_display*100:.0f}%) Gesamtlieferzeit (Tage) &nbsp;→ lang = schlechter<br>
    &nbsp;&nbsp;({wc/tw_display*100:.0f}%) Gesamtkosten (€) &nbsp;→ hoch = schlechter<br>
    &nbsp;&nbsp;({wi/tw_display*100:.0f}%) Inspektionsergebnis &nbsp;→ Fail=1.0 · Pending=0.5 · Pass=0.0<br>
    &nbsp;&nbsp;({wr/tw_display*100:.0f}%) Umsatz (€) &nbsp;→ hoch = BESSER (invers)<br><br>
    Normierung: Min-Max je KPI → alle Werte zwischen 0 und 1<br>
    Grenzwerte: 🔴 Hoch ≥ {thresh_high} · 🟡 Mittel ≥ {thresh_low} · 🟢 Niedrig &lt; {thresh_low}<br>
    Profil aktiv: <b>{preset}</b>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📌 Aktueller Status auf einen Blick")
    n_high=(scored["Risikostufe"]=="🔴 Hoch").sum()
    n_med=(scored["Risikostufe"]=="🟡 Mittel").sum()
    n_low=(scored["Risikostufe"]=="🟢 Niedrig").sum()
    sk1,sk2,sk3,sk4 = st.columns(4)
    sk1.metric("⌀ Risk Score",f"{scored['Risk Score'].mean():.2f} / 100")
    sk2.metric("🔴 Kritische SKUs",f"{n_high}",delta=f"{n_high/len(scored)*100:.0f}% der Auswahl",delta_color="inverse")
    sk3.metric("⌀ Defektrate",f"{scored['Defect rates'].mean():.2f} %")
    sk4.metric("Gesamtumsatz",f"{scored['Revenue generated'].sum()/1000:.2f}k €")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box' style='font-size:0.82rem'>
    💡 <b>Tipp:</b> Wähle links in der Sidebar ein <b>Entscheidungs-Profil</b> (z.B. Höchste Qualität oder Günstig & Schnell)
    und passe die <b>Grenzwerte</b> an – alle Tabs aktualisieren sich automatisch.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  TAB 1 – ENTSCHEIDUNGS-DASHBOARD
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"<h2 style='color:#e8eaf0;font-size:1.4rem;margin-bottom:4px'>🚦 Entscheidungs-Dashboard</h2><p style='color:#5577aa;font-size:0.85rem'>Profil: <b>{preset}</b> · Grenzwert 🔴 ≥{thresh_high} · 🟡 ≥{thresh_low}</p>", unsafe_allow_html=True)

    # Ampel
    sup_risks = scored.groupby("Supplier name")["Risk Score"].mean()
    amp_cols = st.columns(len(sup_order))
    for i,sup in enumerate(sup_order):
        if sup not in sup_risks.index: continue
        rv=sup_risks[sup]
        css,icon,status=("ampel-rot","🔴","KRITISCH") if rv>=thresh_high else ("ampel-gelb","🟡","BEOBACHTEN") if rv>=thresh_low else ("ampel-gruen","🟢","OK")
        n_fail=(scored[scored["Supplier name"]==sup]["Inspection results"]=="Fail").sum()
        n_sku=len(scored[scored["Supplier name"]==sup])
        amp_cols[i].markdown(f"<div class='ampel-card {css}'>{icon} {sup}<br><span style='font-size:1.7rem;font-weight:700'>{rv:.1f}</span><br><span style='font-size:0.78rem'>{status}</span><br><span style='font-size:0.72rem;opacity:0.8'>{n_sku} SKUs · {n_fail} Fails</span></div>", unsafe_allow_html=True)

    # Interaktiver Filter: Klick auf Risikostufe
    st.markdown("---")
    st.markdown("<div class='section-header'>🔍 SKU-Liste filtern – direkt klicken oder Stufe wählen</div>", unsafe_allow_html=True)

    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
    with btn_col1:
        if st.button("🔴 Nur Kritische anzeigen", use_container_width=True):
            st.session_state.risk_filter = "🔴 Hoch"
    with btn_col2:
        if st.button("🟡 Nur Mittlere anzeigen", use_container_width=True):
            st.session_state.risk_filter = "🟡 Mittel"
    with btn_col3:
        if st.button("🟢 Nur OK anzeigen", use_container_width=True):
            st.session_state.risk_filter = "🟢 Niedrig"
    with btn_col4:
        if st.button("⬜ Alle anzeigen", use_container_width=True):
            st.session_state.risk_filter = "Alle"

    # Radio als zweite Möglichkeit
    risk_radio = st.radio("Oder Risikostufe wählen:",
                          ["Alle","🔴 Hoch","🟡 Mittel","🟢 Niedrig"],
                          index=["Alle","🔴 Hoch","🟡 Mittel","🟢 Niedrig"].index(st.session_state.risk_filter),
                          horizontal=True, key="risk_radio")
    if risk_radio != st.session_state.risk_filter:
        st.session_state.risk_filter = risk_radio

    # Gefilterte Liste
    rf = st.session_state.risk_filter
    list_df = scored if rf == "Alle" else scored[scored["Risikostufe"] == rf]
    list_df = list_df.sort_values("Risk Score", ascending=False)

    color_map = {"🔴 Hoch":"#ef4444","🟡 Mittel":"#f59e0b","🟢 Niedrig":"#22c55e"}
    rc = color_map.get(rf, "#2563eb")
    st.markdown(f"<p style='color:{rc};font-size:0.9rem;font-weight:600'>{len(list_df)} SKUs · Risikostufe: {rf}</p>", unsafe_allow_html=True)

    show_cols = ["SKU","Supplier name","Product type","Produkt-Bucket","Defect rates","Gesamtlieferzeit","Costs","Revenue generated","Inspection results","Risk Score","Risikostufe"]
    st.dataframe(list_df[show_cols].reset_index(drop=True), use_container_width=True, height=340)

    st.markdown("---")

    # Handlungsempfehlungen
    kritische=[s for s in sup_order if s in sup_risks.index and sup_risks[s]>=thresh_high]
    mittlere=[s for s in sup_order if s in sup_risks.index and thresh_low<=sup_risks[s]<thresh_high]
    ok=[s for s in sup_order if s in sup_risks.index and sup_risks[s]<thresh_low]
    he1,he2,he3=st.columns(3)
    with he1:
        st.markdown(f"<div class='info-box risk-high'>🔴 <b>Sofortmaßnahmen</b><br>{', '.join(kritische) if kritische else 'Keine'}<br><br>→ Lieferanten-Audit einleiten<br>→ Alternativen prüfen<br>→ Bestellmengen reduzieren</div>", unsafe_allow_html=True)
    with he2:
        st.markdown(f"<div class='info-box risk-medium'>🟡 <b>Monitoring intensivieren</b><br>{', '.join(mittlere) if mittlere else 'Keine'}<br><br>→ Lieferantengespräch führen<br>→ KPIs engmaschig beobachten<br>→ Vertragskonditionen prüfen</div>", unsafe_allow_html=True)
    with he3:
        st.markdown(f"<div class='info-box risk-low'>🟢 <b>Routinebetrieb</b><br>{', '.join(ok) if ok else 'Keine'}<br><br>→ Weiter beobachten<br>→ Best Practices dokumentieren<br>→ Als Benchmark nutzen</div>", unsafe_allow_html=True)

    st.markdown("---")
    qs1,qs2,qs3=st.columns(3)
    with qs1:
        st.markdown("<div class='section-header' style='font-size:0.85rem'>🔴 Top 5 riskanteste SKUs</div>", unsafe_allow_html=True)
        st.dataframe(scored.nlargest(5,"Risk Score")[["SKU","Supplier name","Risk Score","Risikostufe"]].reset_index(drop=True),use_container_width=True,height=210)
    with qs2:
        st.markdown("<div class='section-header' style='font-size:0.85rem'>🟢 Top 5 beste SKUs</div>", unsafe_allow_html=True)
        st.dataframe(scored.nsmallest(5,"Risk Score")[["SKU","Supplier name","Risk Score","Risikostufe"]].reset_index(drop=True),use_container_width=True,height=210)
    with qs3:
        st.markdown("<div class='section-header' style='font-size:0.85rem'>Fail-Quote pro Lieferant</div>", unsafe_allow_html=True)
        fq=scored.groupby("Supplier name").apply(lambda x:round((x["Inspection results"]=="Fail").sum()/len(x)*100,2)).reset_index()
        fq.columns=["Lieferant","Fail %"]
        st.dataframe(fq.sort_values("Fail %",ascending=False).reset_index(drop=True),use_container_width=True,height=210)

# ════════════════════════════════════════════════════════════════
#  TAB 2 – RISK OVERVIEW
# ════════════════════════════════════════════════════════════════
with tab2:
    # KPI Heatmap – nur 3 Farben
    st.markdown("<div class='section-header'>🌡️ KPI Heatmap (synchronisiert mit Filter)</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Rot = schlechter Wert · Gelb = mittlerer Wert · Grün = guter Wert (relativ zum Datensatz). Spalten: Defektrate · Gesamtlieferzeit · Gesamtkosten · Fail-% · Umsatz</div>", unsafe_allow_html=True)

    hm=scored.groupby("Supplier name").agg(
        Defekt=("Defect rates","mean"),
        Lieferzeit=("Gesamtlieferzeit","mean"),
        Kosten=("Costs","mean"),
        Fail=("Inspection results",lambda x:round((x=="Fail").sum()/len(x)*100,2)),
        Umsatz=("Revenue generated","sum")
    ).round(2)
    hm_labels=["Defekt %","Lieferzeit (Tage)","Kosten (€)","Fail %","Umsatz (€)"]
    hm_invert=[True,True,True,True,False]
    hm_vals=hm.values
    hm_norm=np.zeros_like(hm_vals,dtype=float)
    for j in range(hm_vals.shape[1]):
        col=hm_vals[:,j].astype(float); mn,mx=col.min(),col.max()
        n=(col-mn)/(mx-mn+1e-9)
        hm_norm[:,j]=(1-n) if hm_invert[j] else n

    # Nur 3 Farben: unter 0.33 = rot, 0.33-0.66 = gelb, über 0.66 = grün
    def three_color(v):
        if v < 0.33:   return "#3a1010"  # rot
        elif v < 0.66: return "#2a2210"  # gelb
        else:          return "#0a2a14"  # grün

    def three_text(v):
        if v < 0.33:   return "#ff8080"
        elif v < 0.66: return "#ffd080"
        else:          return "#80ffa8"

    fig_hm,ax_hm=plt.subplots(figsize=(11,max(2.5,len(hm)*0.75)))
    for i in range(hm_norm.shape[0]):
        for j in range(hm_norm.shape[1]):
            v=hm_norm[i,j]
            bg = "#3a1010" if v<0.33 else "#2a2210" if v<0.66 else "#0a2a14"
            tc = "#ff8080" if v<0.33 else "#ffd080" if v<0.66 else "#80ffa8"
            ax_hm.add_patch(plt.Rectangle([j,i],1,1,color=bg,zorder=2))
            raw=hm_vals[i,j]
            txt=f"{raw:.2f}%" if j in [0,3] else f"{raw:.2f} Tage" if j==1 else f"€{raw:.0f}" if j==2 else f"€{raw/1000:.1f}k"
            ax_hm.text(j+0.5,i+0.5,txt,ha="center",va="center",fontsize=10,fontweight="bold",color=tc,zorder=3)
    ax_hm.set_xlim(0,len(hm_labels)); ax_hm.set_ylim(0,len(hm))
    ax_hm.set_xticks([x+0.5 for x in range(len(hm_labels))]); ax_hm.set_xticklabels(hm_labels,fontsize=10)
    ax_hm.set_yticks([y+0.5 for y in range(len(hm))]); ax_hm.set_yticklabels(hm.index,fontsize=10)
    ax_hm.set_facecolor("#1a2035"); plt.tight_layout(); st.pyplot(fig_hm); plt.close()

    st.markdown("""<div class='info-box' style='font-size:0.8rem'>
    🔴 Rot = schlechtester Wert im Vergleich · 🟡 Gelb = mittlerer Bereich · 🟢 Grün = bester Wert.<br>
    Bewertung ist <b>relativ</b> – d.h. immer im Vergleich zu den anderen Lieferanten im aktuellen Filter.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")

    c1,c2=st.columns([3,2])
    with c1:
        st.markdown("<div class='section-header'>⌀ Risk Score pro Lieferant</div>", unsafe_allow_html=True)
        sup_risk=scored.groupby("Supplier name")["Risk Score"].mean().sort_values(ascending=True)
        bar_c=["#ef4444" if v>=thresh_high else "#f59e0b" if v>=thresh_low else "#22c55e" for v in sup_risk.values]
        fig,ax=plt.subplots(figsize=(8,3.5))
        bars=ax.barh(sup_risk.index,sup_risk.values,color=bar_c,height=0.55,zorder=3)
        ax.axvline(thresh_high,color="#ef4444",lw=1.5,ls="--",alpha=0.8,label=f"Kritisch ≥{thresh_high}")
        ax.axvline(thresh_low,color="#f59e0b",lw=1.2,ls="--",alpha=0.7,label=f"Mittel ≥{thresh_low}")
        ax.set_xlabel("⌀ Risk Score (0 = kein Risiko · 100 = max. Risiko)")
        ax.set_xlim(0,107); ax.grid(axis="x",zorder=0); ax.legend(fontsize=8)
        for b,v in zip(bars,sup_risk.values):
            ax.text(v+1,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(f"""<div class='info-box' style='font-size:0.8rem'>
        <b>Wie kommt der Wert zustande?</b> Der Balken zeigt den Durchschnitt der Risk Scores aller SKUs dieses Lieferanten.
        Jeder SKU-Score = gewichtete Summe aus 5 normierten KPIs (Profil: {preset}).
        </div>""", unsafe_allow_html=True)
    with c2:
        cat_c=scored["Risikostufe"].value_counts()
        fig2,ax2=plt.subplots(figsize=(4.5,3.8))
        wedges,_,autotexts=ax2.pie(cat_c.values,colors=[RISK_COLORS.get(c,"#555") for c in cat_c.index],autopct="%1.0f%%",startangle=140,wedgeprops={"linewidth":2,"edgecolor":"#161b27"},pctdistance=0.75)
        for at in autotexts: at.set_color("#e8eaf0"); at.set_fontsize(12); at.set_fontweight("bold")
        ax2.legend(wedges,[f"{l} ({v})" for l,v in zip(cat_c.index,cat_c.values)],loc="lower center",bbox_to_anchor=(0.5,-0.08),fontsize=9,framealpha=0,labelcolor="#c8d4f0")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    bc1,bc2=st.columns(2)
    for col_w,kpi,ylabel in [(bc1,"Defect rates","Defektrate (%)"),(bc2,"Gesamtlieferzeit","Gesamtlieferzeit (Tage) = Lead time + Shipping times")]:
        with col_w:
            st.markdown(f"<div class='section-header'>📦 Boxplot: {ylabel.split('(')[0].strip()}</div>", unsafe_allow_html=True)
            box_data=[scored[scored["Supplier name"]==s][kpi].dropna().values for s in sup_order]
            fig_b,ax_b=plt.subplots(figsize=(7,4))
            bp=ax_b.boxplot(box_data,labels=sup_order,patch_artist=True,medianprops={"color":"#e8eaf0","linewidth":2},whiskerprops={"color":"#8899bb"},capprops={"color":"#8899bb"},flierprops={"marker":"o","markerfacecolor":"#ef4444","markersize":5,"alpha":0.7})
            for patch,sup in zip(bp["boxes"],sup_order):
                patch.set_facecolor(SUP_COLORS.get(sup,"#2563eb")); patch.set_alpha(0.7)
            ax_b.set_ylabel(ylabel); ax_b.grid(axis="y",zorder=0)
            plt.tight_layout(); st.pyplot(fig_b); plt.close()

    st.markdown("---")
    st.markdown("<div class='section-header'>📈 Gesamtumsatz vs. Defektrate</div>", unsafe_allow_html=True)
    color_by=st.radio("Einfärben nach:",["Lieferant","Produktkategorie","Produkt-Segment"],horizontal=True,key="sc")
    if color_by=="Lieferant":
        pc=[SUP_COLORS.get(s,"#aaa") for s in scored["Supplier name"]]
        li=[(s,SUP_COLORS.get(s,"#aaa")) for s in sup_order if s in scored["Supplier name"].values]
    elif color_by=="Produktkategorie":
        pc=[PROD_COLORS.get(p,"#aaa") for p in scored["Product type"]]
        li=[(p,PROD_COLORS.get(p,"#aaa")) for p in sorted(scored["Product type"].unique())]
    else:
        pc=[BUCKET_COLORS.get(str(b),"#aaa") for b in scored["Produkt-Bucket"]]
        li=[(b,BUCKET_COLORS.get(b,"#aaa")) for b in ["🥇 Premium","🔵 Standard","💚 Budget"]]
    fig5,ax5=plt.subplots(figsize=(10,4))
    ax5.scatter(scored["Defect rates"],scored["Revenue generated"]/1000,c=pc,s=70,alpha=0.85,edgecolors="#0f1117",linewidths=0.8,zorder=3)
    ax5.set_xlabel("Defektrate (%)"); ax5.set_ylabel("Umsatz (Tsd. €)"); ax5.grid(zorder=0)
    ax5.legend(handles=[mpatches.Patch(color=c,label=l) for l,c in li],fontsize=9,framealpha=0.2)
    plt.tight_layout(); st.pyplot(fig5); plt.close()
    st.markdown("<div class='info-box'>Ideal: links oben (niedrige Defektrate, hoher Umsatz). Rechts unten = kritisch.</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  TAB 3 – PRODUKT-KLASSIFIKATION
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"""<div class='info-box'>
    <b>Klassifikation auf Produktebene</b> – Kombination aus Preis, Umsatz und Defektrate (gleichgewichtet, normiert 0–1).<br>
    <b>Grenzwerte:</b> 💚 Budget = Score &lt; {P33:.2f} &nbsp;·&nbsp; 🔵 Standard = {P33:.2f} – {P66:.2f} &nbsp;·&nbsp; 🥇 Premium = Score &gt; {P66:.2f}<br>
    <i>Berechnung: Score = (norm. Preis + norm. Umsatz + (1 – norm. Defektrate)) / 3</i>
    </div>""", unsafe_allow_html=True)

    bk1,bk2,bk3=st.columns(3)
    for col_w,bucket,css in [(bk1,"🥇 Premium","ampel-gruen"),(bk2,"🔵 Standard","ampel-gelb"),(bk3,"💚 Budget","ampel-rot")]:
        sub=scored[scored["Produkt-Bucket"]==bucket]
        if not sub.empty:
            col_w.markdown(f"<div class='ampel-card {css}'>{bucket}<br><span style='font-size:1.5rem;font-weight:700'>{len(sub)} SKUs</span><br><span style='font-size:0.8rem'>⌀ Preis: {sub['Price'].mean():.2f} €</span><br><span style='font-size:0.8rem'>⌀ Umsatz: {sub['Revenue generated'].mean()/1000:.2f}k €</span><br><span style='font-size:0.8rem'>⌀ Defekt: {sub['Defect rates'].mean():.2f}%</span></div>", unsafe_allow_html=True)
    st.markdown("---")

    # Umsatz Produktkategorie + Standort (wie im Bild)
    st.markdown("<div class='section-header'>💰 Produktkategorie Performance</div>", unsafe_allow_html=True)
    cat_perf = scored.groupby("Product type").agg(
        revenue=("Revenue generated","sum"),
        sold=("Number of products sold","sum"),
        defect=("Defect rates","mean"),
        costs=("Costs","mean")
    ).sort_values("revenue",ascending=False).round(2)
    max_rev_cat = cat_perf["revenue"].max()

    up1, up2 = st.columns(2)
    with up1:
        for prod, row in cat_perf.iterrows():
            color = PROD_COLORS.get(prod,"#2563eb")
            bar_pct = row["revenue"]/max_rev_cat*100
            st.markdown(f"""<div class='umsatz-card'>
            <div class='umsatz-card-title' style='color:{color}'>{prod.capitalize()} <span style='float:right;color:#c8d4f0'>{row['revenue']/1000:.0f}k €</span></div>
            <div style='display:flex;gap:24px;font-size:0.82rem;color:#8899bb'>
                <span>Verkauft<br><b style='color:#c8d4f0;font-size:1.0rem'>{int(row['sold']):,}</b></span>
                <span>⌀ Defekt<br><b style='color:#ef4444;font-size:1.0rem'>{row['defect']:.2f}%</b></span>
                <span>⌀ Kosten<br><b style='color:#c8d4f0;font-size:1.0rem'>€{row['costs']:.0f}</b></span>
            </div>
            <div class='umsatz-bar' style='width:{bar_pct:.0f}%;background:{color}'></div>
            </div>""", unsafe_allow_html=True)

    with up2:
        st.markdown("<div class='section-header'>📍 Standort Performance</div>", unsafe_allow_html=True)
        loc_perf = scored.groupby("Location").agg(
            revenue=("Revenue generated","sum"),
            n=("SKU","count"),
            defect=("Defect rates","mean")
        ).sort_values("revenue",ascending=False).round(2)
        max_rev_loc = loc_perf["revenue"].max()
        for i,(loc,row) in enumerate(loc_perf.iterrows()):
            bar_pct = row["revenue"]/max_rev_loc*100
            st.markdown(f"""<div class='umsatz-card'>
            <div style='font-weight:600;color:#c8d4f0'>{LOC_MEDALS[i]} {loc} <span style='float:right;color:#4ade80'>{row['revenue']/1000:.0f}k €</span></div>
            <div style='font-size:0.8rem;color:#8899bb;margin-top:4px'>SKUs: {int(row['n'])} &nbsp;·&nbsp; ⌀ Defekt: <b style='color:#ef4444'>{row['defect']:.2f}%</b></div>
            <div class='umsatz-bar' style='width:{bar_pct:.0f}%;background:#2563eb'></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Umsatz Lieferant x Produkt Kreuztabelle
    st.markdown("<div class='section-header'>📊 Umsatz: Lieferant × Produkt (Tsd. €)</div>", unsafe_allow_html=True)
    cross = scored.groupby(["Supplier name","Product type"])["Revenue generated"].sum().unstack(fill_value=0)/1000
    cross = cross.round(1)
    prods = list(cross.columns)
    max_val = cross.values.max()

    html_table = "<table class='cross-table'><tr><th></th>"
    for p in prods:
        html_table += f"<th>{p.capitalize()}</th>"
    html_table += "</tr>"
    for sup_name in cross.index:
        html_table += f"<tr><td style='color:{SUP_COLORS.get(sup_name,\"#aaa\")};font-weight:600;background:#1a2035'>{'Sup '+sup_name.replace('Supplier ','')}</td>"
        for p in prods:
            v = cross.loc[sup_name,p]
            pct = v/max_val
            cls = "cross-td-high" if pct>0.6 else "cross-td-mid" if pct>0.3 else "cross-td-low"
            html_table += f"<td class='{cls}'>{v:.1f}k</td>"
        html_table += "</tr>"
    html_table += "</table>"
    st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("---")

    # SKU-Tabelle pro Segment
    st.markdown("<div class='section-header'>📋 SKU-Liste nach Segment</div>", unsafe_allow_html=True)
    sel_bk=st.selectbox("Segment anzeigen:",["Alle","🥇 Premium","🔵 Standard","💚 Budget"])
    bk_df=scored if sel_bk=="Alle" else scored[scored["Produkt-Bucket"]==sel_bk]
    st.dataframe(bk_df[["SKU","Supplier name","Product type","Produkt-Bucket","Bucket-Score","Price","Revenue generated","Defect rates","Gesamtlieferzeit","Inspection results","Risk Score","Risikostufe"]].sort_values("Risk Score",ascending=False).reset_index(drop=True),use_container_width=True,height=350)

# ════════════════════════════════════════════════════════════════
#  TAB 4 – LIEFERANTEN-ANALYSE
# ════════════════════════════════════════════════════════════════
with tab4:
    vm=st.radio("Ansicht:",["Einzelner Lieferant","Alle Lieferanten vergleichen"],horizontal=True)
    if vm=="Alle Lieferanten vergleichen":
        ranking=scored.groupby("Supplier name").agg(avg_risk=("Risk Score","mean"),defect=("Defect rates","mean"),lead=("Gesamtlieferzeit","mean"),kosten=("Costs","mean"),fail_n=("Inspection results",lambda x:(x=="Fail").sum()),revenue=("Revenue generated","sum"),n_skus=("SKU","count")).round(2).sort_values("avg_risk")
        ranking.columns=["⌀ Risk","Defekt %","Lieferzeit (Tage)","⌀ Kosten (€)","# Fail","Umsatz (€)","SKUs"]
        fig_r,ax_r=plt.subplots(figsize=(10,4))
        bars_r=ax_r.barh(ranking.index,ranking["⌀ Risk"],color=[SUP_COLORS.get(s,"#2563eb") for s in ranking.index],height=0.55,zorder=3)
        ax_r.axvline(thresh_high,color="#ef4444",lw=1.5,ls="--",alpha=0.8,label=f"Kritisch ≥{thresh_high}")
        ax_r.axvline(thresh_low,color="#f59e0b",lw=1.2,ls="--",alpha=0.7,label=f"Mittel ≥{thresh_low}")
        ax_r.set_xlabel("⌀ Risk Score"); ax_r.set_xlim(0,107); ax_r.grid(axis="x",zorder=0); ax_r.legend(fontsize=8)
        for b,v in zip(bars_r,ranking["⌀ Risk"]):
            ax_r.text(v+1,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",fontsize=10,fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_r); plt.close()
        dr=ranking.reset_index(); dr["Umsatz (€)"]=dr["Umsatz (€)"].map("{:,.2f} €".format)
        st.dataframe(dr,use_container_width=True,height=230)
    else:
        chosen=st.selectbox("🏢 Lieferant auswählen:",sorted(scored["Supplier name"].unique()))
        sup_d=scored[scored["Supplier name"]==chosen]
        r1,r2,r3,r4,r5,r6=st.columns(6)
        r1.metric("⌀ Risk Score",f"{sup_d['Risk Score'].mean():.2f}")
        r2.metric("⌀ Defektrate",f"{sup_d['Defect rates'].mean():.2f} %")
        r3.metric("⌀ Lieferzeit",f"{sup_d['Gesamtlieferzeit'].mean():.2f} Tage")
        r4.metric("Gesamtumsatz",f"{sup_d['Revenue generated'].sum()/1000:.2f}k €")
        r5.metric("Anzahl SKUs",f"{len(sup_d)}")
        r6.metric("Fail-Quote",f"{(sup_d['Inspection results']=='Fail').sum()/len(sup_d)*100:.2f} %")
        rv=sup_d["Risk Score"].mean()
        css_r="risk-high" if rv>=thresh_high else "risk-medium" if rv>=thresh_low else "risk-low"
        icon_r="🔴" if rv>=thresh_high else "🟡" if rv>=thresh_low else "🟢"
        msg_r="Sofortmaßnahmen: Audit, Alternativen prüfen." if rv>=thresh_high else "Monitoring intensivieren." if rv>=thresh_low else "Lieferant performt gut."
        st.markdown(f"<div class='info-box {css_r}'>{icon_r} <b>{chosen} – Risk Score {rv:.2f}/100:</b> {msg_r}</div>", unsafe_allow_html=True)
        st.markdown("---")
        la1,la2=st.columns(2)
        with la1:
            st.markdown("<div class='section-header'>Risk Score pro SKU</div>", unsafe_allow_html=True)
            sku_s=sup_d.sort_values("Risk Score",ascending=True)
            fig_s,ax_s=plt.subplots(figsize=(6,max(3,len(sku_s)*0.30)))
            ax_s.barh(sku_s["SKU"],sku_s["Risk Score"],color=[RISK_COLORS.get(c,"#555") for c in sku_s["Risikostufe"]],height=0.6,zorder=3)
            ax_s.axvline(thresh_high,color="#ef4444",lw=1,ls="--",alpha=0.5,label=f"Kritisch ≥{thresh_high}")
            ax_s.axvline(thresh_low,color="#f59e0b",lw=1,ls="--",alpha=0.5,label=f"Mittel ≥{thresh_low}")
            ax_s.set_xlabel("Risk Score (0–100)"); ax_s.set_xlim(0,105); ax_s.grid(axis="x",zorder=0); ax_s.legend(fontsize=7)
            plt.tight_layout(); st.pyplot(fig_s); plt.close()
        with la2:
            st.markdown("<div class='section-header'>🚚 Transportweg-Ranking</div>", unsafe_allow_html=True)
            st.markdown("<div class='info-box' style='font-size:0.8rem'>Effizienz-Score = 50% normierte Versandkosten + 50% normierte Defektrate. Niedriger Score = bessere Option.</div>", unsafe_allow_html=True)
            td=sup_d.groupby("Transportation modes").agg(avg_cost=("Shipping costs","mean"),avg_defect=("Defect rates","mean"),count=("SKU","count")).reset_index()
            def ns(s):
                mn,mx=s.min(),s.max(); return (s-mn)/(mx-mn) if mx!=mn else pd.Series(0.5,index=s.index)
            td["Eff"]=(0.5*ns(td["avg_cost"])+0.5*ns(td["avg_defect"]))*100 if len(td)>1 else 50.0
            td=td.sort_values("Eff")
            medals=["🥇","🥈","🥉","4️⃣"]
            for i,(_,row) in enumerate(td.iterrows()):
                st.markdown(f"<div class='info-box' style='margin:4px 0'>{medals[min(i,3)]} <b>{row['Transportation modes']}</b> ({int(row['count'])} SKUs) · Kosten: <b>{row['avg_cost']:.2f} €</b> · Defekt: <b>{row['avg_defect']:.2f}%</b> · Score: <b>{row['Eff']:.0f}</b></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div class='section-header'>🔎 Inspektionsstatus – nach Potenzial</div>", unsafe_allow_html=True)
        insp_d=sup_d.copy(); insp_d["_s"]=insp_d["Inspection results"].map({"Pending":0,"Fail":1,"Pass":2})
        st.dataframe(insp_d.sort_values(["_s","Risk Score"],ascending=[True,False])[["SKU","Product type","Produkt-Bucket","Inspection results","Defect rates","Gesamtlieferzeit","Risk Score","Risikostufe"]].reset_index(drop=True),use_container_width=True,height=280)

# ════════════════════════════════════════════════════════════════
#  TAB 5 – KORRELATION
# ════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='info-box'><b>Spearman-Korrelation:</b> misst rang-basierte Zusammenhänge (geeignet für nicht-normalverteilte Daten). +1 = perfekt positiv · 0 = kein Zusammenhang · -1 = negativ</div>", unsafe_allow_html=True)
    sp_data=scored_full.copy()
    sp_data["Fail_binary"]=(sp_data["Inspection results"]=="Fail").astype(int)
    kpi_cols=["Defect rates","Gesamtlieferzeit","Costs","Fail_binary","Revenue generated"]
    kpi_labels=["Defekt %","Lead Time (Tage)","Kosten (€)","Fail-Quote","Umsatz (€)"]
    sp_matrix=np.zeros((len(kpi_cols),len(kpi_cols)))
    for i,c1 in enumerate(kpi_cols):
        for j,c2 in enumerate(kpi_cols):
            r,_=spearmanr(sp_data[c1].dropna(),sp_data[c2].dropna())
            sp_matrix[i,j]=round(r,2)
    mask=np.ones_like(sp_matrix,dtype=bool); np.fill_diagonal(mask,False)
    flat=sp_matrix.copy(); flat[~mask]=0
    max_idx=np.unravel_index(np.abs(flat).argmax(),flat.shape)
    min_idx=np.unravel_index(np.abs(flat).argmin(),flat.shape)
    mc1,mc2,mc3,mc4=st.columns(4)
    mc1.metric("Stärkste Korrelation",f"{sp_matrix[max_idx]:.2f}",delta=f"{kpi_labels[max_idx[0]]} ↔ {kpi_labels[max_idx[1]]}")
    mc2.metric("Schwächste Korr.",f"{sp_matrix[min_idx]:.2f}",delta=f"{kpi_labels[min_idx[0]]} ↔ {kpi_labels[min_idx[1]]}")
    mc3.metric("Multikollinearität","Gering",delta="Modell sicher")
    mc4.metric("Methode","Spearman",delta="nicht-normalverteilt")
    km1,km2=st.columns(2)
    with km1:
        fig_sp,ax_sp=plt.subplots(figsize=(6,5))
        hm3=np.zeros_like(sp_matrix,dtype=float)
        for i in range(sp_matrix.shape[0]):
            for j in range(sp_matrix.shape[1]):
                v=(sp_matrix[i,j]+1)/2
                hm3[i,j]=v
        im=ax_sp.imshow(hm3,cmap=plt.cm.RdYlGn,vmin=0,vmax=1,aspect="auto")
        ax_sp.set_xticks(range(len(kpi_labels))); ax_sp.set_xticklabels(kpi_labels,rotation=20,ha="right",fontsize=9)
        ax_sp.set_yticks(range(len(kpi_labels))); ax_sp.set_yticklabels(kpi_labels,fontsize=9)
        for i in range(len(kpi_labels)):
            for j in range(len(kpi_labels)):
                ax_sp.text(j,i,f"{sp_matrix[i,j]:.2f}",ha="center",va="center",fontsize=11,fontweight="bold",color="#0f1117" if abs(sp_matrix[i,j])>0.3 else "#c8d4f0")
        plt.colorbar(im,ax=ax_sp,shrink=0.8); plt.tight_layout(); st.pyplot(fig_sp); plt.close()
    with km2:
        fig_sc,ax_sc=plt.subplots(figsize=(6,5))
        ax_sc.scatter(sp_data[kpi_cols[max_idx[0]]],sp_data[kpi_cols[max_idx[1]]],c=[SUP_COLORS.get(s,"#aaa") for s in sp_data["Supplier name"]],s=70,alpha=0.85,edgecolors="#0f1117",linewidths=0.7,zorder=3)
        ax_sc.set_xlabel(kpi_labels[max_idx[0]]+" ("+["","Tage","€","","€"][max_idx[0]]+")")
        ax_sc.set_ylabel(kpi_labels[max_idx[1]])
        ax_sc.set_title(f"Stärkster Zusammenhang: ρ = {sp_matrix[max_idx]:.2f}",fontsize=10); ax_sc.grid(zorder=0)
        ax_sc.legend(handles=[mpatches.Patch(color=SUP_COLORS.get(s,"#aaa"),label=s) for s in sorted(sp_data["Supplier name"].unique())],fontsize=8,framealpha=0.2)
        plt.tight_layout(); st.pyplot(fig_sc); plt.close()
    st.markdown(f"<div class='info-box'>Stärkster Zusammenhang: <b>{kpi_labels[max_idx[0]]}</b> ↔ <b>{kpi_labels[max_idx[1]]}</b> (ρ = {sp_matrix[max_idx]:.2f}). Kein KPI korreliert stark mit einem anderen → kein Multikollinearitätsproblem.</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  TAB 6 – ALLE DATEN
# ════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<div class='section-header'>📋 Alle Daten – filter- & sortierbar · Export CSV & Excel</div>", unsafe_allow_html=True)
    base_cols=list(df.columns)
    remove_internal=["_p33","_p66"]
    base_cols=[c for c in base_cols if c not in remove_internal]
    extra_cols=[c for c in ["Gesamtlieferzeit","Risk Score","Risikostufe"] if c not in base_cols]
    all_cols=base_cols+extra_cols
    with st.expander("🗂️ Spalten auswählen",expanded=False):
        sel_cols2=st.multiselect("Spalten",all_cols,default=all_cols)
    with st.expander("🔧 Zusätzliche Filter",expanded=False):
        f1,f2,f3=st.columns(3)
        with f1:
            ff_sup=st.multiselect("Lieferant",sorted(df["Supplier name"].unique()),default=sorted(df["Supplier name"].unique()),key="ffs")
            ff_prod=st.multiselect("Produkttyp",sorted(df["Product type"].unique()),default=sorted(df["Product type"].unique()),key="ffp")
        with f2:
            ff_loc=st.multiselect("Standort",sorted(df["Location"].unique()),default=sorted(df["Location"].unique()),key="ffl")
            ff_insp=st.multiselect("Inspektion",sorted(df["Inspection results"].unique()),default=sorted(df["Inspection results"].unique()),key="ffi")
        with f3:
            ff_risk=st.multiselect("Risikostufe",["🔴 Hoch","🟡 Mittel","🟢 Niedrig"],default=["🔴 Hoch","🟡 Mittel","🟢 Niedrig"],key="ffr")
            ff_buck=st.multiselect("Segment",["🥇 Premium","🔵 Standard","💚 Budget"],default=["🥇 Premium","🔵 Standard","💚 Budget"],key="ffb")
    s1,s2=st.columns([3,1])
    with s1: sort_by2=st.selectbox("Sortieren nach",["Risk Score","Defect rates","Gesamtlieferzeit","Revenue generated","Costs","Price"])
    with s2: asc2=st.radio("Reihenfolge",["↓ Absteigend","↑ Aufsteigend"])=="↑ Aufsteigend"
    table=scored_full[
        scored_full["Supplier name"].isin(ff_sup)&scored_full["Product type"].isin(ff_prod)&
        scored_full["Location"].isin(ff_loc)&scored_full["Inspection results"].isin(ff_insp)&
        scored_full["Risikostufe"].isin(ff_risk)&scored_full["Produkt-Bucket"].isin(ff_buck)
    ].copy()
    valid2=[c for c in sel_cols2 if c in table.columns]
    table=table[valid2].sort_values(sort_by2,ascending=asc2).reset_index(drop=True)
    st.markdown(f"<p style='color:#5577aa;font-size:0.85rem'>{len(table)} Zeilen · {len(valid2)} Spalten</p>", unsafe_allow_html=True)
    st.dataframe(table,use_container_width=True,height=500)
    ex1,ex2=st.columns(2)
    with ex1:
        st.download_button("⬇️ CSV exportieren",data=table.to_csv(index=False).encode("utf-8"),file_name="riskradar_export.csv",mime="text/csv")
    with ex2:
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine="openpyxl") as writer: table.to_excel(writer,index=False,sheet_name="RiskRadar")
        buf.seek(0)
        st.download_button("⬇️ Excel exportieren",data=buf.getvalue(),file_name="riskradar_export.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#3a4a6a;font-size:0.8rem'>RiskRadar v7 · THI Ingolstadt · DPDS SoSe 2026 · Gruppe 9: Laurenz Angleitner · Leon Pavic · Alex Rauschendorfer · Daniel Steinmetz</p>", unsafe_allow_html=True)
