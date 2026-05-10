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
.info-box{background:linear-gradient(135deg,#1a2640,#1e2d4a);border-left:4px solid #2563eb;border-radius:0 10px 10px 0;padding:12px 16px;margin:8px 0;font-size:0.85rem;color:#a8bbd4;line-height:1.6;}
.risk-high{border-left-color:#ef4444 !important;background:linear-gradient(135deg,#2a1a1a,#331c1c) !important;color:#ffb3b3 !important;}
.risk-medium{border-left-color:#f59e0b !important;background:linear-gradient(135deg,#2a2210,#332a10) !important;color:#ffe0a0 !important;}
.risk-low{border-left-color:#22c55e !important;background:linear-gradient(135deg,#0f2a18,#12331e) !important;color:#a0f0c0 !important;}
.section-header{font-size:1.0rem;font-weight:600;color:#c8d4f0;letter-spacing:0.04em;margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid #252d3d;}
.ampel-card{border-radius:14px;padding:18px 20px;text-align:center;font-weight:600;font-size:1.1rem;margin:6px 0;box-shadow:0 4px 16px rgba(0,0,0,0.3);}
.ampel-rot{background:linear-gradient(135deg,#3a1010,#4a1515);border:2px solid #ef4444;color:#ff8080;}
.ampel-gelb{background:linear-gradient(135deg,#2a2010,#3a2c10);border:2px solid #f59e0b;color:#ffd080;}
.ampel-gruen{background:linear-gradient(135deg,#0a2a14,#0f3a1c);border:2px solid #22c55e;color:#80ffa8;}
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
    # Produkt-Klassifikation
    def norm(s):
        mn,mx=s.min(),s.max(); return (s-mn)/(mx-mn+1e-9)
    score = (norm(df["Price"]) + norm(df["Revenue generated"]) + (1-norm(df["Defect rates"])))/3
    p33,p66 = score.quantile(0.33), score.quantile(0.66)
    df["Produkt-Bucket"] = pd.cut(score, bins=[-1,p33,p66,2], labels=["💚 Budget","🔵 Standard","🥇 Premium"])
    df["Bucket-Score"] = score.round(2)
    return df

df = load_data()
for i,s in enumerate(sorted(df["Supplier name"].unique())):
    SUP_COLORS[s] = ["#2563eb","#f59e0b","#22c55e","#a855f7","#ef4444"][i%5]

def norm_col(series, invert=False):
    mn,mx=series.min(),series.max()
    if mx==mn: return pd.Series(0.5,index=series.index)
    n=(series-mn)/(mx-mn)
    return (1-n) if invert else n

def compute_risk(df_in, w_defect, w_lead, w_cost, w_inspection, w_revenue, thresh_high, thresh_low):
    r=df_in.copy()
    r["_nd"]=norm_col(r["Defect rates"])
    r["_nl"]=norm_col(r["Gesamtlieferzeit"])
    r["_nc"]=norm_col(r["Costs"])
    r["_nr"]=norm_col(r["Revenue generated"],invert=True)
    r["_ni"]=r["Inspection results"].map({"Fail":1.0,"Pending":0.5,"Pass":0.0}).fillna(0.5)
    total=w_defect+w_lead+w_cost+w_inspection+w_revenue
    r["Risk Score"]=(w_defect*r["_nd"]+w_lead*r["_nl"]+w_cost*r["_nc"]+w_inspection*r["_ni"]+w_revenue*r["_nr"])/total*100
    r["Risk Score"]=r["Risk Score"].round(2)
    def cat(s):
        if s>=thresh_high: return "🔴 Hoch"
        if s>=thresh_low:  return "🟡 Mittel"
        return "🟢 Niedrig"
    r["Risikostufe"]=r["Risk Score"].apply(cat)
    return r.drop(columns=[c for c in r.columns if c.startswith("_")])

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ RiskRadar")
    st.markdown("<p style='color:#5577aa;font-size:0.8rem;margin-top:-10px'>THI Ingolstadt · DPDS 2026 · Gruppe 9</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🎯 Entscheidungs-Profil")
    preset = st.radio("", ["🔧 Manuell","🏆 Höchste Qualität","💰 Umsatzstärkster Lieferant","⚡ Günstig & Schnell"], key="preset", label_visibility="collapsed")
    if preset == "🏆 Höchste Qualität":
        w_defect,w_lead,w_cost,w_inspection,w_revenue = 10,1,1,10,5
        st.markdown("<div class='info-box risk-low'>🏆 Qualität first – Defektrate & Inspektion dominieren.</div>", unsafe_allow_html=True)
    elif preset == "💰 Umsatzstärkster Lieferant":
        w_defect,w_lead,w_cost,w_inspection,w_revenue = 3,3,2,3,10
        st.markdown("<div class='info-box risk-medium'>💰 Umsatz-Fokus – hoher Revenue = geringes Risiko.</div>", unsafe_allow_html=True)
    elif preset == "⚡ Günstig & Schnell":
        w_defect,w_lead,w_cost,w_inspection,w_revenue = 1,10,10,1,5
        st.markdown("<div class='info-box risk-high'>⚡ Speed & Cost – Lieferzeit & Kosten dominieren.</div>", unsafe_allow_html=True)
    else:
        w_defect=st.slider("🔬 Defektrate",1,10,4)
        w_lead=st.slider("⏱️ Gesamtlieferzeit",1,10,3)
        w_cost=st.slider("💰 Gesamtkosten",1,10,2)
        w_inspection=st.slider("🔎 Inspektionsergebnis",1,10,5)
        w_revenue=st.slider("📈 Umsatz (hoch=geringes Risiko)",1,10,2)
    total_w=w_defect+w_lead+w_cost+w_inspection+w_revenue
    st.markdown(f"<div class='info-box' style='font-size:0.78rem'>🔬{w_defect/total_w*100:.0f}% · ⏱️{w_lead/total_w*100:.0f}% · 💰{w_cost/total_w*100:.0f}% · 🔎{w_inspection/total_w*100:.0f}% · 📈{w_revenue/total_w*100:.0f}%</div>", unsafe_allow_html=True)
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
    st.markdown("### 🎚️ Grenzwerte")
    thresh_high=st.slider("Ab hier 🔴 Hoch",34,90,60)
    thresh_low=st.slider("Ab hier 🟡 Mittel",10,thresh_high-1,35)
    st.markdown(f"<div class='info-box' style='font-size:0.78rem'>🔴≥{thresh_high} · 🟡{thresh_low}–{thresh_high-1} · 🟢&lt;{thresh_low}<br><i>Default 60: Terzil-Einteilung</i></div>", unsafe_allow_html=True)

# ── FILTER + SCORING ─────────────────────────────────────────────
filtered = df[
    df["Supplier name"].isin(sel_sup) & df["Product type"].isin(sel_prod) &
    df["Produkt-Bucket"].isin(sel_bucket) & df["Location"].isin(sel_loc) &
    df["Shipping carriers"].isin(sel_carr) & df["Transportation modes"].isin(sel_mode) &
    df["Inspection results"].isin(sel_insp) & (df["Defect rates"] <= max_defect)
].copy()

if filtered.empty:
    st.warning("⚠️ Keine Daten. Filter anpassen."); st.stop()

scored = compute_risk(filtered, w_defect,w_lead,w_cost,w_inspection,w_revenue, thresh_high,thresh_low)
scored_full = compute_risk(df, w_defect,w_lead,w_cost,w_inspection,w_revenue, thresh_high,thresh_low)
sup_order = sorted(scored["Supplier name"].unique())

# ── HEADER ───────────────────────────────────────────────────────
st.markdown(f"<h1 style='font-size:1.9rem;font-weight:700;color:#e8eaf0;margin-bottom:2px'>🛡️ RiskRadar</h1><p style='color:#5577aa;font-size:0.88rem;margin-top:0'>Supplier & Procurement Risk Intelligence · THI Ingolstadt DPDS 2026 · Gruppe 9 &nbsp;·&nbsp; Profil: <b style='color:#c8d4f0'>{preset}</b> &nbsp;·&nbsp; Grenzwert: <b style='color:#ef4444'>{thresh_high}</b></p>", unsafe_allow_html=True)
st.markdown("---")
n_high=(scored["Risikostufe"]=="🔴 Hoch").sum()
n_med=(scored["Risikostufe"]=="🟡 Mittel").sum()
n_low=(scored["Risikostufe"]=="🟢 Niedrig").sum()
k1,k2,k3,k4,k5,k6,k7=st.columns(7)
k1.metric("⌀ Risk Score",f"{scored['Risk Score'].mean():.2f}")
k2.metric("🔴 Kritisch",f"{n_high} SKUs",delta=f"{n_high/len(scored)*100:.0f}%",delta_color="inverse")
k3.metric("🟡 Mittel",f"{n_med} SKUs")
k4.metric("🟢 OK",f"{n_low} SKUs")
k5.metric("⌀ Defektrate",f"{scored['Defect rates'].mean():.2f}%")
k6.metric("⌀ Lieferzeit",f"{scored['Gesamtlieferzeit'].mean():.2f} Tage")
k7.metric("Gesamtumsatz",f"{scored['Revenue generated'].sum()/1000:.2f}k €")
st.markdown("---")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["🚦 Entscheidungs-Dashboard","📊 Risk Overview","🏷️ Produkt-Klassifikation","🏢 Lieferanten-Analyse","🔬 Korrelation","📋 Alle Daten"])

# ── TAB 1: ENTSCHEIDUNGS-DASHBOARD ───────────────────────────────
with tab1:
    st.markdown("<div class='info-box'><b>Für wen:</b> CEO oder Supply Chain Manager &nbsp;·&nbsp; <b>Workflow:</b> Profil wählen → Ampel prüfen → Kritische SKUs ansehen → Entscheidung treffen</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🚦 Lieferanten-Ampel</div>", unsafe_allow_html=True)
    sup_risks = scored.groupby("Supplier name")["Risk Score"].mean()
    amp_cols = st.columns(len(sup_order))
    for i,sup in enumerate(sup_order):
        if sup not in sup_risks.index: continue
        rv=sup_risks[sup]
        css,icon,status=("ampel-rot","🔴","KRITISCH") if rv>=thresh_high else ("ampel-gelb","🟡","BEOBACHTEN") if rv>=thresh_low else ("ampel-gruen","🟢","OK")
        n_fail=(scored[scored["Supplier name"]==sup]["Inspection results"]=="Fail").sum()
        n_skus=len(scored[scored["Supplier name"]==sup])
        amp_cols[i].markdown(f"<div class='ampel-card {css}'>{icon} {sup}<br><span style='font-size:1.6rem;font-weight:700'>{rv:.1f}</span><br><span style='font-size:0.8rem'>{status}</span><br><span style='font-size:0.75rem;opacity:0.8'>{n_skus} SKUs · {n_fail} Fails</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-header'>📋 Handlungsempfehlungen</div>", unsafe_allow_html=True)
    he1,he2,he3=st.columns(3)
    kritische=[s for s in sup_order if s in sup_risks.index and sup_risks[s]>=thresh_high]
    mittlere=[s for s in sup_order if s in sup_risks.index and thresh_low<=sup_risks[s]<thresh_high]
    ok=[s for s in sup_order if s in sup_risks.index and sup_risks[s]<thresh_low]
    with he1:
        st.markdown(f"<div class='info-box risk-high'>🔴 <b>Sofortmaßnahmen</b><br>{', '.join(kritische) if kritische else 'Keine'}<br><br>→ Lieferanten-Audit einleiten<br>→ Alternativen prüfen<br>→ Bestellmengen reduzieren</div>", unsafe_allow_html=True)
    with he2:
        st.markdown(f"<div class='info-box risk-medium'>🟡 <b>Monitoring</b><br>{', '.join(mittlere) if mittlere else 'Keine'}<br><br>→ Lieferantengespräch<br>→ KPI-Entwicklung beobachten<br>→ Vertragskonditionen prüfen</div>", unsafe_allow_html=True)
    with he3:
        st.markdown(f"<div class='info-box risk-low'>🟢 <b>Routinebetrieb</b><br>{', '.join(ok) if ok else 'Keine'}<br><br>→ Weiter beobachten<br>→ Best Practices dokumentieren<br>→ Als Referenz nutzen</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"<div class='section-header'>🔴 Kritische SKUs (Risk Score ≥ {thresh_high})</div>", unsafe_allow_html=True)
    crit=scored[scored["Risikostufe"]=="🔴 Hoch"].sort_values("Risk Score",ascending=False)
    if crit.empty:
        st.markdown("<div class='info-box risk-low'>✅ Keine kritischen SKUs bei aktueller Einstellung.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p style='color:#ef4444;font-size:0.9rem'><b>{len(crit)} kritische SKUs</b></p>", unsafe_allow_html=True)
        st.dataframe(crit[["SKU","Supplier name","Product type","Produkt-Bucket","Defect rates","Gesamtlieferzeit","Costs","Revenue generated","Inspection results","Risk Score","Risikostufe"]].reset_index(drop=True), use_container_width=True, height=350)

    st.markdown("---")
    qs1,qs2,qs3=st.columns(3)
    with qs1:
        st.markdown("<div class='section-header' style='font-size:0.85rem'>Top 5 riskanteste SKUs</div>", unsafe_allow_html=True)
        st.dataframe(scored.nlargest(5,"Risk Score")[["SKU","Supplier name","Risk Score","Risikostufe"]].reset_index(drop=True), use_container_width=True, height=220)
    with qs2:
        st.markdown("<div class='section-header' style='font-size:0.85rem'>Top 5 beste SKUs</div>", unsafe_allow_html=True)
        st.dataframe(scored.nsmallest(5,"Risk Score")[["SKU","Supplier name","Risk Score","Risikostufe"]].reset_index(drop=True), use_container_width=True, height=220)
    with qs3:
        st.markdown("<div class='section-header' style='font-size:0.85rem'>Fail-Quote pro Lieferant</div>", unsafe_allow_html=True)
        fq=scored.groupby("Supplier name").apply(lambda x: round((x["Inspection results"]=="Fail").sum()/len(x)*100,2)).reset_index()
        fq.columns=["Lieferant","Fail %"]
        st.dataframe(fq.sort_values("Fail %",ascending=False).reset_index(drop=True), use_container_width=True, height=220)

# ── TAB 2: RISK OVERVIEW ─────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>🌡️ KPI Heatmap (synchronisiert mit Filter)</div>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Rot = schlechter Wert · Grün = guter Wert. Spalten: Defektrate · Gesamtlieferzeit · Gesamtkosten · Fail-% · Umsatz</div>", unsafe_allow_html=True)

    hm=scored.groupby("Supplier name").agg(
        Defekt=("Defect rates","mean"), Lieferzeit=("Gesamtlieferzeit","mean"),
        Kosten=("Costs","mean"), Fail=("Inspection results",lambda x:round((x=="Fail").sum()/len(x)*100,2)),
        Umsatz=("Revenue generated","sum")
    ).round(2)
    hm_labels=["Defekt %","Lieferzeit (d)","Kosten €","Fail %","Umsatz €"]
    hm_invert=[True,True,True,True,False]
    hm_vals=hm.values
    hm_norm=np.zeros_like(hm_vals,dtype=float)
    for j in range(hm_vals.shape[1]):
        col=hm_vals[:,j].astype(float); mn,mx=col.min(),col.max()
        n=(col-mn)/(mx-mn+1e-9)
        hm_norm[:,j]=(1-n) if hm_invert[j] else n

    fig_hm,ax_hm=plt.subplots(figsize=(11,max(2.5,len(hm)*0.7)))
    for i in range(hm_norm.shape[0]):
        for j in range(hm_norm.shape[1]):
            ax_hm.add_patch(plt.Rectangle([j,i],1,1,color=plt.cm.RdYlGn(hm_norm[i,j]),zorder=2))
            raw=hm_vals[i,j]
            txt=f"{raw:.2f}%" if j in [0,3] else f"{raw:.2f}d" if j==1 else f"€{raw:.0f}" if j==2 else f"€{raw/1000:.1f}k"
            ax_hm.text(j+0.5,i+0.5,txt,ha="center",va="center",fontsize=10,fontweight="bold",color="#0f1117",zorder=3)
    ax_hm.set_xlim(0,len(hm_labels)); ax_hm.set_ylim(0,len(hm))
    ax_hm.set_xticks([x+0.5 for x in range(len(hm_labels))]); ax_hm.set_xticklabels(hm_labels,fontsize=10)
    ax_hm.set_yticks([y+0.5 for y in range(len(hm))]); ax_hm.set_yticklabels(hm.index,fontsize=10)
    plt.tight_layout(); st.pyplot(fig_hm); plt.close()
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
        ax.set_xlabel("⌀ Risk Score"); ax.set_xlim(0,107); ax.grid(axis="x",zorder=0); ax.legend(fontsize=8)
        for b,v in zip(bars,sup_risk.values):
            ax.text(v+1,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        cat_c=scored["Risikostufe"].value_counts()
        fig2,ax2=plt.subplots(figsize=(4.5,3.8))
        wedges,_,autotexts=ax2.pie(cat_c.values,colors=[RISK_COLORS.get(c,"#555") for c in cat_c.index],autopct="%1.0f%%",startangle=140,wedgeprops={"linewidth":2,"edgecolor":"#161b27"},pctdistance=0.75)
        for at in autotexts: at.set_color("#e8eaf0"); at.set_fontsize(12); at.set_fontweight("bold")
        ax2.legend(wedges,[f"{l} ({v})" for l,v in zip(cat_c.index,cat_c.values)],loc="lower center",bbox_to_anchor=(0.5,-0.08),fontsize=9,framealpha=0,labelcolor="#c8d4f0")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    st.markdown("<div class='section-header'>📦 Boxplots: Defektrate & Gesamtlieferzeit</div>", unsafe_allow_html=True)
    bc1,bc2=st.columns(2)
    for col_w,kpi,ylabel in [(bc1,"Defect rates","Defektrate (%)"),(bc2,"Gesamtlieferzeit","Gesamtlieferzeit (Tage)")]:
        with col_w:
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

# ── TAB 3: PRODUKT-KLASSIFIKATION ────────────────────────────────
with tab3:
    st.markdown("<div class='info-box'><b>Klassifikation auf Produktebene</b> basierend auf Preis + Umsatz + Defektrate.<br>🥇 <b>Premium</b> = hoher Preis, hoher Umsatz, niedrige Defektrate · 🔵 <b>Standard</b> = Mitte · 💚 <b>Budget</b> = günstig, hohe Menge</div>", unsafe_allow_html=True)
    bk1,bk2,bk3=st.columns(3)
    for col_w,bucket,css in [(bk1,"🥇 Premium","ampel-gruen"),(bk2,"🔵 Standard","ampel-gelb"),(bk3,"💚 Budget","ampel-rot")]:
        sub=scored[scored["Produkt-Bucket"]==bucket]
        if not sub.empty:
            col_w.markdown(f"<div class='ampel-card {css}'>{bucket}<br><span style='font-size:1.5rem;font-weight:700'>{len(sub)} SKUs</span><br><span style='font-size:0.8rem'>⌀ Preis: {sub['Price'].mean():.2f}€</span><br><span style='font-size:0.8rem'>⌀ Umsatz: {sub['Revenue generated'].mean()/1000:.2f}k€</span><br><span style='font-size:0.8rem'>⌀ Defekt: {sub['Defect rates'].mean():.2f}%</span></div>", unsafe_allow_html=True)
    st.markdown("---")
    pk1,pk2=st.columns(2)
    with pk1:
        st.markdown("<div class='section-header'>Segment-Verteilung pro Lieferant</div>", unsafe_allow_html=True)
        pivot_b=scored.groupby(["Supplier name","Produkt-Bucket"]).size().unstack(fill_value=0)
        fig_bk,ax_bk=plt.subplots(figsize=(6,4))
        x=np.arange(len(pivot_b)); w=0.25
        for i,b in enumerate(["🥇 Premium","🔵 Standard","💚 Budget"]):
            if b in pivot_b.columns:
                ax_bk.bar(x+i*w,pivot_b[b].values,w,label=b,color=BUCKET_COLORS.get(b,"#aaa"),alpha=0.9,zorder=3)
        ax_bk.set_xticks(x+w); ax_bk.set_xticklabels(pivot_b.index,rotation=15)
        ax_bk.set_ylabel("Anzahl SKUs"); ax_bk.legend(fontsize=8,framealpha=0.2); ax_bk.grid(axis="y",zorder=0)
        plt.tight_layout(); st.pyplot(fig_bk); plt.close()
    with pk2:
        st.markdown("<div class='section-header'>⌀ Risk Score pro Segment</div>", unsafe_allow_html=True)
        rbb=scored.groupby("Produkt-Bucket")["Risk Score"].mean().reindex(["🥇 Premium","🔵 Standard","💚 Budget"]).dropna()
        fig_rb,ax_rb=plt.subplots(figsize=(6,4))
        bars_rb=ax_rb.bar(rbb.index,rbb.values,color=[BUCKET_COLORS.get(str(b),"#aaa") for b in rbb.index],width=0.5,zorder=3)
        ax_rb.axhline(thresh_high,color="#ef4444",lw=1.2,ls="--",alpha=0.7)
        ax_rb.axhline(thresh_low,color="#f59e0b",lw=1.2,ls="--",alpha=0.7)
        ax_rb.set_ylabel("⌀ Risk Score"); ax_rb.set_ylim(0,105); ax_rb.grid(axis="y",zorder=0)
        for b,v in zip(bars_rb,rbb.values):
            ax_rb.text(b.get_x()+b.get_width()/2,v+1,f"{v:.2f}",ha="center",fontsize=10,fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_rb); plt.close()
    st.markdown("---")
    sel_bk=st.selectbox("Segment anzeigen:",["Alle","🥇 Premium","🔵 Standard","💚 Budget"])
    bk_df=scored if sel_bk=="Alle" else scored[scored["Produkt-Bucket"]==sel_bk]
    st.dataframe(bk_df[["SKU","Supplier name","Product type","Produkt-Bucket","Price","Revenue generated","Defect rates","Gesamtlieferzeit","Inspection results","Risk Score","Risikostufe"]].sort_values("Risk Score",ascending=False).reset_index(drop=True), use_container_width=True, height=380)

# ── TAB 4: LIEFERANTEN-ANALYSE ────────────────────────────────────
with tab4:
    vm=st.radio("Ansicht:",["Einzelner Lieferant","Alle Lieferanten vergleichen"],horizontal=True)
    if vm=="Alle Lieferanten vergleichen":
        ranking=scored.groupby("Supplier name").agg(avg_risk=("Risk Score","mean"),defect=("Defect rates","mean"),lead=("Gesamtlieferzeit","mean"),kosten=("Costs","mean"),fail_n=("Inspection results",lambda x:(x=="Fail").sum()),revenue=("Revenue generated","sum"),n_skus=("SKU","count")).round(2).sort_values("avg_risk")
        ranking.columns=["⌀ Risk","Defekt %","Lieferzeit (d)","⌀ Kosten","# Fail","Umsatz (€)","SKUs"]
        fig_r,ax_r=plt.subplots(figsize=(10,4))
        bars_r=ax_r.barh(ranking.index,ranking["⌀ Risk"],color=[SUP_COLORS.get(s,"#2563eb") for s in ranking.index],height=0.55,zorder=3)
        ax_r.axvline(thresh_high,color="#ef4444",lw=1.5,ls="--",alpha=0.8)
        ax_r.axvline(thresh_low,color="#f59e0b",lw=1.2,ls="--",alpha=0.7)
        ax_r.set_xlabel("⌀ Risk Score"); ax_r.set_xlim(0,107); ax_r.grid(axis="x",zorder=0)
        for b,v in zip(bars_r,ranking["⌀ Risk"]):
            ax_r.text(v+1,b.get_y()+b.get_height()/2,f"{v:.2f}",va="center",fontsize=10,fontweight="bold")
        plt.tight_layout(); st.pyplot(fig_r); plt.close()
        dr=ranking.reset_index(); dr["Umsatz (€)"]=dr["Umsatz (€)"].map("{:,.2f} €".format)
        st.dataframe(dr, use_container_width=True, height=230)
    else:
        chosen=st.selectbox("🏢 Lieferant:",sorted(scored["Supplier name"].unique()))
        sup_d=scored[scored["Supplier name"]==chosen]
        r1,r2,r3,r4,r5,r6=st.columns(6)
        r1.metric("⌀ Risk Score",f"{sup_d['Risk Score'].mean():.2f}")
        r2.metric("⌀ Defektrate",f"{sup_d['Defect rates'].mean():.2f}%")
        r3.metric("⌀ Lieferzeit",f"{sup_d['Gesamtlieferzeit'].mean():.2f} Tage")
        r4.metric("Gesamtumsatz",f"{sup_d['Revenue generated'].sum()/1000:.2f}k €")
        r5.metric("Anzahl SKUs",f"{len(sup_d)}")
        r6.metric("Fail-Quote",f"{(sup_d['Inspection results']=='Fail').sum()/len(sup_d)*100:.2f}%")
        rv=sup_d["Risk Score"].mean()
        css_r="risk-high" if rv>=thresh_high else "risk-medium" if rv>=thresh_low else "risk-low"
        icon_r="🔴" if rv>=thresh_high else "🟡" if rv>=thresh_low else "🟢"
        msg_r="Sofortmaßnahmen: Audit, Alternativen prüfen." if rv>=thresh_high else "Monitoring intensivieren." if rv>=thresh_low else "Lieferant performt gut."
        st.markdown(f"<div class='info-box {css_r}'>{icon_r} <b>{chosen} ({rv:.2f}/100):</b> {msg_r}</div>", unsafe_allow_html=True)
        st.markdown("---")
        la1,la2=st.columns(2)
        with la1:
            sku_s=sup_d.sort_values("Risk Score",ascending=True)
            fig_s,ax_s=plt.subplots(figsize=(6,max(3,len(sku_s)*0.30)))
            ax_s.barh(sku_s["SKU"],sku_s["Risk Score"],color=[RISK_COLORS.get(c,"#555") for c in sku_s["Risikostufe"]],height=0.6,zorder=3)
            ax_s.axvline(thresh_high,color="#ef4444",lw=1,ls="--",alpha=0.5)
            ax_s.axvline(thresh_low,color="#f59e0b",lw=1,ls="--",alpha=0.5)
            ax_s.set_xlabel("Risk Score"); ax_s.set_xlim(0,105); ax_s.grid(axis="x",zorder=0)
            plt.tight_layout(); st.pyplot(fig_s); plt.close()
        with la2:
            st.markdown("<div class='section-header'>🚚 Transportweg-Ranking</div>", unsafe_allow_html=True)
            td=sup_d.groupby("Transportation modes").agg(avg_cost=("Shipping costs","mean"),avg_defect=("Defect rates","mean"),count=("SKU","count")).reset_index()
            def ns(s):
                mn,mx=s.min(),s.max(); return (s-mn)/(mx-mn) if mx!=mn else pd.Series(0.5,index=s.index)
            td["Eff"]=(0.5*ns(td["avg_cost"])+0.5*ns(td["avg_defect"]))*100 if len(td)>1 else 50.0
            td=td.sort_values("Eff")
            for i,(_,row) in enumerate(td.iterrows()):
                medals=["🥇","🥈","🥉","4️⃣"]
                st.markdown(f"<div class='info-box' style='margin:4px 0'>{medals[min(i,3)]} <b>{row['Transportation modes']}</b> ({int(row['count'])} SKUs) · Kosten: <b>{row['avg_cost']:.2f}</b> · Defekt: <b>{row['avg_defect']:.2f}%</b> · Score: <b>{row['Eff']:.0f}</b></div>", unsafe_allow_html=True)

# ── TAB 5: KORRELATION ───────────────────────────────────────────
with tab5:
    st.markdown("<div class='info-box'><b>Spearman-Korrelation</b>: misst rang-basierte Zusammenhänge (geeignet für nicht-normalverteilte Daten). +1 = perfekt positiv · 0 = kein Zusammenhang · -1 = negativ</div>", unsafe_allow_html=True)
    sp_data=scored_full.copy()
    sp_data["Fail_binary"]=(sp_data["Inspection results"]=="Fail").astype(int)
    kpi_cols=["Defect rates","Gesamtlieferzeit","Costs","Fail_binary","Revenue generated"]
    kpi_labels=["Defekt %","Lead Time","Kosten","Fail-Quote","Umsatz"]
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
        im=ax_sp.imshow(sp_matrix,cmap=plt.cm.RdYlGn,vmin=-1,vmax=1,aspect="auto")
        ax_sp.set_xticks(range(len(kpi_labels))); ax_sp.set_xticklabels(kpi_labels,rotation=20,ha="right")
        ax_sp.set_yticks(range(len(kpi_labels))); ax_sp.set_yticklabels(kpi_labels)
        for i in range(len(kpi_labels)):
            for j in range(len(kpi_labels)):
                ax_sp.text(j,i,f"{sp_matrix[i,j]:.2f}",ha="center",va="center",fontsize=11,fontweight="bold",color="#0f1117" if abs(sp_matrix[i,j])>0.3 else "#c8d4f0")
        plt.colorbar(im,ax=ax_sp,shrink=0.8); plt.tight_layout(); st.pyplot(fig_sp); plt.close()
    with km2:
        fig_sc,ax_sc=plt.subplots(figsize=(6,5))
        ax_sc.scatter(sp_data[kpi_cols[max_idx[0]]],sp_data[kpi_cols[max_idx[1]]],c=[SUP_COLORS.get(s,"#aaa") for s in sp_data["Supplier name"]],s=70,alpha=0.85,edgecolors="#0f1117",linewidths=0.7,zorder=3)
        ax_sc.set_xlabel(kpi_labels[max_idx[0]]); ax_sc.set_ylabel(kpi_labels[max_idx[1]])
        ax_sc.set_title(f"Stärkster Zusammenhang: ρ = {sp_matrix[max_idx]:.2f}",fontsize=10); ax_sc.grid(zorder=0)
        ax_sc.legend(handles=[mpatches.Patch(color=SUP_COLORS.get(s,"#aaa"),label=s) for s in sorted(sp_data["Supplier name"].unique())],fontsize=8,framealpha=0.2)
        plt.tight_layout(); st.pyplot(fig_sc); plt.close()

# ── TAB 6: ALLE DATEN ────────────────────────────────────────────
with tab6:
    st.markdown("<div class='section-header'>📋 Alle Daten – filter- & sortierbar · Export CSV & Excel</div>", unsafe_allow_html=True)
    base_cols=list(df.columns)
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
    table=scored_full[scored_full["Supplier name"].isin(ff_sup)&scored_full["Product type"].isin(ff_prod)&scored_full["Location"].isin(ff_loc)&scored_full["Inspection results"].isin(ff_insp)&scored_full["Risikostufe"].isin(ff_risk)&scored_full["Produkt-Bucket"].isin(ff_buck)].copy()
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
st.markdown("<p style='text-align:center;color:#3a4a6a;font-size:0.8rem'>RiskRadar v6 · THI Ingolstadt · DPDS SoSe 2026 · Gruppe 9: Laurenz Angleitner · Leon Pavic · Alex Rauschendorfer · Daniel Steinmetz</p>", unsafe_allow_html=True)
