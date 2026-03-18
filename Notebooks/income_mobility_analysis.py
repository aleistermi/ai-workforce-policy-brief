"""
Income Mobility Analysis for Section 2 Policy Brief
====================================================
Generates interactive visualizations and a standalone HTML document
for the new "Income Mobility" section.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────
BASE = "/Users/aleistermontfort/Documentos_HD/future_impact_group/Future Impact Group"
DATA = os.path.join(BASE, "DATA")
VIZ_OUT = os.path.join(BASE, "Viz", "Section2")
os.makedirs(VIZ_OUT, exist_ok=True)

# ── Occupation group full names ────────────────────────────────────
OCC_NAMES = {
    "MGR": "Management",
    "BUS": "Business & Finance",
    "FIN": "Financial Specialists",
    "CMM": "Computer & Math",
    "ENG": "Engineering",
    "SCI": "Science",
    "CMS": "Community & Social Svc",
    "LGL": "Legal",
    "EDU": "Education & Library",
    "ENT": "Arts & Entertainment",
    "MED": "Healthcare Practitioners",
    "HLS": "Healthcare Support",
    "PRT": "Protective Service",
    "EAT": "Food Preparation",
    "CLN": "Cleaning & Grounds",
    "PRS": "Personal Care",
    "SAL": "Sales",
    "OFF": "Office & Admin Support",
    "FFF": "Farming & Forestry",
    "CON": "Construction",
    "EXT": "Extraction",
    "RPR": "Maintenance & Repair",
    "PRD": "Production",
    "TRN": "Transportation",
    "MIL": "Military",
}

GOOGLE_FONTS = '<link href="https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;600;700&display=swap" rel="stylesheet">'

def save_plotly(fig, path):
    fig.write_html(path, include_plotlyjs='cdn')
    with open(path, 'r') as f:
        html = f.read()
    html = html.replace('<head>', f'<head>\n{GOOGLE_FONTS}', 1)
    with open(path, 'w') as f:
        f.write(html)

# ── 1. Load enriched transitions ──────────────────────────────────
print("Loading enriched transitions...")
df = pd.read_csv(os.path.join(DATA, "Clean_data", "TOTAL_changes_enriched.csv"),
                 low_memory=False)
print(f"  Loaded {len(df):,} transitions")

# Map to full names
df['curr_occ_full'] = df['curr_occ_group'].map(OCC_NAMES).fillna(df['curr_occ_group'])
df['prev_occ_full'] = df['prev_occ_group'].map(OCC_NAMES).fillna(df['prev_occ_group'])

# ── 2. Build income lookup from raw data ──────────────────────────
print("Building income lookup from data_with_income.gz...")
colspecs = [
    (0, 4), (4, 9), (9, 11), (11, 21), (21, 35), (35, 36), (36, 47),
    (47, 49), (49, 63), (63, 77), (77, 92), (92, 103), (103, 108),
    (108, 112), (112, 115), (115, 125), (125, 134), (134, 142)
]
column_names = [
    "YEAR", "SERIAL", "MONTH", "HWTFINL", "CPSID", "ASECFLAG", "ASECWTH",
    "PERNUM", "WTFINL", "CPSIDP", "CPSIDV", "ASECWT", "HOURWAGE2", "OCC",
    "IND1990", "FTOTVAL", "INCTOT", "INCWAGE"
]

df_income = pd.read_fwf(
    os.path.join(DATA, "data_with_income.gz"),
    colspecs=colspecs, names=column_names, compression='gzip'
)

valid = df_income[
    (df_income['MONTH'] == 3) & (df_income['INCWAGE'] > 0) &
    (df_income['ASECWT'] > 0) & (df_income['OCC'] > 0)
].copy()

weighted_inc = (
    valid.groupby(['YEAR', 'OCC'])
    .apply(lambda g: (g['INCWAGE'] * g['ASECWT']).sum() / g['ASECWT'].sum())
    .reset_index(name='avg_weighted_incwage')
)
income_lookup = weighted_inc.set_index(['YEAR', 'OCC'])['avg_weighted_incwage'].to_dict()
print(f"  Income lookup: {len(income_lookup):,} year-occupation pairs")

# ── 3. Merge income (with fallback to nearest available year) ─────
max_asec_year = int(weighted_inc['YEAR'].max())
print(f"  Max ASEC year: {max_asec_year}")

def get_income(year, occ):
    """Look up income, falling back to the closest prior year if exact match missing."""
    val = income_lookup.get((year, occ), np.nan)
    if pd.notna(val):
        return val
    # Fallback: try previous years (up to 2 years back)
    for fallback in range(1, 3):
        val = income_lookup.get((year - fallback, occ), np.nan)
        if pd.notna(val):
            return val
    return np.nan

df['previous_wage'] = df.apply(
    lambda r: get_income(r['Year'] - 1, r['Previous_OCC']), axis=1
)
df['current_wage'] = df.apply(
    lambda r: get_income(r['Year'], r['Current_OCC']), axis=1
)
df['wage_change_pct'] = ((df['current_wage'] - df['previous_wage']) / df['previous_wage']) * 100

def classify_mobility(pct):
    if pd.isna(pct): return np.nan
    if pct > 10: return 'Upward'
    elif pct < -10: return 'Downward'
    else: return 'Lateral'

df['income_mobility'] = df['wage_change_pct'].apply(classify_mobility)

# Now include 2025 (uses 2024 ASEC income as fallback for destinations)
# Exclude 2026 which has very few observations
df_valid = df.dropna(subset=['wage_change_pct', 'income_mobility']).copy()
df_valid = df_valid[df_valid['Year'] <= 2025]
df_escape = df_valid[df_valid['prev_ai_aggressive'] > 0.45].copy()
print(f"  Valid income transitions: {len(df_valid):,}")
print(f"  High-AI escape transitions: {len(df_escape):,}")

# ── Colors & style (aligned with policy brief palette) ────────────
colors = {'Upward': '#2b6cb0', 'Lateral': '#cbd5e0', 'Downward': '#e53e3e'}
font_family = 'Raleway, sans-serif'
layout_base = dict(
    template='plotly_white',
    font=dict(family=font_family, size=11, color='#2d3748'),
    title_font=dict(color='#1a365d'),
    paper_bgcolor='#fff',
    plot_bgcolor='#fff',
)

# ══════════════════════════════════════════════════════════════════
# VIZ 1: Mobility rates by destination (ALL transitions)
# ══════════════════════════════════════════════════════════════════
print("\n--- Viz 1: Mobility by destination (all) ---")

mob_dest = (
    df_valid.groupby(['curr_occ_full', 'income_mobility'])
    .agg(n=('Weight', 'sum')).reset_index()
)
totals = df_valid.groupby('curr_occ_full')['Weight'].sum().reset_index(name='total')
mob_dest = mob_dest.merge(totals, on='curr_occ_full')
mob_dest['pct'] = mob_dest['n'] / mob_dest['total'] * 100

up_rates = mob_dest[mob_dest['income_mobility'] == 'Upward'].set_index('curr_occ_full')['pct']
sort_order = up_rates.sort_values(ascending=True).index.tolist()

fig1 = go.Figure()
for mob in ['Downward', 'Lateral', 'Upward']:
    s = mob_dest[mob_dest['income_mobility'] == mob].set_index('curr_occ_full').reindex(sort_order)
    fig1.add_trace(go.Bar(
        y=s.index, x=s['pct'], name=mob, orientation='h',
        marker_color=colors[mob],
        hovertemplate='%{y}<br>' + mob + ': %{x:.1f}%<extra></extra>'
    ))
fig1.update_layout(
    barmode='stack',
    title='Income Mobility by Destination Occupation Group<br><span style="font-size:12px;color:#666">All occupational transitions, weighted | Upward: >+10% | Lateral: ±10% | Downward: <−10%</span>',
    xaxis_title='Share of Transitions (%)', yaxis_title='',
    height=750, margin=dict(l=220),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    **layout_base
)
save_plotly(fig1, os.path.join(VIZ_OUT, "income_mobility_by_destination.html"))
print("  Saved.")

# ══════════════════════════════════════════════════════════════════
# VIZ 2: Mobility rates by destination (HIGH-AI ESCAPE)
# ══════════════════════════════════════════════════════════════════
print("--- Viz 2: Mobility by destination (high-AI escape) ---")

mob_esc = (
    df_escape.groupby(['curr_occ_full', 'income_mobility'])
    .agg(n=('Weight', 'sum')).reset_index()
)
esc_totals = df_escape.groupby('curr_occ_full')['Weight'].sum().reset_index(name='total')
mob_esc = mob_esc.merge(esc_totals, on='curr_occ_full')
mob_esc['pct'] = mob_esc['n'] / mob_esc['total'] * 100

up_esc = mob_esc[mob_esc['income_mobility'] == 'Upward'].set_index('curr_occ_full')['pct']
sort_esc = up_esc.sort_values(ascending=True).index.tolist()

fig2 = go.Figure()
for mob in ['Downward', 'Lateral', 'Upward']:
    s = mob_esc[mob_esc['income_mobility'] == mob].set_index('curr_occ_full').reindex(sort_esc)
    fig2.add_trace(go.Bar(
        y=s.index, x=s['pct'], name=mob, orientation='h',
        marker_color=colors[mob],
        hovertemplate='%{y}<br>' + mob + ': %{x:.1f}%<extra></extra>'
    ))
fig2.update_layout(
    barmode='stack',
    title='Income Mobility for Workers Escaping High-AI Occupations<br><span style="font-size:12px;color:#666">Origin AI exposure > 45% (aggressive) | Weighted transitions</span>',
    xaxis_title='Share of Transitions (%)', yaxis_title='',
    height=750, margin=dict(l=220),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    **layout_base
)
save_plotly(fig2, os.path.join(VIZ_OUT, "income_mobility_escape_highAI.html"))
print("  Saved.")

# ══════════════════════════════════════════════════════════════════
# VIZ 3: Demographics
# ══════════════════════════════════════════════════════════════════
print("--- Viz 3: Demographics ---")

demo_dims = {
    'SEX_LABEL': 'Gender',
    'AGE_BIN': 'Age Group',
    'EDUC_CAT': 'Education Level',
    'RACE_ETHN': 'Race / Ethnicity'
}

# Custom sort orders
EDUC_ORDER = ['Graduate', "Bachelor's", 'Some College', 'High School', 'Less than HS']

# Compute race upward mobility rates for sorting
race_up = df_escape[df_escape['income_mobility'] == 'Upward'].groupby('RACE_ETHN')['Weight'].sum()
race_tot = df_escape.groupby('RACE_ETHN')['Weight'].sum()
race_up_pct = (race_up / race_tot * 100).sort_values(ascending=False)
RACE_ORDER = race_up_pct.index.tolist()

CUSTOM_ORDERS = {
    'EDUC_CAT': EDUC_ORDER,
    'RACE_ETHN': RACE_ORDER,
}

fig3 = make_subplots(rows=2, cols=2, subplot_titles=list(demo_dims.values()),
                     horizontal_spacing=0.15, vertical_spacing=0.12)

for idx, (col, label) in enumerate(demo_dims.items()):
    row, col_idx = idx // 2 + 1, idx % 2 + 1
    dm = df_escape.groupby([col, 'income_mobility']).agg(n=('Weight', 'sum')).reset_index()
    dt = df_escape.groupby(col)['Weight'].sum().reset_index(name='total')
    dm = dm.merge(dt, on=col)
    dm['pct'] = dm['n'] / dm['total'] * 100
    dm = dm[dm[col].notna() & (dm['total'] > 1000)]

    # Apply custom sort order if defined, otherwise sort alphabetically
    if col in CUSTOM_ORDERS:
        order = CUSTOM_ORDERS[col]
        dm[col] = pd.Categorical(dm[col], categories=order, ordered=True)
        dm = dm.sort_values(col)
    else:
        dm = dm.sort_values(col)

    for mob in ['Downward', 'Lateral', 'Upward']:
        s = dm[dm['income_mobility'] == mob]
        fig3.add_trace(go.Bar(
            x=s[col], y=s['pct'], name=mob,
            marker_color=colors[mob],
            showlegend=(idx == 0), legendgroup=mob,
            hovertemplate='%{x}<br>' + mob + ': %{y:.1f}%<extra></extra>'
        ), row=row, col=col_idx)

fig3.update_layout(
    barmode='stack',
    title='Income Mobility by Demographics, Workers Escaping High-AI Occupations<br><span style="font-size:12px;color:#666">Origin AI exposure > 45% (aggressive) | Weighted transitions</span>',
    height=850,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
    **layout_base
)
save_plotly(fig3, os.path.join(VIZ_OUT, "income_mobility_demographics.html"))
print("  Saved.")

# ══════════════════════════════════════════════════════════════════
# VIZ 4: Wage change distributions (box plots)
# ══════════════════════════════════════════════════════════════════
print("--- Viz 4: Wage change distributions ---")

df_viz = df_escape[df_escape['wage_change_pct'].between(-100, 300)].copy()
medians = df_viz.groupby('curr_occ_full')['wage_change_pct'].median().sort_values()

fig4 = go.Figure()
for grp in medians.index:
    s = df_viz[df_viz['curr_occ_full'] == grp]
    fig4.add_trace(go.Box(
        y=s['wage_change_pct'], name=grp,
        marker_color='#2b6cb0' if medians[grp] >= 0 else '#e53e3e',
        boxmean=True,
        hovertemplate='%{y:.0f}%<extra>' + grp + '</extra>'
    ))

fig4.add_hline(y=0, line_dash="dash", line_color="#718096", line_width=1)
fig4.add_hline(y=10, line_dash="dot", line_color="#2b6cb0", line_width=0.5,
               annotation_text="+10%", annotation_position="right")
fig4.add_hline(y=-10, line_dash="dot", line_color="#e53e3e", line_width=0.5,
               annotation_text="-10%", annotation_position="right")

fig4.update_layout(
    title='Wage Change Distribution by Destination, Workers Escaping High-AI Occupations<br><span style="font-size:12px;color:#666">Box: IQR | Diamond: mean | Whiskers: 1.5×IQR | Capped at [−100%, +300%]</span>',
    yaxis_title='Wage Change (%)', xaxis_title='',
    height=600, showlegend=False,
    xaxis_tickangle=-45, margin=dict(b=180),
    **layout_base
)
save_plotly(fig4, os.path.join(VIZ_OUT, "income_mobility_distributions.html"))
print("  Saved.")

# ══════════════════════════════════════════════════════════════════
# VIZ 5: Temporal evolution
# ══════════════════════════════════════════════════════════════════
print("--- Viz 5: Temporal ---")

tm = df_escape.groupby(['Year', 'income_mobility']).agg(n=('Weight', 'sum')).reset_index()
tt = df_escape.groupby('Year')['Weight'].sum().reset_index(name='total')
tm = tm.merge(tt, on='Year')
tm['pct'] = tm['n'] / tm['total'] * 100

fig5 = go.Figure()
for mob in ['Downward', 'Lateral', 'Upward']:
    s = tm[tm['income_mobility'] == mob].sort_values('Year')
    fig5.add_trace(go.Scatter(
        x=s['Year'], y=s['pct'], name=mob,
        mode='lines+markers',
        line=dict(color=colors[mob], width=2.5),
        marker=dict(size=8),
        hovertemplate='Year %{x}<br>' + mob + ': %{y:.1f}%<extra></extra>'
    ))

fig5.update_layout(
    title='Income Mobility Over Time, Workers Escaping High-AI Occupations<br><span style="font-size:12px;color:#666">Share of weighted transitions by mobility category, per year</span>',
    xaxis_title='Year', yaxis_title='Share of Transitions (%)',
    height=450,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    **layout_base
)
save_plotly(fig5, os.path.join(VIZ_OUT, "income_mobility_temporal.html"))
print("  Saved.")


# ══════════════════════════════════════════════════════════════════
# SUMMARY STATS
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

overall = df_escape.groupby('income_mobility')['Weight'].sum()
overall_pct = overall / overall.sum() * 100
print(f"\nOverall mobility (high-AI escape, n={len(df_escape):,}):")
for cat in ['Upward', 'Lateral', 'Downward']:
    if cat in overall_pct:
        print(f"  {cat}: {overall_pct[cat]:.1f}%")

print(f"\nMedian wage change by destination:")
med = df_escape.groupby('curr_occ_full').agg(
    median_pct=('wage_change_pct', 'median'),
    n_weighted=('Weight', 'sum'),
    median_prev=('previous_wage', 'median'),
    median_curr=('current_wage', 'median')
).sort_values('median_pct', ascending=False)
print(med.to_string())

print(f"\nUpward mobility by gender:")
for sex in df_escape['SEX_LABEL'].dropna().unique():
    sub = df_escape[df_escape['SEX_LABEL'] == sex]
    up = sub[sub['income_mobility'] == 'Upward']['Weight'].sum()
    print(f"  {sex}: {up/sub['Weight'].sum()*100:.1f}%")

print(f"\nUpward mobility by education:")
for ed in sorted(df_escape['EDUC_CAT'].dropna().unique()):
    sub = df_escape[df_escape['EDUC_CAT'] == ed]
    up = sub[sub['income_mobility'] == 'Upward']['Weight'].sum()
    print(f"  {ed}: {up/sub['Weight'].sum()*100:.1f}%")

print(f"\nUpward mobility by age:")
for age in sorted(df_escape['AGE_BIN'].dropna().unique()):
    sub = df_escape[df_escape['AGE_BIN'] == age]
    up = sub[sub['income_mobility'] == 'Upward']['Weight'].sum()
    print(f"  {age}: {up/sub['Weight'].sum()*100:.1f}%")

print(f"\nUpward mobility by race/ethnicity:")
for race in sorted(df_escape['RACE_ETHN'].dropna().unique()):
    sub = df_escape[df_escape['RACE_ETHN'] == race]
    up = sub[sub['income_mobility'] == 'Upward']['Weight'].sum()
    print(f"  {race}: {up/sub['Weight'].sum()*100:.1f}%")

print("\nDone!")
