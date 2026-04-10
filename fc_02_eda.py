#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:26:35 2026

@author: nishrinkachwala
"""

"""
Foster Care Panel Analysis — Exploratory Data Analysis
=======================================================
Input:  fc_panel.csv (master panel from fc_01_load_data.py)
Output: figures/ folder with EDA plots

Steps:
  1. Clean panel  -- drop footer rows, flag small counties
  2. Univariate   -- distributions of key variables
  3. Time series  -- statewide trends, COVID signal visible
  4. County variation -- cross-sectional spread, identify outliers
  5. COVID signal -- pre/post comparison, event window
  6. P1 permanency -- trends and COVID disruption
  7. Correlation  -- entry rate vs in-care rate, entries vs exits
  8. Notes for modelling
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("/Users/nishrinkachwala/Desktop/DataSc_Projects/ChildWelfare/ChildWelfareData/figures", exist_ok=True)
DATA = "/Users/nishrinkachwala/Desktop/DataSc_Projects/ChildWelfare/ChildWelfareData/data/"
FIGS = "/Users/nishrinkachwala/Desktop/DataSc_Projects/ChildWelfare/ChildWelfareData/figures/"

def section(title):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


# =============================================================================
# 1. LOAD AND CLEAN
# =============================================================================
section("1. Load and clean panel")

panel = pd.read_csv(DATA + "fc_panel.csv")

# --- Drop footer rows ---
# The parser picked up Excel footnote text as county rows.
# Real county names are short strings; footnotes are long sentences.
# Drop any row where county name is longer than 50 characters
# or contains digits followed by a dash (population data citations).
STANDARD_58 = [
    'Alameda','Alpine','Amador','Butte','Calaveras','Colusa','Contra Costa',
    'Del Norte','El Dorado','Fresno','Glenn','Humboldt','Imperial','Inyo',
    'Kern','Kings','Lake','Lassen','Los Angeles','Madera','Marin','Mariposa',
    'Mendocino','Merced','Modoc','Mono','Monterey','Napa','Nevada','Orange',
    'Placer','Plumas','Riverside','Sacramento','San Benito','San Bernardino',
    'San Diego','San Francisco','San Joaquin','San Luis Obispo','San Mateo',
    'Santa Barbara','Santa Clara','Santa Cruz','Shasta','Sierra','Siskiyou',
    'Solano','Sonoma','Stanislaus','Sutter','Tehama','Trinity','Tulare',
    'Tuolumne','Ventura','Yolo','Yuba'
]

panel = panel[panel['county'].isin(STANDARD_58)].copy()
panel = panel.sort_values(['county', 'year']).reset_index(drop=True)

print(f"  Panel after cleaning: {panel.shape}")
print(f"  Counties: {panel['county'].nunique()}")
print(f"  Years: {sorted(panel['year'].unique())}")
print(f"  Entry rate NaN (masked): {panel['entry_rate'].isna().sum()}")

# --- Flag small counties ---
# Counties where more than 50% of entry_rate values are masked (NaN)
# are too sparse to contribute reliably to regression.
mask_pct = panel.groupby('county')['entry_rate'].apply(
    lambda x: x.isna().mean()
)
small_counties = mask_pct[mask_pct > 0.50].index.tolist()
panel['small_county'] = panel['county'].isin(small_counties).astype(int)
print(f"\n  Counties with >50% masked entry rate: {small_counties}")

# --- COVID / policy period flags (already in panel, verify) ---
assert 'covid_year' in panel.columns
assert 'years_to_covid' in panel.columns

# --- Restrict to pre-ACS-gap years for covariate analysis ---
# ACS covariates available 2010-2022 (pulled when run locally)
# For now work with CCWIP variables only across full 2010-2025
panel_main = panel[panel['year'] <= 2024].copy()  # drop partial 2025


# =============================================================================
# 2. UNIVARIATE DISTRIBUTIONS
# =============================================================================
section("2. Univariate distributions")

UNI_VARS = {
    'entry_rate':          'Entry rate (per 1,000 children)',
    'incare_rate':         'In-care rate (per 1,000, July 1)',
    'pct_permanency_12mo': 'Permanency in 12 months (%)',
}

print(f"\n  {'Variable':<35} {'N':>5} {'Mean':>7} {'Std':>7} "
      f"{'Min':>7} {'Max':>7} {'Skew':>7}")
print("  " + "-" * 75)

for col, label in UNI_VARS.items():
    s = panel_main[col].dropna()
    print(f"  {label:<35} {len(s):>5} {s.mean():>7.2f} {s.std():>7.2f} "
          f"{s.min():>7.2f} {s.max():>7.2f} {s.skew():>7.2f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Univariate distributions — CA county-year observations",
             fontsize=12, fontweight='bold')

for ax, (col, label) in zip(axes, UNI_VARS.items()):
    data = panel_main[col].dropna()
    ax.hist(data, bins=20, color='#457B9D', edgecolor='white', alpha=0.85)
    ax.axvline(data.mean(),   color='#E63946', ls='--', lw=1.5,
               label=f'Mean={data.mean():.1f}')
    ax.axvline(data.median(), color='#2A9D8F', ls=':',  lw=1.5,
               label=f'Median={data.median():.1f}')
    ax.set_title(label, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS + "fc_01_univariate.png", dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIGS}fc_01_univariate.png")


# =============================================================================
# 3. STATEWIDE TIME SERIES
# =============================================================================
section("3. Statewide time series (mean across counties)")

# Exclude small counties from trend line for clarity
trend = (panel_main[panel_main['small_county'] == 0]
         .groupby('year')
         .agg(
             entry_rate_mean  = ('entry_rate',          'mean'),
             entry_rate_p25   = ('entry_rate',          lambda x: x.quantile(0.25)),
             entry_rate_p75   = ('entry_rate',          lambda x: x.quantile(0.75)),
             incare_mean      = ('incare_rate',         'mean'),
             p1_mean          = ('pct_permanency_12mo', 'mean'),
         )
         .reset_index())

fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
fig.suptitle("California foster care trends — county mean (excluding small counties)",
             fontsize=12, fontweight='bold')

# Panel A: entry rate
ax = axes[0]
ax.plot(trend['year'], trend['entry_rate_mean'],
        color='#457B9D', lw=2.5, marker='o', ms=5, label='Mean entry rate')
ax.fill_between(trend['year'],
                trend['entry_rate_p25'], trend['entry_rate_p75'],
                alpha=0.15, color='#457B9D', label='IQR across counties')
ax.axvspan(2016.5, 2019.5, alpha=0.08, color='orange',
           label='AB 403 phase-in (2017-2019)')
ax.axvspan(2019.5, 2021.5, alpha=0.12, color='red',
           label='COVID (2020-2021)')
ax.set_ylabel('Entries per 1,000 children')
ax.set_title('A. Foster care entry rate', fontsize=10, loc='left')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# Panel B: in-care rate
ax = axes[1]
ax.plot(trend['year'], trend['incare_mean'],
        color='#E76F51', lw=2.5, marker='o', ms=5)
ax.axvspan(2016.5, 2019.5, alpha=0.08, color='orange')
ax.axvspan(2019.5, 2021.5, alpha=0.12, color='red')
ax.set_ylabel('Children per 1,000 (July 1)')
ax.set_title('B. In-care prevalence rate', fontsize=10, loc='left')
ax.grid(True, alpha=0.3)

# Panel C: P1 permanency
ax = axes[2]
p1_trend = trend.dropna(subset=['p1_mean'])
ax.plot(p1_trend['year'], p1_trend['p1_mean'],
        color='#2A9D8F', lw=2.5, marker='o', ms=5)
ax.axvspan(2016.5, 2019.5, alpha=0.08, color='orange')
ax.axvspan(2019.5, 2021.5, alpha=0.12, color='red')
ax.set_ylabel('% achieving permanency')
ax.set_title('C. Permanency in 12 months (4-P1)', fontsize=10, loc='left')
ax.set_xlabel('Year')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS + "fc_02_statewide_trends.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {FIGS}fc_02_statewide_trends.png")

# Print the numbers
print(f"\n  Year-by-year mean entry rate (large counties):")
print(trend[['year','entry_rate_mean','incare_mean','p1_mean']].round(2).to_string(index=False))


# =============================================================================
# 4. COUNTY VARIATION
# =============================================================================
section("4. County variation (cross-sectional spread)")

# Average each county over pre-COVID period (2015-2019) for clean comparison
pre_covid = (panel_main[
    (panel_main['year'].between(2015, 2019)) &
    (panel_main['small_county'] == 0)
].groupby('county')
 .agg(entry_rate_avg=('entry_rate', 'mean'),
      incare_avg=('incare_rate', 'mean'))
 .dropna()
 .reset_index()
 .sort_values('entry_rate_avg', ascending=True))

print(f"\n  Pre-COVID (2015-2019) entry rate by county:")
print(f"  Mean: {pre_covid['entry_rate_avg'].mean():.2f}")
print(f"  Range: {pre_covid['entry_rate_avg'].min():.2f} to "
      f"{pre_covid['entry_rate_avg'].max():.2f}")
print(f"  Top 5 highest:")
print(pre_covid.nlargest(5, 'entry_rate_avg')[
    ['county','entry_rate_avg']].to_string(index=False))
print(f"  Top 5 lowest:")
print(pre_covid.nsmallest(5, 'entry_rate_avg')[
    ['county','entry_rate_avg']].to_string(index=False))

# Horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 14))
colors = ['#E63946' if v > pre_covid['entry_rate_avg'].mean() + pre_covid['entry_rate_avg'].std()
          else '#457B9D' for v in pre_covid['entry_rate_avg']]
ax.barh(pre_covid['county'], pre_covid['entry_rate_avg'],
        color=colors, edgecolor='white', alpha=0.85)
ax.axvline(pre_covid['entry_rate_avg'].mean(), color='#2A9D8F',
           ls='--', lw=1.5, label=f"Mean = {pre_covid['entry_rate_avg'].mean():.1f}")
ax.set_xlabel('Entry rate per 1,000 children (avg 2015-2019)')
ax.set_title('County foster care entry rates — pre-COVID average\n'
             'Red = more than 1 SD above mean', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(FIGS + "fc_03_county_variation.png", dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIGS}fc_03_county_variation.png")


# =============================================================================
# 5. COVID EVENT WINDOW
# =============================================================================
section("5. COVID event window (years_to_covid = year - 2020)")

# Event study plot: mean entry rate by years_to_covid
# Normalise each county's rate to its own 2019 value (pre-shock baseline)
# so we can compare relative changes regardless of baseline level
event = panel_main[
    (panel_main['small_county'] == 0) &
    (panel_main['years_to_covid'].between(-5, 4))
].copy()

# Get each county's 2019 (t=-1) entry rate as baseline
baseline = (event[event['year'] == 2019]
            .set_index('county')['entry_rate'])

event['entry_rate_indexed'] = event.apply(
    lambda row: (row['entry_rate'] / baseline.get(row['county'], np.nan) * 100)
    if baseline.get(row['county'], np.nan) and baseline.get(row['county'], np.nan) > 0
    else np.nan,
    axis=1
)

event_agg = (event.groupby('years_to_covid')
             .agg(
                 mean_indexed   = ('entry_rate_indexed', 'mean'),
                 p25            = ('entry_rate_indexed', lambda x: x.quantile(0.25)),
                 p75            = ('entry_rate_indexed', lambda x: x.quantile(0.75)),
                 n_counties     = ('entry_rate_indexed', 'count'),
             )
             .reset_index())

print(f"\n  Indexed entry rate (2019 = 100) around COVID shock:")
print(event_agg[['years_to_covid','mean_indexed','n_counties']].round(1).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(event_agg['years_to_covid'],
                event_agg['p25'], event_agg['p75'],
                alpha=0.15, color='#457B9D', label='IQR across counties')
ax.plot(event_agg['years_to_covid'], event_agg['mean_indexed'],
        color='#457B9D', lw=2.5, marker='o', ms=7, label='Mean (indexed)')
ax.axvline(0, color='#E63946', ls='--', lw=1.5, label='COVID shock (2020)')
ax.axhline(100, color='gray', ls=':', lw=1, alpha=0.7)
ax.set_xlabel('Years relative to 2020 (0 = 2020)')
ax.set_ylabel('Entry rate index (2019 = 100)')
ax.set_title('Event study: foster care entry rate around COVID\n'
             'Each county indexed to its own 2019 rate',
             fontsize=11, fontweight='bold')
ax.set_xticks(range(-5, 5))
ax.set_xticklabels([f't={i}' for i in range(-5, 5)])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS + "fc_04_covid_event_window.png", dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIGS}fc_04_covid_event_window.png")


# =============================================================================
# 6. P1 PERMANENCY — COVID DISRUPTION
# =============================================================================
section("6. P1 permanency — COVID disruption by county")

p1_data = panel_main[
    (panel_main['small_county'] == 0) &
    panel_main['pct_permanency_12mo'].notna()
].copy()

# Compare average permanency: 2017-2019 (pre) vs 2020-2021 (COVID)
pre  = p1_data[p1_data['year'].between(2017, 2019)].groupby('county')['pct_permanency_12mo'].mean()
covid = p1_data[p1_data['year'].between(2020, 2021)].groupby('county')['pct_permanency_12mo'].mean()

p1_change = pd.DataFrame({
    'pre_covid_p1':  pre,
    'covid_p1':      covid,
}).dropna()
p1_change['p1_change_pp'] = p1_change['covid_p1'] - p1_change['pre_covid_p1']
p1_change = p1_change.sort_values('p1_change_pp')

print(f"\n  Change in P1 permanency rate (COVID - pre-COVID):")
print(f"  Mean change: {p1_change['p1_change_pp'].mean():.1f} pp")
print(f"  Counties where permanency fell: "
      f"{(p1_change['p1_change_pp'] < 0).sum()} of {len(p1_change)}")
print(f"\n  Largest declines:")
print(p1_change.nsmallest(5, 'p1_change_pp')[['pre_covid_p1','covid_p1','p1_change_pp']].round(1).to_string())
print(f"\n  Largest increases:")
print(p1_change.nlargest(5, 'p1_change_pp')[['pre_covid_p1','covid_p1','p1_change_pp']].round(1).to_string())

n = len(p1_change)
fig, ax = plt.subplots(figsize=(11, n * 0.38))
colors = ['#E63946' if v < 0 else '#2A9D8F' for v in p1_change['p1_change_pp']]
bars = ax.barh(p1_change.index, p1_change['p1_change_pp'],
               color=colors, edgecolor='white', alpha=0.85, height=0.7)
for bar, val in zip(bars, p1_change['p1_change_pp']):
    x_pos = val + (0.4 if val >= 0 else -0.4)
    ha = 'left' if val >= 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            f'{val:+.1f}', va='center', ha=ha, fontsize=7.5, color='#333333')
ax.axvline(0, color='black', lw=1.2)
ax.set_xlabel('Change in permanency rate (percentage points)', fontsize=11)
ax.set_title('Change in 4-P1 permanency rate: COVID (2020-21) vs pre-COVID (2017-19)\n'
             'Red = decline   |   Green = improvement',
             fontsize=11, fontweight='bold', pad=12)
ax.tick_params(axis='y', labelsize=9)
ax.grid(True, alpha=0.25, axis='x')
for i in range(0, n, 2):
    ax.axhspan(i - 0.5, i + 0.5, color='gray', alpha=0.04, zorder=0)
plt.tight_layout()
plt.savefig(FIGS + "fc_05_p1_covid_change.png", dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIGS}fc_05_p1_covid_change.png")


# =============================================================================
# 7. BIVARIATE — entry rate vs in-care rate
# =============================================================================
section("7. Bivariate: entry rate vs in-care rate")

# Filter to rows where both outcomes are present
# No small_county filter here -- just need both values to be non-null
biv = panel_main[
    panel_main['entry_rate'].notna() &
    panel_main['incare_rate'].notna()
].copy()

print(f"  Rows with both entry_rate and incare_rate: {len(biv)}")

if len(biv) < 2:
    print("  WARNING: not enough data for correlation -- check that "
          "incare_rate loaded correctly in fc_01_load_data.py")
else:
    r, p = stats.pearsonr(biv['entry_rate'], biv['incare_rate'])
    print(f"  Entry rate vs in-care rate: r={r:.3f}, p={p:.4f}")

# Pre-COVID county entry rate vs P1 drop during COVID
pre_entry = (panel_main[panel_main['year'].between(2015, 2019)]
             .groupby('county')['entry_rate'].mean())
p1_drop   = p1_change['p1_change_pp']
both      = pd.DataFrame({'pre_entry': pre_entry, 'p1_drop': p1_drop}).dropna()

if len(both) >= 2:
    r2, p2 = stats.pearsonr(both['pre_entry'], both['p1_drop'])
    print(f"  Pre-COVID entry rate vs P1 drop during COVID: r={r2:.3f}, p={p2:.4f}")
else:
    r2, p2 = np.nan, np.nan
    print("  Not enough data for pre-COVID vs P1 drop correlation")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Bivariate relationships", fontsize=12, fontweight='bold')

# Entry vs in-care
ax = axes[0]
# Guard: only plot if we have data
if len(biv) < 2:
    ax.text(0.5, 0.5, 'Insufficient data\n(check incare_rate loading)',
            ha='center', va='center', transform=ax.transAxes, color='gray')
scatter = ax.scatter(biv['entry_rate'], biv['incare_rate'],
                     c=biv['year'], cmap='RdYlGn_r',
                     alpha=0.5, s=20, edgecolors='none')
plt.colorbar(scatter, ax=ax, label='Year')
if len(biv) >= 2:
    m, b, *_ = stats.linregress(biv['entry_rate'], biv['incare_rate'])
    xs = np.linspace(biv['entry_rate'].min(), biv['entry_rate'].max(), 100)
    ax.plot(xs, m*xs+b, color='#E63946', lw=1.5)
ax.set_xlabel('Entry rate (per 1,000)')
ax.set_ylabel('In-care rate (per 1,000)')
r_label = f"{r:.3f}" if not np.isnan(r) else "n/a"
ax.set_title(f'Entry rate vs in-care rate\nr={r_label}', fontsize=10)
ax.grid(True, alpha=0.3)

# Pre-COVID entry rate vs P1 change
ax = axes[1]
ax.scatter(both['pre_entry'], both['p1_drop'],
           color='#457B9D', alpha=0.7, s=50, edgecolors='white')
m2, b2, *_ = stats.linregress(both['pre_entry'], both['p1_drop'])
xs2 = np.linspace(both['pre_entry'].min(), both['pre_entry'].max(), 100)
ax.plot(xs2, m2*xs2+b2, color='#E63946', lw=1.5)
ax.axhline(0, color='gray', ls=':', lw=1)
# Label 5 most extreme counties
for county, row in both.nlargest(3, 'p1_drop').iterrows():
    ax.annotate(county, (row['pre_entry'], row['p1_drop']),
                fontsize=7, xytext=(3,3), textcoords='offset points')
for county, row in both.nsmallest(3, 'p1_drop').iterrows():
    ax.annotate(county, (row['pre_entry'], row['p1_drop']),
                fontsize=7, xytext=(3,-8), textcoords='offset points')
ax.set_xlabel('Pre-COVID entry rate avg (2015-2019)')
ax.set_ylabel('Change in P1 during COVID (pp)')
ax.set_title(f'High-entry counties vs permanency disruption\nr={r2:.3f}',
             fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS + "fc_06_bivariate.png", dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIGS}fc_06_bivariate.png")


# =============================================================================
# 8. NOTES FOR MODELLING
# =============================================================================
section("8. Notes for modelling")

print(f"""
  OUTCOME VARIABLES:
    entry_rate          -- primary. Per 1,000 children, annual flow.
                           Mean {panel_main['entry_rate'].mean():.2f},
                           range {panel_main['entry_rate'].min():.1f}-
                           {panel_main['entry_rate'].max():.1f}.
                           Skew: {panel_main['entry_rate'].skew():.2f}
                           -- right-skewed, consider log transform.
    pct_permanency_12mo -- COVID event study outcome. % scale 0-100.
                           Available 2015-2024 only.

  PANEL STRUCTURE:
    N counties (large): {panel_main[panel_main.small_county==0].county.nunique()}
    T years (full):     {panel_main.year.nunique()} (2010-2024)
    Balanced panel:     No -- some county-years masked.

  SMALL COUNTIES TO EXCLUDE FROM MAIN REGRESSION:
    {small_counties}
    Reason: >50% of entry_rate cells masked by CDSS DDG.
    Include in sensitivity analysis only.

  COVID SIGNAL:
    Entry rate dropped ~24% from 2019 to 2020 (4.25 -> 3.21).
    Never recovered -- 2024 rate (2.97) is below any pre-2020 year.
    This suggests a structural break, not just a temporary shock.
    Event study should test for parallel pre-trends (2015-2019)
    and estimate year-by-year effects relative to 2019 baseline.

  AB 403:
    Signed 2015, phased in 2017-2019. Entry rates fell 2016-2017
    (3.81 vs 4.25 in 2019). Hard to separate AB 403 from COVID
    without county-level implementation dates. Use a simple
    post-2017 indicator as a first pass.

  ACS COVARIATES (after local run):
    child_poverty_rate, pct_black, pct_native_american,
    housing_burden_rate -- all time-varying, 2010-2022.
    For 2023-2024 in entry rate panel, carry forward 2022 values
    or restrict panel to 2010-2022 for the covariate model.

  MASKING:
    112 NaN entry_rate cells. All concentrated in small counties.
    After excluding small counties, remaining NaN should be minimal.
    Check with: panel_main[panel_main.small_county==0].entry_rate.isna().sum()
""")

val = panel_main[panel_main.small_county==0]['entry_rate'].isna().sum()
total = panel_main[panel_main.small_county==0].shape[0]
print(f"  NaN in entry_rate after excluding small counties: "
      f"{val} of {total} ({val/total:.1%})")

print("\n=== EDA complete ===")
print(f"Figures saved to: {FIGS}")