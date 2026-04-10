#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:38:56 2026

@author: nishrinkachwala
"""

"""
Foster Care Panel Analysis — Pre-Modelling Checks
==================================================
Five checks that inform modelling decisions before
running the panel regression and event study.

  1. Pre-trend parallel check  -- visual test for TWFE validity
  2. Outcome skew & log transform -- decide whether to log entry_rate
  3. AB 403 signal check       -- is the 2016-17 drop policy or trend?
  4. Masked cell pattern       -- understand missingness structure
  5. Entry rate vs permanency  -- are the two outcomes measuring the same thing?

Input:  fc_panel.csv, fc_entry_rates.csv
Output: figures/fc_07 through fc_11
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("figures", exist_ok=True)
DATA = "/Users/nishrinkachwala/Desktop/DataSc_Projects/ChildWelfare/ChildWelfareData/data/"
FIGS = "/Users/nishrinkachwala/Desktop/DataSc_Projects/ChildWelfare/ChildWelfareData/figures/"

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


def section(title):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


# --- Load ---
panel = pd.read_csv(DATA + "fc_panel.csv")
panel = panel[panel['county'].isin(STANDARD_58)].copy()
panel_main = panel[panel['year'] <= 2024].copy()

# Rebuild small_county flag (not stored in panel CSV)
mask_pct = panel_main.groupby('county')['entry_rate'].apply(
    lambda x: x.isna().mean()
)
small_counties = mask_pct[mask_pct > 0.50].index.tolist()
panel_main['small_county'] = panel_main['county'].isin(small_counties).astype(int)

large = panel_main[panel_main['small_county'] == 0].copy()


# =============================================================================
# 1. PRE-TREND PARALLEL CHECK
#
# The validity of TWFE (two-way fixed effects) and the event study rests on
# the parallel trends assumption: in the absence of the shock (COVID in 2020),
# all counties would have continued on similar trajectories.
# We cannot test this directly -- we can only inspect the pre-period visually.
#
# Approach: divide counties into terciles by their pre-COVID entry rate
# (2015-2019 average). Plot the mean trajectory for each group from 2010-2024.
# If the lines were roughly parallel before 2020, the assumption is plausible.
# If the High group was already declining faster than Low, TWFE estimates
# of COVID effects will be biased.
# =============================================================================
section("1. Pre-trend parallel check")

# Assign terciles based on 2015-2019 average
pre_avg = (large[large['year'].between(2015, 2019)]
           .groupby('county')['entry_rate'].mean()
           .dropna())
labels  = ['Low (bottom third)', 'Mid (middle third)', 'High (top third)']
terciles = pd.qcut(pre_avg, 3, labels=labels)
large    = large.merge(
    terciles.rename('entry_tercile').reset_index(),
    on='county', how='left'
)

trend_by_tercile = (large.groupby(['year', 'entry_tercile'])
                    ['entry_rate'].mean()
                    .reset_index())

print(f"  County tercile groups:")
for label in labels:
    counties = terciles[terciles == label].index.tolist()
    print(f"  {label}: {', '.join(counties[:5])}{'...' if len(counties)>5 else ''}")
    avg = pre_avg[terciles == label].mean()
    print(f"    Pre-COVID avg rate: {avg:.2f}")

# Test for parallel pre-trends numerically:
# Regress entry_rate on year*tercile interaction in 2010-2019
# If interaction is non-significant, pre-trends are parallel
pre_data = large[large['year'].between(2010, 2019)].copy()
pre_data['year_c'] = pre_data['year'] - 2019   # centre on last pre year
pre_data['high']   = (pre_data['entry_tercile'] == labels[2]).astype(float)
pre_data['low']    = (pre_data['entry_tercile'] == labels[0]).astype(float)
pre_data['year_x_high'] = pre_data['year_c'] * pre_data['high']
pre_data['year_x_low']  = pre_data['year_c'] * pre_data['low']

valid = pre_data[['entry_rate','year_c','high','low',
                  'year_x_high','year_x_low']].dropna()
X = np.column_stack([
    np.ones(len(valid)),
    valid['year_c'], valid['high'], valid['low'],
    valid['year_x_high'], valid['year_x_low']
])
y = valid['entry_rate'].values
coef = np.linalg.lstsq(X, y, rcond=None)[0]
resid = y - X @ coef
se    = np.sqrt(np.diag(
    np.linalg.inv(X.T @ X) * (resid**2).mean()
))
t_high = coef[4] / se[4] if se[4] > 0 else np.nan
t_low  = coef[5] / se[5] if se[5] > 0 else np.nan
p_high = 2 * stats.t.sf(abs(t_high), df=len(y)-6)
p_low  = 2 * stats.t.sf(abs(t_low),  df=len(y)-6)

print(f"\n  Pre-trend test (2010-2019):")
print(f"  Year x High tercile: coef={coef[4]:+.3f}  p={p_high:.3f}"
      f"  {'-- NON-PARALLEL' if p_high < 0.05 else '-- parallel OK'}")
print(f"  Year x Low tercile:  coef={coef[5]:+.3f}  p={p_low:.3f}"
      f"  {'-- NON-PARALLEL' if p_low < 0.05 else '-- parallel OK'}")

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
colors = {'Low (bottom third)': '#2A9D8F',
          'Mid (middle third)': '#E9C46A',
          'High (top third)':   '#E63946'}
lws    = {'Low (bottom third)': 2,
          'Mid (middle third)': 2,
          'High (top third)':   2.5}

for label in labels:
    grp = trend_by_tercile[trend_by_tercile['entry_tercile'] == label]
    ax.plot(grp['year'], grp['entry_rate'],
            color=colors[label], lw=lws[label],
            marker='o', ms=5, label=label)

ax.axvspan(2019.5, 2021.5, alpha=0.12, color='red', label='COVID (2020-21)')
ax.axvline(2019.5, color='red', ls='--', lw=1, alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Mean entry rate (per 1,000 children)')
ax.set_title(
    'Pre-trend parallel check: entry rate by county tercile 2010-2024\n'
    'If lines were parallel before 2020, TWFE assumption is plausible',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS + "fc_07_parallel_trends.png", dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIGS}fc_07_parallel_trends.png")


# =============================================================================
# 2. OUTCOME SKEW AND LOG TRANSFORM DECISION
#
# Entry rate skew ranged from 0.43 to 3.20 across years (from EDA).
# Inconsistent skew suggests the problem is not a stable distributional shape
# but rather a few counties with intermittently extreme values.
#
# We test three options:
#   (a) Raw entry rate
#   (b) log(entry_rate + 1)  -- adds 1 to handle zeros
#   (c) sqrt(entry_rate)     -- lighter transform, preserves more signal
#
# Decision criterion: which transformation produces the most consistent
# skew across years and the most normal-looking residual distribution?
# For TWFE the outcome transformation also affects interpretation:
#   raw   -> coefficient = pp change per unit X
#   log   -> coefficient = % change per unit X (semi-log)
#   sqrt  -> harder to interpret, usually not preferred
# =============================================================================
section("2. Outcome skew and log transform decision")

large['log_entry_rate']  = np.log(large['entry_rate'] + 0.1)
large['sqrt_entry_rate'] = np.sqrt(large['entry_rate'])

print(f"\n  Skewness across all county-year observations:")
for col, label in [('entry_rate',      'Raw'),
                   ('log_entry_rate',  'log(x + 0.1)'),
                   ('sqrt_entry_rate', 'sqrt(x)')]:
    s = large[col].dropna()
    print(f"  {label:<20} skew={s.skew():.2f}  mean={s.mean():.2f}  "
          f"std={s.std():.2f}")

print(f"\n  Skewness by year (raw vs log):")
print(f"  {'Year':>6}  {'Raw skew':>10}  {'Log skew':>10}  {'Max raw':>10}")
for yr, grp in large.groupby('year'):
    raw_s = grp['entry_rate'].dropna().skew()
    log_s = grp['log_entry_rate'].dropna().skew()
    mx    = grp['entry_rate'].max()
    print(f"  {yr:>6}  {raw_s:>10.2f}  {log_s:>10.2f}  {mx:>10.1f}")

# Visual comparison
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Entry rate transformation comparison",
             fontsize=12, fontweight='bold')

for ax, (col, label) in zip(axes, [
    ('entry_rate',      'Raw entry rate'),
    ('log_entry_rate',  'log(entry rate + 0.1)'),
    ('sqrt_entry_rate', 'sqrt(entry rate)'),
]):
    data = large[col].dropna()
    ax.hist(data, bins=25, color='#457B9D', edgecolor='white', alpha=0.85)
    ax.axvline(data.mean(), color='#E63946', ls='--', lw=1.5,
               label=f'Mean={data.mean():.2f}')
    ax.set_title(f"{label}\nskew={data.skew():.2f}", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS + "fc_08_log_transform_decision.png", dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIGS}fc_08_log_transform_decision.png")


# =============================================================================
# 3. AB 403 SIGNAL CHECK
#
# AB 403 (Continuum of Care Reform) was signed October 2015 and phased in
# county by county from 2017-2019. It eliminated group homes for children
# under 12 and restricted them for older youth, requiring counties to shift
# toward family-based placements. The hypothesis is that this reform reduced
# foster care entries by diverting children to less formal placements.
#
# Evidence: statewide entry rate was 4.50 in 2015 and dropped to 3.81 in 2016,
# before COVID. We test whether this drop was:
#   (a) Concentrated in high-volume counties that used more group homes
#       (and therefore had more children to divert)
#   (b) Uniform across all counties (which would suggest a different cause,
#       e.g. statewide policy like mandatory reporter training)
#
# We compute the 2015-to-2018 change for each county and check whether it
# correlates with the county's pre-reform (2013-2015) entry rate level.
# Higher-rate counties using more placements should show larger drops.
# =============================================================================
section("3. AB 403 signal check")

pre_reform  = (large[large['year'].between(2013, 2015)]
               .groupby('county')['entry_rate'].mean())
post_reform = (large[large['year'].between(2017, 2018)]
               .groupby('county')['entry_rate'].mean())

ab403 = pd.DataFrame({
    'pre_reform':  pre_reform,
    'post_reform': post_reform,
}).dropna()
ab403['change_pp']  = ab403['post_reform'] - ab403['pre_reform']
ab403['change_pct'] = (ab403['change_pp'] / ab403['pre_reform'] * 100).round(1)

r403, p403 = stats.pearsonr(ab403['pre_reform'], ab403['change_pp'])

print(f"\n  Pre-reform entry rate vs change 2013-15 to 2017-18:")
print(f"  r={r403:.3f}  p={p403:.4f}")
print(f"  Mean change: {ab403['change_pp'].mean():.2f} pp "
      f"({ab403['change_pct'].mean():.1f}%)")
print(f"  Counties with decrease: {(ab403['change_pp'] < 0).sum()} of {len(ab403)}")
print(f"\n  Largest decreases:")
print(ab403.nsmallest(5,'change_pp')[['pre_reform','post_reform','change_pp']].round(2).to_string())
print(f"\n  Largest increases:")
print(ab403.nlargest(5,'change_pp')[['pre_reform','post_reform','change_pp']].round(2).to_string())

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(ab403['pre_reform'], ab403['change_pp'],
           color='#457B9D', alpha=0.7, s=50, edgecolors='white')
m, b, *_ = stats.linregress(ab403['pre_reform'], ab403['change_pp'])
xs = np.linspace(ab403['pre_reform'].min(), ab403['pre_reform'].max(), 100)
ax.plot(xs, m*xs+b, color='#E63946', lw=1.5,
        label=f'r={r403:.2f}, p={p403:.3f}')
ax.axhline(0, color='gray', ls=':', lw=1)

# Label top outliers
for county, row in ab403.nsmallest(4, 'change_pp').iterrows():
    ax.annotate(county, (row['pre_reform'], row['change_pp']),
                fontsize=7, xytext=(3, -10), textcoords='offset points')
for county, row in ab403.nlargest(3, 'change_pp').iterrows():
    ax.annotate(county, (row['pre_reform'], row['change_pp']),
                fontsize=7, xytext=(3, 4), textcoords='offset points')

ax.set_xlabel('Pre-reform entry rate avg (2013-2015)', fontsize=10)
ax.set_ylabel('Change in entry rate (pp)', fontsize=10)
ax.set_title(
    'AB 403 signal: did high-entry counties drop more after reform?\n'
    'Pre-reform rate (2013-15) vs change to post-reform (2017-18)',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS + "fc_09_ab403_signal.png", dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIGS}fc_09_ab403_signal.png")


# =============================================================================
# 4. MASKED CELL PATTERN
#
# CCWIP masks cells with values 1-10 per CDSS de-identification guidelines.
# In the entry RATE file (our primary outcome) there are no masked cells
# because the rate is computed from the unduplicated child count and
# population -- if the underlying count is suppressed, the rate is too.
#
# But in the raw counts file (Entries) there were masked cells. We check
# which counties and years they affect to understand whether our panel
# has a missingness problem for the covariate model (when ACS is added).
#
# We also check the permanency (P1) file which has fewer years of coverage.
# =============================================================================
section("4. Masked cell pattern")

# Entry rate -- check by county
er = pd.read_csv(DATA + "fc_entry_rates.csv")
er = er[er['county'].isin(STANDARD_58)]
print(f"\n  Entry rate (primary outcome):")
print(f"  Total obs: {len(er)}")
print(f"  Missing (masked): {er['entry_rate'].isna().sum()}")
print(f"  Complete county-year obs: {er['entry_rate'].notna().sum()}")

# P1 permanency -- check coverage
p1 = pd.read_csv(DATA + "fc_p1_permanency.csv")
p1 = p1[p1['county'].isin(STANDARD_58)]
print(f"\n  P1 permanency:")
print(f"  Years available: {sorted(p1['year'].unique())}")
print(f"  Total obs: {len(p1)}")
print(f"  Missing: {p1['pct_permanency_12mo'].isna().sum()}")

# Counties with any missing P1
missing_p1 = (p1.groupby('county')['pct_permanency_12mo']
              .apply(lambda x: x.isna().sum()))
counties_missing_p1 = missing_p1[missing_p1 > 0].sort_values(ascending=False)
print(f"\n  Counties with missing P1 values:")
print(counties_missing_p1.to_string())

# Pivot P1 to wide format: counties x years, values = permanency rate
p1_wide = p1.pivot_table(
    index='county', columns='year',
    values='pct_permanency_12mo', aggfunc='mean'
)
# Sort by pre-COVID average for meaningful ordering
pre_avg = p1_wide[[c for c in p1_wide.columns if c <= 2019]].mean(axis=1)
p1_wide = p1_wide.loc[pre_avg.sort_values().index]

fig, ax = plt.subplots(figsize=(13, 14))
im = ax.imshow(p1_wide.values, aspect='auto',
               cmap='RdYlGn', vmin=15, vmax=65)

for i in range(len(p1_wide.index)):
    for j in range(len(p1_wide.columns)):
        val = p1_wide.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=6.5,
                    color='black' if 25 < val < 55 else 'white')
        else:
            ax.text(j, i, '—', ha='center', va='center',
                    fontsize=8, color='#999999')

years = list(p1_wide.columns)
for covid_yr in [2020, 2021]:
    if covid_yr in years:
        j = years.index(covid_yr)
        rect = plt.Rectangle((j-0.5, -0.5), 1, len(p1_wide),
                               linewidth=2, edgecolor='#E63946',
                               facecolor='none', zorder=3)
        ax.add_patch(rect)

ax.set_xticks(range(len(p1_wide.columns)))
ax.set_xticklabels(p1_wide.columns, fontsize=9)
ax.set_yticks(range(len(p1_wide.index)))
ax.set_yticklabels(p1_wide.index, fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.35,
             label='% achieving permanency in 12 months')
ax.set_title(
    '4-P1 Permanency in 12 months by county and year (%)\n'
    'Sorted by pre-COVID average  |  Red box = COVID years  |  — = suppressed',
    fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(FIGS + 'fc_10_p1_heatmap.png', dpi=150, bbox_inches='tight')
print(f'  Saved: {FIGS}fc_10_p1_heatmap.png')


# =============================================================================
# 5. ENTRY RATE vs PERMANENCY RATE
#
# Do counties with higher foster care entry rates tend to have lower
# permanency rates? If yes, this suggests system strain -- overwhelmed
# counties struggle to achieve permanency. If no relationship, the two
# outcomes are measuring different dimensions and should be modelled
# separately rather than as a single system strain index.
#
# We also check whether this relationship changed during COVID:
# did counties with high entry rates see disproportionate permanency drops,
# or was the COVID permanency disruption uniform across counties regardless
# of entry volume?
# =============================================================================
section("5. Entry rate vs permanency rate")

# Pre-COVID relationship
pre_both = large[
    large['year'].between(2016, 2019) &
    large['entry_rate'].notna() &
    large['pct_permanency_12mo'].notna()
].copy()

r_pre, p_pre = stats.pearsonr(
    pre_both['entry_rate'], pre_both['pct_permanency_12mo']
)
print(f"\n  Pre-COVID (2016-2019):")
print(f"  Entry rate vs P1: r={r_pre:.3f}, p={p_pre:.4f}, n={len(pre_both)}")

# COVID relationship
cov_both = large[
    large['year'].between(2020, 2021) &
    large['entry_rate'].notna() &
    large['pct_permanency_12mo'].notna()
].copy()
if len(cov_both) >= 2:
    r_cov, p_cov = stats.pearsonr(
        cov_both['entry_rate'], cov_both['pct_permanency_12mo']
    )
    print(f"\n  COVID (2020-2021):")
    print(f"  Entry rate vs P1: r={r_cov:.3f}, p={p_cov:.4f}, n={len(cov_both)}")
else:
    r_cov, p_cov = np.nan, np.nan

# County-level: pre-COVID avg entry rate vs avg P1
county_pre = large[large['year'].between(2016, 2019)].groupby('county').agg(
    avg_entry=('entry_rate', 'mean'),
    avg_p1=('pct_permanency_12mo', 'mean')
).dropna()
r_county, p_county = stats.pearsonr(county_pre['avg_entry'], county_pre['avg_p1'])
print(f"\n  County-level cross-section (pre-COVID avgs):")
print(f"  r={r_county:.3f}, p={p_county:.4f}, n={len(county_pre)}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Entry rate vs permanency rate",
             fontsize=12, fontweight='bold')

# Panel A: county-year scatter, coloured by period
ax = axes[0]
for period, colour, label in [
    (pre_both,  '#2A9D8F', 'Pre-COVID (2016-19)'),
    (cov_both,  '#E63946', 'COVID (2020-21)'),
]:
    ax.scatter(period['entry_rate'], period['pct_permanency_12mo'],
               color=colour, alpha=0.4, s=25, label=label, edgecolors='none')
# Regression line for pre-COVID
m, b, *_ = stats.linregress(pre_both['entry_rate'],
                              pre_both['pct_permanency_12mo'])
xs = np.linspace(pre_both['entry_rate'].min(),
                  pre_both['entry_rate'].max(), 100)
ax.plot(xs, m*xs+b, color='#2A9D8F', lw=1.5, ls='--')
ax.set_xlabel('Entry rate (per 1,000 children)')
ax.set_ylabel('Permanency in 12 months (%)')
ax.set_title(f'County-year observations\nPre-COVID r={r_pre:.3f}',
             fontsize=10)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel B: county-level cross-section
ax = axes[1]
ax.scatter(county_pre['avg_entry'], county_pre['avg_p1'],
           color='#457B9D', alpha=0.7, s=50, edgecolors='white')
m2, b2, *_ = stats.linregress(county_pre['avg_entry'], county_pre['avg_p1'])
xs2 = np.linspace(county_pre['avg_entry'].min(),
                   county_pre['avg_entry'].max(), 100)
ax.plot(xs2, m2*xs2+b2, color='#E63946', lw=1.5,
        label=f'r={r_county:.2f}')
for county, row in county_pre.nlargest(4, 'avg_entry').iterrows():
    ax.annotate(county, (row['avg_entry'], row['avg_p1']),
                fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Avg entry rate 2016-2019')
ax.set_ylabel('Avg permanency rate 2016-2019 (%)')
ax.set_title(f'County averages (pre-COVID)\nr={r_county:.3f}',
             fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS + "fc_11_entry_vs_permanency.png", dpi=150, bbox_inches='tight')
print(f"\n  Saved: {FIGS}fc_11_entry_vs_permanency.png")


# =============================================================================
# SUMMARY OF MODELLING DECISIONS
# =============================================================================
section("Summary of modelling decisions")

print("""
  1. PARALLEL PRE-TRENDS
     Formal test passed: Year x High tercile p=0.744, Year x Low p=0.227.
     Pre-trends were parallel across county terciles 2010-2019.
     TWFE assumption is plausible. No need for county-specific time trends
     in the baseline model, but include as a robustness check.

  2. OUTCOME TRANSFORMATION
     Log transform overcorrects: skew goes from +1.82 to -1.35.
     sqrt(entry_rate) brings skew to 0.32 and is consistent across years.
     DECISION: use sqrt(entry_rate) as the regression outcome.
     Report raw rates in all descriptive tables and figures.
     Coefficients interpreted as change in sqrt(rate) per unit X --
     convert back to rate scale for reporting using: fitted^2.

  3. AB 403
     Strong signal confirmed: r=-0.594, p<0.0001.
     High-entry counties dropped significantly more after AB 403.
     Trinity (-9.7 pp), Calaveras (-4.1 pp), Tehama (-3.9 pp).
     DECISION: include post-2017 indicator interacted with pre-reform
     entry rate rather than a simple post-2017 dummy. This captures
     the differential effect of AB 403 by county volume.

  4. MASKED CELLS
     No missingness problem. Entry rate panel 100% complete (928 obs).
     P1 permanency: 10 missing cells in Alpine (6), Sierra (3), Mono (1).
     These three counties should be excluded from the P1 event study.

  5. ENTRY VS PERMANENCY
     Weak positive correlation -- not significant at county level
     (r=0.200, p=0.132). The two outcomes measure different dimensions.
     DECISION: model separately.
       - Entry rate (sqrt): TWFE panel regression with ACS covariates.
       - P1 permanency: COVID event study, no ACS covariates needed.
""")