"""
Foster Care Panel Analysis — Data Loading
==========================================
Loads all CCWIP files and ACS poverty/demographics into clean
long-format CSVs ready for EDA and panel regression.

CCWIP files (wide format: counties as rows, years as columns):
  EntryRates    -- entries per 1,000 children, 2010-2025, county x year
  InCareRates   -- in-care prevalence per 1,000, July 1, 2010-2025
  P1            -- % achieving permanency in 12 months, 2015-2024
  Entries       -- raw entry counts, 2016-2025
  PIT           -- point-in-time count Jan 1, 2016-2026
  Exits         -- raw exit counts, 2016-2025

ACS (Census API):
  B17001        -- poverty by age (child poverty rate)
  B09001        -- child population under 18
  B02001        -- race/ethnicity composition
  B25070        -- housing cost burden (rent > 30% of income)

Outputs saved to data/:
  fc_entry_rates.csv      -- primary outcome panel
  fc_incare_rates.csv     -- secondary outcome panel
  fc_p1_permanency.csv    -- COVID event study outcome
  fc_entries_raw.csv      -- raw counts
  fc_pit_raw.csv          -- point-in-time counts
  fc_exits_raw.csv        -- exit counts
  fc_acs_panel.csv        -- ACS covariates 2010-2023
  fc_panel.csv            -- merged master panel
"""

import pandas as pd
import numpy as np
import os
import json
import re
import urllib.request
import warnings
warnings.filterwarnings('ignore')

os.makedirs("data", exist_ok=True)

UPLOADS = "/mnt/user-data/uploads/"   # <-- change this to your local folder

# Map each dataset to a keyword substring of its filename.
# The script searches UPLOADS for any .xlsx whose name contains
# the keyword (case-insensitive), so you do NOT need exact filenames.
# Only edit the right-hand side if your filenames use different words.
FILE_KEYWORDS = {
    "entry_rates":  "EntryRates",   # incidence per 1,000 children
    "incare_rates": "InCareRates",  # prevalence per 1,000, July 1
    "p1":           "P1",           # 4-P1 permanency in 12 months
    "entries":      "Entries",      # raw entry counts
    "pit":          "PIT",          # point-in-time counts
    "exits":        "Exits",        # raw exit counts
}

def find_file(key):
    """
    Return the path of the first .xlsx in UPLOADS whose filename
    contains FILE_KEYWORDS[key] (case-insensitive).
    If multiple files match, the most recently modified is used.
    """
    keyword = FILE_KEYWORDS[key]
    matches = [
        os.path.join(UPLOADS, f) for f in os.listdir(UPLOADS)
        if f.lower().endswith(".xlsx")
        and keyword.lower() in f.lower()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No .xlsx containing '{keyword}' found in:\n  {UPLOADS}\n"
            f"Files present: {os.listdir(UPLOADS)}"
        )
    if len(matches) > 1:
        # When multiple files match (e.g. both a statewide and a county-level
        # version exist), prefer files with more characters in the name --
        # the county-level exports tend to have a longer filename suffix
        # like __1_ appended by the browser on re-download.
        matches.sort(key=lambda f: len(os.path.basename(f)), reverse=True)
        print(f"  WARNING: multiple files match '{keyword}', "
              f"using: {os.path.basename(matches[0])}")
    print(f"  File: {os.path.basename(matches[0])}")
    return matches[0]


def section(title):
    print(f"\n{'=' * 62}\n  {title}\n{'=' * 62}")


# =============================================================================
# CCWIP PARSER
# All six files share the same wide format:
#   - Metadata header block of variable depth
#   - Row where col[0] == 'California' marks start of data
#   - Year labels are two rows above the California row
#   - Counties run down col[0], years run across cols[1:]
#   - 'M' = masked (value 1-10, suppressed per CDSS DDG) -> NaN
#   - '.' = zero or indeterminate (0/0) -> 0 for counts, NaN for rates/pct
#   - Output: long format with columns [county, year, value]
# =============================================================================

def parse_ccwip(path, value_col, year_format="jan_dec"):
    """
    Parse a CCWIP wide-format Excel file into long format.

    Parameters
    ----------
    path        : str   path to Excel file
    value_col   : str   name to give the value column in output
    year_format : str   'jan_dec' for JAN20XX-DEC20XX intervals
                        'jul'     for Jul 1, 20XX point-in-time
                        'jan'     for Jan 1, 20XX point-in-time

    Returns
    -------
    pd.DataFrame with columns: county, year, <value_col>
    """
    raw = pd.read_excel(path, header=None)

    # Find the row index where California (first data row) starts
    data_start = None
    for i, row in raw.iterrows():
        if str(row[0]).strip() == 'California':
            data_start = i
            break
    if data_start is None:
        raise ValueError(f"Could not find 'California' row in {path}")

    # Year labels are two rows above the California row
    year_row = raw.iloc[data_start - 2, 1:]

    # Parse year integers from the label strings
    years = []
    for label in year_row:
        label = str(label).strip()
        if year_format == "jan_dec":
            # Format: JAN2010-DEC2010 -> extract first 4-digit year
            m = re.search(r'(\d{4})', label)
            years.append(int(m.group(1)) if m else None)
        elif year_format in ("jul", "jan"):
            # Format: Jul 1, 2010 or Jan 1, 2016
            m = re.search(r'(\d{4})', label)
            years.append(int(m.group(1)) if m else None)

    # Data block: from California row to last county row
    # Drop trailing empty rows (NaN in col 0)
    data_block = raw.iloc[data_start:].copy()
    data_block = data_block[data_block[0].notna()].copy()

    # Drop the statewide California row -- county-level only
    county_data = data_block[data_block[0] != 'California'].copy()

    # Build wide DataFrame
    county_data.columns = ['county'] + years
    county_data = county_data.reset_index(drop=True)

    # Melt to long format
    long = county_data.melt(id_vars='county', var_name='year',
                             value_name=value_col)
    long['year'] = pd.to_numeric(long['year'], errors='coerce')

    # Handle suppression codes
    # 'M' = masked (1-10), treat as NaN
    # '.' = zero or 0/0, treat as NaN (conservative -- don't assume zero)
    long[value_col] = long[value_col].astype(str).str.strip()
    long[value_col] = long[value_col].replace({'M': np.nan, '.': np.nan,
                                                'nan': np.nan, '0': 0})
    long[value_col] = pd.to_numeric(long[value_col], errors='coerce')

    # Standardise county names
    long['county'] = long['county'].str.strip()

    long = long.dropna(subset=['year']).sort_values(
        ['county', 'year']).reset_index(drop=True)

    return long


# =============================================================================
# 1. ENTRY RATES — primary outcome
#    Entries per 1,000 children, 2010-2025, annual (Jan-Dec)
# =============================================================================
section("1. Entry rates (primary outcome)")

entry_rates = parse_ccwip(
    find_file("entry_rates"),
    value_col="entry_rate",
    year_format="jan_dec"
)

print(f"  Shape: {entry_rates.shape}")
print(f"  Counties: {entry_rates['county'].nunique()}")
print(f"  Years: {sorted(entry_rates['year'].unique())}")
print(f"  Masked cells: {entry_rates['entry_rate'].isna().sum()}")
print(f"\n  Sample:")
print(entry_rates[entry_rates['county'] == 'Los Angeles'].to_string(index=False))

entry_rates.to_csv("data/fc_entry_rates.csv", index=False)
print("\n  Saved: data/fc_entry_rates.csv")


# =============================================================================
# 2. IN-CARE RATES — secondary outcome
#    Prevalence per 1,000 children, July 1 snapshots, 2010-2025
#    NOTE: point-in-time (stock), not annual flow like entry rates
# =============================================================================
section("2. In-care rates (secondary outcome)")

incare_rates = parse_ccwip(
    find_file("incare_rates"),
    value_col="incare_rate",
    year_format="jul"
)

print(f"  Shape: {incare_rates.shape}")
print(f"  Years: {sorted(incare_rates['year'].unique())}")
print(f"  Masked cells: {incare_rates['incare_rate'].isna().sum()}")

incare_rates.to_csv("data/fc_incare_rates.csv", index=False)
print("  Saved: data/fc_incare_rates.csv")


# =============================================================================
# 3. 4-P1 PERMANENCY — COVID event study outcome
#    % of entry cohort achieving permanency within 12 months
#    2015-2024, annual (Jan-Dec)
#    Permanency = reunification + adoption + guardianship
# =============================================================================
section("3. 4-P1 Permanency in 12 months (COVID event study)")

p1 = parse_ccwip(
    find_file("p1"),
    value_col="pct_permanency_12mo",
    year_format="jan_dec"
)

print(f"  Shape: {p1.shape}")
print(f"  Years: {sorted(p1['year'].unique())}")
print(f"  Masked cells: {p1['pct_permanency_12mo'].isna().sum()}")
print(f"  CA mean by year:")
print(entry_rates.groupby('year')['entry_rate'].mean().round(2).to_string())

p1.to_csv("data/fc_p1_permanency.csv", index=False)
print("\n  Saved: data/fc_p1_permanency.csv")


# =============================================================================
# 4. RAW ENTRIES — for validation and rate computation cross-check
# =============================================================================
section("4. Raw entry counts")

entries_raw = parse_ccwip(
    find_file("entries"),
    value_col="entries_n",
    year_format="jan_dec"
)

print(f"  Shape: {entries_raw.shape}")
print(f"  Years: {sorted(entries_raw['year'].unique())}")

entries_raw.to_csv("data/fc_entries_raw.csv", index=False)
print("  Saved: data/fc_entries_raw.csv")


# =============================================================================
# 5. POINT-IN-TIME COUNT — Jan 1 snapshots
# =============================================================================
section("5. Point-in-time counts (Jan 1)")

pit = parse_ccwip(
    find_file("pit"),
    value_col="pit_count",
    year_format="jan"
)

print(f"  Shape: {pit.shape}")
print(f"  Years: {sorted(pit['year'].unique())}")

pit.to_csv("data/fc_pit_raw.csv", index=False)
print("  Saved: data/fc_pit_raw.csv")


# =============================================================================
# 6. EXITS — for system flow analysis
# =============================================================================
section("6. Exit counts")

exits = parse_ccwip(
    find_file("exits"),
    value_col="exits_n",
    year_format="jan_dec"
)

print(f"  Shape: {exits.shape}")
print(f"  Years: {sorted(exits['year'].unique())}")

exits.to_csv("data/fc_exits_raw.csv", index=False)
print("  Saved: data/fc_exits_raw.csv")


# =============================================================================
# 7. ACS COVARIATES — Census API
#
#    Variables pulled for all CA counties, 5-year estimates 2010-2022:
#
#    B17001_001E  total population for whom poverty determined
#    B17001_004E  male under 6, below poverty
#    B17001_005E  male 6-11, below poverty
#    B17001_006E  male 12-17, below poverty
#    B17001_018E  female under 6, below poverty
#    B17001_019E  female 6-11, below poverty
#    B17001_020E  female 12-17, below poverty
#    -- sum of above / child population = child poverty rate
#
#    B09001_001E  total children under 18 (denominator for rates)
#
#    B02001_003E  Black or African American alone
#    B02001_005E  American Indian and Alaska Native alone
#    B03003_003E  Hispanic or Latino
#
#    B25070_007E  gross rent 30-34.9% of income
#    B25070_008E  gross rent 35-39.9% of income
#    B25070_009E  gross rent 40-49.9% of income
#    B25070_010E  gross rent 50%+ of income
#    B25070_001E  total renter-occupied units
#    -- sum 007-010 / 001 = severe housing cost burden rate
#
#    Why these covariates?
#    Child poverty rate: primary structural predictor of foster care entry.
#    Research consistently shows poverty is the strongest county-level
#    predictor of child welfare involvement, independent of maltreatment risk.
#    Black and Native American share: racial disproportionality in foster
#    care is well-documented; controlling for demographics separates
#    structural poverty effects from racial bias in reporting and removal.
#    Housing cost burden: housing instability is a proximal risk factor
#    for child welfare involvement and a target of prevention programs.
# =============================================================================
section("7. ACS covariates (Census API)")

ACS_VARS = ",".join([
    "NAME",
    "B17001_001E",  # total pop for poverty determination
    "B17001_004E","B17001_005E","B17001_006E",   # male children below poverty
    "B17001_018E","B17001_019E","B17001_020E",   # female children below poverty
    "B09001_001E",  # total children under 18
    "B02001_002E",  # White alone
    "B02001_003E",  # Black alone
    "B02001_005E",  # American Indian / Alaska Native alone
    "B03003_003E",  # Hispanic or Latino
    "B25070_001E",  # total renter-occupied units
    "B25070_007E","B25070_008E",
    "B25070_009E","B25070_010E",  # rent-burdened households
])

def fetch_acs(year):
    """Pull ACS 5-year estimates for all CA counties for a given year."""
    url = (f"https://api.census.gov/data/{year}/acs/acs5"
           f"?get={ACS_VARS}&for=county:*&in=state:06")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.loads(r.read())
        df = pd.DataFrame(data[1:], columns=data[0])
        df['year'] = year
        return df
    except Exception as e:
        print(f"    {year}: {e}")
        return None

acs_frames = []
# ACS 5-year estimates available from 2009 onward
# Use 2010-2022 to align with CCWIP panel
for yr in range(2010, 2023):
    print(f"  Fetching ACS {yr}...", end=" ", flush=True)
    df = fetch_acs(yr)
    if df is not None:
        acs_frames.append(df)
        print(f"OK ({len(df)} counties)")
    else:
        print("skipped")

if acs_frames:
    acs = pd.concat(acs_frames, ignore_index=True)

    # Convert all numeric columns
    num_cols = [c for c in acs.columns
                if c not in ('NAME', 'state', 'county', 'year')]
    for col in num_cols:
        acs[col] = pd.to_numeric(acs[col], errors='coerce')
        acs[col] = acs[col].where(acs[col] >= 0, np.nan)

    # County FIPS
    acs['county_fips'] = acs['state'] + acs['county']

    # Clean county name
    acs['county_name'] = acs['NAME'].str.replace(
        ' County, California', '', regex=False).str.strip()

    # --- Derived covariates ---

    # Child poverty rate: children under 18 below poverty / total children
    child_poverty_n = (
        acs['B17001_004E'] + acs['B17001_005E'] + acs['B17001_006E'] +
        acs['B17001_018E'] + acs['B17001_019E'] + acs['B17001_020E']
    )
    acs['child_poverty_rate'] = (
        child_poverty_n / acs['B09001_001E'] * 100
    ).round(2)

    # Child population (denominator for rate validation)
    acs['child_pop'] = acs['B09001_001E']

    # Racial composition (% of total population)
    total_pop = acs['B17001_001E']
    acs['pct_black']         = (acs['B02001_003E'] / total_pop * 100).round(2)
    acs['pct_native_american'] = (acs['B02001_005E'] / total_pop * 100).round(2)
    acs['pct_hispanic']      = (acs['B03003_003E'] / total_pop * 100).round(2)

    # Housing cost burden: % of renters paying 30%+ of income on rent
    rent_burdened = (
        acs['B25070_007E'] + acs['B25070_008E'] +
        acs['B25070_009E'] + acs['B25070_010E']
    )
    acs['housing_burden_rate'] = (
        rent_burdened / acs['B25070_001E'] * 100
    ).round(2)

    # Keep only derived columns
    acs_clean = acs[[
        'county_fips', 'county_name', 'year',
        'child_pop', 'child_poverty_rate',
        'pct_black', 'pct_native_american', 'pct_hispanic',
        'housing_burden_rate',
    ]].copy()

    print(f"\n  ACS panel: {len(acs_clean)} county-year rows")
    print(f"  Years: {sorted(acs_clean['year'].unique())}")
    print(f"\n  Descriptive stats:")
    print(acs_clean[['child_poverty_rate','pct_black',
                      'pct_native_american','housing_burden_rate']]
          .describe().round(2).to_string())

    acs_clean.to_csv("data/fc_acs_panel.csv", index=False)
    print("\n  Saved: data/fc_acs_panel.csv")
    has_acs = True
else:
    print("  Census API not reachable -- run locally to get ACS data")
    has_acs = False


# =============================================================================
# 8. MERGE — MASTER PANEL
#
#    Unit of analysis: county x year
#    Primary join: entry_rates (broadest time coverage 2010-2025)
#    Left join ACS (2010-2022), P1 (2015-2024), exits/PIT (2016-2025)
#    ACS 5-year estimates for 2023+ are not yet available so those
#    years will have NaN covariates -- flagged in analysis
# =============================================================================
section("8. Building master panel")

panel = entry_rates.copy()

# Join in-care rates (same years, July 1 vs Jan-Dec -- note in comments)
panel = panel.merge(incare_rates, on=['county','year'], how='left')

# Join permanency (2015-2024 only)
panel = panel.merge(p1, on=['county','year'], how='left')

# Join exits
panel = panel.merge(exits, on=['county','year'], how='left')

# Join PIT
panel = panel.merge(pit, on=['county','year'], how='left')

# Join ACS -- match on county name (strip to bare name for both)
if has_acs:
    acs_join = acs_clean.rename(columns={'county_name': 'county'})
    panel = panel.merge(acs_join, on=['county','year'], how='left')

# COVID indicator flags
panel['covid_year']      = panel['year'].isin([2020, 2021]).astype(int)
panel['post_covid']      = (panel['year'] >= 2022).astype(int)
panel['ab403_phase_in']  = panel['year'].isin([2017, 2018, 2019]).astype(int)
# AB 403 fully implemented statewide by 2020
panel['ab403_full']      = (panel['year'] >= 2020).astype(int)

# Event study time variable relative to COVID shock (2020 = 0)
panel['years_to_covid']  = panel['year'] - 2020

print(f"\n  Master panel shape: {panel.shape}")
print(f"  Counties: {panel['county'].nunique()}")
print(f"  Years: {sorted(panel['year'].unique())}")
print(f"\n  Coverage by variable:")
for col in ['entry_rate','incare_rate','pct_permanency_12mo',
            'exits_n','pit_count','child_poverty_rate']:
    if col in panel.columns:
        n_obs   = panel[col].notna().sum()
        n_total = len(panel)
        pct     = n_obs / n_total * 100
        print(f"    {col:<30} {n_obs:>4} / {n_total} ({pct:.0f}%)")

panel.to_csv("data/fc_panel.csv", index=False)
print("\n  Saved: data/fc_panel.csv")


# =============================================================================
# SUMMARY
# =============================================================================
section("Summary")

saved = [
    ("data/fc_entry_rates.csv",   "Entry rate per 1,000 children, 2010-2025"),
    ("data/fc_incare_rates.csv",  "In-care rate per 1,000, July 1, 2010-2025"),
    ("data/fc_p1_permanency.csv", "% permanency in 12 months, 2015-2024"),
    ("data/fc_entries_raw.csv",   "Raw entry counts, 2016-2025"),
    ("data/fc_pit_raw.csv",       "Point-in-time counts, Jan 1, 2016-2026"),
    ("data/fc_exits_raw.csv",     "Raw exit counts, 2016-2025"),
    ("data/fc_acs_panel.csv",     "ACS covariates, 2010-2022 (if API reachable)"),
    ("data/fc_panel.csv",         "Master panel, all sources merged"),
]
for path, desc in saved:
    status = "SAVED  " if os.path.exists(path) else "MISSING"
    print(f"  [{status}] {path}")
    print(f"           {desc}")

print("""
  Masked cell note:
    CCWIP masks cells with values 1-10 per CDSS de-identification rules.
    These appear as 'M' in the raw files and are set to NaN here.
    Small counties (Alpine, Sierra, Mono, Modoc) will have many masked
    cells. Consider dropping counties with >50% masked entry_rate values
    before running panel regression.

  ACS note:
    5-year estimates for 2023 are not yet published by Census Bureau.
    Years 2023-2025 in the master panel will have NaN for all ACS
    covariates. Panel regression should be restricted to 2010-2022
    or ACS covariates treated as time-invariant using the most recent
    available year (2022) for all subsequent years.

  Next: run fc_02_eda.py
""")
