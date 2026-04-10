# Data Download Instructions

This folder holds the raw data files used in the analysis.
CCWIP files cannot be included in the repository — download them
following the instructions below.

---

## File 1: Entry Rates (primary outcome)

**URL:** https://ccwip.berkeley.edu/childwelfare/reports/EntryRates

1. Under **Rate Type**, select: `Incidence per 1,000 Children`
2. Under **Agency Type**, select: `Child Welfare`
3. Under **Episode Count**, select: `All Children Entering`
4. Under **Geography**, select: `All Counties`
5. Set date range: `2010` to present
6. Click **Download** → Excel
7. Save as: `CCWIP Data Load EntryRates.xlsx` (or any name containing "EntryRates")

**Expected:** 58 county rows, 15+ year columns, values like 3.2, 4.1 etc.
**Note:** Requires CCWIP secure site login for county-level breakdown.

---

## File 2: In-Care Rates

**URL:** https://ccwip.berkeley.edu/childwelfare/reports/InCareRates

1. Select: `Prevalence per 1,000 Children`
2. Agency Type: `Child Welfare`
3. Geography: `All Counties`
4. Date range: `2010` to present
5. Download → Excel
6. Save as any filename containing "InCareRates"

**Expected:** 58 county rows, July 1 snapshots per year.

---

## File 3: 4-P1 Permanency

**URL:** https://ccwip.berkeley.edu → Federal Measures → 4-P1

1. Click **Multi County View** button
2. Agency Type: `Child Welfare`
3. All available years
4. Download → Excel
5. Save as any filename containing "P1"

**Expected:** 58 county rows, 2015–2024, values as percentages (35.8, 40.2 etc).

---

## File 4: Point-in-Time Count

**URL:** https://ccwip.berkeley.edu/childwelfare/reports/PointInTime

1. Agency Type: `Child Welfare`
2. Geography: `All Counties`
3. Date range: `2016` to present (Jan 1 snapshots)
4. Download → Excel
5. Save as any filename containing "PIT"

**Expected:** 58 county rows, Jan 1 counts per year (raw numbers, not rates).

---

## File 5: Exits

**URL:** https://ccwip.berkeley.edu/childwelfare/reports/Exits

1. Agency Type: `Child Welfare`
2. Episode Count: `Children Exiting`
3. Days in care: `8 days or more`
4. Geography: `All Counties`
5. Date range: `2016` to present
6. Download → Excel
7. Save as any filename containing "Exits"

**Expected:** 58 county rows, 2016–2025, raw exit counts.

---

## ACS Data (auto-generated)

The file `fc_acs_panel.csv` is generated automatically by running:

```bash
python fc_01_load_data.py
```

This pulls data from the Census API — no manual download needed.
Requires internet access. Covers all California counties, 2010–2022.

A pre-generated copy is included in the repository for convenience.

---

## File Naming

The loading script (`fc_01_load_data.py`) auto-detects files by keyword:

| Dataset | Keyword it looks for |
|---|---|
| Entry Rates | `EntryRates` |
| In-Care Rates | `InCareRates` |
| 4-P1 Permanency | `P1` |
| Point-in-Time | `PIT` |
| Exits | `Exits` |

File names do not need to match exactly — just contain the keyword.
Spaces and underscores are treated the same.
If multiple files match, the one with the longer filename is used
(the county-level secure site downloads typically have longer names).

---

## De-identification Note

CCWIP applies CDSS Data De-identification Guidelines — cells with values
between 1 and 10 are masked and appear as "M" in downloads. The loading
script treats these as NaN. The entry rate panel has zero masked cells.
Three counties (Alpine, Sierra, Mono) are excluded from the permanency
event study due to sparse data.

Per CCWIP terms of use, raw data files should not be redistributed.
This repository does not include them.
