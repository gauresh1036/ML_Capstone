# Datasets

All three raw datasets are stored here unchanged. The notebooks read them from this folder with paths relative to the
repository root, so no download step is needed.

| File | Track | Rows × columns | Target | Source |
|------|-------|----------------|--------|--------|
| `housing.csv` | Regression | 2,930 × 82 | `SalePrice` (USD, continuous) | Ames Housing — De Cock, D. (2011), *Journal of Statistics Education* 19(3) |
| `online_shoppers_intention.csv` | Classification | 12,330 × 18 | `Revenue` (True / False, 15.5 % True) | UCI ML Repository #468 — Sakar *et al.* (2019) |
| `Online Retail.xlsx` | Clustering | 541,909 × 8 | — (unlabelled) | UCI ML Repository #352 — Chen, Sain & Guo (2012) |

## Ames Housing (`housing.csv`)
Residential sales in Ames, Iowa, 2006–2010. 79 explanatory variables describe size, quality, condition, age, location
and amenities, plus `Order`/`PID` identifiers. 27 columns contain missing values. In most of them (pool, alley, fence,
garage, basement fields) a missing value means *the house does not have that feature*.

## Online Shoppers Purchasing Intention (`online_shoppers_intention.csv`)
One row per browsing session over one year: pages visited and time spent per page type, Google-Analytics bounce/exit
rates and page values, closeness to special days, month, operating system, browser, region, traffic source, visitor
type and a weekend flag. There are no missing values and 125 exact duplicates (removed in the notebook).

## Online Retail (`Online Retail.xlsx`)
Every invoice line of a UK-based online gift retailer from 01-Dec-2010 to 09-Dec-2011: `InvoiceNo` (prefix `C` =
cancellation), `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice` (£), `CustomerID` (25 % missing) and
`Country`. There are no labels: the notebook aggregates the lines into per-customer RFM features for clustering.
