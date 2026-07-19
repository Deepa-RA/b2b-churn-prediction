# B2B Account Churn & Renewal Risk Prediction

## Overview
Predicting which client accounts are at risk of churning, using account-level usage, tenure, and relationship signals — framed around the kind of renewal-risk analysis a customer success or account management team would use to prioritise intervention.

## Business Context
A marketing agency producing ads for client websites has noticed significant customer churn. Account managers are currently assigned **randomly** rather than based on risk — the business goal is to replace this with a model that flags at-risk accounts so managers can be allocated where they matter most. This is a scarce-resource-allocation problem, not just a prediction exercise.

## Dataset
Source: [Kaggle - Customer Churn by hassanamin](https://www.kaggle.com/datasets/hassanamin/customer-churn)

900 client accounts, 10 columns:
- `Name` — name of the latest contact at the client company
- `Age` — customer age
- `Total_Purchase` — total ads purchased
- `Account_Manager` — binary, whether a manager is currently assigned (assigned **randomly**, not risk-based)
- `Years` — total years as a customer
- `Num_Sites` — number of websites using the service
- `Onboard_date` — date the contact was onboarded
- `Location` — client HQ address
- `Company` — client company name
- `Churn` — target variable (binary)

No missing values across any column.

## Exploratory Findings

**Churn rate: ~17%** (150 of 900 accounts) — a meaningful class imbalance, which affects both modeling approach and evaluation metrics used later (accuracy alone would be misleading; recall on the churn class matters more).

**Key drivers identified** (via boxplots, t-tests, and chi-square testing):

| Variable | Direction | Strength | Notes |
|---|---|---|---|
| `Num_Sites` (websites using the service) | Churned accounts show *higher* usage | Strong (t=18.50, p<0.0001) | Counterintuitive — more embedded accounts churn more, not less |
| `Years` (tenure) | Churned accounts are *longer*-tenured | Strong (t=6.58, p<0.0001) | Counterintuitive — longer relationships don't predict safety |
| `Account_Manager` | Managed accounts churn slightly more | Weak/borderline (χ²=4.12, p=0.0425) | See note below on random assignment |
| `Total_Purchase` (ad volume) | No meaningful difference | Not significant | Spend level alone doesn't predict churn |
| `Age` | Churned accounts are slightly *older* | Weak-to-modest (t=2.58, p=0.0099) | Significant but minor — ~1.4 year average difference |

**Note on `Account_Manager`:** per the source problem statement, managers are currently assigned *randomly*, not based on any existing risk assessment. This means the weak positive correlation between having a manager and churning isn't a reverse-causation artifact (i.e. "risky accounts already get flagged") — it's a genuine, if modest, signal from a natural random-assignment setup.

**Correlation check:** `Years` and `Num_Sites` are essentially uncorrelated with each other (r=0.05) — these are two independent risk signals, not two measurements of the same underlying "account size" factor.

**Interpretation:** Predictior Ranking - `Num_Sites` > `Years` > `Age` > `Account_Manager` > `Total_Purchase` (not significant). Churn risk is concentrated in large, established accounts — with a minor additional signal from contact age — rather than new or small ones. — measured by number of websites deployed and tenure — rather than new or small ones. This is consistent with vendor reviews, renegotiation leverage, or competitive displacement at renewal for bigger, more valuable accounts, rather than poor onboarding of small ones. This is a hypothesis suggested by the pattern, not a proven causal mechanism.


## Approach
1. Exploratory data analysis (complete) — distribution checks, statistical testing, correlation analysis
2. Feature engineering — TBD
3. Modeling — TBD, with attention to class imbalance given the 83/17 split
4. Evaluation — TBD, likely prioritizing recall on the churn class over raw accuracy

## Tech Stack
Python 3.12, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Feature Engineering

- Dropped high-cardinality identifier columns (`Names`, `Company`, raw `Location`) — near-unique per row, unusable for generalisation
- Derived `Region` from `Location` by extracting state, then grouping ~50 sparse state categories into 4 mainland regions (Northeast, Midwest, South, West) plus a `Territories` bucket (PR, Guam, US Virgin Islands, etc.) — done to avoid overfitting on states with as few as 7-20 accounts each
- Extracted `Onboard_year` and `Onboard_month` from `Onboard_date`
- Resulting model-ready dataset: 9 columns, 900 rows, no missing values