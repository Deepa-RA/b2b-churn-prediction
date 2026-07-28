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

## Modeling

**Model: Logistic Regression** — chosen as an interpretable baseline, since coefficients can be directly explained to business stakeholders and any more complex model would need to justify its added complexity against this benchmark.

**Preprocessing:** features scaled with `StandardScaler` (fit on training data only, applied to test data — avoiding data leakage from test into training). `Region` one-hot encoded; identifier columns (`Names`, `Company`, `Location`) dropped; `Onboard_date` decomposed into year/month.

**Train/test split:** 80/20, stratified by `Churn` to preserve the ~17% churn rate in both sets.

### Baseline results (default 0.5 threshold, unweighted)
- Recall (churn): 0.50 — missed half of actual churners
- Precision (churn): 0.79
- Accuracy: 0.89 (misleading in isolation, given the 83/17 class split)

### Addressing class imbalance: class weighting
Applying `class_weight='balanced'` shifted the trade-off substantially:
- Recall (churn): 0.50 → **0.77**
- Precision (churn): 0.79 → 0.50
- Missed churners: 15 → 7 (out of 30)

This trade-off is appropriate given the business context: missing an at-risk account (false negative) is costlier than flagging a safe account for review (false positive), since the whole point is replacing random account manager assignment with risk-based prioritization.

### Threshold tuning
Testing decision thresholds from 0.3–0.7 on the weighted model's predicted probabilities:

| Threshold | Precision | Recall |
|---|---|---|
| 0.3 | 0.39 | 0.93 |
| 0.4 | 0.44 | 0.93 |
| 0.5 | 0.50 | 0.77 |
| 0.6 | 0.56 | 0.73 |
| 0.7 | 0.66 | 0.70 |

**Recommended threshold: 0.4** — catches 93% of actual churners (28/30) at 44% precision. Chosen over the default 0.5 because the business's current process (random account manager assignment) has zero risk signal at all — redirecting that same existing account manager capacity toward a 44%-precision flagged list represents a clear improvement, even accounting for false alarms. This assumes account managers have some spare capacity to absorb additional flagged reviews; if capacity is highly constrained, a higher threshold (e.g. 0.6) would be a more conservative choice, trading some recall for fewer wasted reviews.

Note: 0.3 was tested but dominated by 0.4 (identical recall, worse precision) — excluded from consideration.