# B2B Account Churn & Renewal Risk Prediction

## Overview
Predicting which client accounts are at risk of churning, using account-level usage, tenure, and relationship signals — framed around the kind of renewal-risk analysis a customer success or account management team would use to prioritize intervention.

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
| `Age` | Churned accounts are slightly *older* | Weak-to-modest (t=2.58, p=0.0099) | Significant but minor — ~1.4 year average difference |
| `Account_Manager` | Managed accounts churn slightly more | Weak/borderline (χ²=4.12, p=0.0425) | See note below on random assignment |
| `Total_Purchase` (ad volume) | No meaningful difference | Not significant | Spend level alone doesn't predict churn |

**Note on `Account_Manager`:** per the source problem statement, managers are currently assigned *randomly*, not based on any existing risk assessment. This means the weak positive correlation between having a manager and churning isn't a reverse-causation artifact (i.e. "risky accounts already get flagged") — it's a genuine, if modest, signal from a natural random-assignment setup.

**Correlation check:** `Years` and `Num_Sites` are essentially uncorrelated with each other (r=0.05) — these are two independent risk signals, not two measurements of the same underlying "account size" factor.

**Updated predictor ranking (strongest to weakest):** `Num_Sites` > `Years` > `Age` > `Account_Manager` > `Total_Purchase` (not significant). Churn risk is concentrated in large, established accounts — with a minor additional signal from contact age — rather than new or small ones. This is a hypothesis suggested by the pattern (consistent with vendor reviews, renegotiation leverage, or competitive displacement at renewal for bigger accounts), not a proven causal mechanism.

## Feature Engineering
- Dropped `Names`, `Company` — near-unique identifiers, no learnable pattern for a model
- Extracted `Onboard_year` / `Onboard_month` from `Onboard_date`, then dropped the raw date column
- Extracted `State` from `Location` via regex; found individual states too sparse (many with <20 accounts) to use reliably, so grouped into 4 mainland regions plus a `Territories` bucket (covering Puerto Rico, Guam, Marshall Islands, etc.), then dropped both `State` and `Location`
- One-hot encoded `Region` (`drop_first=True`, avoiding redundant/multicollinear columns)
- **Investigated but rejected:** a `Purchase_per_Site` ratio and a binned `Num_Sites` interaction check (see Feature Importance section below) — both dropped after testing showed they didn't add genuine independent signal

## Modeling

**Approach:** Logistic regression as an interpretable baseline, compared against a random forest. Features scaled with `StandardScaler` for logistic regression (fit on training data only, applied to test data, to avoid data leakage); random forest used unscaled features, since tree-based splits are scale-invariant. 80/20 train/test split, stratified to preserve the ~17% churn ratio in both sets.

**Baseline logistic regression (unweighted):**

| Metric (Churn class) | Value |
|---|---|
| Precision | 0.79 |
| Recall | 0.50 |
| Missed churners (False Negatives) | 15 of 30 |

The baseline missed half of all actual churners — a direct consequence of the class imbalance found in EDA, since the model defaults to favoring the majority (non-churn) class.

**Class-weighted logistic regression:** applying `class_weight='balanced'` reweights training penalties inversely to class frequency, penalizing missed churners more heavily.

| Metric (Churn class) | Baseline | Weighted |
|---|---|---|
| Precision | 0.79 | 0.50 |
| Recall | 0.50 | 0.77 |
| Missed churners | 15 | 7 |
| False alarms | 4 | 23 |

Recall improved substantially at the cost of precision — an expected trade-off given the business context: missing an at-risk account is costlier than flagging a safe one for review.

**Random forest (class-weighted):** trained as a second model to test whether a more complex, non-linear approach improves on logistic regression.

| Metric (Churn class) | Logistic (weighted) | Random Forest (weighted) |
|---|---|---|
| Precision | 0.50 | 0.66 |
| Recall | 0.77 | 0.63 |
| Missed churners | 7 | 11 |
| False alarms | 23 | 10 |

Random forest lands at a different point on the precision/recall trade-off rather than being a strict improvement — better precision, lower recall. Added model complexity did not automatically produce a better outcome; it produced a different trade-off.

**Threshold tuning:** rather than accepting the default 0.5 cutoff, tested precision/recall across several thresholds for both models.

| Threshold | Logistic (weighted) P / R | Random Forest P / R |
|---|---|---|
| 0.3 | 0.39 / 0.93 | 0.51 / 0.80 |
| 0.4 | 0.44 / 0.93 | 0.57 / 0.70 |
| 0.5 | 0.50 / 0.77 | 0.67 / 0.67 |
| 0.6 | 0.56 / 0.73 | 0.78 / 0.60 |
| 0.7 | 0.66 / 0.70 | 0.80 / 0.40 |

**Reading the trade-off:** logistic regression can reach recall as high as 0.93 (catching nearly all churners), a ceiling random forest doesn't reach in this range (its best recall here is 0.80). Random forest offers a more balanced trade-off at moderate thresholds (e.g. 0.67 precision / 0.67 recall at 0.5), better than logistic regression's options at similar recall levels.

**Which model/threshold to recommend depends on account manager capacity, a business assumption rather than something the data confirms:**
- If maximum recall is the priority (catch nearly everyone, tolerate many false alarms) → logistic regression at threshold 0.3–0.4
- If a more balanced trade-off is preferred (catch a solid majority, keep false alarms more contained) → random forest at threshold 0.4–0.5

Given that account managers are currently assigned with **zero risk signal at all** (random assignment), either option is very likely an improvement over the status quo — the choice between them is about how aggressively to reallocate manager time, not whether to.

### Feature Importance & Investigating a Discrepancy

Random forest feature importance broadly confirmed the EDA findings — `Num_Sites` and `Years` ranked highest in both approaches, in the same order. This is a useful cross-check: two independent methods (single-variable statistical testing vs. a model learning from all features jointly) converged on the same top drivers.

One discrepancy was investigated rather than dismissed: `Total_Purchase` ranked 3rd in random forest importance (0.11) despite showing **no statistically significant relationship with churn** in EDA (t-test, p>0.05). Two possible explanations were tested:

1. **A `Total_Purchase`/`Num_Sites` interaction (purchase-per-site)** — tested via a derived ratio feature. Result: strongly "significant" (t=-8.26, p<0.0001), but investigation showed this was mechanically confounded — since `Total_Purchase` is roughly flat between groups while `Num_Sites` is meaningfully higher for churned accounts, the ratio is largely restating `Num_Sites` in a different form, not adding independent signal. Feature dropped after confirming this.

2. **A conditional interaction, tested via binning `Num_Sites` into low/high tiers** and re-testing `Total_Purchase` within each tier separately. Result: not significant in either tier (Low_Sites: p=0.56; High_Sites: p=0.41) — no evidence of a genuine interaction effect.

**Conclusion:** neither explanation held up under direct testing. The most likely remaining explanation is that random forest importance can reflect scattered, non-generalizable patterns picked up across many individual tree splits, without there being one clean, statistically describable relationship behind it — a known limitation of tree-based feature importance. `Total_Purchase` is treated as a weak/unreliable predictor going forward, consistent with the original EDA finding, not the random forest ranking.

## Approach
1. `notebooks/01_eda.ipynb` — exploratory data analysis: distribution checks, statistical testing, correlation analysis, feature importance discrepancy investigation
2. `notebooks/02_feature_engineering.ipynb` — dropping identifiers, date/region feature extraction, one-hot encoding
3. `notebooks/03_modeling.ipynb` — train/test split, logistic regression, random forest, threshold tuning, feature importance, final model export

## Tech Stack
Python 3.12, pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Key Findings & Recommendation

**The core finding is counterintuitive and worth leading with:** churn risk is concentrated in the agency's *largest, most established* accounts — high website usage (`Num_Sites`) and long tenure (`Years`) are the strongest predictors, both in the opposite direction of the common assumption that big, embedded accounts are "safe." This points toward vendor reviews, renegotiation leverage, or competitive displacement at renewal for high-value accounts, rather than an onboarding or new-customer retention problem.

**Recommended model: Random Forest, threshold 0.4**

| Metric | Value |
|---|---|
| Precision (Churn) | 0.57 |
| Recall (Churn) | 0.70 |
| Catches | 21 of 30 actual churners in the test set |
| False alarms | 16 accounts flagged that were actually safe |

**Why this specific model and threshold, over the alternatives tested:** logistic regression could reach higher recall (up to 0.93) but only at substantially lower precision (as low as 0.39-0.44) — meaning more than half of every flagged account would be a false alarm. The random forest at 0.4 offers a more efficient use of account manager time: catching 70% of actual churners while keeping false alarms more contained. Since account managers are currently assigned with **zero risk signal at all** (random assignment, per the source problem statement), this model represents a clear improvement over the status quo regardless of which threshold ultimately gets adopted in practice — the choice between thresholds is about how aggressively to reallocate manager time, not whether to.

**Caveats, stated honestly rather than glossed over:**
- This is a single, relatively small dataset (900 accounts, 150 churned) — results should be treated as a strong proof of concept, not a production-ready guarantee, without validation on more data
- The `Account_Manager` finding is weak and borderline (p=0.0425) — real, but not something to build a strong causal claim around without further testing (a dedicated causal inference follow-up project is planned to examine this properly)
- Correlation and predictive value were both extensively tested, but this remains a predictive model, not a causal one — it identifies accounts *at risk*, not necessarily *why*, beyond what the exploratory analysis suggests as a plausible business story

**If deployed:** the model would flag roughly 37 of 180 accounts (in the test set) as at-risk, versus the current process of assigning managers with no risk signal at all — a meaningfully more targeted starting point for account management prioritization.