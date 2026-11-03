# 🛡️ Fraud Detection in Online Payments

An end-to-end fraud-detection pipeline for a digital-payments network (UPI, cards, net banking, wallets): validate 50,000 transactions, find the real fraud signals, engineer 30+ behaviour features, compare four models with time-ordered cross-validation, pick a threshold by cost, and turn probabilities into a risk score with four actions: allow, OTP challenge, manual review, hard block.

The most useful result is an honest one: **three signals carry almost everything**, and the final model ties a simple two-flag rule rather than beating it.

**Interactive case study:** https://devam-shah.com/projects/fraud-detection/

---

## Results (test set: the last two weeks, 7,500 transactions, 756 fraud)

| Metric | Value |
|---|---|
| ROC-AUC | **0.718** |
| PR-AUC | **0.231** (base rate 0.101, so 2.3x random) |
| Top 20% riskiest transactions | catch **47.5%** of fraud (2.4x lift) |
| At the cost-optimal threshold (0.085) | fraud recall **74.9%**, precision 15.8%, 47.8% of transactions challenged |
| Confusion matrix | TN 3,722 · FP 3,022 · FN 190 · TP 566 |
| Modelled net saving | **₹30.4 lakh** over two weeks, under the stated cost assumptions (₹300 chargeback fee, ₹1,000 per false alarm) |

Risk tiers on the test set:

| Tier | Action | Share of transactions | Fraud rate |
|---|---|---|---|
| 1 | Allow | 52.2% | 4.9% |
| 2 | OTP challenge | 42.1% | 13.8% |
| 3 | Manual review | 4.2% | 28.3% |
| 4 | Hard block (+ device lock if new device) | 1.5% | 35.7% |

---

## What the Data Actually Says

- **Fraud is 10.06%** of transactions (5,028 of 50,000), not the under-0.5% the problem description suggested. Resampling and class weights made no difference.
- **Three signals matter:** a device change (28.4% fraud vs 8.5%), a location change (24.0% vs 8.5%), and IP risk (fraud rises from 5.4% in the lowest tenth to 14.3% in the highest). Both changes together: 42.8% fraud.
- **Everything else is flat:** amount, hour, weekday, merchant category, payment method, device trust and 9 customer-history features all sit near AUC 0.50.
- **About 57% of fraud is "quiet":** no device or location change at all. The model misses all of it, which caps recall without new data sources.

## Honest Findings

- **The model ties a simple rule.** Flagging "device OR location changed" (17.3% of test transactions) gives 25.7% precision and 44.2% recall; the model flagging the same number gives 25.3% and 43.5%. The model's value is a smooth score for tiered actions, not better ranking.
- **Feature engineering didn't help.** An ablation adding feature groups one at a time moved validation PR-AUC by under 0.01. The final model is logistic regression on four raw columns: IP risk, device change, location change, international.
- **The data looks synthetic:** most numeric columns are near-uniform, every merchant appears under all six categories, and customers average 12.5 devices. Conclusions are about this dataset, not real payments traffic.

---

## Pipeline

1. **Validation and cleaning:** 22 logged checks (keys, nulls, ranges, business rules, dates); 4,121 impossible 24-hour averages set to missing; rows sorted by time.
2. **EDA + statistical tests:** chi-square / Mann-Whitney with Bonferroni correction (α = 0.0026); 5 of 19 columns significant.
3. **Features:** 24 row-level + 9 past-only customer-history features, an Isolation Forest anomaly score (AUC 0.50, dropped) and a combined risk index.
4. **Chronological 70/15/15 split** (Jan–Mar 2024), so the model is always tested on the future.
5. **Models:** Logistic Regression, Decision Tree, Random Forest, Histogram Gradient Boosting; `TimeSeriesSplit` 5-fold CV; tuning; all land near PR-AUC 0.22.
6. **Platt calibration**, cost-based threshold, sensitivity table across false-alarm costs.
7. **Risk score (0–100), tiers, a `score_transactions()` function**, permutation importance, odds ratios, error analysis.

## Data

50,000 transactions × 25 columns (Jan 1 – Mar 30, 2024) for **PaySphere**, a fictional payments company, in `Datasets/transactions_fraud.csv` (7.8 MB), with column descriptions in `Data Definition.xlsx`. The data is synthetic and contains no personal data. The notebook writes a cleaned copy to `Datasets/transactions_clean.csv` and the scoring artifacts to `fraud_model.pkl` when it runs.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook online_payment_fraud_detection.ipynb
```

## Tech Stack

Python · pandas · NumPy · SciPy · scikit-learn · Matplotlib · Seaborn · joblib

---

**Devam Shah** · [devam-shah.com](https://devam-shah.com) · [LinkedIn](https://www.linkedin.com/in/devam-a-shah) · [GitHub](https://github.com/dshah-ds)
