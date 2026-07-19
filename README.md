# FraudGuard

A Machine Learning fraud detection system for banking transactions. It scores  every transaction in real time and flags the ones likely to be fraudulen
before money leaves the bank.

## The problem this solves

Banks lose money to fraudulent transactions such as stolen cards, account takeovers, money-laundering transfers. Reviewing every transaction manually doesn't scale, and rule-based checks e.g ("block if amount > $1000") are easy for fraudsters to get around and cause too many false alarms.

FraudGuard uses machine learning to learn the real patterns behind fraud like device changes, location mismatches, unusual transaction timing, risk scores and scores each transaction automatically, so the bank's fraud team can focus their attention where it matters most.

## How it works

```
transaction → model → fraud probability (0–100%) → decision → Fraud / Not Fraud
```

## The business tradeoff

There are two ways this system can be wrong:
- **Missed fraud** — a real fraud case slips through undetected.
- **False alarm** — a legitimate customer gets wrongly flagged.

Missing real fraud is the costlier mistake as it means direct financial loss and regulatory risk. A false alarm, by contrast, just means a transaction goes to manual review. So FraudGuard is tuned to **catch as much real fraud as possible**, even if that means more transactions get flagged to be double-check.

**Current performance (decision threshold: 0.40):**

| Metric | Value | What it means |
|---|---|---|
| Recall | 51.3% | Catches just over half of all real fraud cases |
| Precision | 3.7% | Most flagged transactions are false alarms, needing manual review |
| Accuracy | 73.3% | Not the priority metric here see note below |
| AUC | 0.67 | Model has real signal, with room to improve |

**Why accuracy isn't the focus:** because fraud is rare compared to normal
transactions, a model can score high accuracy just by predicting "Not
Fraud" almost every time. Optimizing for accuracy alone would mean
catching *less* fraud, not more — the opposite of this project's goal.
That's why threshold 0.40 was chosen over higher-accuracy options: it's
the one that catches the most fraud.


## Project structure

```
bank_fraud/
├── data/                  # transaction data (raw + processed)
├── models/                # trained model files
├── src/
│   ├── data_ingestion.py       # loads and validates data
│   ├── preprocessing.py        # cleans and prepares data
│   ├── feature_engineering.py  # builds fraud-signal features
│   ├── model_building.py       # trains the model
│   └── model_evaluation.py     # tests performance, reports results
├── app.py                 # interactive app — check any transaction live
└── requirements.txt
```

## Quick start

```bash
conda create -n fraud python=3.11 -y
conda activate fraud
pip install -r requirements.txt

python src/data_ingestion.py
python src/preprocessing.py
python src/feature_engineering.py
python src/model_building.py
python src/model_evaluation.py
```

## Try it live

```bash
streamlit run app.py
```

Enter a transaction's details and get an instant fraud probability and
verdict — the same logic the bank's system would use in production.

## Roadmap

- Improve the model's accuracy with richer fraud-signal features
- Compare against stronger models (XGBoost, LightGBM)
- Add a manual-review queue for borderline cases, instead of one hard cutoff
- Track performance over time as the model retrains on new data

## Open for collaboration

This project is open to collaborators — data scientists, fraud/risk domain
experts, or engineers interested in fraud detection systems are welcome to
contribute. Open an issue or reach out if you'd like to get involved.

**Contact:**
- LinkedIn: [linkedin.com/in/donatusvictor](https://www.linkedin.com/in/donatusvictor)
- Email: donatusvictor76@gmail.com
- Phone: +234 813 779 0780




# ML Pipeline

A modular, production-style ML pipeline: **Ingestion → Preprocessing → model_building → Evaluation**.

## Structure
```
ml_pipeline/
├── configs/config.yaml       # all settings — no hardcoded values in code
├── data/raw/                 # put your input file here (csv/xlsx/parquet)
├── data/processed/           # (optional) intermediate outputs
├── models/                   # trained model + preprocessor .pkl artifacts land here
├── logs/                     # pipeline.log + evaluation_report.json
├── src/
│   ├── data_ingestion.py          # load + validate raw data
│   ├── preprocessing.py           # impute/encode/scale, train/test split
│   ├── feature_engineering.py     # model training + cross-validation
│   ├── model_building.py                # metrics + JSON report + deployment gate
│   ├── model_evaluation.py                # orchestrates all stages end-to-end
└── requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt

```
# Build & Track ML Pipelines with DVC

## How to run?

conda create -n test python=3.11 -y

conda activate test

## pip install -r requirements.txt
python -m pip install -r requirements.txt

## DVC Commands

git init

dvc init

dvc repro

dvc dag

dvc metrics show# FraudGuard