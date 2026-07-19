import os
import logging
import pandas as pd


# logging configure
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# fetch the data from data/raw
train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')


# Extract hour, day-of-week, and weekend flag from the raw timestamp
def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    df["hour"] = df["hour"].fillna(-1)
    df["day_of_week"] = df["day_of_week"].fillna("unknown")

    return df


# Strip stray whitespace out of amount_src so it can be cast to numeric
def cleaning_amount_src(text) -> str:
    text = str(text).replace(' ', '')
    return text


# amount_src arrives as a string column (with stray whitespace), so it has to be
# cleaned to numeric before any mean can be computed on it.
def clean_amount_src_column(df: pd.DataFrame) -> pd.DataFrame:
    df["amount_src"] = df["amount_src"].apply(cleaning_amount_src)
    df["amount_src"] = pd.to_numeric(df["amount_src"], errors="coerce")
    return df


# FIX: compute the fill values (means) from the TRAINING data only, then reuse the
# same values on the test set. Previously each dataframe computed its own mean
# independently, which means the test set's fill values leaked information about
# its own distribution — not something you'd know in a real deployment scenario.
def compute_numeric_fill_values(train_df: pd.DataFrame) -> dict:
    train_df = clean_amount_src_column(train_df.copy())
    num_fill_cols = ["amount_usd", "fee", "device_trust_score", "amount_src"]
    fill_values = {}
    for col in num_fill_cols:
        if col in train_df.columns:
            fill_values[col] = round(train_df[col].mean(), 2)
    return fill_values


def apply_numeric_fill(df: pd.DataFrame, fill_values: dict) -> pd.DataFrame:
    for col, mean_value in fill_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(mean_value)
    return df


# Fill missing categorical fields (ip_country, kyc_tier, etc.) with "unknown"
def filling_categorical_missing(df: pd.DataFrame) -> pd.DataFrame:
    cat_fill_cols = ["ip_address", "ip_country", "kyc_tier"]
    for col in cat_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")
    return df


# Fix inconsistent spellings/casing in the kyc_tier field and drop invalid values.
# FIX: log how many rows (and how much fraud) get dropped instead of silently
# discarding them — this matters because dropping fraud rows worsens an already
# severe class imbalance.
def standardizing_kyc_tier(df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
    df["kyc_tier"] = (
        df["kyc_tier"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"standrd": "standard", "enhancd": "enhanced", "nan": "unknown"})
    )
    valid_kyc_tiers = ["standard", "enhanced", "low", "unknown"]
    invalid_mask = ~df["kyc_tier"].isin(valid_kyc_tiers)

    if invalid_mask.any():
        dropped_fraud = df.loc[invalid_mask, "is_fraud"].sum() if "is_fraud" in df.columns else "n/a"
        logger.debug(
            "%s: dropping %d rows with unexpected kyc_tier values (%s of them fraud)",
            dataset_name, invalid_mask.sum(), dropped_fraud
        )

    df = df[df["kyc_tier"].isin(valid_kyc_tiers)]
    return df


# Fix inconsistent spellings/casing in the channel field (mobile/web/ATM)
def standardizing_channel(df: pd.DataFrame) -> pd.DataFrame:
    channel_replacements = {
        "mobille": "mobile", " mobile": "mobile", "MOBILE": "mobile",
        "WEB": "web", " web": "web", "weeb": "web",
        "ATm": "ATM", " ATM": "ATM", "unknown": "unknown",
    }
    df["channel"] = df["channel"].replace(channel_replacements)
    df["channel"] = df["channel"].astype(str).str.strip().str.lower()
    return df


# Fix inconsistent spellings/casing in the home_country field
def standardizing_home_country(df: pd.DataFrame) -> pd.DataFrame:
    home_country_replacements = {" US": "US", " UK": "UK", " CA": "CA", "unknown": "unknown"}
    df["home_country"] = df["home_country"].replace(home_country_replacements)
    df["home_country"] = df["home_country"].astype(str).str.strip().str.lower()
    return df


# FIX: previously dropped ip_risk_score, location_mismatch, new_device, corridor_risk,
# chargeback_history_count, txn_velocity_1h, hour, and is_weekend — exactly the kind of
# behavioral/risk signals that matter most for fraud detection. Those are kept now;
# only true identifiers and clearly redundant columns are dropped.
def removing_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "customer_id", "transaction_id", "timestamp",
        "device_id", "ip_address", "ip_country",
        "exchange_rate_src_to_dest", "year",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


# Delete rows that are completely empty
def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="all")


# Clean and normalize a single dataframe using all rules above.
# `fill_values` must come from the TRAINING set (see compute_numeric_fill_values below)
# so test data never leaks its own statistics into preprocessing.
def normalize_data(df: pd.DataFrame, fill_values: dict, dataset_name: str = "dataset") -> pd.DataFrame:
    try:
        df = engineer_time_features(df)
        df = clean_amount_src_column(df)
        df = apply_numeric_fill(df, fill_values)
        df = filling_categorical_missing(df)
        df = standardizing_kyc_tier(df, dataset_name=dataset_name)
        df = standardizing_channel(df)
        df = standardizing_home_country(df)
        df["day_of_week"] = df["day_of_week"].replace(-1, "unknown")
        df = removing_unwanted_columns(df)
        df = remove_empty_rows(df)
        logger.debug("%s: preprocessing completed successfully (%d rows remain)", dataset_name, len(df))
        return df
    except Exception as e:
        logger.error("An unexpected error occurred while preprocessing %s.\n%s", dataset_name, e)
        raise


# Move the target column to the end for a consistent feature/target layout
def reorder_target(df: pd.DataFrame) -> pd.DataFrame:
    df["is_fraud"] = df.pop("is_fraud")
    return df


def main():
    try:
        # Compute numeric fill values from TRAIN only, then reuse on test — no leakage
        train_fill_values = compute_numeric_fill_values(train_data)
        logger.debug("Computed numeric fill values from training data: %s", train_fill_values)

        train_processed_data = normalize_data(train_data, train_fill_values, dataset_name="train")
        test_processed_data = normalize_data(test_data, train_fill_values, dataset_name="test")

        # NOTE: categorical encoding (OneHotEncoder) now happens in feature_engineering.py,
        # bundled inside the same ColumnTransformer that scales numeric columns. Keeping
        # encoding out of this stage means the raw, human-readable category strings are
        # still visible in train_processed.csv / test_processed.csv for debugging.

        train_processed_data = reorder_target(train_processed_data)
        test_processed_data = reorder_target(test_processed_data)

        # storing the data inside data/processed
        data_path = os.path.join("data", "processed")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error("Failed to complete the data preprocessing process: %s", e)
        raise


if __name__ == "__main__":
    main()
