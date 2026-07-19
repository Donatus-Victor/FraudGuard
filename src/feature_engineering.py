import os
import logging
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# logging configure
logger = logging.getLogger('feature_engineering')
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


CATEGORICAL_COLS = ['channel', 'home_country', 'kyc_tier', 'source_currency', 'dest_currency', 'day_of_week']
SCALE_COLS = ['amount_usd', 'fee', 'account_age_days', 'amount_src']
# Everything else numeric (risk scores, velocity counts, flags, hour, is_weekend, etc.)
# passes through unscaled — BalancedRandomForestClassifier doesn't need scaling to
# split effectively, so this only matters if the model is ever swapped for something
# scale-sensitive (e.g. logistic regression).


def build_preprocessor() -> ColumnTransformer:
    """
    FIX: replaces the previous manual LabelEncoder + StandardScaler approach.
    - OneHotEncoder(handle_unknown='ignore') avoids the false ordering LabelEncoder
      implies (e.g. mobile=2 > ATM=0), and gracefully handles any category that
      shows up in test/production but wasn't seen during training.
    - remainder='passthrough' keeps every other numeric column (the restored risk
      features included) flowing through untouched instead of being dropped.
    """
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLS),
            ('scale', StandardScaler(), SCALE_COLS),
        ],
        remainder='passthrough'
    )


def main():
    try:
        # fetch the data from data/processed
        train_data = pd.read_csv('./data/processed/train_processed.csv')
        test_data = pd.read_csv('./data/processed/test_processed.csv')

        y_train = train_data['is_fraud'].values
        y_test = test_data['is_fraud'].values

        X_train = train_data.drop(columns=['is_fraud'])
        X_test = test_data.drop(columns=['is_fraud'])

        # FIX: fit the preprocessor (encoder + scaler) on TRAIN ONLY, then transform
        # both train and test with it — no leakage of test-set categories/statistics
        # into how encoding or scaling is learned.
        preprocessor = build_preprocessor()
        X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        X_test_transformed = preprocessor.transform(X_test)

        feature_names = preprocessor.get_feature_names_out()

        train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        test_df['label'] = y_test

        # store the transformed features inside data/features
        data_path = os.path.join("data", "features")
        os.makedirs(data_path, exist_ok=True)

        train_df.to_csv(os.path.join(data_path, "train_features.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_features.csv"), index=False)

        # FIX: save the FITTED preprocessor itself. Previously the scaler/encoders
        # weren't saved at all, so new raw data at inference time had no way to be
        # transformed consistently with how the model was trained. model_building.py
        # bundles this back together with the trained classifier into one deployable
        # pipeline.
        with open('preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)

        logger.debug(
            "Feature engineering completed successfully. %d features produced.",
            len(feature_names)
        )
    except Exception as e:
        logger.error("Failed to complete feature engineering: %s", e)
        raise


if __name__ == "__main__":
    main()
