import logging
import pickle

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline


# logging configure
logger = logging.getLogger('model_building')
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


#TARGET_RECALL = 0.60


def tune_threshold(clf, X_train: pd.DataFrame, y_train: np.ndarray) -> float:
    """
    Fixed, standard decision threshold — chosen manually, not tuned by sweeping
    candidates. Kept as a function (rather than a bare constant) so the reporting
    log line below still runs for visibility.
    """
    best_thresh = 0.40

    oof_probs = cross_val_predict(clf, X_train, y_train, cv=5, method='predict_proba', n_jobs=-1)[:, 1]
    preds = (oof_probs >= best_thresh).astype(int)
    train_recall = recall_score(y_train, preds)

    logger.debug("Using fixed threshold: %.2f (cross-validated training recall: %.2f)", best_thresh, train_recall)
    return best_thresh


def main():
    try:
        # fetch the data from data/features
        train_data = pd.read_csv('./data/features/train_features.csv')

        # FIX: select by column name instead of iloc[:, 0:-1] / iloc[:, -1] — positional
        # slicing silently breaks if the column order ever changes upstream.
        X_train = train_data.drop(columns=['label'])
        y_train = train_data['label'].values

        # Define and train the Balanced Random Forest model
        # (balances the class distribution internally, which matters for
        # fraud data where "not fraud" heavily outnumbers "fraud")
        clf = BalancedRandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=10,
            sampling_strategy='all'
        )
        clf.fit(X_train, y_train)
        logger.debug("Model trained successfully on %d rows.", len(X_train))

        best_thresh = tune_threshold(clf, X_train, y_train)

        # save the raw classifier + threshold (used by model_evaluation.py, which
        # reads already-preprocessed data/features/test_features.csv)
        with open('model.pkl', 'wb') as f:
            pickle.dump(clf, f)

        with open('threshold.pkl', 'wb') as f:
            pickle.dump(best_thresh, f)

        # The fitted preprocessor (from feature_engineering.py) with
        # this classifier into ONE pipeline. This is what makes inference on new, raw
        # (unencoded, unscaled) transactions work correctly — no manual re-encoding
        # required, unlike the earlier version where the scaler/encoders were never
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)

        full_pipeline = Pipeline(steps=[
            ('preprocess', preprocessor),
            ('classifier', clf),
        ])
        with open('fraud_pipeline.pkl', 'wb') as f:
            pickle.dump(full_pipeline, f)

        logger.debug("Model building completed successfully.")
    except Exception as e:
        logger.error("Failed to complete model building: %s", e)
        raise


if __name__ == "__main__":
    main()
