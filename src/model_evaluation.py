import logging
import pickle
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)


# logging configure
logger = logging.getLogger('model_evaluation')
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


def main():
    try:
        clf = pickle.load(open('model.pkl', 'rb'))

        # FIX: use the threshold tuned on training-set cross-validation (model_building.py),
        # instead of sklearn's default 0.5 cutoff — 0.5 is rarely a sensible choice for a
        # problem this imbalanced.
        best_thresh = pickle.load(open('threshold.pkl', 'rb'))

        test_data = pd.read_csv('./data/features/test_features.csv')

        # FIX: select by column name instead of iloc[:, 0:-1] / iloc[:, -1] — positional
        # slicing silently breaks if column order ever changes upstream.
        X_test = test_data.drop(columns=['label'])
        y_test = test_data['label'].values

        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= best_thresh).astype(int)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        logger.debug("Confusion matrix (threshold=%.2f):\n%s", best_thresh, cm)
        logger.debug(
            "accuracy=%.4f precision=%.4f recall=%.4f auc=%.4f",
            accuracy, precision, recall, auc
        )

        metrics_dict = {
            'threshold': best_thresh,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
        }

        with open('metrics.json', 'w') as file:
            json.dump(metrics_dict, file, indent=4)

        logger.debug("Model evaluation completed successfully.")
    except Exception as e:
        logger.error("Failed to complete model evaluation: %s", e)
        raise


if __name__ == "__main__":
    main()
