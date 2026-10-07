"""V4FinBench h2 file / paper horizon h=1: exploratory grouped 5-fold CV.

Put this script in V4FinBench_Project and run with that project's Python.
Reuses existing folds without claiming they are the official released indices.
Each rotation chooses weight, tree count and threshold using validation only.
The dataset has already been explored; this is not a fresh external evaluation.
Grouped evaluation does not establish chronological forecasting validity or
point-in-time availability of operational_status or other supplied features.
"""

from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent


def validate_folds(metadata, folds):
    if folds.ndim != 1 or len(folds) != len(metadata):
        raise ValueError('Fold assignments must match dataset rows exactly.')
    if set(np.unique(folds)) != set(range(5)):
        raise ValueError('Expected exactly folds 0, 1, 2, 3, 4.')
    if metadata[['country', 'company', 'main_label']].isna().any().any():
        raise ValueError('Missing country, company, or target values.')
    if not metadata['main_label'].isin([0, 1]).all():
        raise ValueError('main_label must contain only 0 and 1.')
    groups = metadata[['country', 'company']].copy()
    groups['fold'] = folds
    if (groups.groupby(['country', 'company'], observed=True)['fold'].nunique() > 1).any():
        raise ValueError('A company-country group occurs in multiple folds.')
    for fold in range(5):
        if metadata.loc[folds == fold, 'main_label'].nunique() != 2:
            raise ValueError(f'Fold {fold} must contain both classes.')


def select_threshold(labels, scores):
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite prediction scores.')
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator != 0)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(labels, scores, threshold):
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'roc_auc': roc_auc_score(labels, scores),
        # Average precision, not trapezoidal integration of the PR curve.
        'average_precision': average_precision_score(labels, scores),
        'true_negatives': int(tn), 'false_positives': int(fp),
        'false_negatives': int(fn), 'true_positives': int(tp),
    }


def main():
    import joblib
    import lightgbm as lgb
    import pyarrow.parquet as pq
    import sklearn

    started = time.perf_counter()
    data_path = ROOT / 'data/raw/company_years_h2.parquet'
    fold_path = ROOT / 'data/folds/h2/fold_assignments.txt'
    feature_path = ROOT / 'data/processed/feature_columns.txt'
    for path in [data_path, fold_path, feature_path]:
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}')
    features = [s.strip() for s in feature_path.read_text(encoding='utf-8-sig').splitlines()
                if s.strip()]
    if not features or len(set(features)) != len(features):
        raise ValueError('Feature names must be nonempty and unique.')
    if set(features) & {'main_label', 'company', 'emis_id', 'link', 'num'}:
        raise ValueError('Target or identifier detected in feature list.')

    print('STEP 1: Checking saved folds against company groups', flush=True)
    metadata = pq.read_table(data_path, columns=['country', 'company', 'main_label']).to_pandas()
    metadata = metadata.reset_index(drop=True)
    folds = np.loadtxt(fold_path, dtype=np.int64)
    validate_folds(metadata, folds)
    labels = metadata['main_label'].to_numpy(dtype=np.int8)
    print(pd.crosstab(folds, labels).to_string(), flush=True)
    print('Company-country grouping check passed.', flush=True)
    del metadata
    gc.collect()

    print('\nSTEP 2: Loading features', flush=True)
    frame = pq.read_table(data_path, columns=features).to_pandas()
    X = frame.to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
    del frame
    # Process one column at a time to avoid a full-size Boolean temporary.
    for column in range(X.shape[1]):
        X[np.isinf(X[:, column]), column] = np.nan
    gc.collect()
    print(f'Rows: {len(X):,}; features: {len(features)}', flush=True)

    run_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = ROOT / 'results' / f'h1_lightgbm_cv_{run_name}'
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'dataset': data_path.name, 'paper_horizon': 1,
        'feature_columns': features,
        'fold_file_sha256': hashlib.sha256(fold_path.read_bytes()).hexdigest(),
        'lightgbm_version': lgb.__version__, 'sklearn_version': sklearn.__version__,
        'selection': 'highest validation average precision; validation F1 threshold',
        'weight_candidates': ['1', '10', 'sqrt(training imbalance)', '50', '100', 'training imbalance'],
        'evaluation': 'exploratory grouped CV on previously explored data, not chronological holdout',
        'feature_timing': 'operational_status and upstream feature timing not verified',
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows = []
    tested = np.zeros(len(X), dtype=np.int8)

    for validation_fold in range(5):
        rotation_start = time.perf_counter()
        test_fold = (validation_fold + 1) % 5
        train_folds = [f for f in range(5) if f not in [validation_fold, test_fold]]
        train_indices = np.flatnonzero(np.isin(folds, train_folds))
        validation_indices = np.flatnonzero(folds == validation_fold)
        test_indices = np.flatnonzero(folds == test_fold)
        X_train, y_train = X[train_indices], labels[train_indices]
        X_validation, y_validation = X[validation_indices], labels[validation_indices]
        ratio = float((y_train == 0).sum() / (y_train == 1).sum())
        candidates = [1.0, 10.0, float(np.sqrt(ratio)), 50.0, 100.0, ratio]
        destination = output / f'validation_{validation_fold}_test_{test_fold}'
        destination.mkdir()
        print(f'\nROTATION {validation_fold + 1}/5: train={train_folds}, '
              f'validation={validation_fold}, test={test_fold}', flush=True)
        search = []
        best_model = None
        best_ap = -1.0
        best = None

        for weight in candidates:
            print(f'  Training weight {weight:.4f} ...', flush=True)
            model = lgb.LGBMClassifier(
                objective='binary', boosting_type='gbdt', metric='average_precision',
                n_estimators=1500, learning_rate=0.05, num_leaves=31,
                max_depth=-1, min_child_samples=100,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                reg_alpha=0.0, reg_lambda=1.0, max_bin=255,
                scale_pos_weight=weight, random_state=42, n_jobs=-1,
                verbosity=-1, deterministic=True, force_col_wise=True,
            )
            fit_args = {
                'eval_names': ['validation'], 'eval_metric': 'average_precision',
                'feature_name': features,
                'callbacks': [lgb.early_stopping(100, first_metric_only=True, verbose=False)],
            }
            # Support both the user's newer API and released older LightGBM.
            if 'eval_X' in inspect.signature(model.fit).parameters:
                fit_args.update(eval_X=X_validation, eval_y=y_validation)
            else:
                fit_args['eval_set'] = [(X_validation, y_validation)]
            model.fit(X_train, y_train, **fit_args)
            iteration = int(model.best_iteration_)
            scores = model.booster_.predict(X_validation, raw_score=True, num_iteration=iteration)
            threshold = select_threshold(y_validation, scores)
            metrics = evaluate(y_validation, scores, threshold)
            candidate = {
                'weight': weight, 'best_iteration': iteration, 'threshold': threshold,
                **{f'validation_{key}': value for key, value in metrics.items()},
            }
            search.append(candidate)
            pd.DataFrame(search).to_csv(destination / 'weight_search.csv', index=False)
            print(f"    Trees={iteration}; validation AP={metrics['average_precision']:.6f}; "
                  f"F1={metrics['f1']:.6f}", flush=True)
            if metrics['average_precision'] > best_ap:
                best_ap, best_model, best = metrics['average_precision'], model, candidate

        del X_train, X_validation, model
        gc.collect()
        # No test scores were used to choose the model or its threshold.
        X_test = X[test_indices]
        scores = best_model.booster_.predict(
            X_test, raw_score=True, num_iteration=best['best_iteration'])
        test_metrics = evaluate(labels[test_indices], scores, best['threshold'])
        tested[test_indices] += 1
        row = {
            'validation_fold': validation_fold, 'test_fold': test_fold,
            'train_folds': ','.join(map(str, train_folds)),
            'selected_weight': best['weight'], 'best_iteration': best['best_iteration'],
            'threshold': best['threshold'], 'validation_average_precision': best_ap,
            'validation_f1': best['validation_f1'], 'test_rows': len(test_indices),
            'test_positives': int(labels[test_indices].sum()),
            **test_metrics, 'rotation_seconds': time.perf_counter() - rotation_start,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(output / 'fold_metrics.csv', index=False)
        pd.DataFrame({
            'row_index': test_indices, 'actual_label': labels[test_indices],
            'decision_score': scores,
            'predicted_label': (scores >= best['threshold']).astype(np.int8),
        }).to_parquet(destination / 'test_predictions.parquet', index=False)
        joblib.dump({
            'feature_columns': features, 'model': best_model,
            'threshold': best['threshold'], 'score_type': 'raw_margin',
            'selected_weight': best['weight'], 'best_iteration': best['best_iteration'],
        }, destination / 'model.joblib')
        pd.DataFrame({
            'feature': features,
            'gain_importance': best_model.booster_.feature_importance(importance_type='gain'),
        }).sort_values('gain_importance', ascending=False).to_csv(
            destination / 'feature_importance.csv', index=False)
        print(f"  SELECTED weight={best['weight']:.4f}; test AP={test_metrics['average_precision']:.6f}; "
              f"F1={test_metrics['f1']:.6f}", flush=True)
        del X_test, best_model, scores
        gc.collect()

    if not np.all(tested == 1):
        raise RuntimeError('Every row must be evaluated exactly once as test data.')
    results = pd.DataFrame(rows)
    names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
    summary = results[names].agg(['mean', 'std']).T.rename_axis('metric').reset_index()
    summary.to_csv(output / 'cv_summary.csv', index=False)
    print('\nFIVE-FOLD RESULTS (std = sample standard deviation)', flush=True)
    print(results[['validation_fold', 'test_fold', 'selected_weight', 'best_iteration'] + names]
          .to_string(index=False), flush=True)
    print('\nMEAN AND STANDARD DEVIATION', flush=True)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'), flush=True)
    print(f'\nTotal runtime: {time.perf_counter() - started:.2f} seconds', flush=True)
    print(f'Results saved to: {output}', flush=True)


if __name__ == '__main__':
    main()
