from __future__ import annotations

import logging
from typing import Tuple, Dict
import pandas as pd
import numpy as np
from .indicators import calculate_rsi, calculate_macd, calculate_sma

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    # Computing the indicators
    df['rsi'] = calculate_rsi(df['Close'])
    df['sma10'] = calculate_sma(df['Close'], 10)
    df['sma20'] = calculate_sma(df['Close'], 20)
    df['sma50'] = calculate_sma(df['Close'], 50)
    macd_line, signal_line, hist = calculate_macd(df['Close'])
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df["macd_hist"] = hist
    # Adding lag features
    df["rsi_lag1"] = df["rsi"].shift(1)
    df["macd_hist_lag1"] = df["macd_hist"].shift(1)
    df["close_lag1"] = df["Close"].shift(1)
    # Adding ROC
    df["roc"] = df["Close"].pct_change(periods=3)
    # Adding rolling volatility
    df["volatility"] = df["Close"].rolling(window=5).std()
    # Adding volume based features
    df["vol_change"] = df["Volume"].pct_change()
    df["vol_sma10"] = df["Volume"].rolling(10).mean()
    # Future returns label: 1 if next day's close > today's close
    df["close_shift"] = df["Close"].shift(-1)
    df["target"] = (df["close_shift"] > df["Close"] * 1.002).astype(int)

    # Droping rows with NaNs (due to indicators and shift)
    df.dropna(inplace=True)
    # Features
    feature_cols = [
        'rsi', 'sma10', 'sma20', 'sma50', 'macd_line', 'macd_signal', 'macd_hist',
        'Volume', 'rsi_lag1', 'macd_hist_lag1', 'close_lag1', 'roc', 'volatility',
        'vol_change', 'vol_sma10'
    ]
    X = df[feature_cols]
    y = df['target']
    return X, y


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, float]:

    # Time-series splitting
    tscv = TimeSeriesSplit(n_splits=5)
    all_logistic_metrics = []
    all_tree_metrics = []

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scaling features for Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic regression model
        log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
        log_reg.fit(X_train_scaled, y_train)
        y_pred_lr = log_reg.predict(X_test_scaled)
        logistic_accuracy = accuracy_score(y_test, y_pred_lr)
        logistic_precision = precision_score(y_test, y_pred_lr, zero_division=0)
        logistic_recall = recall_score(y_test, y_pred_lr, zero_division=0)
        logistic_f1 = f1_score(y_test, y_pred_lr, zero_division=0)
        logistic_roc_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1])
        all_logistic_metrics.append({
            'accuracy': logistic_accuracy,
            'precision': logistic_precision,
            'recall': logistic_recall,
            'f1': logistic_f1,
            'roc_auc': logistic_roc_auc,
        })

        # Decision tree model
        tree = DecisionTreeClassifier(max_depth=5, random_state=random_state, class_weight='balanced')
        tree.fit(X_train, y_train)
        y_pred_tree = tree.predict(X_test)
        tree_accuracy = accuracy_score(y_test, y_pred_tree)
        tree_precision = precision_score(y_test, y_pred_tree, zero_division=0)
        tree_recall = recall_score(y_test, y_pred_tree, zero_division=0)
        tree_f1 = f1_score(y_test, y_pred_tree, zero_division=0)
        tree_roc_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
        all_tree_metrics.append({
            'accuracy': tree_accuracy,
            'precision': tree_precision,
            'recall': tree_recall,
            'f1': tree_f1,
            'roc_auc': tree_roc_auc,
        })

        # Feature importance analysis for Decision Tree model
        importances = tree.feature_importances_
        features = X.columns
        feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        logger.info("Decision Tree feature importances:\n%s", feature_importance_df)

    avg_logistic_metrics = pd.DataFrame(all_logistic_metrics).mean().to_dict()
    avg_tree_metrics = pd.DataFrame(all_tree_metrics).mean().to_dict()

    logger.info(
        "Average Trained models (Cross-Validation): logistic_accuracy=%.3f, tree_accuracy=%.3f",
        avg_logistic_metrics['accuracy'],
        avg_tree_metrics['accuracy'],
    )
    return {
        'logistic_accuracy': avg_logistic_metrics['accuracy'],
        'logistic_precision': avg_logistic_metrics['precision'],
        'logistic_recall': avg_logistic_metrics['recall'],
        'logistic_f1': avg_logistic_metrics['f1'],
        'logistic_roc_auc': avg_logistic_metrics['roc_auc'],
        'tree_accuracy': avg_tree_metrics['accuracy'],
        'tree_precision': avg_tree_metrics['precision'],
        'tree_recall': avg_tree_metrics['recall'],
        'tree_f1': avg_tree_metrics['f1'],
        'tree_roc_auc': avg_tree_metrics['roc_auc'],
    }