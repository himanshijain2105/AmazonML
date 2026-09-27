"""Pair classifiers. LightGBM (MIT) or Logistic Regression (scikit-learn, BSD; stored as JSON).
The saved metadata records the feature set; loading with a different feature list raises."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

DEFAULT_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      seed=42, deterministic=True, force_row_wise=True, num_threads=0, verbose=-1)


class LRModel:
    def __init__(self, mean=None, scale=None, coef=None, intercept=0.0):
        self.mean, self.scale, self.coef, self.intercept = mean, scale, coef, intercept

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.mean = X.mean(0)
        self.scale = X.std(0) + 1e-6
        m = LogisticRegression(C=1.0, max_iter=500).fit((X - self.mean) / self.scale, y)
        self.coef, self.intercept = m.coef_.ravel(), float(m.intercept_[0])
        return self

    def predict(self, X):
        z = ((X - self.mean) / self.scale) @ self.coef + self.intercept
        return (1 / (1 + np.exp(-z))).astype(np.float32)

    def to_json(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}


def train(X, y, X_es, y_es, feature_names, kind="lgbm", scale_pos_weight=1.0, rounds=2000, params=None):
    if kind == "lr":
        return LRModel().fit(X, y)
    p = dict(DEFAULT_PARAMS, scale_pos_weight=scale_pos_weight, **(params or {}))
    dtr = lgb.Dataset(X, y, feature_name=list(feature_names), free_raw_data=True)
    des = lgb.Dataset(X_es, y_es, reference=dtr)
    return lgb.train(p, dtr, rounds, valid_sets=[des], callbacks=[lgb.early_stopping(50, verbose=False)])


def predict(model, X):
    if isinstance(model, LRModel):
        return model.predict(X)
    it = model.best_iteration if getattr(model, "best_iteration", 0) > 0 else None
    return model.predict(X, num_iteration=it).astype(np.float32)


def save(model, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, LRModel):
        meta = dict(meta, kind="lr", lr=model.to_json())
        path.write_text("see .json (logistic regression)")
    else:
        model.save_model(str(path), num_iteration=model.best_iteration)
        meta = dict(meta, kind="lgbm", best_iteration=model.best_iteration)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))


def load(path, expected_features=None):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if expected_features is not None and meta["feature_names"] != list(expected_features):
        raise ValueError("model feature list differs from the code's feature set — train/inference mismatch")
    if meta.get("kind") == "lr":
        d = meta["lr"]
        return LRModel(np.array(d["mean"]), np.array(d["scale"]), np.array(d["coef"]), d["intercept"]), meta
    return lgb.Booster(model_file=str(path)), meta


def feature_importance(model, names):
    if isinstance(model, LRModel):
        return dict(sorted(zip(names, np.abs(model.coef).tolist()), key=lambda x: -x[1]))
    g = model.feature_importance("gain")
    return dict(sorted(zip(names, (g / max(g.sum(), 1e-9)).round(5).tolist()), key=lambda x: -x[1]))
