from typing import List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import mean_squared_log_error
from sklearn.utils import shuffle

import config


def NWRMSLE(preds, train_data):
    return "nwrmsle", mean_squared_log_error(train_data.get_label(), preds), False


def rmsle(preds, data):
    y_true = data.get_label()
    score = np.sqrt(mean_squared_log_error(y_true, preds))
    return "RMSLE", score, False


def train_lgb(
    train_features, train_targets, train_df
) -> Tuple[List[lgb.Booster], pd.DataFrame, pd.DataFrame]:

    models: List[lgb.Booster] = []
    oof_preds = pd.DataFrame()
    cv = GroupKFold(n_splits=7)
    fold_ids = list(cv.split(train_features, groups=train_df.series_Name))

    for train_idx, val_idx in fold_ids:
        train_x = train_features.iloc[train_idx]
        train_y = np.log1p(train_targets.iloc[train_idx])
        val_x = train_features.iloc[val_idx]
        val_y = np.log1p(train_targets.iloc[val_idx])

        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(val_x, label=val_y, reference=dtrain)

        num_round = 30000
        param = {
            "objective": "rmse",
            "verbosity": -100,
            "boosting": "gbdt",
            "seed": config.SEED,
            "num_leaves": 32,
            "learning_rate": 0.025,
            "max_depth": 7,
            "feature_fraction": 0.3,
            # "bagging_fraction": 0.7,
            "min_data_in_leaf": 10,
            "max_bin": 127,
        }

        model = lgb.train(
            param,
            dtrain,
            num_round,
            valid_sets=dval,
            early_stopping_rounds=500,
        )

        models.append(model)
        _oof_preds = pd.DataFrame(
            {
                "pred": predict_models([model], val_x),
                "val_idx": val_idx,
            }
        )
        oof_preds = pd.concat([oof_preds, _oof_preds])

    oof_preds = oof_preds.set_index("val_idx").sort_index()
    feature_importances = pd.DataFrame(
        {
            f"cv_{i}": m.feature_importance(importance_type="gain")
            for i, m in enumerate(models)
        },
        index=models[0].feature_name(),
    )

    return models, oof_preds, feature_importances.T


def predict_models(models: List[lgb.Booster], features: pd.DataFrame) -> np.ndarray:
    return np.mean(
        [
            np.expm1(model.predict(features, num_iteration=model.best_iteration))
            for model in models
        ],
        axis=0,
    )
