import datetime
import preprocessing
import config
from train import predict_models, train_lgb
import numpy as np
from sklearn.metrics import mean_squared_log_error
import pandas as pd

if __name__ == "__main__":
    train_df = pd.read_csv(config.DATA_DIR.joinpath("train.csv"))
    test_df = pd.read_csv(config.DATA_DIR.joinpath("test.csv"))

    train_targets = train_df["Global_Sales"]
    train_features, test_features = preprocessing.run(train_df, test_df)

    models, oof_preds, feature_importance = train_lgb(
        train_features, train_targets, train_df
    )

    oof_score = np.sqrt(mean_squared_log_error(train_targets, oof_preds))
    print(f"oofでのRMSLE: {oof_score}")

    # ============================================
    # === for notebook visualization
    # ============================================

    # plt.figure(figsize=(12, 8))
    # sns.scatterplot(
    #     x=np.log1p(oof_preds.sum(axis=1)),
    #     y=np.log1p(train_targets),
    # )
    # plt.show()

    # features = feature_importance.mean().sort_values(ascending=False).head(50).index
    # plt.figure(figsize=(12,8))
    # sns.violinplot(data=feature_importance[features])
    # plt.xticks(rotation=90)
    # plt.tight_layout()
    # plt.show()

    DATE = datetime.datetime.now()
    config.SUBMISSION_CSV = f"{DATE.strftime('%Y-%m-%d-%H:%M:%S')}_submission.csv"

    pred = predict_models(models, test_features)
    pd.DataFrame(pred, columns=["Global_Sales"]).to_csv(
        config.DATA_DIR.joinpath(config.SUBMISSION_CSV), index=0
    )
