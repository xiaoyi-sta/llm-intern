"""建模层：LightGBM 训练与指标"""
import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def train(X, y, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = lgb.LGBMClassifier(
        # TODO 把 notebook cell 2 / cell 6 里的模型参数抄到这里
    )
    model.fit(X_train, y_train)
    prob = np.asarray(model.predict_proba(X_test))[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "auc": roc_auc_score(y_test, prob),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
    }
    return model, metrics
