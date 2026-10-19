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
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        class_weight="balanced",  # 风控场景违约样本极少，强制关注少数类
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
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
