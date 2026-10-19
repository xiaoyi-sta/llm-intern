"""特征层：跨表 SQL 特征 + 预处理"""
import numpy as np
import pandas as pd

SQL_FEATURE_V1 = """
把 notebook cell 1 里 feature_sql 的内容整段粘到这里
"""

SQL_FEATURE_V2 = """
把 notebook cell 6 里 feature_sql_v2 的内容整段粘到这里
"""


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """cell 2 开头的预处理：年龄换算、未就业异常值、类别转换"""
    df = df.copy()
    df["age"] = df["DAYS_BIRTH"] / -365
    df["years_employed"] = df["DAYS_EMPLOYED"] / -365
    df["years_employed"] = df["years_employed"].replace(-1000.6657534246575, np.nan)
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col] = df[col].astype("category")
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["SK_ID_CURR", "TARGET", "DAYS_BIRTH", "DAYS_EMPLOYED"])
    y = df["TARGET"]
    return X, y
