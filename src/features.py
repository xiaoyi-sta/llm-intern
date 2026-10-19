"""特征层：跨表 SQL 特征 + 预处理"""
import numpy as np
import pandas as pd

SQL_FEATURE_V1 = """
SELECT 
    -- 1. 主表核心字段
    a.SK_ID_CURR,
    a.TARGET,
    a.CODE_GENDER,
    a.FLAG_OWN_CAR,
    a.FLAG_OWN_REALTY,
    a.AMT_INCOME_TOTAL AS income,
    a.AMT_CREDIT AS credit_amount,
    a.DAYS_BIRTH,
    a.DAYS_EMPLOYED,
    
    -- 2. 外部征信表 (bureau) 衍生特征：历史负债与逾期情况
    IFNULL(b.bureau_loan_count, 0) AS bureau_loan_count,
    IFNULL(b.total_debt, 0) AS total_debt,
    IFNULL(b.max_overdue_days, 0) AS max_overdue_days,
    
    -- 3. 内部历史申请表 (previous_application) 衍生特征：曾经被拒贷的次数
    IFNULL(p.prev_refused_count, 0) AS prev_refused_count
    
FROM application_train a

-- 【关键逻辑 1】聚合 bureau 表，提取每个人的外部贷款总数、总负债、最大逾期天数
LEFT JOIN (
    SELECT 
        SK_ID_CURR,
        COUNT(SK_ID_BUREAU) AS bureau_loan_count,
        SUM(AMT_CREDIT_SUM) AS total_debt,
        MAX(CREDIT_DAY_OVERDUE) AS max_overdue_days
    FROM bureau
    GROUP BY SK_ID_CURR
) b ON a.SK_ID_CURR = b.SK_ID_CURR

-- 【关键逻辑 2】聚合 previous_application 表，用 CASE WHEN 统计历史被拒贷次数
LEFT JOIN (
    SELECT 
        SK_ID_CURR,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) AS prev_refused_count
    FROM previous_application
    GROUP BY SK_ID_CURR
) p ON a.SK_ID_CURR = p.SK_ID_CURR
"""

SQL_FEATURE_V2 = """
SELECT 
    a.SK_ID_CURR, a.TARGET, a.CODE_GENDER, a.FLAG_OWN_CAR, a.FLAG_OWN_REALTY,
    a.AMT_INCOME_TOTAL AS income, a.AMT_CREDIT AS credit_amount, a.DAYS_BIRTH, a.DAYS_EMPLOYED,
    
    -- 外部征信表进阶特征
    IFNULL(b.bureau_loan_count, 0) AS bureau_loan_count,
    IFNULL(b.active_loan_count, 0) AS active_loan_count,
    IFNULL(b.total_debt, 0) AS total_debt,
    IFNULL(b.max_overdue_days, 0) AS max_overdue_days,
    -- 核心风控特征：信贷使用率 (分母加 0.001 防止除以 0 报错)
    IFNULL(b.total_debt, 0) / (IFNULL(b.total_credit, 0) + 0.001) AS debt_to_credit_ratio,

    -- 内部历史表进阶特征
    IFNULL(p.prev_refused_count, 0) AS prev_refused_count,
    IFNULL(p.prev_approved_count, 0) AS prev_approved_count,
    IFNULL(p.mean_prev_annuity, 0) AS mean_prev_annuity

FROM application_train a

-- 聚合外部征信表 (bureau)
LEFT JOIN (
    SELECT 
        SK_ID_CURR,
        COUNT(SK_ID_BUREAU) AS bureau_loan_count,
        SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END) AS active_loan_count,
        SUM(AMT_CREDIT_SUM) AS total_credit,
        SUM(AMT_CREDIT_SUM_DEBT) AS total_debt,
        MAX(CREDIT_DAY_OVERDUE) AS max_overdue_days
    FROM bureau
    GROUP BY SK_ID_CURR
) b ON a.SK_ID_CURR = b.SK_ID_CURR

-- 聚合内部历史表 (previous_application)
LEFT JOIN (
    SELECT 
        SK_ID_CURR,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) AS prev_refused_count,
        SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS prev_approved_count,
        AVG(AMT_ANNUITY) AS mean_prev_annuity
    FROM previous_application
    GROUP BY SK_ID_CURR
) p ON a.SK_ID_CURR = p.SK_ID_CURR
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
