"""数据层：CSV → SQLite，以及按 SQL 取特征表"""
import os
import sqlite3

import pandas as pd

CSV_TABLES = {
    "application_train.csv": "application_train",
    "bureau.csv": "bureau",
    "previous_application.csv": "previous_application",
}


def db_path(data_dir: str) -> str:
    return os.path.join(data_dir, "home_credit.db")


def build_db(data_dir: str, chunksize: int = 10000) -> None:
    """把三张 CSV 分批导入 SQLite（对应 notebook 的 cell 0）"""
    conn = sqlite3.connect(db_path(data_dir))
    try:
        for csv_name, table in CSV_TABLES.items():
            df = pd.read_csv(os.path.join(data_dir, csv_name))
            df.to_sql(table, conn, if_exists="replace", index=False, chunksize=chunksize)
            print(f"导入 {table}: {len(df)} 行")
    finally:
        conn.close()


def load_features(data_dir: str, sql: str) -> pd.DataFrame:
    """执行一条特征 SQL，返回 DataFrame"""
    conn = sqlite3.connect(db_path(data_dir))
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()
