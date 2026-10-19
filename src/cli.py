"""命令行入口：python -m src.cli --data-dir <数据目录>"""
import argparse

from src import data, features, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Home Credit 风控模型流水线")
    parser.add_argument("--data-dir", required=True, help="CSV / SQLite 所在目录")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--rebuild-db", action="store_true", help="从 CSV 重建 SQLite")
    args = parser.parse_args()

    if args.rebuild_db:
        data.build_db(args.data_dir)

    sql = features.SQL_FEATURE_V1 if args.version == "v1" else features.SQL_FEATURE_V2
    df = data.load_features(args.data_dir, sql)
    df = features.prepare(df)
    X, y = features.split_xy(df)
    _, metrics = train.train(X, y)
    print({k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
