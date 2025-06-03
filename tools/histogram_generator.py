"""
CSVファイルからヒストグラムを生成するツール.

特徴量が記載されたCSVファイルを読み込み、class列による分類で色分けした
ヒストグラムを生成し、CSVファイルと同じフォルダのhistサブフォルダに保存します。
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class HistogramGenerator:
    """CSVファイルからヒストグラムを生成するクラス."""

    def __init__(self, csv_path: str, class_column: str = "class") -> None:
        """
        HistogramGeneratorのコンストラクタ.

        Args:
            csv_path (str): CSVファイルのパス.
            class_column (str): クラス分類に使用する列名. デフォルトは"class".
        """
        self.csv_path = Path(csv_path)
        self.class_column = class_column
        self.output_dir = self.csv_path.parent / "hist"
        self.df: Optional[pd.DataFrame] = None
        self.numeric_columns: List[str] = []

    def load_csv(self) -> None:
        """
        CSVファイルを読み込み、数値列を特定する.

        Raises:
            FileNotFoundError: CSVファイルが見つからない場合.
            ValueError: CSVファイルの読み込みに失敗した場合.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {self.csv_path}")

        try:
            # CSVファイルを読み込み
            self.df = pd.read_csv(self.csv_path)
            print(f"CSVファイルを読み込みました: {self.csv_path}")
            print(f"データ形状: {self.df.shape}")

            # クラス列の存在確認
            if self.class_column not in self.df.columns:
                print(f"警告: クラス列 '{self.class_column}' が見つかりません")
                print(f"利用可能な列: {list(self.df.columns)}")
                # クラス列がない場合は全データを一つのクラスとして扱う
                self.df[self.class_column] = "all_data"

            # 数値列を特定（クラス列、filename列、timestamp列を除く）
            exclude_columns = {self.class_column, "filename", "timestamp"}
            self.numeric_columns = [
                col
                for col in self.df.columns
                if col not in exclude_columns
                and pd.api.types.is_numeric_dtype(self.df[col])
            ]

            if not self.numeric_columns:
                raise ValueError("数値データの列が見つかりません")

            print(f"数値列を特定しました: {len(self.numeric_columns)}列")
            print(f"クラス数: {self.df[self.class_column].nunique()}")

        except Exception as e:
            raise ValueError(f"CSVファイルの読み込みに失敗しました: {e}")

    def create_output_directory(self) -> None:
        """出力ディレクトリを作成する."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"出力ディレクトリを作成しました: {self.output_dir}")

    def _get_color_palette(self, n_classes: int) -> List[str]:
        """
        クラス数に応じたカラーパレットを取得する.

        Args:
            n_classes (int): クラス数.

        Returns:
            List[str]: カラーパレット.
        """
        if n_classes <= 10:
            # 10色以下の場合はtab10パレットを使用
            return sns.color_palette("tab10", n_classes)
        else:
            # 10色を超える場合はhslパレットを使用
            return sns.color_palette("hsl", n_classes)

    def _calculate_bins(self, data: pd.Series) -> int:
        """
        データに適したビン数を計算する.

        Args:
            data (pd.Series): データ系列.

        Returns:
            int: ビン数.
        """
        # Sturgesの公式とFreedman-Diaconisルールの中間値を使用
        n = len(data.dropna())
        if n == 0:
            return 10

        # Sturgesの公式
        sturges_bins = int(np.ceil(np.log2(n) + 1))

        # Freedman-Diaconisルール
        q75, q25 = np.percentile(data.dropna(), [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            fd_bins = int(
                np.ceil((data.max() - data.min()) / (2 * iqr * (n ** (-1 / 3))))
            )
        else:
            fd_bins = sturges_bins

        # 最小10、最大50の範囲で調整
        return max(10, min(50, int((sturges_bins + fd_bins) / 2)))

    def generate_histogram(self, column: str) -> None:
        """
        指定された列のヒストグラムを生成する.

        Args:
            column (str): ヒストグラムを生成する列名.
        """
        if self.df is None:
            raise ValueError("CSVファイルが読み込まれていません")

        # データの準備
        data = self.df[[column, self.class_column]].dropna()
        if data.empty:
            print(f"警告: 列 '{column}' にデータがありません")
            return

        # クラスごとのデータを取得
        classes = sorted(data[self.class_column].unique())
        colors = self._get_color_palette(len(classes))

        # ビン数の計算
        bins = self._calculate_bins(data[column])

        # 図のサイズと設定
        plt.figure(figsize=(12, 8))
        plt.style.use("default")

        # 各クラスのヒストグラムを重ねて描画
        for i, class_name in enumerate(classes):
            class_data = data[data[self.class_column] == class_name][column]
            plt.hist(
                class_data,
                bins=bins,
                alpha=0.7,
                label=f"{class_name} (n={len(class_data)})",
                color=colors[i],
                edgecolor="black",
                linewidth=0.5,
            )

        # グラフの装飾
        plt.title(f"Histogram of {column}", fontsize=16, fontweight="bold")
        plt.xlabel(column, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # 統計情報をテキストで追加
        stats_text = f"Total samples: {len(data)}\n"
        stats_text += f"Mean: {data[column].mean():.3f}\n"
        stats_text += f"Std: {data[column].std():.3f}\n"
        stats_text += f"Min: {data[column].min():.3f}\n"
        stats_text += f"Max: {data[column].max():.3f}"

        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=10,
        )

        # レイアウトの調整
        plt.tight_layout()

        # ファイル名の生成（特殊文字を除去）
        safe_column_name = "".join(
            c for c in column if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_column_name = safe_column_name.replace(" ", "_")
        output_path = self.output_dir / f"{safe_column_name}_histogram.png"

        # 保存
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"ヒストグラムを保存しました: {output_path}")

    def generate_all_histograms(self) -> None:
        """すべての数値列のヒストグラムを生成する."""
        if not self.numeric_columns:
            print("警告: 生成するヒストグラムがありません")
            return

        print(f"=== {len(self.numeric_columns)}個のヒストグラムを生成します ===")

        for i, column in enumerate(self.numeric_columns, 1):
            print(f"進行状況 ({i}/{len(self.numeric_columns)}): {column}")
            try:
                self.generate_histogram(column)
            except Exception as e:
                print(f"エラー: 列 '{column}' のヒストグラム生成に失敗しました: {e}")

        print("=== ヒストグラム生成が完了しました ===")

    def generate_summary_report(self) -> None:
        """生成されたヒストグラムのサマリーレポートを作成する."""
        if self.df is None:
            return

        report_path = self.output_dir / "histogram_summary.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== ヒストグラム生成サマリーレポート ===\n\n")
            f.write(f"CSVファイル: {self.csv_path}\n")
            f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"データ形状: {self.df.shape}\n")
            f.write(f"クラス列: {self.class_column}\n")
            f.write(f"クラス数: {self.df[self.class_column].nunique()}\n")
            f.write(f"生成されたヒストグラム数: {len(self.numeric_columns)}\n\n")

            f.write("=== クラス分布 ===\n")
            class_counts = self.df[self.class_column].value_counts()
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}件\n")

            f.write("\n=== 数値列の統計情報 ===\n")
            for column in self.numeric_columns:
                data = self.df[column].dropna()
                f.write(f"\n{column}:\n")
                f.write(f"  サンプル数: {len(data)}\n")
                f.write(f"  平均: {data.mean():.6f}\n")
                f.write(f"  標準偏差: {data.std():.6f}\n")
                f.write(f"  最小値: {data.min():.6f}\n")
                f.write(f"  最大値: {data.max():.6f}\n")
                f.write(f"  中央値: {data.median():.6f}\n")

        print(f"サマリーレポートを保存しました: {report_path}")

    def run(self) -> None:
        """ヒストグラム生成を実行する."""
        try:
            print("=== CSVヒストグラム生成を開始します ===")

            # CSVファイルの読み込み
            self.load_csv()

            # 出力ディレクトリの作成
            self.create_output_directory()

            # すべてのヒストグラムを生成
            self.generate_all_histograms()

            # サマリーレポートの生成
            self.generate_summary_report()

            print("=== CSVヒストグラム生成が完了しました ===")

        except Exception as e:
            print(f"エラー: ヒストグラム生成中にエラーが発生しました: {e}")
            sys.exit(1)


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(
        description="CSVファイルからヒストグラムを生成します"
    )
    parser.add_argument("csv_path", help="CSVファイルのパス")
    parser.add_argument(
        "--class-column",
        default="class",
        help="クラス分類に使用する列名 (デフォルト: class)",
    )

    args = parser.parse_args()

    # ヒストグラム生成の実行
    generator = HistogramGenerator(args.csv_path, args.class_column)
    generator.run()


if __name__ == "__main__":
    main()
