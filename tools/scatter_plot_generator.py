"""
CSVファイルから散布図を生成するツール.

特徴量が記載されたCSVファイルを読み込み、数値列の全ての組み合わせで
class列による分類で色分けした散布図を生成し、CSVファイルと同じフォルダの
scatterサブフォルダに保存します。
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class ScatterPlotGenerator:
    """CSVファイルから散布図を生成するクラス."""

    def __init__(self, csv_path: str, class_column: str = "class") -> None:
        """
        ScatterPlotGeneratorのコンストラクタ.

        Args:
            csv_path (str): CSVファイルのパス.
            class_column (str): クラス分類に使用する列名. デフォルトは"class".
        """
        self.csv_path = Path(csv_path)
        self.class_column = class_column
        self.output_dir = self.csv_path.parent / "scatter"
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

            if len(self.numeric_columns) < 2:
                raise ValueError("散布図生成には最低2つの数値列が必要です")

            print(f"数値列を特定しました: {len(self.numeric_columns)}列")
            print(f"クラス数: {self.df[self.class_column].nunique()}")

            # 散布図の組み合わせ数を計算
            n_combinations = len(list(combinations(self.numeric_columns, 2)))
            print(f"生成される散布図数: {n_combinations}個")

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
            return list(sns.color_palette("tab10", n_classes))
        else:
            # 10色を超える場合はhslパレットを使用
            return list(sns.color_palette("hsl", n_classes))

    def _calculate_correlation(
        self, x_data: pd.Series, y_data: pd.Series
    ) -> Tuple[float, float]:
        """
        2つのデータ系列間の相関係数とp値を計算する.

        Args:
            x_data (pd.Series): X軸データ.
            y_data (pd.Series): Y軸データ.

        Returns:
            Tuple[float, float]: 相関係数とp値.
        """
        # 欠損値を除去
        valid_data = pd.DataFrame({"x": x_data, "y": y_data}).dropna()
        if len(valid_data) < 3:
            return 0.0, 1.0

        try:
            correlation, p_value = stats.pearsonr(valid_data["x"], valid_data["y"])
            return correlation, p_value
        except Exception:
            return 0.0, 1.0

    def generate_scatter_plot(self, x_column: str, y_column: str) -> None:
        """
        指定された2つの列の散布図を生成する.

        Args:
            x_column (str): X軸に使用する列名.
            y_column (str): Y軸に使用する列名.
        """
        if self.df is None:
            raise ValueError("CSVファイルが読み込まれていません")

        # データの準備
        data = self.df[[x_column, y_column, self.class_column]].dropna()
        if data.empty:
            print(f"警告: 列 '{x_column}' と '{y_column}' にデータがありません")
            return

        # クラスごとのデータを取得
        classes = sorted(data[self.class_column].unique())
        colors = self._get_color_palette(len(classes))

        # 図のサイズと設定
        plt.figure(figsize=(12, 10))
        plt.style.use("default")

        # 各クラスの散布図を描画
        for i, class_name in enumerate(classes):
            class_data = data[data[self.class_column] == class_name]
            plt.scatter(
                class_data[x_column],
                class_data[y_column],
                alpha=0.7,
                label=f"{class_name} (n={len(class_data)})",
                color=colors[i],
                s=50,
                edgecolors="black",
                linewidth=0.5,
            )

        # 回帰直線を追加（全データ）
        if len(data) > 1:
            try:
                # 線形回帰
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    data[x_column], data[y_column]
                )
                line_x = np.array([data[x_column].min(), data[x_column].max()])
                line_y = slope * line_x + intercept
                plt.plot(
                    line_x,
                    line_y,
                    color="red",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=2,
                    label=f"Regression Line (R²={r_value**2:.3f})",
                )
            except Exception:
                pass  # 回帰直線の描画に失敗した場合はスキップ

        # グラフの装飾
        plt.title(
            f"Scatter Plot: {x_column} vs {y_column}", fontsize=16, fontweight="bold"
        )
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # 統計情報を計算
        correlation, p_value = self._calculate_correlation(
            data[x_column], data[y_column]
        )

        # 統計情報をテキストで追加（凡例の下に配置）
        stats_text = f"Total samples: {len(data)}\n"
        stats_text += f"Correlation: {correlation:.3f}\n"
        stats_text += f"P-value: {p_value:.3e}\n"
        stats_text += (
            f"X range: [{data[x_column].min():.3f}, {data[x_column].max():.3f}]\n"
        )
        stats_text += (
            f"Y range: [{data[y_column].min():.3f}, {data[y_column].max():.3f}]"
        )

        plt.text(
            1.05,
            0.7,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            fontsize=10,
        )

        # レイアウトの調整
        plt.tight_layout()

        # ファイル名の生成（特殊文字を除去）
        safe_x_name = "".join(
            c for c in x_column if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_y_name = "".join(
            c for c in y_column if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_x_name = safe_x_name.replace(" ", "_")
        safe_y_name = safe_y_name.replace(" ", "_")
        output_path = self.output_dir / f"{safe_x_name}_vs_{safe_y_name}_scatter.png"

        # 保存
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"散布図を保存しました: {output_path}")

    def generate_all_scatter_plots(self) -> None:
        """すべての数値列の組み合わせで散布図を生成する."""
        if len(self.numeric_columns) < 2:
            print("警告: 散布図生成には最低2つの数値列が必要です")
            return

        # 全ての組み合わせを生成（重複なし）
        combinations_list = list(combinations(self.numeric_columns, 2))
        total_combinations = len(combinations_list)

        print(f"=== {total_combinations}個の散布図を生成します ===")

        for i, (x_col, y_col) in enumerate(combinations_list, 1):
            print(f"進行状況 ({i}/{total_combinations}): {x_col} vs {y_col}")
            try:
                self.generate_scatter_plot(x_col, y_col)
            except Exception as e:
                print(f"エラー: 散布図 '{x_col} vs {y_col}' の生成に失敗しました: {e}")

        print("=== 散布図生成が完了しました ===")

    def generate_correlation_matrix(self) -> None:
        """数値列間の相関行列を生成して保存する."""
        if self.df is None or len(self.numeric_columns) < 2:
            return

        # 数値列のみを抽出
        numeric_data = self.df[self.numeric_columns].dropna()
        if numeric_data.empty:
            print("警告: 相関行列生成用のデータがありません")
            return

        # 相関行列を計算
        correlation_matrix = numeric_data.corr()

        # ヒートマップを作成
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # 上三角をマスク

        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".3f",
            cbar_kws={"shrink": 0.8},
        )

        plt.title("Correlation Matrix", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # 保存
        output_path = self.output_dir / "correlation_matrix.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"相関行列を保存しました: {output_path}")

        # 相関行列をCSVとしても保存
        csv_path = self.output_dir / "correlation_matrix.csv"
        correlation_matrix.to_csv(csv_path)
        print(f"相関行列CSVを保存しました: {csv_path}")

    def generate_summary_report(self) -> None:
        """生成された散布図のサマリーレポートを作成する."""
        if self.df is None:
            return

        report_path = self.output_dir / "scatter_summary.txt"
        combinations_list = list(combinations(self.numeric_columns, 2))

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== 散布図生成サマリーレポート ===\n\n")
            f.write(f"CSVファイル: {self.csv_path}\n")
            f.write(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"データ形状: {self.df.shape}\n")
            f.write(f"クラス列: {self.class_column}\n")
            f.write(f"クラス数: {self.df[self.class_column].nunique()}\n")
            f.write(f"数値列数: {len(self.numeric_columns)}\n")
            f.write(f"生成された散布図数: {len(combinations_list)}個\n\n")

            f.write("=== クラス分布 ===\n")
            class_counts = self.df[self.class_column].value_counts()
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}件\n")

            f.write("\n=== 数値列一覧 ===\n")
            for i, column in enumerate(self.numeric_columns, 1):
                f.write(f"{i:2d}. {column}\n")

            f.write("\n=== 散布図組み合わせ一覧 ===\n")
            for i, (x_col, y_col) in enumerate(combinations_list, 1):
                # 相関係数を計算
                correlation, p_value = self._calculate_correlation(
                    self.df[x_col], self.df[y_col]
                )
                f.write(f"{i:2d}. {x_col} vs {y_col} (相関係数: {correlation:.3f})\n")

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
        """散布図生成を実行する."""
        try:
            print("=== CSV散布図生成を開始します ===")

            # CSVファイルの読み込み
            self.load_csv()

            # 出力ディレクトリの作成
            self.create_output_directory()

            # すべての散布図を生成
            self.generate_all_scatter_plots()

            # 相関行列を生成
            self.generate_correlation_matrix()

            # サマリーレポートの生成
            self.generate_summary_report()

            print("=== CSV散布図生成が完了しました ===")

        except Exception as e:
            print(f"エラー: 散布図生成中にエラーが発生しました: {e}")
            sys.exit(1)


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(description="CSVファイルから散布図を生成します")
    parser.add_argument("csv_path", help="CSVファイルのパス")
    parser.add_argument(
        "--class-column",
        default="class",
        help="クラス分類に使用する列名 (デフォルト: class)",
    )

    args = parser.parse_args()

    # 散布図生成の実行
    generator = ScatterPlotGenerator(args.csv_path, args.class_column)
    generator.run()


if __name__ == "__main__":
    main()
