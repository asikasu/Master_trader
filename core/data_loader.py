import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO


class DataLoader:

    def __init__(self, data_dir="data"):

        self.project_root = Path(__file__).resolve().parents[1]
        self.data_dir = Path(data_dir)

        if not self.data_dir.is_absolute():
            self.data_dir = (self.project_root / self.data_dir).resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

    # =====================
    # CLEAN COLUMN NAMES
    # =====================

    def _fix_dataframe(self, df):

        # Хэрэв бүх өгөгдөл нэг баганад байвал
        if len(df.columns) == 1:

            col = df.columns[0]

            text = "\n".join(
                [col]
                + df.iloc[:, 0]
                .astype(str)
                .tolist()
            )

            df = pd.read_csv(
                StringIO(text),
                sep="\t"
            )

        df.columns = (
            df.columns
            .str.strip()
            .str.replace("<", "", regex=False)
            .str.replace(">", "", regex=False)
            .str.upper()
        )

        required = [
            "OPEN",
            "HIGH",
            "LOW",
            "CLOSE"
        ]

        missing = [
            c
            for c in required
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        return df

    # =====================
    # DEMO DATA FALLBACK
    # =====================

    def _create_demo_data(self, file):

        print(
            f"[WARNING] No dataset found at {file}. Creating demo dataset instead."
        )

        rng = np.random.default_rng(42)
        n = 600
        base = 100 + np.cumsum(rng.normal(0, 0.5, n))

        df = pd.DataFrame({
            "DATE": pd.date_range("2020-01-01", periods=n, freq="h").strftime("%Y%m%d"),
            "TIME": pd.date_range("2020-01-01", periods=n, freq="h").strftime("%H%M%S"),
            "OPEN": base,
            "HIGH": base + 0.8,
            "LOW": base - 0.8,
            "CLOSE": base + np.where(rng.random(n) < 0.5, -0.3, 0.3),
        })

        self.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(file, index=False)

        return df

    # =====================
    # GENERIC PARQUET LOADER
    # =====================

    def _load_parquet(
        self,
        filename
    ):

        file = (
            self.data_dir /
            filename
        )

        if not file.exists():
            df = self._create_demo_data(file)
        else:
            print(f"Loading {filename}...")
            df = pd.read_parquet(file)

        df = self._fix_dataframe(df)

        print(f"Loaded {len(df):,} rows")

        return df

    # =====================
    # GOLD
    # =====================

    def load_gold_data(self):

        return self._load_parquet(
            "MASTER_GOLD_10Y.parquet"
        )

    # =====================
    # EURO
    # =====================

    def load_euro_data(self):

        return self._load_parquet(
            "MASTER_EURO_10Y.parquet"
        )