# features.py
"""
Module de feature engineering pour le système de trading algorithmique.
Crée 15 features techniques avec prévention du look-ahead bias.
"""
import numpy as np
from statsmodels.tsa.stattools import adfuller


class FeatureEngine:
    """
    Moteur de création et validation de features pour le trading algorithmique.

    Cette classe génère 15 features techniques à partir de données de marché
    (SPY, VIX, TNX) et valide leur stationnarité via le test ADF.

    Attributes:
        selected_features (list): Liste des 15 features utilisées par le modèle

    Example:
        >>> engine = FeatureEngine()
        >>> df_features = engine.create_all_features(df_raw)
        >>> non_stat = engine.check_stationarity(df_features)
    """

    def __init__(self):
        """
        Initialise le moteur avec la liste des features sélectionnées.

        Les features incluent :
        - Tendance : SMA ratios, distance logarithmique
        - Momentum : Rendements multi-périodes
        - Volatilité : Vol_Shock, divergence vol réalisée/implicite
        - Technique : RSI, Bollinger Bands
        - Macro : Momentum des taux (TNX)
        - Interactions : RSI*BB, corrélation Prix/VIX
        """
        self.selected_features = [
            "SMA_Cross_Ratio",
            "SMA200_LogDist",
            "Return_5days",
            "Return_20days",
            "RSI",
            "BB_Position",
            "Vol_Shock",
            "Momentum_20",
            "Dist_From_High",
            "VIX_SMA20_Ratio",
            "VIX_Change_5d",
            "TNX_Momentum",
            "RSI_BB_Interaction",
            "Vol_VIX_Divergence",
            "Price_VIX_Corr",
        ]

    def create_all_features(self, df):
        """
        Crée toutes les features techniques à partir des données brutes.

        IMPORTANT : Toutes les features utilisent un décalage (shift) pour
        éviter le look-ahead bias et garantir la validité du backtest.

        Args:
            df (pd.DataFrame): DataFrame avec SPY_Close, vix_Close, tnx_Close

        Returns:
            pd.DataFrame: DataFrame enrichi avec 15+ features techniques

        Example:
            >>> df_features = engine.create_all_features(df_raw)
            >>> print(df_features.shape)
            (7334, 35)
        """
        df = df.copy()

        # Extraction des colonnes
        close = df["SPY_Close"].squeeze()
        vix = df["vix_Close"].squeeze()
        tnx = df["tnx_Close"].squeeze()

        # Décalage systématique pour éviter le look-ahead bias
        close_lag = close.shift(1)
        vix_lag = vix.shift(1)
        tnx_lag = tnx.shift(1)

        # Moyennes mobiles (tendance)
        sma20 = close_lag.rolling(window=20).mean()
        sma50 = close_lag.rolling(window=50).mean()
        sma200 = close_lag.rolling(window=200).mean()

        df["SMA20"] = sma20
        df["SMA50"] = sma50
        df["SMA200"] = sma200
        df["SMA_Cross_Ratio"] = sma20 / sma50
        df["SMA200_LogDist"] = np.log(close_lag / sma200)

        # Rendements
        daily_return = close_lag.pct_change(periods=1)
        df["Daily_Return"] = daily_return
        df["Return_5days"] = close_lag.pct_change(periods=5)
        df["Return_20days"] = close_lag.pct_change(periods=20)

        # RSI (Relative Strength Index)
        delta = close_lag.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        std20 = close_lag.rolling(20).std()
        upper = sma20 + std20 * 2
        lower = sma20 - std20 * 2
        df["Upper"] = upper
        df["Lower"] = lower
        df["BB_Position"] = (close_lag - lower) / (upper - lower)

        # Volatilité
        vol_10 = daily_return.rolling(window=10).std()
        vol_50 = daily_return.rolling(window=50).std()
        df["Vol_10"] = vol_10
        df["Vol_50"] = vol_50
        df["Vol_Shock"] = vol_10 / vol_50

        # Momentum
        df["Momentum_10"] = close_lag.pct_change(10)
        df["Momentum_20"] = close_lag.pct_change(20)

        # Distance au plus haut annuel
        rolling_max = close_lag.rolling(window=252, min_periods=50).max()
        df["Dist_From_High"] = (close_lag - rolling_max) / rolling_max

        # VIX (Fear Index)
        vix_sma20 = vix_lag.rolling(20).mean()
        df["VIX_SMA20"] = vix_sma20
        df["VIX_SMA20_Ratio"] = vix_lag / vix_sma20
        df["VIX_Change_5d"] = vix_lag.pct_change(5)

        # TNX (Taux d'intérêt 10 ans)
        df["TNX_Change_5d"] = tnx_lag.diff(5)
        df["TNX_Momentum"] = tnx_lag.diff(20) / tnx_lag.shift(20)

        # Features d'interaction
        df["RSI_BB_Interaction"] = df["RSI"] * df["BB_Position"]
        df["Vol_VIX_Divergence"] = vol_10 / (vix_lag / 100)
        df["Price_VIX_Corr"] = close_lag.rolling(20).corr(vix_lag)

        df = df.dropna()
        print(f"✓ {len(self.selected_features)} features créées")
        print(f"  - Observations après nettoyage : {len(df)}")
        return df

    def check_stationarity(self, df, threshold=0.05):
        """
        Vérifie la stationnarité des features via le test ADF.

        Le test ADF (Augmented Dickey-Fuller) teste l'hypothèse nulle de
        présence d'une racine unitaire (non-stationnarité).

        Args:
            df (pd.DataFrame): DataFrame contenant les features
            threshold (float): Seuil de p-value. Défaut: 0.05

        Returns:
            list: Features non-stationnaires (si p-value > threshold)
        """
        print(f"\n{'Feature':<25} | {'p-value':<10} | {'Stationnaire?'}")
        print("-" * 50)
        non_stationary = []

        for col in self.selected_features:
            if col not in df.columns:
                continue
            try:
                result = adfuller(df[col].dropna())
                p_value = result[1]
                is_stationary = p_value < threshold
                status = "✓" if is_stationary else "✗"
                print(f"{col:<25} | {p_value:<10.4f} | {status}")
                if not is_stationary:
                    non_stationary.append(col)
            except Exception:
                pass

        return non_stationary