# data_loader.py
"""
Module de chargement et fusion de données financières depuis Yahoo Finance.
"""
import yfinance as yf


class DataLoader:
    """
    Gestionnaire de téléchargement et fusion de données financières.

    Cette classe permet de télécharger automatiquement des données
    historiques depuis Yahoo Finance et de les fusionner sur leurs
    dates communes via une jointure INNER.

    Attributes:
        period (str): Période de téléchargement (ex: "30y", "5y", "1y")
        interval (str): Intervalle des données (ex: "1d", "1h", "1wk")

    Example:
        >>> loader = DataLoader(period="10y", interval="1d")
        >>> df = loader.load_all_data()
        >>> print(df.shape)
        (2520, 3)
    """

    def __init__(self, period="30y", interval="1d"):
        """
        Initialise le chargeur de données.

        Args:
            period (str): Période de téléchargement. Défaut: "30y"
            interval (str): Intervalle temporel. Défaut: "1d" (journalier)
        """
        self.period = period
        self.interval = interval

    def load_all_data(self):
        """
        Charge et fusionne toutes les sources de données financières.

        Télécharge SPY, VIX et TNX depuis Yahoo Finance puis réalise
        une jointure INNER sur les dates communes pour garantir l'alignement
        temporel des trois séries.

        Returns:
            pd.DataFrame: DataFrame avec index DatetimeIndex et colonnes :
                - SPY_Close (float): Prix de clôture du S&P 500 ETF
                - vix_Close (float): Niveau de l'indice de volatilité VIX
                - tnx_Close (float): Rendement du Treasury 10 ans (en %)

        Raises:
            ValueError: Si le téléchargement échoue ou si aucune date commune

        Example:
            >>> loader = DataLoader(period="5y")
            >>> df = loader.load_all_data()
            >>> print(df.columns.tolist())
            ['SPY_Close', 'vix_Close', 'tnx_Close']

        Note:
            La jointure INNER garantit que seules les dates où les 3 marchés
            sont ouverts sont conservées (pas de valeurs manquantes).
        """
        try:
            print("Téléchargement SPY...")
            spy = yf.download(
                "SPY", period=self.period, interval=self.interval, progress=False
            )
            if spy.empty:
                raise ValueError("Échec du téléchargement de SPY")
            if "Close" in spy.columns:
                spy = spy[["Close"]].rename(columns={"Close": "SPY_Close"})

            print("Téléchargement VIX...")
            vix = yf.download(
                "^VIX", period=self.period, interval=self.interval, progress=False
            )
            if vix.empty:
                raise ValueError("Échec du téléchargement de VIX")
            if "Close" in vix.columns:
                vix = vix[["Close"]].rename(columns={"Close": "vix_Close"})

            print("Téléchargement TNX...")
            tnx = yf.download(
                "^TNX", period=self.period, interval=self.interval, progress=False
            )
            if tnx.empty:
                raise ValueError("Échec du téléchargement de TNX")
            if "Close" in tnx.columns:
                tnx = tnx[["Close"]].rename(columns={"Close": "tnx_Close"})

            df = spy.join([vix, tnx], how="inner").dropna().sort_index()

            if df.empty:
                raise ValueError("Aucune date commune après jointure")

            print(f"✓ {len(df)} observations chargées")
            print(f"  Période: {df.index[0]} → {df.index[-1]}")
            return df

        except Exception as e:
            print(f"❌ Erreur lors du chargement des données : {e}")
            raise