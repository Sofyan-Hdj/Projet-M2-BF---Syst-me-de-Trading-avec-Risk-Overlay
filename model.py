# model.py
"""
Module de machine learning et backtesting pour le système de trading algorithmique.

Ce module implémente un modèle Random Forest avec validation croisée time-series
pour prédire la direction du marché, puis applique un risk overlay dynamique
pour gérer l'exposition au risque.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score


class TradingModel:
    """
    Modèle de trading algorithmique avec Random Forest et risk overlay.

    Cette classe gère l'entraînement d'un modèle Random Forest pour prédire
    la direction du marché (hausse/baisse), puis applique un système de risk
    overlay sophistiqué pour gérer dynamiquement l'exposition au risque.

    Le risk overlay combine 4 mécanismes :
    - Exposition graduelle basée sur la conviction du modèle
    - Filtre de tendance (SMA50) pour éviter les faux signaux
    - Lissage temporel pour réduire le churn
    - Socle d'exposition (Buy & Hold partiel) pour éviter le timing risk

    Attributes:
        features (list): Liste des noms de features utilisées
        test_size (float): Proportion du test set (0.0 à 1.0)
        random_state (int): Seed pour reproductibilité
        model (RandomForestClassifier): Modèle ML entraîné
        scaler (StandardScaler): Normaliseur des features
        X_train, X_test (pd.DataFrame): Features train/test
        y_train, y_test (pd.Series): Target train/test
        X_train_scaled, X_test_scaled (np.ndarray): Features normalisées

    Example:
        >>> model = TradingModel(features=selected_features)
        >>> model.prepare_data(df_features)
        >>> opt_low, opt_high = model.optimize_thresholds()
        >>> accuracy = model.train()
        >>> probs, preds = model.predict()
        >>> backtest_df = model.run_backtest(df, probs, opt_low, opt_high)
    """

    def __init__(self, features, test_size=0.20, random_state=42):
        """
        Initialise le modèle de trading.

        Args:
            features (list): Noms des features à utiliser pour l'entraînement
            test_size (float): Proportion du test set. Défaut: 0.20 (20%)
            random_state (int): Seed pour reproductibilité. Défaut: 42
        """
        self.features = features
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None

    def prepare_data(self, df):
        """
        Prépare les données pour l'entraînement (split temporel + normalisation).

        Effectue un split temporel (pas aléatoire) pour respecter la nature
        séquentielle des séries financières. Les données anciennes servent
        à l'entraînement, les données récentes au test. Les features sont
        ensuite normalisées via StandardScaler.

        Args:
            df (pd.DataFrame): DataFrame avec features et colonne 'Direction'

        Note:
            Le split temporel est CRUCIAL en finance pour éviter le look-ahead
            bias. Ne jamais utiliser train_test_split() avec shuffle=True !
        """
        X = df[self.features]
        y = df["Direction"]
        split_idx = int(len(df) * (1 - self.test_size))

        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("✓ Données préparées")
        print(f"  - Train : {len(self.X_train)} observations")
        print(f"  - Test  : {len(self.X_test)} observations")
        print(f"  - Distribution train : {self.y_train.value_counts().to_dict()}")

    def optimize_thresholds(self, n_splits=5):
        """
        Optimise les seuils de décision via validation croisée time-series.

        Utilise TimeSeriesSplit pour respecter l'ordre temporel des données.
        Entraîne un modèle sur chaque fold et collecte les probabilités
        prédites pour optimiser les seuils bas/haut maximisant la séparation
        des classes.

        Args:
            n_splits (int): Nombre de folds pour la cross-validation. Défaut: 5

        Returns:
            tuple: (seuil_bas, seuil_haut) optimisés
                - seuil_bas : en-dessous, exposition = 0% (cash)
                - seuil_haut : au-dessus, exposition = 100% (full invest)
                - entre les deux : exposition = 50% (prudent)

        Example:
            >>> opt_low, opt_high = model.optimize_thresholds(n_splits=5)
            >>> print(f"Seuils optimisés : {opt_low} / {opt_high}")
            Seuils optimisés : 0.58 / 0.63
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_val_probs = []

        for i, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            X_train_cv = self.X_train.iloc[train_idx]
            X_val_cv = self.X_train.iloc[val_idx]
            y_train_cv = self.y_train.iloc[train_idx]

            scaler_cv = StandardScaler()
            X_train_cv_scaled = scaler_cv.fit_transform(X_train_cv)
            X_val_cv_scaled = scaler_cv.transform(X_val_cv)

            model_cv = RandomForestClassifier(
                n_estimators=500, min_samples_leaf=10, random_state=self.random_state
            )
            model_cv.fit(X_train_cv_scaled, y_train_cv)

            probs_val = model_cv.predict_proba(X_val_cv_scaled)[:, 1]
            all_val_probs.extend(probs_val)

            print(f"  Fold {i+1}/{n_splits} terminé")

        return self._search_best_thresholds(all_val_probs)

    def _search_best_thresholds(self, probs):
        """
        Recherche les meilleurs seuils via grid search.

        Teste différentes combinaisons de seuils (bas, haut) et sélectionne
        celle maximisant la séparation entre convictions fortes et faibles.

        Args:
            probs (list): Liste des probabilités prédites sur validation

        Returns:
            tuple: (seuil_bas, seuil_haut) optimaux

        Note:
            L'objectif est de maximiser : n_high + n_low - 0.5 * n_mid
            pour favoriser les décisions claires (très confiant ou pas du tout).
        """
        best_spread = 0
        best_thresholds = (0.50, 0.60)
        probs_array = np.array(probs)

        for low in np.linspace(0.48, 0.58, 11):
            for high in np.linspace(low + 0.05, 0.70, 11):
                n_high = (probs_array > high).sum()
                n_low = (probs_array < low).sum()
                n_mid = len(probs_array) - n_high - n_low
                spread = n_high + n_low - 0.5 * n_mid

                if spread > best_spread:
                    best_spread = spread
                    best_thresholds = (round(low, 3), round(high, 3))

        return best_thresholds

    def train(self):
        """
        Entraîne le modèle Random Forest sur les données d'entraînement.

        Utilise un Random Forest avec 1000 arbres et class_weight='balanced'
        pour gérer le déséquilibre potentiel entre hausses et baisses.

        Returns:
            float: Précision (accuracy) sur le test set

        Raises:
            ValueError: Si prepare_data() n'a pas été appelé

        Note:
            Hyperparamètres choisis :
            - n_estimators=1000 : nombreux arbres pour stabilité
            - max_features='sqrt' : réduit la corrélation entre arbres
            - min_samples_leaf=10 : évite le sur-apprentissage
            - class_weight='balanced' : compense le déséquilibre de classes
            - n_jobs=-1 : utilise tous les CPU disponibles
        """
        if self.X_train_scaled is None:
            raise ValueError(
                "Les données doivent être préparées. Appelez prepare_data() d'abord."
            )

        self.model = RandomForestClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(self.X_train_scaled, self.y_train)
        y_pred = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)

        print(f"\n✓ Modèle entraîné")
        print(f"  - Précision : {accuracy:.2%}")
        print(f"  - Arbres : {self.model.n_estimators}")

        return accuracy

    def predict(self, threshold=0.50):
        """
        Génère les prédictions sur le test set.

        Args:
            threshold (float): Seuil de décision. Défaut: 0.50

        Returns:
            tuple: (probabilités, prédictions)
                - probabilités (np.ndarray): Probabilités de hausse [0-1]
                - prédictions (np.ndarray): Prédictions binaires {0, 1}

        Raises:
            ValueError: Si le modèle n'a pas été entraîné

        Example:
            >>> probs, preds = model.predict(threshold=0.50)
            >>> print(f"Probabilité moyenne : {probs.mean():.2f}")
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné. Appelez train() d'abord.")

        probs = self.model.predict_proba(self.X_test_scaled)[:, 1]
        preds = (probs > threshold).astype(int)

        return probs, preds

    def get_feature_importance(self):
        """
        Retourne l'importance des features du modèle entraîné.

        Returns:
            pd.DataFrame: DataFrame avec colonnes 'Feature' et 'Importance',
                         trié par importance décroissante

        Raises:
            ValueError: Si le modèle n'a pas été entraîné

        Example:
            >>> importance_df = model.get_feature_importance()
            >>> print(importance_df.head(5))
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné. Appelez train() d'abord.")

        return pd.DataFrame(
            {"Feature": self.features, "Importance": self.model.feature_importances_}
        ).sort_values("Importance", ascending=False)

    def run_backtest(
        self,
        df,
        probs,
        threshold_low,
        threshold_high,
        base_exposure=0.30,
        smoothing_window=5,
        transaction_cost=0.001,
    ):
        """
        Execute le backtest avec risk overlay dynamique à 4 couches.

        Applique une stratégie sophistiquée d'exposition au risque basée sur :

        1. Exposition brute (conviction du modèle) :
           - Haute conviction (prob > threshold_high) : 100% exposé
           - Conviction moyenne (threshold_low ≤ prob ≤ threshold_high) : 50%
           - Faible conviction (prob < threshold_low) : 0% (cash)

        2. Filtre de tendance :
           - Coupe l'exposition si prix < SMA50
           - Protection contre les faux signaux en tendance baissière

        3. Lissage temporel :
           - Moyenne mobile sur N jours pour réduire le churn
           - Évite les ajustements trop fréquents et coûteux

        4. Socle d'exposition :
           - Maintient X% toujours investi (Buy & Hold partiel)
           - Évite le risque de manquer le marché (timing risk)

        Args:
            df (pd.DataFrame): DataFrame avec features, prix et SMA50
            probs (np.ndarray): Probabilités prédites par le modèle
            threshold_low (float): Seuil bas (< cash, > partiel)
            threshold_high (float): Seuil haut (> full invest, < partiel)
            base_exposure (float): Exposition minimale. Défaut: 0.30 (30%)
            smoothing_window (int): Fenêtre de lissage. Défaut: 5 jours
            transaction_cost (float): Coût par transaction. Défaut: 0.001 (10 bps)

        Returns:
            pd.DataFrame: Résultats du backtest avec colonnes :
                - Market_Return : Rendements Buy & Hold journaliers
                - Strategy_Return_Net : Rendements stratégie net de frais
                - Exposure_final : Exposition appliquée (0.0 à 1.0)
                - Cum_Market : Performance cumulée Buy & Hold
                - Cum_Overlay : Performance cumulée avec overlay

        Example:
            >>> backtest_df = model.run_backtest(df, probs, 0.58, 0.63)
            >>> final_perf = backtest_df['Cum_Overlay'].iloc[-1]
            >>> print(f"Performance finale : {final_perf:.2f}x")

        Note:
            Les coûts de transaction sont proportionnels au changement
            d'exposition : 10 bps pour un changement de 100% d'exposition.
        """
        split_idx = int(len(df) * (1 - self.test_size))
        backtest_df = df.iloc[split_idx:].copy()

        # Extraction propre des Series
        spy_close = backtest_df["SPY_Close"].squeeze()
        sma50 = backtest_df["SMA50"].squeeze()

        # Rendements du marché (Buy & Hold)
        backtest_df["Market_Return"] = spy_close.pct_change()
        probs_series = pd.Series(probs, index=backtest_df.index)

        # Couche 1 : Exposition brute basée sur la conviction
        backtest_df["Exposure_raw"] = np.select(
            [
                probs_series > threshold_high,  # Haute conviction : full invest
                (probs_series >= threshold_low)
                & (probs_series <= threshold_high),  # Conviction moyenne : prudent
                probs_series < threshold_low,  # Faible conviction : cash
            ],
            [1.0, 0.5, 0.0],
            default=0.0,
        )

        # Couche 2 : Filtre de tendance (SMA50)
        # Protection : ne pas investir en tendance baissière claire
        backtest_df["Trend_Filter"] = (spy_close > sma50).astype(int)

        # Couche 3 : Application avec décalage (décision T appliquée en T+1)
        # Simule un délai d'exécution réaliste
        backtest_df["Exposure"] = (
            (backtest_df["Exposure_raw"] * backtest_df["Trend_Filter"])
            .shift(1)
            .fillna(0.0)
        )

        # Couche 4 : Lissage temporel pour réduire le churn
        backtest_df["Exposure_smooth"] = (
            backtest_df["Exposure"].rolling(window=smoothing_window, min_periods=1).mean()
        )

        # Couche 5 : Socle d'exposition (Buy & Hold partiel)
        # Formule : 30% fixe + 70% variable selon le modèle
        backtest_df["Exposure_final"] = (
            base_exposure + (1 - base_exposure) * backtest_df["Exposure_smooth"]
        )

        # Calcul des transactions (changements d'exposition)
        backtest_df["Trades"] = backtest_df["Exposure_final"].diff().abs()

        # Rendements de la stratégie (bruts)
        backtest_df["Strategy_Return"] = (
            backtest_df["Market_Return"] * backtest_df["Exposure_final"]
        )

        # Coûts de transaction (proportionnels aux trades)
        backtest_df["Transaction_Cost"] = backtest_df["Trades"] * transaction_cost

        # Rendements nets de frais
        backtest_df["Strategy_Return_Net"] = (
            backtest_df["Strategy_Return"] - backtest_df["Transaction_Cost"]
        )

        # Performance cumulée
        backtest_df["Cum_Market"] = (1 + backtest_df["Market_Return"].fillna(0)).cumprod()
        backtest_df["Cum_Overlay"] = (
            1 + backtest_df["Strategy_Return_Net"].fillna(0)
        ).cumprod()

        return backtest_df

    @staticmethod
    def calculate_metrics(returns, name):
        """
        Calcule les métriques de performance pour une stratégie.

        Métriques calculées :
        - Rendement total : Performance globale sur la période
        - Rendement annualisé : CAGR (Compound Annual Growth Rate)
        - Volatilité annualisée : Écart-type des rendements annualisé
        - Sharpe ratio : Rendement ajusté au risque (reward/risk)
        - Sortino ratio : Comme Sharpe mais pénalise uniquement volatilité baissière
        - Calmar ratio : Rendement / Max drawdown
        - Max drawdown : Perte maximale depuis le sommet
        - Win rate : Proportion de jours positifs

        Args:
            returns (pd.Series): Série de rendements journaliers
            name (str): Nom de la stratégie

        Returns:
            dict: Dictionnaire avec toutes les métriques formatées en %

        Example:
            >>> metrics = TradingModel.calculate_metrics(returns, "Ma Stratégie")
            >>> print(metrics['Sharpe ratio'])
            '0.95'
        """
        returns = returns.dropna()

        # Rendement total
        total_return = (1 + returns).prod() - 1

        # Rendement annualisé (méthode CAGR)
        mean_daily_return = returns.mean()
        annualized_return = (1 + mean_daily_return) ** 252 - 1

        # Volatilité annualisée
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio (sans taux sans risque)
        sharpe = (
            (mean_daily_return / returns.std() * np.sqrt(252))
            if returns.std() > 0
            else np.nan
        )

        # Sortino ratio (pénalise uniquement volatilité baissière)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (mean_daily_return * 252 / downside_std) if downside_std > 0 else np.nan

        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Calmar ratio (rendement / abs(max_drawdown))
        calmar = (annualized_return / abs(max_drawdown)) if max_drawdown != 0 else np.nan

        # Win rate (% de jours positifs)
        win_rate = (returns > 0).sum() / len(returns)

        return {
            "Stratégie": name,
            "Rendement total": f"{total_return:.2%}",
            "Rendement annualisé": f"{annualized_return:.2%}",
            "Volatilité annualisée": f"{volatility:.2%}",
            "Sharpe ratio": f"{sharpe:.2f}",
            "Sortino ratio": f"{sortino:.2f}",
            "Calmar ratio": f"{calmar:.2f}",
            "Max drawdown": f"{max_drawdown:.2%}",
            "Win rate": f"{win_rate:.2%}",
        }

    @staticmethod
    def analyze_crisis_periods(backtest_df):
        """
        Analyse la performance durant les périodes de crise du marché.

        Identifie les périodes où le drawdown du marché dépasse 10% et
        compare la performance de la stratégie overlay vs Buy & Hold
        pendant ces phases de stress.

        Args:
            backtest_df (pd.DataFrame): Résultats du backtest

        Returns:
            pd.DataFrame: Statistiques comparatives crise vs hors-crise

        Example:
            >>> crisis_stats = TradingModel.analyze_crisis_periods(backtest_df)
            >>> print(crisis_stats)
                        Période  Jours  Rendement Market  Rendement Overlay
            Crise (DD > 10%)     322          -0.068%            -0.021%
            Hors crise          1145           0.111%             0.037%
        """
        # Calcul du drawdown du marché
        market_dd = backtest_df["Cum_Market"] / backtest_df["Cum_Market"].cummax() - 1

        # Masque des périodes de crise (drawdown > 10%)
        crisis_mask = market_dd < -0.10

        return pd.DataFrame(
            {
                "Période": ["Crise (DD > 10%)", "Hors crise"],
                "Jours": [crisis_mask.sum(), (~crisis_mask).sum()],
                "Rendement Market": [
                    f"{backtest_df.loc[crisis_mask, 'Market_Return'].mean():.3%}",
                    f"{backtest_df.loc[~crisis_mask, 'Market_Return'].mean():.3%}",
                ],
                "Rendement Overlay": [
                    f"{backtest_df.loc[crisis_mask, 'Strategy_Return_Net'].mean():.3%}",
                    f"{backtest_df.loc[~crisis_mask, 'Strategy_Return_Net'].mean():.3%}",
                ],
            }
        )