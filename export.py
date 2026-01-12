# export.py
"""
Module d'export des résultats du système de trading algorithmique.

Ce module gère l'export des résultats vers différents formats :
- CSV pour les données détaillées
- Excel pour les rapports multi-feuilles
- TXT pour les rapports synthétiques

Tous les fichiers sont versionnés automatiquement avec timestamp.
"""

import pandas as pd
import numpy as np
import os
import re
import glob
from pathlib import Path
from datetime import datetime


class Exporter:
    """
    Gestionnaire d'exports pour les résultats du système de trading.

    Cette classe centralise tous les exports de résultats vers différents
    formats (CSV, Excel, TXT) avec versionning automatique par timestamp.

    Attributes:
        output_dir (Path): Chemin du dossier de destination
        timestamp (str): Timestamp au format YYYYMMDD_HHMMSS

    Example:
        >>> exporter = Exporter(output_dir="results")
        >>> exporter.export_backtest_results(backtest_df)
        >>> exporter.export_metrics(metrics_df)
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialise l'exporteur et crée le dossier de destination.

        Args:
            output_dir (str): Nom du dossier de destination. Défaut: "results"

        Note:
            Le dossier est créé automatiquement s'il n'existe pas.
            Le timestamp est généré à l'initialisation pour synchroniser
            tous les exports d'une même session.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Timestamp pour versionnage (fixé à l'initialisation)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"✓ Dossier d'export créé : {self.output_dir}")

    def cleanup_old_results(self, keep_last_n: int = 1) -> None:
        """
        Nettoie les anciens résultats et garde uniquement les N dernières exécutions.

        Cette méthode supprime automatiquement les fichiers CSV des exécutions
        précédentes pour éviter l'accumulation de résultats obsolètes.

        Args:
            keep_last_n (int): Nombre d'exécutions à conserver. Défaut: 1

        Example:
            >>> exporter = Exporter(output_dir="results")
            >>> exporter.cleanup_old_results(keep_last_n=1)
            ✓ 28 anciens fichiers supprimés
            ✓ Conservation : 4 fichiers de la dernière exécution

        Note:
            Chaque exécution génère 4 fichiers CSV. Cette méthode supprime
            automatiquement les fichiers au-delà du seuil défini.
        """
        # Récupère tous les fichiers CSV dans le dossier results
        all_files = glob.glob(str(self.output_dir / "*.csv"))

        if not all_files:
            print("  ℹ️  Aucun fichier à nettoyer")
            return

        # Extrait les timestamps uniques des noms de fichiers
        # Pattern regex pour matcher YYYYMMDD_HHMMSS
        pattern = r"_(\d{8}_\d{6})\.csv$"
        timestamps = set()

        for file in all_files:
            filename = Path(file).name
            match = re.search(pattern, filename)
            if match:
                timestamps.add(match.group(1))

        if not timestamps:
            print("  ℹ️  Aucun fichier avec timestamp valide trouvé")
            return

        # Trie les timestamps (du plus ancien au plus récent)
        sorted_timestamps = sorted(list(timestamps))

        # Détermine les timestamps à supprimer
        if len(sorted_timestamps) <= keep_last_n:
            print(
                f"  ℹ️  {len(sorted_timestamps)} exécution(s) trouvée(s), rien à supprimer"
            )
            return

        timestamps_to_delete = sorted_timestamps[:-keep_last_n]

        # Supprime les fichiers correspondant aux anciens timestamps
        deleted_count = 0
        for file in all_files:
            filename = Path(file).name
            for old_timestamp in timestamps_to_delete:
                if old_timestamp in filename:
                    try:
                        os.remove(file)
                        deleted_count += 1
                    except Exception as e:
                        print(f"  ⚠️  Impossible de supprimer {filename}: {e}")
                    break

        kept_count = len(all_files) - deleted_count
        print(f"  ✓ {deleted_count} ancien(s) fichier(s) supprimé(s)")
        print(
            f"  ✓ Conservation : {kept_count} fichier(s) "
            f"de(s) {keep_last_n} dernière(s) exécution(s)"
        )

    def _get_filepath(self, filename: str, extension: str = "csv") -> Path:
        """
        Génère un chemin de fichier avec timestamp.

        Méthode privée utilisée pour créer des noms de fichiers uniques
        en ajoutant le timestamp avant l'extension.

        Args:
            filename (str): Nom du fichier (sans extension)
            extension (str): Extension du fichier (csv, xlsx, txt). Défaut: "csv"

        Returns:
            Path: Chemin complet du fichier avec timestamp

        Example:
            >>> path = self._get_filepath("backtest", "csv")
            >>> print(path)
            results/backtest_20260112_143022.csv
        """
        return self.output_dir / f"{filename}_{self.timestamp}.{extension}"

    def export_backtest_results(
        self, backtest_df: pd.DataFrame, filename: str = "backtest_results"
    ) -> None:
        """
        Export les résultats complets du backtest vers CSV.

        Exporte les colonnes essentielles du backtest : prix, rendements,
        exposition, performances cumulées et coûts de transaction.

        Args:
            backtest_df (pd.DataFrame): DataFrame du backtest
            filename (str): Nom du fichier (sans extension). Défaut: "backtest_results"

        Colonnes exportées:
            - SPY_Close : Prix de clôture du SPY
            - Market_Return : Rendements Buy & Hold
            - Strategy_Return_Net : Rendements stratégie (net de frais)
            - Exposure_final : Exposition appliquée
            - Cum_Market : Performance cumulée marché
            - Cum_Overlay : Performance cumulée overlay
            - Trades : Montant des transactions
            - Transaction_Cost : Coûts de transaction

        Example:
            >>> exporter.export_backtest_results(backtest_df)
            ✓ Backtest exporté : backtest_results_20260112_143022.csv
        """
        filepath = self._get_filepath(filename, "csv")

        # Sélection des colonnes pertinentes
        export_cols = [
            "SPY_Close",
            "Market_Return",
            "Strategy_Return_Net",
            "Exposure_final",
            "Cum_Market",
            "Cum_Overlay",
            "Trades",
            "Transaction_Cost",
        ]

        # Filtrage des colonnes existantes (robustesse)
        available_cols = [col for col in export_cols if col in backtest_df.columns]

        backtest_df[available_cols].to_csv(filepath, index=True)

        print(f"  ✓ Backtest exporté : {filepath.name}")

    def export_metrics(
        self, metrics_df: pd.DataFrame, filename: str = "performance_metrics"
    ) -> None:
        """
        Export les métriques de performance vers CSV.

        Exporte le tableau comparatif des métriques entre Buy & Hold
        et la stratégie overlay (Sharpe, Sortino, Calmar, etc.).

        Args:
            metrics_df (pd.DataFrame): DataFrame avec les métriques comparatives
            filename (str): Nom du fichier. Défaut: "performance_metrics"

        Example:
            >>> metrics_df = pd.DataFrame([metrics_market, metrics_overlay])
            >>> exporter.export_metrics(metrics_df)
            ✓ Métriques exportées : performance_metrics_20260112_143022.csv
        """
        filepath = self._get_filepath(filename, "csv")
        metrics_df.to_csv(filepath, index=False)

        print(f"  ✓ Métriques exportées : {filepath.name}")

    def export_feature_importance(
        self, importance_df: pd.DataFrame, filename: str = "feature_importance"
    ) -> None:
        """
        Export l'importance des features du Random Forest vers CSV.

        Exporte le classement des features par ordre d'importance décroissante.
        Utile pour l'interprétabilité du modèle.

        Args:
            importance_df (pd.DataFrame): DataFrame avec colonnes
                'Feature' et 'Importance'
            filename (str): Nom du fichier. Défaut: "feature_importance"

        Example:
            >>> importance_df = model.get_feature_importance()
            >>> exporter.export_feature_importance(importance_df)
            ✓ Importance des features exportée : feature_importance_20260112_143022.csv
        """
        filepath = self._get_filepath(filename, "csv")
        importance_df.to_csv(filepath, index=False)

        print(f"  ✓ Importance des features exportée : {filepath.name}")

    def export_predictions(
        self,
        dates: pd.DatetimeIndex,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        actual: np.ndarray,
        filename: str = "predictions",
    ) -> None:
        """
        Export les prédictions détaillées du modèle vers CSV.

        Crée un fichier avec les prédictions jour par jour, incluant :
        - Date
        - Probabilité prédite [0-1]
        - Prédiction binaire {0, 1}
        - Valeur réelle {0, 1}
        - Indicateur de prédiction correcte

        Args:
            dates (pd.DatetimeIndex): Index temporel des prédictions
            probabilities (np.ndarray): Probabilités prédites [0-1]
            predictions (np.ndarray): Prédictions binaires {0, 1}
            actual (np.ndarray): Valeurs réelles {0, 1}
            filename (str): Nom du fichier. Défaut: "predictions"

        Example:
            >>> exporter.export_predictions(
            ...     model.X_test.index, probs, preds, model.y_test
            ... )
            ✓ Prédictions exportées : predictions_20260112_143022.csv
        """
        filepath = self._get_filepath(filename, "csv")

        predictions_df = pd.DataFrame(
            {
                "Date": dates,
                "Probability": probabilities,
                "Prediction": predictions,
                "Actual": actual,
                "Correct": (predictions == actual).astype(int),
            }
        )

        predictions_df.to_csv(filepath, index=False)

        print(f"  ✓ Prédictions exportées : {filepath.name}")

    def export_to_excel(
        self,
        backtest_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        importance_df: pd.DataFrame,
        filename: str = "trading_report",
    ) -> None:
        """
        Export complet vers un fichier Excel multi-feuilles.

        Crée un fichier Excel avec 3 feuilles :
        1. Metrics : Métriques de performance comparatives
        2. Feature_Importance : Importance des features
        3. Backtest_Sample : 1000 dernières lignes du backtest

        Args:
            backtest_df (pd.DataFrame): Résultats du backtest
            metrics_df (pd.DataFrame): Métriques de performance
            importance_df (pd.DataFrame): Importance des features
            filename (str): Nom du fichier. Défaut: "trading_report"

        Example:
            >>> exporter.export_to_excel(backtest_df, metrics_df, importance_df)
            ✓ Rapport Excel généré : trading_report_20260112_143022.xlsx

        Note:
            Le backtest est limité aux 1000 dernières lignes pour éviter
            des fichiers Excel trop volumineux (>30 ans de données).
        """
        filepath = self._get_filepath(filename, "xlsx")

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Feuille 1 : Métriques comparatives
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

            # Feuille 2 : Importance des features
            importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)

            # Feuille 3 : Échantillon du backtest (1000 dernières lignes)
            backtest_cols = [
                "SPY_Close",
                "Market_Return",
                "Strategy_Return_Net",
                "Exposure_final",
                "Cum_Market",
                "Cum_Overlay",
            ]
            
            # Filtrage des colonnes existantes
            available_cols = [col for col in backtest_cols if col in backtest_df.columns]
            
            backtest_export = backtest_df[available_cols].tail(1000)
            backtest_export.to_excel(writer, sheet_name="Backtest_Sample", index=True)

        print(f"  ✓ Rapport Excel généré : {filepath.name}")

    def export_summary_report(
        self,
        metrics_df: pd.DataFrame,
        backtest_df: pd.DataFrame,
        importance_df: pd.DataFrame,
        filename: str = "summary_report",
    ) -> None:
        """
        Crée un rapport textuel synthétique (.txt).

        Génère un rapport lisible avec :
        1. Métriques de performance
        2. Top 10 features par importance
        3. Statistiques du backtest
        4. Performances finales et surperformance

        Args:
            metrics_df (pd.DataFrame): Métriques de performance
            backtest_df (pd.DataFrame): Résultats du backtest
            importance_df (pd.DataFrame): Importance des features
            filename (str): Nom du fichier. Défaut: "summary_report"

        Example:
            >>> exporter.export_summary_report(metrics_df, backtest_df, importance_df)
            ✓ Rapport de synthèse généré : summary_report_20260112_143022.txt

        Note:
            Idéal pour une lecture rapide des résultats sans ouvrir Excel.
        """
        filepath = self._get_filepath(filename, "txt")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("RAPPORT DE SYNTHÈSE - SYSTÈME DE TRADING SPY\n")
            f.write("=" * 70 + "\n\n")

            # Date de génération
            f.write(
                f"Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Section 1 : Métriques de performance
            f.write("-" * 70 + "\n")
            f.write("1. MÉTRIQUES DE PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            f.write(metrics_df.to_string(index=False))
            f.write("\n\n")

            # Section 2 : Top 10 features
            f.write("-" * 70 + "\n")
            f.write("2. TOP 10 FEATURES PAR IMPORTANCE\n")
            f.write("-" * 70 + "\n")
            f.write(importance_df.head(10).to_string(index=False))
            f.write("\n\n")

            # Section 3 : Statistiques du backtest
            f.write("-" * 70 + "\n")
            f.write("3. STATISTIQUES DU BACKTEST\n")
            f.write("-" * 70 + "\n")
            f.write(f"Période : {backtest_df.index[0]} → {backtest_df.index[-1]}\n")
            f.write(f"Nombre de jours : {len(backtest_df)}\n")
            f.write(
                f"Exposition moyenne : {backtest_df['Exposure_final'].mean():.2%}\n"
            )
            f.write(
                f"Nombre de trades (notionnel) : {backtest_df['Trades'].sum():.2f}\n"
            )
            f.write(
                f"Coût total transactions : {backtest_df['Transaction_Cost'].sum():.4%}\n"
            )
            f.write("\n")

            # Performances finales
            final_market = backtest_df["Cum_Market"].iloc[-1]
            final_overlay = backtest_df["Cum_Overlay"].iloc[-1]

            f.write(f"Valeur finale Buy & Hold : ${final_market:.2f}\n")
            f.write(f"Valeur finale Overlay : ${final_overlay:.2f}\n")
            f.write(
                f"Surperformance : {((final_overlay / final_market - 1) * 100):.2f}%\n"
            )
            f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("FIN DU RAPPORT\n")
            f.write("=" * 70 + "\n")

        print(f"  ✓ Rapport de synthèse généré : {filepath.name}")

    def export_all(
        self,
        backtest_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        importance_df: pd.DataFrame,
        probs: np.ndarray,
        preds: np.ndarray,
        y_test: np.ndarray,
        dates: pd.DatetimeIndex,
    ) -> None:
        """
        Export tous les résultats en une seule fois.

        Méthode pratique qui exécute tous les exports disponibles :
        - Résultats de backtest (CSV)
        - Métriques (CSV)
        - Importance des features (CSV)
        - Prédictions détaillées (CSV)
        - Rapport Excel multi-feuilles
        - Rapport texte synthétique

        Args:
            backtest_df (pd.DataFrame): Résultats du backtest
            metrics_df (pd.DataFrame): Métriques de performance
            importance_df (pd.DataFrame): Importance des features
            probs (np.ndarray): Probabilités prédites
            preds (np.ndarray): Prédictions binaires
            y_test (np.ndarray): Valeurs réelles
            dates (pd.DatetimeIndex): Dates des prédictions

        Example:
            >>> exporter.export_all(
            ...     backtest_df, metrics_df, importance_df,
            ...     probs, preds, model.y_test, model.X_test.index
            ... )
            Export de tous les résultats...
            ✓ Backtest exporté : ...
            ✓ Métriques exportées : ...
            ✓ Tous les exports terminés dans : results
        """
        print("\nExport de tous les résultats...")

        self.export_backtest_results(backtest_df)
        self.export_metrics(metrics_df)
        self.export_feature_importance(importance_df)
        self.export_predictions(dates, probs, preds, y_test)
        self.export_to_excel(backtest_df, metrics_df, importance_df)
        self.export_summary_report(metrics_df, backtest_df, importance_df)

        print(f"\n✓ Tous les exports terminés dans : {self.output_dir}")


if __name__ == "__main__":
    # Test du module
    print("Test du module export...")
    
    exporter = Exporter(output_dir="test_results")

    # Création de données factices pour test
    test_df = pd.DataFrame(
        {
            "SPY_Close": [100, 101, 102],
            "Market_Return": [0.01, 0.01, 0.01],
            "Strategy_Return_Net": [0.02, 0.02, 0.02],
            "Exposure_final": [0.5, 0.5, 0.5],
            "Cum_Market": [1.0, 1.01, 1.02],
            "Cum_Overlay": [1.0, 1.02, 1.04],
            "Trades": [0.1, 0.1, 0.1],
            "Transaction_Cost": [0.001, 0.001, 0.001],
        }
    )

    test_metrics = pd.DataFrame(
        {
            "Stratégie": ["Test"],
            "Rendement total": ["10%"],
            "Sharpe ratio": ["1.5"],
        }
    )

    test_importance = pd.DataFrame(
        {
            "Feature": ["Feature1", "Feature2"],
            "Importance": [0.5, 0.3],
        }
    )

    exporter.export_backtest_results(test_df)
    exporter.export_metrics(test_metrics)
    exporter.export_feature_importance(test_importance)

    print("\n✓ Test du module export réussi")