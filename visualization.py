# visualization.py
"""
Module de visualisation des résultats du modèle de trading algorithmique.

Ce module génère 6 types de graphiques professionnels pour analyser
les performances du modèle et du risk overlay :
- Matrice de confusion (évaluation du modèle)
- Importance des features (interprétabilité)
- Performance cumulée comparée (résultats)
- Distribution des probabilités (calibration)
- Évolution de l'exposition (gestion du risque)
- Distribution des rendements (analyse statistique)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional


class Visualizer:
    """
    Gestionnaire de visualisations pour le système de trading.

    Cette classe crée des graphiques professionnels avec matplotlib et seaborn
    pour analyser les performances du modèle et du risk overlay.

    Attributes:
        figsize_default (tuple): Taille par défaut pour grands graphiques (14, 8)
        figsize_small (tuple): Taille par défaut pour petits graphiques (10, 6)

    Example:
        >>> viz = Visualizer()
        >>> viz.plot_confusion_matrix(y_true, y_pred, threshold=0.50)
        >>> viz.plot_performance_comparison(backtest_df)
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialise le visualiseur avec un style matplotlib.

        Args:
            style (str): Style matplotlib à utiliser.
                Défaut: "seaborn-v0_8-darkgrid"
                Fallback: "default" si le style n'existe pas

        Note:
            Configure également la palette de couleurs seaborn sur "husl"
            pour des couleurs harmonieuses.
        """
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("default")

        # Configuration par défaut
        self.figsize_default = (14, 8)
        self.figsize_small = (10, 6)
        sns.set_palette("husl")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Affiche la matrice de confusion du modèle.

        Visualise les prédictions correctes et incorrectes sous forme de
        heatmap. Permet d'évaluer la précision du modèle sur chaque classe
        (hausse/baisse).

        Args:
            y_true (np.ndarray): Valeurs réelles (0=Baisse, 1=Hausse)
            y_pred (np.ndarray): Prédictions du modèle (0=Baisse, 1=Hausse)
            threshold (float): Seuil de décision utilisé
            save_path (Optional[str]): Chemin pour sauvegarder l'image

        Example:
            >>> viz.plot_confusion_matrix(model.y_test, preds, threshold=0.50)
            >>> # Affiche une heatmap 2x2 avec les prédictions

        Note:
            Diagonale : prédictions correctes
            Hors-diagonale : erreurs (faux positifs/négatifs)
        """
        from sklearn.metrics import confusion_matrix

        conf_matrix = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=self.figsize_small)
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Baisse", "Hausse"],
            yticklabels=["Baisse", "Hausse"],
            cbar_kws={"label": "Nombre de prédictions"},
        )

        plt.xlabel("Prédictions", fontsize=12, fontweight="bold")
        plt.ylabel("Réalité", fontsize=12, fontweight="bold")
        plt.title(
            f"Matrice de Confusion (Seuil = {threshold})",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Affiche l'importance des features du modèle Random Forest.

        Visualise les features les plus importantes pour les prédictions
        du modèle sous forme de barres horizontales avec gradient de couleur.

        Args:
            importance_df (pd.DataFrame): DataFrame avec colonnes
                'Feature' et 'Importance'
            top_n (int): Nombre de features à afficher. Défaut: 15
            save_path (Optional[str]): Chemin pour sauvegarder l'image

        Example:
            >>> importance_df = model.get_feature_importance()
            >>> viz.plot_feature_importance(importance_df, top_n=10)

        Note:
            Plus une feature est importante, plus elle influence
            les décisions du modèle. Utile pour l'interprétabilité.
        """
        # Sélection des top N features
        plot_df = importance_df.head(top_n).copy()

        plt.figure(figsize=self.figsize_small)

        # Barres horizontales avec gradient de couleur
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(plot_df)))

        plt.barh(
            plot_df["Feature"],
            plot_df["Importance"],
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )

        plt.xlabel("Importance", fontsize=12, fontweight="bold")
        plt.ylabel("Feature", fontsize=12, fontweight="bold")
        plt.title(
            f"Top {top_n} Features par Importance",
            fontsize=14,
            fontweight="bold",
        )

        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_performance_comparison(
        self, backtest_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Compare les performances Buy & Hold vs Risk Overlay.

        Génère 2 graphiques superposés :
        1. Performance cumulée ($1 initial) pour comparer les rendements
        2. Drawdowns comparés pour évaluer la gestion du risque

        Args:
            backtest_df (pd.DataFrame): Résultats du backtest avec colonnes
                'Cum_Market', 'Cum_Overlay' (performances cumulées)
            save_path (Optional[str]): Chemin pour sauvegarder l'image

        Example:
            >>> backtest_df = model.run_backtest(...)
            >>> viz.plot_performance_comparison(backtest_df)

        Note:
            Les annotations affichent les valeurs finales et max drawdowns
            pour faciliter la comparaison.
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize_default)

        # 1. Performance cumulée
        ax1 = axes[0]

        ax1.plot(
            backtest_df.index,
            backtest_df["Cum_Market"],
            label="Buy & Hold SPY",
            color="#95a5a6",
            linewidth=2.5,
            alpha=0.8,
        )

        ax1.plot(
            backtest_df.index,
            backtest_df["Cum_Overlay"],
            label="Buy & Hold + RF Overlay (net)",
            color="#3498db",
            linewidth=2.5,
        )

        ax1.set_title(
            "Performance cumulée : Buy & Hold vs Buy & Hold + Overlay de risque",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax1.set_ylabel("Valeur du portefeuille ($1 initial)", fontsize=11)
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Ajout des valeurs finales
        final_market = backtest_df["Cum_Market"].iloc[-1]
        final_overlay = backtest_df["Cum_Overlay"].iloc[-1]

        ax1.text(
            0.02,
            0.98,
            f"Final: Market ${final_market:.2f} | Overlay ${final_overlay:.2f}",
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 2. Drawdowns comparés
        ax2 = axes[1]

        market_dd = (
            backtest_df["Cum_Market"] / backtest_df["Cum_Market"].cummax() - 1
        ) * 100

        overlay_dd = (
            backtest_df["Cum_Overlay"] / backtest_df["Cum_Overlay"].cummax() - 1
        ) * 100

        ax2.fill_between(
            backtest_df.index,
            market_dd,
            0,
            color="#95a5a6",
            alpha=0.4,
            label="Buy & Hold SPY",
        )

        ax2.fill_between(
            backtest_df.index,
            overlay_dd,
            0,
            color="#3498db",
            alpha=0.4,
            label="Buy & Hold + RF Overlay",
        )

        ax2.set_title("Drawdowns comparés", fontsize=14, fontweight="bold", pad=15)
        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.legend(loc="lower left", fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Ajout des max drawdowns
        max_dd_market = market_dd.min()
        max_dd_overlay = overlay_dd.min()

        ax2.text(
            0.02,
            0.02,
            f"Max DD: Market {max_dd_market:.2f}% | Overlay {max_dd_overlay:.2f}%",
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_probability_distribution(
        self,
        probs: np.ndarray,
        threshold_low: float,
        threshold_high: float,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Affiche la distribution des probabilités prédites par le modèle.

        Visualise comment les probabilités sont distribuées et où se situent
        les seuils de décision. Permet de vérifier la calibration du modèle.

        Args:
            probs (np.ndarray): Probabilités prédites [0-1]
            threshold_low (float): Seuil bas (exposition réduite)
            threshold_high (float): Seuil haut (exposition complète)
            save_path (Optional[str]): Chemin pour sauvegarder l'image

        Example:
            >>> viz.plot_probability_distribution(probs, 0.58, 0.63)

        Note:
            Une bonne calibration montre une distribution étalée avec
            des zones claires de haute/basse conviction.
        """
        plt.figure(figsize=self.figsize_small)

        # Histogramme
        plt.hist(
            probs,
            bins=50,
            alpha=0.75,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

        # Lignes de seuil
        plt.axvline(
            threshold_low,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Seuil bas ({threshold_low})",
        )

        plt.axvline(
            threshold_high,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Seuil haut ({threshold_high})",
        )

        plt.xlabel("Probabilité prédite de hausse", fontsize=12, fontweight="bold")
        plt.ylabel("Fréquence", fontsize=12, fontweight="bold")
        plt.title(
            "Distribution des probabilités prédites par le modèle",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Statistiques
        stats_text = f"Médiane: {np.median(probs):.3f}\nMoyenne: {np.mean(probs):.3f}"
        plt.text(
            0.02,
            0.98,
            stats_text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_exposure_over_time(
        self, backtest_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Affiche l'évolution de l'exposition au fil du temps.

        Visualise comment le risk overlay ajuste dynamiquement l'exposition
        au marché en fonction des conditions et des prédictions du modèle.

        Args:
            backtest_df (pd.DataFrame): Résultats du backtest
            save_path (Optional[str]): Chemin pour sauvegarder l'image

        Example:
            >>> viz.plot_exposure_over_time(backtest_df)

        Note:
            Permet de comprendre la gestion active du risque et
            d'identifier les périodes de prudence du modèle.
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize_default, sharex=True)

        # 1. Prix SPY
        ax1 = axes[0]

        ax1.plot(
            backtest_df.index,
            backtest_df["SPY_Close"],
            color="black",
            linewidth=1.5,
            label="SPY",
        )

        ax1.set_title("Prix SPY et exposition du modèle", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Prix SPY ($)", fontsize=11)
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 2. Exposition finale
        ax2 = axes[1]

        ax2.fill_between(
            backtest_df.index,
            backtest_df["Exposure_final"],
            0,
            alpha=0.5,
            color="steelblue",
            label="Exposition finale",
        )

        ax2.axhline(
            y=0.30,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Socle d'exposition (30%)",
        )

        ax2.set_title(
            "Évolution de l'exposition au fil du temps",
            fontsize=14,
            fontweight="bold",
        )
        ax2.set_ylabel("Exposition (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylim([0, 1.1])
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_returns_distribution(
        self, backtest_df: pd.DataFrame, save_path: Optional[str] = None
    ) -> None:
        """
        Compare les distributions de rendements Buy & Hold vs Overlay.

        Visualise la différence de profil de risque entre les deux stratégies.
        L'overlay devrait montrer une distribution plus concentrée autour
        de zéro (moins de volatilité).

        Args:
            backtest_df (pd.DataFrame): Résultats du backtest
            save_path (Optional[str]): Chemin pour sauvegarder l'image

        Example:
            >>> viz.plot_returns_distribution(backtest_df)

        Note:
            Une distribution plus étroite indique une meilleure
            gestion du risque (moins de volatilité).
        """
        plt.figure(figsize=self.figsize_small)

        # Histogrammes superposés
        plt.hist(
            backtest_df["Market_Return"].dropna() * 100,
            bins=50,
            alpha=0.5,
            color="gray",
            edgecolor="black",
            label="Buy & Hold",
            density=True,
        )

        plt.hist(
            backtest_df["Strategy_Return_Net"].dropna() * 100,
            bins=50,
            alpha=0.5,
            color="blue",
            edgecolor="black",
            label="Overlay",
            density=True,
        )

        plt.xlabel("Rendement journalier (%)", fontsize=12, fontweight="bold")
        plt.ylabel("Densité", fontsize=12, fontweight="bold")
        plt.title(
            "Distribution des rendements journaliers",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    # Test du module
    print("Module de visualisation chargé avec succès")
    print("Utilisez Visualizer() pour créer les graphiques")