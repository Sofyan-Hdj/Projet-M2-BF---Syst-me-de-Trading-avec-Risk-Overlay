# main.py
"""
Point d'entrée principal du système de trading SPY avec Random Forest
et risk overlay intelligent.
"""

import pandas as pd
from data_loader import DataLoader
from features import FeatureEngine
from model import TradingModel
from visualization import Visualizer
from export import Exporter


def main():
    """
    Pipeline principal d'exécution du système de trading.
    
    Étapes :
        1. Chargement des données (SPY, VIX, TNX)
        2. Construction de 15 features techniques
        3. Création de la target (direction du marché)
        4. Entraînement du modèle Random Forest
        5. Backtest avec risk overlay
        6. Visualisation et export des résultats
    
    Raises:
        Exception: Si une étape du pipeline échoue
    """
    print("=" * 70)
    print("SYSTÈME DE TRADING SPY - RANDOM FOREST + RISK OVERLAY")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. CHARGEMENT DES DONNÉES
    # -------------------------------------------------------------------------
    print("\n[1/6] Chargement des données...")
    loader = DataLoader(period="30y")
    df_raw = loader.load_all_data()
    print(f"✓ Données chargées : {len(df_raw)} observations")
    print(f"✓ Période : {df_raw.index[0]} → {df_raw.index[-1]}")

    # -------------------------------------------------------------------------
    # 2. CONSTRUCTION DES FEATURES
    # -------------------------------------------------------------------------
    print("\n[2/6] Construction des features...")
    feature_engine = FeatureEngine()
    df_features = feature_engine.create_all_features(df_raw)

    # Audit de stationnarité
    print("\n→ Audit de stationnarité des features...")
    non_stationary = feature_engine.check_stationarity(df_features)
    if non_stationary:
        print(f"⚠️  Features non-stationnaires détectées : {non_stationary}")
    else:
        print("✓ Toutes les features sont stationnaires")

    print(f"✓ Features créées : {len(feature_engine.selected_features)}")

    # -------------------------------------------------------------------------
    # 3. CRÉATION DE LA TARGET
    # -------------------------------------------------------------------------
    print("\n[3/6] Création de la target...")
    spy_close = df_features["SPY_Close"].squeeze()
    next_close = spy_close.shift(-1)
    df_features["Next_Close"] = next_close
    df_features["Direction"] = (next_close > spy_close).astype(int)
    df_features = df_features.dropna()
    print(
        f"✓ Target créée - Distribution : {df_features['Direction'].value_counts().to_dict()}"
    )

    # -------------------------------------------------------------------------
    # 4. ENTRAÎNEMENT DU MODÈLE
    # -------------------------------------------------------------------------
    print("\n[4/6] Entraînement du modèle...")
    model = TradingModel(
        features=feature_engine.selected_features, test_size=0.20, random_state=42
    )

    # Préparation des données
    model.prepare_data(df_features)

    # Optimisation des seuils via validation croisée
    print("\n→ Optimisation des seuils (validation croisée)...")
    opt_low, opt_high = model.optimize_thresholds()
    print(f"✓ Seuils optimisés : Bas = {opt_low}, Haut = {opt_high}")

    # Entraînement final
    print("\n→ Entraînement du modèle final...")
    accuracy = model.train()
    print(f"✓ Précision sur le test : {accuracy:.2%}")

    # Prédictions
    probs, preds = model.predict()

    # Importance des features
    print("\n→ Importance des features :")
    importance_df = model.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. BACKTEST & MÉTRIQUES
    # -------------------------------------------------------------------------
    print("\n[5/6] Backtest et calcul des métriques...")
    backtest_df = model.run_backtest(
        df_features,
        probs,
        opt_low,
        opt_high,
        base_exposure=0.30,
        smoothing_window=5,
        transaction_cost=0.001,
    )

    # Calcul des métriques
    metrics_market = model.calculate_metrics(
        backtest_df["Market_Return"], "Buy & Hold SPY"
    )

    metrics_overlay = model.calculate_metrics(
        backtest_df["Strategy_Return_Net"], "Buy & Hold + RF Overlay (net)"
    )

    metrics_df = pd.DataFrame([metrics_market, metrics_overlay])
    print("\n→ Métriques de performance :")
    print(metrics_df.to_string(index=False))

    # Analyse des périodes de crise
    print("\n→ Performance durant les corrections (drawdown > 10%) :")
    crisis_stats = model.analyze_crisis_periods(backtest_df)
    print(crisis_stats)

    # -------------------------------------------------------------------------
    # 6. VISUALISATION & EXPORT
    # -------------------------------------------------------------------------
    print("\n[6/6] Génération des visualisations et export...")

    # Visualisation
    viz = Visualizer()

    viz.plot_confusion_matrix(model.y_test, preds, threshold=0.50)

    viz.plot_feature_importance(importance_df)

    viz.plot_performance_comparison(backtest_df)

    viz.plot_probability_distribution(probs, opt_low, opt_high)

    print("✓ Graphiques générés")

    # Export
    exporter = Exporter(output_dir="results")

    # Nettoyage des anciens résultats AVANT d'exporter les nouveaux
    print("\n→ Nettoyage des anciens résultats...")
    exporter.cleanup_old_results(keep_last_n=1)

    # Export des nouveaux résultats
    exporter.export_backtest_results(backtest_df)
    exporter.export_metrics(metrics_df)
    exporter.export_feature_importance(importance_df)
    exporter.export_predictions(model.X_test.index, probs, preds, model.y_test)

    print("✓ Données exportées dans le dossier 'results/'")

    print("\n" + "=" * 70)
    print("EXÉCUTION TERMINÉE AVEC SUCCÈS")
    print("=" * 70)


if __name__ == "__main__":
    main()