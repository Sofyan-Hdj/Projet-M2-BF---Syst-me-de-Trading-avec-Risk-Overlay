Références et Sources Utilisées

Projet : Système de Trading avec Risk Overlay
Auteurs : HADJ Sofyan & LEVEHANG Johannes
Formation : Master 2 Banque et Finance
Date : 12 janvier 2026

1. Cours DataCamp Utilisés

Les cours DataCamp ci-dessous ont été utilisés comme supports pédagogiques pour l’acquisition des concepts techniques et méthodologiques.
Les implémentations, choix de paramètres et résultats sont propres au projet.

1.1 Finance & Risk Management
Introduction to Portfolio Risk Management in Python

URL : https://www.datacamp.com/courses/introduction-to-portfolio-risk-management-in-python

Concepts abordés :

Mesures de risque de portefeuille

Value at Risk (VaR)

Analyse des rendements et de la volatilité

Application dans le projet :

Calcul des métriques de risque (drawdown, volatilité)

Analyse comparative Buy & Hold vs stratégie avec Risk Overlay

Identification et étude des périodes de stress de marché

1.2 Importation et Gestion des Données Financières
Importing and Managing Financial Data in Python

URL : https://www.datacamp.com/courses/importing-and-managing-financial-data-in-python

Concepts abordés :

Importation de données financières

Manipulation de séries temporelles

Gestion des index de dates

Application dans le projet :

Téléchargement automatisé des données SPY, VIX et TNX

Construction d’un jeu de données cohérent sur longue période

Préparation des séries temporelles pour l’analyse et le backtesting

Intermediate Importing Data in Python

URL : https://www.datacamp.com/courses/intermediate-importing-data-in-python

Concepts abordés :

Importation depuis le web et APIs

Gestion des formats CSV et JSON

Validation et nettoyage des données

Application dans le projet :

Pipeline robuste d’importation des données

Contrôles de cohérence (valeurs manquantes, alignement temporel)

Exports des résultats (CSV, Excel)

1.3 Manipulation de Données
Joining Data with Pandas

URL : https://www.datacamp.com/courses/joining-data-with-pandas

Concepts abordés :

Jointures de DataFrames

Alignement temporel

Gestion des index

Application dans le projet :

Jointure stricte (INNER) des séries SPY, VIX et TNX

Construction d’un dataset final aligné sur 7 532 dates communes

1.4 Machine Learning
Supervised Learning with scikit-learn

URL : https://www.datacamp.com/courses/supervised-learning-with-scikit-learn

Concepts abordés :

Classification supervisée

Random Forest

Validation et overfitting

Application dans le projet :

Modèle RandomForestClassifier

Prédiction de la direction du marché

Évaluation des performances hors-échantillon

Unsupervised Learning in Python

URL : https://www.datacamp.com/courses/unsupervised-learning-in-python

Concepts abordés :

Clustering

Transformation des données

Analyse exploratoire

Apport indirect :

Réflexion sur la structure des données

Prétraitement et normalisation

1.5 Visualisation des Données
Introduction to Data Visualization with Matplotlib

URL : https://www.datacamp.com/courses/introduction-to-data-visualization-with-matplotlib

Concepts abordés :

Graphiques multi-axes

Annotations

Visualisation des performances

Application dans le projet :

Graphiques de performance cumulée

Analyse comparative des drawdowns

Introduction to Data Visualization with Seaborn

URL : https://www.datacamp.com/courses/introduction-to-data-visualization-with-seaborn

Concepts abordés :

Heatmaps

Visualisation des distributions

Application dans le projet :

Matrice de confusion

Distribution des probabilités prédites

Importance des variables

2. Documentation Technique Officielle
2.1 Librairies Python
scikit-learn

RandomForestClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

TimeSeriesSplit
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

StandardScaler
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

pandas

Time Series User Guide
https://pandas.pydata.org/docs/user_guide/timeseries.html

Rolling Windows
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html

Join Operations
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.join.html

statsmodels

Augmented Dickey-Fuller Test
https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html

Time Series Analysis
https://www.statsmodels.org/stable/tsa.html

yfinance

Documentation officielle
https://pypi.org/project/yfinance/

matplotlib & seaborn

Matplotlib User Guide
https://matplotlib.org/stable/users/index.html

Seaborn Tutorial
https://seaborn.pydata.org/tutorial.html

3. Références Académiques
Machine Learning en Finance

López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.

López de Prado, M. (2020). Machine Learning for Asset Managers. Cambridge University Press.

Risk Management

McNeil, A. J., Frey, R., Embrechts, P. (2015). Quantitative Risk Management. Princeton University Press.

Volatilité et Marchés

Derman, E., Miller, M. (2016). The Volatility Smile. Wiley.

Analyse Technique

Murphy, J. J. (1999). Technical Analysis of the Financial Markets. New York Institute of Finance.

4. Attestation d’Originalité

Ce projet a été réalisé de manière autonome par HADJ Sofyan et LEVEHANG Johannes.
Les cours DataCamp ont été utilisés exclusivement comme supports pédagogiques.

L’intégralité :

du code,

des choix méthodologiques,

des paramètres,

des résultats,

et des interprétations

a été développée spécifiquement pour ce projet, conformément aux standards académiques et professionnels.