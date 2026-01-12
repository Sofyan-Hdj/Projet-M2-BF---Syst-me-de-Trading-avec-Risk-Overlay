#description_projet.md
# Projet M2 BF - Système de Trading avec Risk Overlay

## Membres du Groupe
- **HADJ Sofyan**
- **LEVEHANG Johannes**

**Formation** : Master 2 Banque et Finance  
**Année universitaire** : 2025-2026  
**Date de rendu** : 25 janvier 2026

---

## Description du Sujet

Développement d'un système de trading algorithmique appliquant un **"risk overlay"** 
sur l'indice SPY (S&P 500) en utilisant un modèle **Random Forest**.

### Problématique
Comment réduire l'exposition au risque d'un portefeuille actions tout en maintenant 
un rendement ajusté au risque compétitif ?

### Objectif Principal
Réduire le **drawdown maximum** (perte maximale depuis le sommet) tout en maintenant 
un **rendement ajusté au risque supérieur** (Sharpe ratio) par rapport à une 
stratégie Buy & Hold passive.

### Approche
Utilisation d'un modèle de machine learning (Random Forest) pour prédire la 
direction du marché et ajuster dynamiquement l'exposition selon un système 
de **risk overlay à 4 couches** :
1. Exposition graduelle basée sur la conviction du modèle
2. Filtre de tendance (SMA50) pour éviter les faux signaux
3. Lissage temporel pour réduire le churn
4. Socle d'exposition (30%) pour éviter le timing risk

---

## Sources de Données

### API Utilisée
**Yahoo Finance (yfinance)** - Téléchargement automatisé via Python

### Instruments Financiers
- **SPY** : S&P 500 ETF (proxy du marché actions américain)
- **VIX** : Indice de volatilité CBOE (indicateur de peur du marché)
- **TNX** : Taux du Treasury 10 ans US (indicateur macro-économique)

### Période et Fréquence
- **Période** : 30 ans de données historiques (12 janvier 1996 - 9 janvier 2026)
- **Fréquence** : Journalière (1d)
- **Observations brutes** : 7,532 jours de trading
- **Observations utilisables** : 7,332 (après création des features)

### Jointure des Données
Les 3 sources sont fusionnées via une **jointure INNER** sur les dates communes,
garantissant l'alignement temporel et l'absence de valeurs manquantes.

---

## Transformations Principales

### 1. Feature Engineering (15 indicateurs)

**Tendance** :
- SMA20, SMA50, SMA200 (moyennes mobiles)
- SMA_Cross_Ratio (détection de croisements)
- SMA200_LogDist (distance logarithmique au prix moyen LT)

**Momentum** :
- Return_5days, Return_20days (rendements multi-périodes)
- Momentum_20 (momentum 20 jours)
- Dist_From_High (distance au plus haut annuel)

**Technique** :
- RSI (Relative Strength Index)
- BB_Position (position dans les Bandes de Bollinger)

**Volatilité** :
- Vol_Shock (choc de volatilité 10j/50j)
- Vol_VIX_Divergence (vol réalisée vs implicite)

**Macro** :
- TNX_Momentum (momentum des taux d'intérêt)
- VIX_SMA20_Ratio (stress relatif du marché)
- VIX_Change_5d (accélération de la peur)
- Price_VIX_Corr (corrélation prix/volatilité)

**Interactions** :
- RSI_BB_Interaction (surachat en haut de bande)

**Protection contre le look-ahead bias** : Toutes les features utilisent un 
décalage (`shift(1)`) pour garantir qu'elles n'utilisent que des données 
passées disponibles au moment de la décision.

### 2. Validation Statistique
**Test de stationnarité (ADF)** : Les 15 features sont stationnaires (p-value < 0.05),
garantissant la validité statistique du modèle et évitant les régressions fallacieuses.

### 3. Machine Learning

**Modèle** : Random Forest Classifier
- 1000 arbres de décision
- Hyperparamètres optimisés (`max_features='sqrt'`, `min_samples_leaf=10`)
- `class_weight='balanced'` pour gérer le déséquilibre de classes

**Split des données** :
- Train : 5,864 observations (80%)
- Test : 1,467 observations (20%)
- Distribution train : 3,173 hausses (54.1%) / 2,691 baisses (45.9%)

**Validation** :
- Split temporel 80/20 (pas de mélange aléatoire)
- Time-Series Cross-Validation (5 folds) pour optimisation des seuils
- Normalisation des features (StandardScaler)

**Optimisation** : Recherche des seuils de décision optimaux via grid search
- Seuil bas : 0.58 (en-dessous = 0% exposition)
- Seuil haut : 0.63 (au-dessus = 100% exposition)

**Précision du modèle** : 50.51% (attendu pour un marché efficace)

### 4. Backtesting Réaliste

**Risk Overlay** :
- Exposition graduée : 0% (cash) / 50% (prudent) / 100% (full invest)
- Filtre de tendance : coupe l'exposition si prix < SMA50
- Lissage temporel : moyenne mobile sur 5 jours
- Socle de 30% : combine Buy & Hold (30%) + overlay actif (70%)

**Réalisme** :
- Coûts de transaction : 10 bps (0.10%) par transaction
- Décalage d'exécution : décision prise en T appliquée en T+1
- Slippage implicite dans les coûts

---

## Résultats Obtenus

### Métriques de Performance

| Métrique | Buy & Hold SPY | RF Overlay | Amélioration |
|----------|---------------|------------|--------------|
| **Rendement Total** | 160.88% | 40.17% | - |
| **Rendement Annualisé** | 20.35% | 6.19% | - |
| **Volatilité Annualisée** | 20.18% | 6.33% | **-69%** ✅ |
| **Sharpe Ratio** | 0.92 | **0.95** | **+3%** ✅ |
| **Sortino Ratio** | 1.15 | **1.20** | **+4%** ✅ |
| **Calmar Ratio** | 0.83 | 0.79 | -5% |
| **Max Drawdown** | -24.50% | **-7.82%** | **-68%** ✅ |
| **Win Rate** | 55.32% | 55.25% | Similaire |

### Performance en Période de Crise
Lors des corrections de marché > 10% (318 jours sur 1,467) :
- **Buy & Hold** : -0.118% par jour
- **RF Overlay** : -0.035% par jour
- **Protection** : Pertes divisées par **3.4** ✅

En période normale (1,149 jours) :
- **Buy & Hold** : +0.127% par jour
- **RF Overlay** : +0.040% par jour
- **Trade-off** : Participation réduite aux hausses pour protection accrue

### Top 5 Features les Plus Importantes
1. **TNX_Momentum** (7.60%) - Momentum des taux d'intérêt
2. **VIX_SMA20_Ratio** (7.09%) - Stress relatif du marché
3. **SMA200_LogDist** (7.03%) - Sur/sous-valorisation long terme
4. **Return_5days** (6.91%) - Momentum court terme
5. **Price_VIX_Corr** (6.89%) - Corrélation prix/volatilité

### Interprétation des Résultats

**Objectif atteint** ✅ : Le système **n'est pas conçu pour surperformer le marché** 
en termes de rendement absolu, mais pour **protéger le capital durant les crises** 
tout en maintenant un rendement ajusté au risque légèrement supérieur.

**Points clés** :
- ✅ Réduction drastique du drawdown (-68%)
- ✅ Amélioration du Sharpe ratio (+3%)
- ✅ Amélioration du Sortino ratio (+4%)
- ✅ Protection efficace en crise (pertes divisées par 3.4)
- ⚠️ Participation réduite aux hausses (trade-off assumé)

**Profil investisseur** : Ce système convient à un investisseur **risk-averse** 
privilégiant la **préservation du capital** plutôt que la maximisation du rendement.

---

### Concepts Avancés Utilisés
1. **Time-Series Cross-Validation** : Validation temporelle sans mélange
2. **Feature Engineering Financier** : 15 indicateurs techniques
3. **Tests de Stationnarité (ADF)** : Validation statistique (p < 0.05 pour toutes)
4. **Risk Management Dynamique** : Overlay à 4 couches
5. **Backtesting Réaliste** : Coûts de transaction et décalage d'exécution

---

## Langage et Technologies

### Langage Principal
**Python 3.11**

### Librairies Principales
**Data & ML** :
- pandas 1.5+ (manipulation de données)
- numpy 1.23+ (calculs numériques)
- scikit-learn 1.2+ (Random Forest, validation croisée)

**Visualisation** :
- matplotlib 3.5+ (graphiques)
- seaborn 0.12+ (visualisations statistiques)

**Finance** :
- yfinance 0.2+ (téléchargement données Yahoo Finance)
- statsmodels 0.14+ (tests de stationnarité ADF)

**Autres** :
- pathlib (gestion des chemins)
- datetime (gestion temporelle)
- openpyxl 3.0+ (export Excel)

---

## Références DataCamp
Voir fichier `REFERENCES.md`

---

## Conclusion

**Résultats quantifiés** :
- Drawdown réduit de **68%** (-24.5% → -7.82%)
- Sharpe ratio amélioré de **3%** (0.92 → 0.95)
- Protection en crise : pertes divisées par **3.4**

**Le système est production-ready et pourrait être déployé dans un 
environnement professionnel avec adaptations mineures.**

---

**Auteurs** : HADJ Sofyan & LEVEHANG Johannes  
