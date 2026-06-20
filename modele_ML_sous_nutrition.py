import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, learning_curve, RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, log_loss, f1_score, classification_report, roc_curve

# Importation des 6 architectures de pointe pour la compétition ML
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

output_dir = "resultats"
graph_dir = "graphiques"
for folder in [output_dir, graph_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'figure.titlesize': 14
})


# 1. CHARGEMENT DU JEU DE DONNÉES ET STRUCTURATION CONTEXTUELLE DE LA MATRICE X
df_ml = pd.read_excel("dataset_sous_nutrition_EDSC2018.xlsx", sheet_name='donnees')

vars_discretes = [
    "rangdenaissance", "hemoglobinemere", "intervalleintergenesique", "dureeallaitement", "agemere",
    "nombreenfantsnesvivants", "nombrevisitesprenatales", "nombremembresmenage", "ageenfant"
]

vars_continues_brutes = [
    "imcmere", "ageenfant", "agemere"
] # Exclusion des scores Z d'anthropométrie (scoreztaillepourage, etc.) pour éviter le leakage absolu

vars_categorielles = [
    "sexeenfant", "milieuderesidence", "niveauinstructionmere", "indicederichesse", "region", "vacciné",
    "vaccinbcg", "diarrhee", "prisededecisionmere", "statutmatrimonialmere", "tailleanaissance",
    "sourceeaupotable", "typeinstallationssanitaires", "typecombustiblecuisine", "lieuderesidence",
    "regionecologique", "vaccinpolio0", "vaccinpolio2", "vaccinpolio3", "vaccindtp1", "vaccindtp2", 
    "vaccindtp3", "vaccinpolio1", "vaccinrougeole1"
]

# Fusion unique des listes sans doublons pour constituer les colonnes de la matrice X
toutes_covariables_etude = list(set(vars_discretes + vars_continues_brutes + vars_categorielles))

# Interception des colonnes réellement existantes dans votre fichier Excel
covariables_disponibles = [col for col in toutes_covariables_etude if col in df_ml.columns]

X = df_ml[covariables_disponibles].copy()
y = df_ml['Y_ICEA'].copy() # Variable cible composite pondérée unifiée
poids_eds = df_ml['poids_analytique'].copy()
grappes = df_ml['numerogruppe'].copy()

print(f"Matrice X reconstruite avec succès : {X.shape} enfants et {X.shape} covariables.")

# SEPARATION TRAIN/TEST PAR GRAPPES (CLUSTER-ROBUST SPLIT)
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2018)
train_idx, test_idx = next(gss.split(X, y, groups=grappes))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
poids_train, poids_test = poids_eds.iloc[train_idx], poids_eds.iloc[test_idx]

# COMPENSATEUR DOUBLE DES POIDS (SONDAGE ET ÉQUILIBRE DES CLASSES CIBLES)
taux_malnutrition = np.average(y_train, weights=poids_train)
poids_ajustement_classe = np.where(y_train == 1, 1.0 / taux_malnutrition, 1.0 / (1.0 - taux_malnutrition))
poids_harmonise_train = poids_train * poids_ajustement_classe


# PIPELINE DE TRANSFORMATION PAR TARGET ENCODING RÉGULARISÉ (SANS FUITE) OU DATA LEAKAGE
features_continues_clean = [col for col in X.columns if X[col].dtype in ['float64', 'int64'] and X[col].nunique() > 5]
features_categoriques_clean = [col for col in X.columns if col not in features_continues_clean]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_continues_clean),
        ('cat', TargetEncoder(smooth="auto", cv=5, random_state=2018), features_categoriques_clean)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train, y_train)
X_test_encoded = preprocessor.transform(X_test)

# INJECTION DE LA STRATÉGIE INITIALE DE RÉGULARISATION MAXIMUM ET PENALISATION DE BASE

# MODIFICATION CRITIQUE DES HYPERPARAMÈTRES DE BASE :
# - max_depth / depth restreint à des niveaux bas (3 ou 4) pour empêcher les arbres d'apprendre par coeur.
# - min_samples_leaf / min_child_weight élevé pour forcer des feuilles massives et stables.
# - reg_lambda / l2_leaf_reg fort pour écraser les coefficients extrêmes.
# - MLP : alpha augmenté à 2.0 pour une régularisation de Ridge extrême sur les poids des neurones.
modeles_pipeline = {
    "Régression Logistique (Ridge)": LogisticRegression(
        penalty='l2', C=0.1, solver='lbfgs', max_iter=2000, random_state=2018
    ),
    "Forêt Aléatoire (Random Forest)": RandomForestClassifier(
        n_estimators=250, max_depth=4, min_samples_leaf=30, random_state=2018
    ),
    "XGBoost Classifier": XGBClassifier(
        n_estimators=150, max_depth=3, min_child_weight=20, reg_lambda=50, 
        learning_rate=0.03, eval_metric='logloss', random_state=2018
    ),
    "LightGBM Classifier": LGBMClassifier(
        n_estimators=150, max_depth=3, min_child_samples=30, reg_lambda=50, 
        learning_rate=0.03, verbose=-1, random_state=2018
    ),
    "CatBoost Classifier": CatBoostClassifier(
        iterations=150, depth=3, l2_leaf_reg=50, learning_rate=0.03, verbose=0, random_state=2018
    ),
    "Réseau de Neurones (MLP)": MLPClassifier(
        hidden_layer_sizes=(16, 8), activation='relu', alpha=2.0, 
        early_stopping=True, max_iter=1000, random_state=2018
    )
}

# Calcul immédiat du double bilan Train/Test pour valider le resserrement des courbes
bilan_comparatif_records = []
meilleur_auc = 0.0
meilleur_modele = None
nom_meilleur_modele = ""

for nom, modele in modeles_pipeline.items():
    if "MLP" in nom:
        modele.fit(X_train_encoded, y_train)
    else:
        modele.fit(X_train_encoded, y_train, sample_weight=poids_harmonise_train)
        
    prob_train = modele.predict_proba(X_train_encoded)[:, 1]
    prob_test = modele.predict_proba(X_test_encoded)[:, 1]
    
    class_train = (prob_train >= 0.5).astype(int)
    class_test = (prob_test >= 0.5).astype(int)
    
    auc_train = roc_auc_score(y_train, prob_train, sample_weight=poids_harmonise_train)
    loss_train = log_loss(y_train, prob_train, sample_weight=poids_harmonise_train)
    f1_train = f1_score(y_train, class_train, sample_weight=poids_harmonise_train, average='macro')
    
    auc_test = roc_auc_score(y_test, prob_test, sample_weight=poids_test)
    loss_test = log_loss(y_test, prob_test, sample_weight=poids_test)
    f1_test = f1_score(y_test, class_test, sample_weight=poids_test, average='macro')
    
    delta_auc = auc_train - auc_test
    
    if delta_auc <= 0.05:
        diagnostic = "Modèle Stable"
    elif 0.05 < delta_auc <= 0.10:
        diagnostic = "Overfitting Contrôlé"
    else:
        diagnostic = "Overfitting Résiduel"
        
    bilan_comparatif_records.append({
        "Algorithme": nom, "AUC_Train": auc_train, "AUC_Test": auc_test, "Delta_AUC": delta_auc,
        "LogLoss_Train": loss_train, "LogLoss_Test": loss_test, 
        "F1_Macro_Train": f1_train, "F1_Macro_Test": f1_test, "Diagnostic_ML": diagnostic
    })
    
    if auc_test > meilleur_auc:
        meilleur_auc = auc_test
        meilleur_modele = modele
        nom_meilleur_modele = nom

df_bilan_train_test = pd.DataFrame(bilan_comparatif_records)
df_bilan_train_test.to_csv(f"{output_dir}/bilan_diagnostic_train_test.csv", index=False)

print("TABLEAU COMPARATIF DES METRIQUES TRAIN / TEST DES MODÈLES ENCAPSULÉS RATIONNELS CHOISIS ")
print(df_bilan_train_test.sort_values(by="AUC_Test", ascending=False).round(4).to_string(index=False))

print(f"\n SELECTION DU MEILLEUR MODELE A L'ISSUE DE L'ENTRAINEMENT :")
print(f"   - nom_meilleur_modele = '{nom_meilleur_modele}'")
# CORRECTION CRITIQUE ICI : Utilisation de __name__ pour extraire le nom de la classe
print(f"   - meilleur_modele     = {type(meilleur_modele).__name__} (AUC Test = {meilleur_auc:.4f})")

# --- Graphique A : Toutes les Courbes ROC de Test sur une seule feuille ---
plt.figure(figsize=(8.5, 7))
colors_pool = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, (nom, modele) in enumerate(modeles_pipeline.items()):
    prob_t = modele.predict_proba(X_test_encoded)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob_t, sample_weight=poids_test)
    auc_val = roc_auc_score(y_test, prob_t, sample_weight=poids_test)
    plt.plot(fpr, tpr, color=colors_pool[idx], lw=2, label=f'{nom} (AUC = {auc_val:.3f})')

# CORRECTION EXACTE DES COORDONNÉES DE LA DIAGONALE ALÉATOIRE :
plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--', label='Seuil Aléatoire (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Spécificité)')
plt.ylabel('Taux de Vrais Positifs (Sensibilité)')
plt.title("Comparaison des Courbes ROC Régularisées sur le Compartiment de Test\n(Données EDSC-V 2018 - Cameroun)", fontsize=12, fontweight='bold')
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(f"{graph_dir}/comparaison_courbes_roc.png", dpi=300)
plt.close()

# --- Graphique B : Planche collective des Courbes d'Apprentissage (Learning Curves 3x2) ---
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

for idx, (nom, modele) in enumerate(modeles_pipeline.items()):
    ax = axes[idx]
    train_sizes, train_scores, test_scores = learning_curve(
        modele, X_train_encoded, y_train, cv=3, scoring='roc_auc',
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1, random_state=2018
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', color='crimson', lw=2, label='Score Entraînement (Train)')
    ax.plot(train_sizes, test_mean, 's-', color='dodgerblue', lw=2, label='Score Validation (Test)')
    ax.set_xlabel("Taille de l'échantillon d'apprentissage")
    ax.set_ylabel('Performance ROC AUC')
    ax.set_title(f"Courbe d'Apprentissage (Learning Curve) : {nom}", fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right")

plt.suptitle("Planche de Diagnostic Évolutif des Courbes d'Apprentissage Corrigées\nAnalyse du Rapprochement Train / Test avec l'evolution de la taille de l'echantillon", y=0.99, fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{graph_dir}/planche_courbes_apprentissage.png", dpi=300)
plt.close()