import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, log_loss, f1_score, classification_report, roc_curve
from sklearn.model_selection import learning_curve

# Importation des 6 architectures/algorithmes choisis pour la compétition ML
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
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 15
})


# CHARGEMENT DU JEU DE DONNÉES ET STRUCTURATION DE LA MATRICE X (VOTRE LISTE)
print("PHASE 1 : RECONSTRUCTION CONTEXTUELLE DE LA MATRICE X DES COVARIABLES")

df_ml = pd.read_excel("dataset_sous_nutrition_EDSC2018.xlsx", sheet_name='donnees')

# Déclaration des variables disponibles dans le jeu de donnees d'entrainement
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

# Interception des colonnes réellement existantes dans le fichier Excel du jeu de donnees d'entrainement
covariables_disponibles = [col for col in toutes_covariables_etude if col in df_ml.columns]

X = df_ml[covariables_disponibles].copy()
y = df_ml['Y_ICEA'].copy() # Appel de la variable cible composite pondérée unifiée
poids_eds = df_ml['poids_analytique'].copy()
grappes = df_ml['numerogruppe'].copy()

print(f"Matrice X reconstruite avec succès : {X.shape} enfants et {X.shape} covariables.")

# SEPARATION TRAIN/TEST PAR GRAPPES (CLUSTER-ROBUST SPLIT)

print("PHASE 2 : DECOUPAGE ETANCHE PAR GRAPPES SUIVANT LE PLAN D'ECHANTILLONNAGE EDS")
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=2018)
train_idx, test_idx = next(gss.split(X, y, groups=grappes))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
poids_train, poids_test = poids_eds.iloc[train_idx], poids_eds.iloc[test_idx]

# COMPENSATEUR DOUBLE DES POIDS (SONDAGE ET ÉQUILIBRE DES CLASSES CIBLES)
taux_malnutrition = np.average(y_train, weights=poids_train)
poids_ajustement_classe = np.where(y_train == 1, 1.0 / taux_malnutrition, 1.0 / (1.0 - taux_malnutrition))
poids_harmonise_train = poids_train * poids_ajustement_classe

# 4. PIPELINE DE TRANSFORMATION PAR TARGET ENCODING RÉGULARISÉ (SANS FUITE)
print("PIPELINE DE TRANSFORMATION ET DE CODAGE DES FEATURES SANS FUITE OU DATA LEAAKAGE")
# Identification automatique de la nature numérique des colonnes pour le preprocessor
features_continues_clean = [col for col in X.columns if X[col].dtype in ['float64', 'int64'] and X[col].nunique() > 5]
features_categoriques_clean = [col for col in X.columns if col not in features_continues_clean]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_continues_clean),
        ('cat', TargetEncoder(smooth="auto", cv=5, random_state=2018), features_categoriques_clean)
    ]
)

# Ajustement exclusif Train -> Transformation du bloc d'évaluation Test
X_train_encoded = preprocessor.fit_transform(X_train, y_train)
X_test_encoded = preprocessor.transform(X_test)

# ESTIMATION ET ENTRAÎNEMENT DE LA COMPÉTITION DES 6 MODÈLES PUISSANTS
print("MISE EN COMPETITION DES 6 ALGORITHMES ET ÉVALUATION PONDÉRÉE SUR LE TEST SET")

modeles_pipeline = {
    "Régression Logistique (Ridge)": LogisticRegression(penalty='l2', solver='lbfgs', max_iter=2000, random_state=2018),
    "Forêt Aléatoire (Random Forest)": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=2018),
    "XGBoost Classifier": XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.04, eval_metric='logloss', random_state=2018),
    "LightGBM Classifier": LGBMClassifier(n_estimators=250, max_depth=5, learning_rate=0.04, verbose=-1, random_state=2018),
    "CatBoost Classifier": CatBoostClassifier(iterations=300, depth=5, learning_rate=0.04, verbose=0, random_state=2018),
    "Réseau de Neurones (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01, max_iter=1000, random_state=2018)
}

tableau_ml_records = []
meilleur_auc = 0.0

# VARIABLES DEMANDÉES INITIALISÉES : Seront assignées dynamiquement à la fin de la compétition
meilleur_modele = None
nom_meilleur_modele = ""

for nom, modele in modeles_pipeline.items():
    if "MLP" in nom:
        modele.fit(X_train_encoded, y_train)
    else:
        modele.fit(X_train_encoded, y_train, sample_weight=poids_harmonise_train)
        
    probabilites = modele.predict_proba(X_test_encoded)[:, 1]
    classes_predites = (probabilites >= 0.5).astype(int)
    
    # Calcul des performances sous contrôle strict du poids de sondage EDS de Test
    score_auc = roc_auc_score(y_test, probabilites, sample_weight=poids_test)
    score_loss = log_loss(y_test, probabilites, sample_weight=poids_test)
    score_f1 = f1_score(y_test, classes_predites, sample_weight=poids_test, average='macro')
    
    tableau_ml_records.append({
        "Algorithme_ML": nom,
        "ROC_AUC_Global": score_auc,
        "Log_Loss_Unifie": score_loss,
        "F1_Score_Macro": score_f1
    })
    
    # Capture et assignation dynamique du meilleur modèle 
    if score_auc > meilleur_auc:
        meilleur_auc = score_auc
        meilleur_modele = modele
        nom_meilleur_modele = nom

df_comparatif_final = pd.DataFrame(tableau_ml_records)
df_comparatif_final.to_csv(f"{output_dir}/t7_comparatif_performances_ml.csv", index=False)

print("\n CLASSEMENT DES ARCHITECTURES ISSUES DU MODÈLE CONTEXTUEL INTEGRAL ")
print(df_comparatif_final.sort_values(by="ROC_AUC_Global", ascending=False).round(4).to_string(index=False))


print(f" SAUVEGARDE DU MEILLEUR MODELE APRES TRAINING DANS LE PROJET :")
print(f" - nom_meilleur_modele = '{nom_meilleur_modele}'")
print(f" - meilleur_modele     = {type(meilleur_modele).__name__} (AUC Test = {meilleur_auc:.4f})")



print("PHASE D'ÉVALUATION DOUBLE : CALCUL DES MÉTRIQUES SUR LES COMPARTIMENTS TRAIN ET TEST")

# Initialisation du conteneur pour stocker le bilan Train vs Test
bilan_comparatif_records = []

# Itération sur les instances déjà entraînées lors de votre phase précédente
# (Le script utilise le dictionnaire des modèles et les matrices X_train_encoded/X_test_encoded)
for nom, modele in modeles_pipeline.items():
    print(f"Évaluation des courbes et métriques : {nom}...")
    
    # 1. Calcul des probabilités de risque sur le Train et le Test
    prob_train = modele.predict_proba(X_train_encoded)[:, 1]
    prob_test = modele.predict_proba(X_test_encoded)[:, 1]
    
    # Assignation des classes au seuil standard de 0.5
    class_train = (prob_train >= 0.5).astype(int)
    class_test = (prob_test >= 0.5).astype(int)
    
    # 2. Métriques sur le compartiment d'Entraînement (Train)
    auc_train = roc_auc_score(y_train, prob_train, sample_weight=poids_harmonise_train)
    loss_train = log_loss(y_train, prob_train, sample_weight=poids_harmonise_train)
    f1_train = f1_score(y_train, class_train, sample_weight=poids_harmonise_train, average='macro')
    
    # 3. Métriques sur le compartiment d'Évaluation Étanche (Test) - Poids EDS purs
    auc_test = roc_auc_score(y_test, prob_test, sample_weight=poids_test)
    loss_test = log_loss(y_test, prob_test, sample_weight=poids_test)
    f1_test = f1_score(y_test, class_test, sample_weight=poids_test, average='macro')
    
    # Calcul du Delta d'Overfitting
    delta_auc = auc_train - auc_test
    
    # Règle d'interprétation automatisée selon les seuils du Machine Learning
    if delta_auc > 0.10:
        diagnostic = "Overfitting Sévère (À régulariser)"
    elif 0.05 < delta_auc <= 0.10:
        diagnostic = "Overfitting Modéré"
    elif 0.0 <= delta_auc <= 0.05:
        diagnostic = "Modèle Stable (Excellente Généralisation)"
    else:
        diagnostic = "Underfitting ou Instabilité matricielle"
        
    bilan_comparatif_records.append({
        "Algorithme": nom,
        "AUC_Train": auc_train,
        "AUC_Test": auc_test,
        "Delta_AUC": delta_auc,
        "LogLoss_Train": loss_train,
        "LogLoss_Test": loss_test,
        "F1_Macro_Train": f1_train,
        "F1_Macro_Test": f1_test,
        "Diagnostic_ML": diagnostic
    })

# Création et exportation de la table maîtresse comparative Train/Test
df_bilan_train_test = pd.DataFrame(bilan_comparatif_records)
df_bilan_train_test.to_csv(f"{output_dir}/t9_bilan_diagnostic_train_test.csv", index=False)

print(df_bilan_train_test.sort_values(by="AUC_Test", ascending=False).round(4).to_string(index=False))

print("PHASE GRAPHIQUE : TRACÉ DES COURBES ROC ET DES COURBES D'APPRENTISSAGE")

# --- GRAPH_1 : TOUTES LES COURBES ROC SUR LA MÊME FEUILLE (TEST SET) ---
plt.figure(figsize=(8.5, 7))
colors_pool = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for idx, (nom, modele) in enumerate(modeles_pipeline.items()):
    prob_test = modele.predict_proba(X_test_encoded)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob_test, sample_weight=poids_test)
    auc_val = roc_auc_score(y_test, prob_test, sample_weight=poids_test)
    
    plt.plot(fpr, tpr, color=colors_pool[idx], lw=2, label=f'{nom} (AUC = {auc_val:.3f})')

# Ajout des coordonnées [0, 1] pour tracer la ligne diagonale
plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--', label='Seuil Aléatoire (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Spécificité)')
plt.ylabel('Taux de Vrais Positifs (Sensibilité)')
plt.title("Comparaison des Courbes ROC sur le Compartiment de Test\n(Données Réelles EDSC-V 2018 - Cameroun)", fontsize=13, fontweight='bold')
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig(f"{graph_dir}/g5_comparaison_courbes_roc.png", dpi=300)
plt.close()
print(f"-> Planche collective des Courbes ROC enregistrée : '{graph_dir}/g5_comparaison_courbes_roc.png'")

# --- GRAPH_2 : PLANCHE DES COURBES D'ENTRAÎNEMENT (LEARNING CURVES 3x2) ---
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
axes = axes.flatten()

# Simulation et extraction des courbes de perte/score au fil de la taille de l'échantillon
for idx, (nom, modele) in enumerate(modeles_pipeline.items()):
    ax = axes[idx]
    
    # Utilisation de learning_curve de sklearn pour mesurer l'évolution du score
    # (Adaptation simplifiée sur l'espace transformé pour figer la dynamique)
    train_sizes, train_scores, test_scores = learning_curve(
        modele, X_train_encoded, y_train, cv=3, scoring='roc_auc',
        train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1, random_state=2018
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    ax.plot(train_sizes, train_mean, 'o-', color='crimson', lw=2, label='Score Entraînement (Train)')
    ax.plot(train_sizes, test_mean, 's-', color='dodgerblue', lw=2, label='Score Validation (Test)')
    
    ax.set_xlabel('Taille de l\'échantillon d\'apprentissage')
    ax.set_ylabel('Performance ROC AUC')
    ax.set_title(f'Courbe d\'Apprentissage : {nom}', fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right")

plt.suptitle("Planche de Diagnostic Évolutif des Courbes d'Apprentissage (Train vs Test)\nAnalyse de la Convergence des 6 Modèles d'Élite", y=0.99, fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{graph_dir}/g6_planche_courbes_apprentissage.png", dpi=300)
plt.close()
