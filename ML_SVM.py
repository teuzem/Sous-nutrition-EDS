import os
import joblib
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold, learning_curve
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, log_loss, f1_score, roc_curve, classification_report, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC

# Configuration des repertoires du projet
output_dir = "resultats"
graph_dir = "graphiques"
model_dir = "modeles_sauvegardes"
for folder in [output_dir, graph_dir, model_dir]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Configuration visuelle des graphiques
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'figure.titlesize': 14
})

# 1. Enrichissement epidemiologique et feature engineering avance
print("Phase 1 : Enrichissement epidemiologique et feature engineering avance")
df_ml = pd.read_excel("dataset_sous_nutrition_EDSC2018.xlsx", sheet_name='donnees')
print(f"Base de donnees chargee : {df_ml.shape}")

# Recodage de securite des phenotypes WASH
if 'sourceeaupotable' in df_ml.columns:
    df_ml['eau_amelioree'] = df_ml['sourceeaupotable'].isin([11, 12, 13, 21]).astype(int)
else:
    df_ml['eau_amelioree'] = 0

if 'typeinstallationssanitaires' in df_ml.columns:
    df_ml['toilettes_ameliorees'] = df_ml['typeinstallationssanitaires'].isin([11, 12, 21]).astype(int)
else:
    df_ml['toilettes_ameliorees'] = 0

# Creation de l'index synergique WASH
df_ml['index_wash_synergie'] = (df_ml['eau_amelioree'] * df_ml['toilettes_ameliorees']).astype(int)

# Vulnerabilite geo-economique (milieu rural et pauvreté)
if 'milieuderesidence' in df_ml.columns and 'indicederichesse' in df_ml.columns:
    df_ml['pauvreté_rurale'] = ((df_ml['milieuderesidence'] == 2) & (df_ml['indicederichesse'].isin([1, 2]))).astype(int)
else:
    df_ml['pauvreté_rurale'] = 0

# Ratio de charge dependante du menage
if 'nombreenfantsnesvivants' in df_ml.columns and 'nombremembresmenage' in df_ml.columns:
    df_ml['ratio_charge_menage'] = df_ml['nombreenfantsnesvivants'] / (df_ml['nombremembresmenage'].replace(0, np.nan))
    df_ml['ratio_charge_menage'] = df_ml['ratio_charge_menage'].fillna(df_ml['ratio_charge_menage'].median())
else:
    df_ml['ratio_charge_menage'] = 0

# Definition des listes de variables d'etude
vars_discretes = [
    "rangdenaissance", "hemoglobinemere", "intervalleintergenesique", "dureeallaitement", "agemere",
    "nombreenfantsnesvivants", "nombrevisitesprenatales", "nombremembresmenage", "ageenfant"
]

vars_continues_brutes = [
    "imcmere", "ageenfant", "agemere"
]

vars_categorielles = [
    "sexeenfant", "milieuderesidence", "niveauinstructionmere", "indicederichesse", "region", "vacciné",
    "vaccinbcg", "diarrhee", "prisededecisionmere", "statutmatrimonialmere", "tailleanaissance",
    "sourceeaupotable", "typeinstallationssanitaires", "typecombustiblecuisine", "lieuderesidence",
    "regionecologique", "vaccinpolio0", "vaccinpolio2", "vaccinpolio3", "vaccindtp1", "vaccindtp2", 
    "vaccindtp3", "vaccinpolio1", "vaccinrougeole1"
]

toutes_covariables_etude = list(set(vars_discretes + vars_continues_brutes + vars_categorielles + ["eau_amelioree", "toilettes_ameliorees", "index_wash_synergie", "pauvreté_rurale", "ratio_charge_menage"]))
covariables_ml = [col for col in toutes_covariables_etude if col in df_ml.columns]

# 2. Partitionnement etanche par grappes en 3 compartiments
print("Phase 2 : Partitionnement par grappes en 3 compartiments")
X_complet = df_ml[covariables_ml].copy()
y_complet = df_ml['Y_ICEA'].copy()
poids_complet = df_ml['poids_analytique'].copy()
grappes_complet = df_ml['numerogruppe'].copy()

# Isolation du pool de stress test (15% des grappes)
gss_stress = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=2018)
train_val_idx, stress_idx = next(gss_stress.split(X_complet, y_complet, groups=grappes_complet))

X_train_val, X_stress = X_complet.iloc[train_val_idx], X_complet.iloc[stress_idx]
y_train_val, y_stress = y_complet.iloc[train_val_idx], y_complet.iloc[stress_idx]
poids_train_val, poids_stress = poids_complet.iloc[train_val_idx], poids_complet.iloc[stress_idx]
grappes_train_val, grappes_stress = grappes_complet.iloc[train_val_idx], grappes_complet.iloc[stress_idx]

# Decoupage standard train (80%) / test (20%) sur les grappes restantes
gss_standard = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss_standard.split(X_train_val, y_train_val, groups=grappes_train_val))

X_train, X_test = X_train_val.iloc[train_idx], X_train_val.iloc[test_idx]
y_train, y_test = y_train_val.iloc[train_idx], y_train_val.iloc[test_idx]
poids_train, poids_test = poids_train_val.iloc[train_idx], poids_train_val.iloc[test_idx]
grappes_train, grappes_test = grappes_train_val.iloc[train_idx], grappes_train_val.iloc[test_idx]

print(f"Train set : {X_train.shape} observations ({len(np.unique(grappes_train))} grappes)")
print(f"Test set  : {X_test.shape} observations ({len(np.unique(grappes_test))} grappes)")
print(f"Stress set : {X_stress.shape} observations ({len(np.unique(grappes_stress))} grappes)")

# 3. Pipeline de transformation et encodage sans fuite
print("Phase 3 : Pipeline de transformation et encodage")
features_continues = [col for col in X_complet.columns if X_complet[col].dtype in ['float64', 'int64'] and X_complet[col].nunique() > 5]
features_categoriques = [col for col in X_complet.columns if col not in features_continues]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_continues),
        ('cat', TargetEncoder(smooth="auto", cv=5, random_state=2018), features_categoriques)
    ]
)

X_train_clean = preprocessor.fit_transform(X_train, y_train)
X_test_clean = preprocessor.transform(X_test)
X_stress_clean = preprocessor.transform(X_stress)

# Harmonisation des poids pour le desequilibre des classes
taux_malnutrition = np.average(y_train, weights=poids_train)
poids_ajustement_classe = np.where(y_train == 1, 1.0 / taux_malnutrition, 1.0 / (1.0 - taux_malnutrition))
poids_final_train = poids_train * poids_ajustement_classe

noms_colonnes_post_encode = (
    features_continues + 
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(features_categoriques))
)

# 4. Optimisation des hyperparametres du SVM par validation croisee par groupe
print("Phase 4 : Fine-tuning de l'architecture SVM Classifier")
param_distributions = {
    'C': stats.loguniform(1e-2, 1e2),
    'gamma': stats.loguniform(1e-3, 1e1),
    'kernel': ['rbf']
}

cv_groupes = GroupKFold(n_splits=3)
svm_base = SVC(probability=True, class_weight='balanced', cache_size=1000, random_state=2018)

search_cv = RandomizedSearchCV(
    estimator=svm_base,
    param_distributions=param_distributions,
    n_iter=6,
    scoring='roc_auc',
    cv=cv_groupes,
    n_jobs=-1,
    random_state=2018
)

search_cv.fit(X_train_clean, y_train, groups=grappes_train, sample_weight=poids_final_train)
meilleur_modele = search_cv.best_estimator_
nom_meilleur_modele = "SVM Classifier Optimisé"
print(f"Hyperparamètres optimaux retenus pour le SVM : {search_cv.best_params_}")

# 5. Evaluation et stress testing du modele
print("Phase 5 : Stress testing et evaluation de la robustesse")
prob_train = meilleur_modele.predict_proba(X_train_clean)[:, 1]
prob_test = meilleur_modele.predict_proba(X_test_clean)[:, 1]
prob_stress = meilleur_modele.predict_proba(X_stress_clean)[:, 1]

class_train = (prob_train >= 0.5).astype(int)
class_test = (prob_test >= 0.5).astype(int)
class_stress = (prob_stress >= 0.5).astype(int)

auc_train = roc_auc_score(y_train, prob_train, sample_weight=poids_final_train)
loss_train = log_loss(y_train, prob_train, sample_weight=poids_final_train)
f1_train = f1_score(y_train, class_train, sample_weight=poids_final_train, average='macro')

auc_test = roc_auc_score(y_test, prob_test, sample_weight=poids_test)
loss_test = log_loss(y_test, prob_test, sample_weight=poids_test)
f1_test = f1_score(y_test, class_test, sample_weight=poids_test, average='macro')

auc_stress = roc_auc_score(y_stress, prob_stress, sample_weight=poids_stress)
loss_stress = log_loss(y_stress, prob_stress, sample_weight=poids_stress)
f1_stress = f1_score(y_stress, class_stress, sample_weight=poids_stress, average='macro')

df_bilan_final_ml = pd.DataFrame({
    'Compartiment_Echantillon': ['1_Apprentissage (Train)', '2_Evaluation Standard (Test)', '3_Stress_Testing (Grappes Inconnues)'],
    'ROC_AUC_Global': [auc_train, auc_test, auc_stress],
    'Log_Loss_Unifie': [loss_train, loss_test, loss_stress],
    'F1_Score_Macro': [f1_train, f1_test, f1_stress],
    'Ecart_Stabilite_vs_Train': [0.0, auc_train - auc_test, auc_train - auc_stress]
})
df_bilan_final_ml.to_csv(f"{output_dir}/t10_bilan_inference_stress_testing_svm.csv", index=False)
print("Bilan des métriques train, test et stress test genere.")

# 6. Generation des graphiques academiques et d'interpretabilite
print("Phase 6 : Generation des graphiques et calcul de l'importance par permutation")

# Graphique A : Courbe ROC finale du SVM sur le Test et le Stress Test
plt.figure(figsize=(7, 6))
fpr_t, tpr_t, _ = roc_curve(y_test, prob_test, sample_weight=poids_test)
fpr_s, tpr_s, _ = roc_curve(y_stress, prob_stress, sample_weight=poids_stress)

plt.plot(fpr_t, tpr_t, color='dodgerblue', lw=2.5, label=f"Jeu d'Évaluation Standard (AUC = {auc_test:.3f})")
plt.plot(fpr_s, tpr_s, color='crimson', lw=2.5, linestyle='-.', label=f"Stress Test Grappes Inconnues (AUC = {auc_stress:.3f})")
plt.plot([0, 1], [0, 1], color='black', lw=1.2, linestyle='--', label='Seuil Aléatoire (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Spécificité)')
plt.ylabel('Taux de Vrais Positifs (Sensibilité)')
plt.title("Courbe ROC Finale SVM Post-Optimisation et Stress Testing\n(Validation de Généralisation Nationale EDSC-V)")
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()

path_roc_svm = f"{graph_dir}/g7_courbe_roc_optimisee_stress_svm.png"
plt.savefig(path_roc_svm, dpi=300)
plt.close()
print(f"Graphique de la courbe ROC exporte dans : '{path_roc_svm}'")

# Graphique B : Courbe d'apprentissage du SVM pour valider la non-variance
plt.figure(figsize=(7.5, 6))
sizes, t_scores, v_scores = learning_curve(
    meilleur_modele, X_train_clean, y_train, cv=cv_groupes, groups=grappes_train, scoring='roc_auc',
    train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1, random_state=2018
)

plt.plot(sizes, np.mean(t_scores, axis=1), 'o-', color='crimson', lw=2, label="Score d'Entraînement (Train)")
plt.plot(sizes, np.mean(v_scores, axis=1), 's-', color='dodgerblue', lw=2, label='Score de Validation Croisée (CV)')
plt.xlabel("Taille de l'échantillon d'apprentissage")
plt.ylabel('Performance ROC AUC')
plt.title("Courbe d'Apprentissage du Modèle Champion SVM Optimisé\n(Preuve de la convergence sans surapprentissage)")
plt.ylim(0.5, 1.02)
plt.legend(loc="lower right")
plt.tight_layout()

path_lc_svm = f"{graph_dir}/g8_courbe_apprentissage_optimisee_svm.png"
plt.savefig(path_lc_svm, dpi=300)
plt.close()
print(f"Graphique de la courbe d'apprentissage exporte dans : '{path_lc_svm}'")

# Graphique C : Importance des variables par permutation du SVM
resultat_permutation = permutation_importance(
    meilleur_modele, X_test_clean, y_test, scoring='roc_auc', n_repeats=10, random_state=2018, n_jobs=-1
)

# Structuration du tableau d'importance par permutation
df_importance = pd.DataFrame({
    'Variable_DHS_Fr': covariables_ml,
    'Importance_Moyenne_AUC': resultat_permutation.importances_mean,
    'Ecart_Type_Importance': resultat_permutation.importances_std
}).sort_values(by='Importance_Moyenne_AUC', ascending=False)

# Archivage du tableau d'importance
path_table_imp = f"{output_dir}/t8_importance_features_permutation_svm.csv"
df_importance.to_csv(path_table_imp, index=False)
print(f"Tableau de l'importance par permutation enregistre dans : '{path_table_imp}'")

# Traci visuel du Top 15 des facteurs d'influence du SVM
plt.figure(figsize=(10, 7.5))
sns.barplot(
    x='Importance_Moyenne_AUC', y='Variable_DHS_Fr', data=df_importance.head(15), 
    palette='mako', hue='Variable_DHS_Fr', legend=False
)

# Ajout des barres d'erreur exactes de la permutation pour le jury
y_positions = np.arange(15)
plt.errorbar(
    df_importance.head(15)['Importance_Moyenne_AUC'], y_positions, 
    xerr=df_importance.head(15)['Ecart_Type_Importance'], 
    fmt='none', color='black', elinewidth=1.5, capsize=4
)

plt.xlabel('Dégradation moyenne du score ROC AUC sur le jeu de test')
plt.ylabel('Variables Explicatives du Contexte SVM')
plt.title("Top 15 des Facteurs de Malnutrition du Contexte SVM\nAnalyse par Permutation issue du SVM Champion")
plt.tight_layout()

path_imp_graph = f"{graph_dir}/g4_importance_features_permutation_svm.png"
plt.savefig(path_imp_graph, dpi=300)
plt.close()
print(f"Graphique de l'importance par permutation exporte dans : '{path_imp_graph}'")

# 7. Inference clinique marginalisee sur le pool de stress test
print("Phase 7 : Architecture de marginalisation clinique pour les sous-classes OMS")
df_stress_real = df_ml.iloc[stress_idx].copy()
df_stress_real['probabilite_icea_unifiee'] = prob_stress

# Calcul des probabilites conditionnelles historiques de l'EDSC-V 2018
p_y1_sachant_icea = df_ml[df_ml['Y_ICEA'] == 1]['Y1_stunting'].mean()
p_y2_sachant_icea = df_ml[df_ml['Y_ICEA'] == 1]['Y2_wasting'].mean()
p_y3_sachant_icea = df_ml[df_ml['Y_ICEA'] == 1]['Y3_underweight'].mean()

# Applications des probabilites marginales decodees
df_stress_real['prob_retard_croissance'] = df_stress_real['probabilite_icea_unifiee'] * p_y1_sachant_icea
df_stress_real['prob_amaigrissement'] = df_stress_real['probabilite_icea_unifiee'] * p_y2_sachant_icea
df_stress_real['prob_insuffisance_ponderale'] = df_stress_real['probabilite_icea_unifiee'] * p_y3_sachant_icea

colonnes_livrables_cliniques = ['numerogruppe', 'numeromenage', 'ageenfant', 'probabilite_icea_unifiee', 'prob_retard_croissance', 'prob_amaigrissement', 'prob_insuffisance_ponderale']
df_stress_real[colonnes_livrables_cliniques].to_csv(f"{output_dir}/t11_predictions_marginalisees_sous_classes_svm.csv", index=False)
print("Fichier de prediction clinique t11 genere avec succes.")

# 8. Sauvegarde de la matrice propre, metriques de reussite et stockage des objets physiques pour l'API
print("Phase 8 : Sauvegarde finale et archivage des artefacts pour l'API predictive")

df_train_final_matrix = pd.DataFrame(X_train_clean, columns=noms_colonnes_post_encode)
df_train_final_matrix['y_decision_unifiee'] = y_train.values
df_train_final_matrix.to_csv(f"{output_dir}/matrice_enfants_entrainement_final_svm.csv", index=False)

# Evaluation sur des grappes et des strates geographiques non observees
exactitude_generale = accuracy_score(y_stress, class_stress, sample_weight=poids_stress)
precision_clinique = precision_score(y_stress, class_stress, sample_weight=poids_stress, zero_division=0)
rappel_clinique = recall_score(y_stress, class_stress, sample_weight=poids_stress, zero_division=0)

df_metriques_stress_reelles = pd.DataFrame({
    'Metrique_Inference': ['Proportion de predictions correctes (Accuracy)', 'Precision de detection', 'Sensibilite clinique (Recall)'],
    'Valeur_Sur_Grappes_Inconnues': [exactitude_generale, precision_clinique, rappel_clinique]
})
df_metriques_stress_reelles.to_csv(f"{output_dir}/t12_metriques_reelles_stress_testing_svm.csv", index=False)

# Validation des diagnostics specifiques pour chaque sous-classe OMS sur le Stress Test
y1_vrai_stress = df_ml.iloc[stress_idx]['Y1_stunting'].values
y2_vrai_stress = df_ml.iloc[stress_idx]['Y2_wasting'].values
y3_vrai_stress = df_ml.iloc[stress_idx]['Y3_underweight'].values

classes_predites_stunting = (df_stress_real['prob_retard_croissance'] >= df_stress_real['prob_retard_croissance'].median()).astype(int)
classes_predites_wasting = (df_stress_real['prob_amaigrissement'] >= df_stress_real['prob_amaigrissement'].median()).astype(int)
classes_predites_underweight = (df_stress_real['prob_insuffisance_ponderale'] >= df_stress_real['prob_insuffisance_ponderale'].median()).astype(int)

acc_stunting = accuracy_score(y1_vrai_stress, classes_predites_stunting, sample_weight=poids_stress)
acc_wasting = accuracy_score(y2_vrai_stress, classes_predites_wasting, sample_weight=poids_stress)
acc_underweight = accuracy_score(y3_vrai_stress, classes_predites_underweight, sample_weight=poids_stress)

df_comparatif_sous_classes = pd.DataFrame({
    'Pathologie_OMS': ['Retard de croissance (Stunting)', 'Amaigrissement aigu (Wasting)', 'Insuffisance ponderale (Underweight)'],
    'Proportion_Exacte_Predictions': [acc_stunting, acc_wasting, acc_underweight]
})
df_comparatif_sous_classes.to_csv(f"{output_dir}/t13_validation_predictions_sous_classes_svm.csv", index=False)

# Sauvegarde conditionnelle des objets physiques pour FastAPI
if auc_stress >= 0.70:
    print(f"Critere de validation atteint (AUC Stress = {auc_stress:.4f}). Sauvegarde autorisee.")
    joblib.dump(meilleur_modele, f"{model_dir}/svm_inference_core.joblib")
    joblib.dump(preprocessor, f"{model_dir}/column_transformer_api_svm.joblib")
    joblib.dump(features_continues, f"{model_dir}/liste_variables_continues_svm.joblib")
    joblib.dump(features_categoriques, f"{model_dir}/liste_variables_categoriques_svm.joblib")
    joblib.dump(covariables_ml, f"{model_dir}/ordre_exact_variables_entree_svm.joblib")

    with open(f"{model_dir}/svm_inference_core.pkl", "wb") as f:
        pickle.dump(meilleur_modele, f)
    with open(f"{model_dir}/column_transformer_api_svm.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    dictionnaire_taux_cliniques = {
        'p_y1_sachant_icea': float(p_y1_sachant_icea),
        'p_y2_sachant_icea': float(p_y2_sachant_icea),
        'p_y3_sachant_icea': float(p_y3_sachant_icea)
    }
    joblib.dump(dictionnaire_taux_cliniques, f"{model_dir}/taux_conditionnels_cliniques_svm.joblib")
    print("Execution et stockage des objets physiques acheves avec succes.")
else:
    print("Sauvegarde annulee : stabilite insuffisante sur le compartiment de stress test.")
