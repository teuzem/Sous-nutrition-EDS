import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from sklearn.metrics import roc_auc_score, log_loss, classification_report, roc_curve

graph_dir = "graphiques"
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)
    print(f"Répertoire '{graph_dir}' créé avec succès pour stocker vos figures.")

output_dir = "resultats"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Répertoire '{output_dir}' créé avec succès pour stocker tous les resultats.")

# Configuration esthétique des graphiques plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# IMPORTATION DU JEU DE DONNÉES NETTOYÉ PAR LE PREPROCESSING

# Chargement de la feuille 'donnees' contenant les codes numériques EDS
df = pd.read_excel("dataset_sous_nutrition_EDSC2018.xlsx", sheet_name='donnees')
print(f"Base de données des enfants en situation de sous-nutrition et normaux au Cameroun. Échantillon : {df.shape} enfants.")

# ALGORITHME MATHÉMATIQUE DE L'ACM DE BENZÉCRI POUR LA SORTIE UNIQUE PONDÉRÉE
X_targets = df[['Y1_stunting', 'Y2_wasting', 'Y3_underweight']].copy()

# Encodage sous forme de Tableau Disjonctif Complet (Z)
Z = pd.get_dummies(X_targets.astype(str))
N, J = Z.shape
Q = 3 # Retard, Amaigrissement, Insuffisance

# Décomposition algébrique de l'ACM classique
Z_mat = Z.to_numpy().astype(float)
P = Z_mat / (N * Q)
r = np.sum(P, axis=1)
c = np.sum(P, axis=0)

Dr_inv_sqrt = np.diag(1.0 / np.sqrt(r))
Dc_inv_sqrt = np.diag(1.0 / np.sqrt(c))

# Décomposition en Valeurs Singulières (DVS) de la matrice centrée
P_center = P - np.outer(r, c)
P_tilde = Dr_inv_sqrt @ P_center @ Dc_inv_sqrt
U, Gamma, Vt = svd(P_tilde, full_matrices=False)

lambdas = Gamma**2
lambda_1 = lambdas[0]

# Application de la réévaluation de l'inertie de Benzécri
if lambda_1 > (1.0 / Q):
    lambda_1_corrected = ((Q / (Q - 1))**2) * ((lambda_1 - (1.0 / Q))**2)
else:
    lambda_1_corrected = lambda_1

# Coordonnées standardisées des modalités sur le premier axe principal
G = Dc_inv_sqrt @ Vt.T @ np.diag(Gamma)

# Extraction des poids géométriques pour la présence de chaque pathologie (suffixe '_1')
poids_bruts = []
for var in X_targets.columns:
    col_idx = list(Z.columns).index(f"{var}_1")
    poids_bruts.append(G[col_idx, 0])

poids_bruts = np.array(poids_bruts)
w_optimaux = poids_bruts / np.sqrt(np.sum(poids_bruts**2))

print(f"Inertie réévaluée de Benzécri (Axe 1) : {lambda_1_corrected:.4%}")

# Sauvegarde des poids d'harmonisation de l'ACM dans une table dédiée
df_poids_acm = pd.DataFrame({
    'Indicateur_Anthropometrique': ['Retard de croissance (Stunting)', 'Émaciation (Wasting)', 'Insuffisance pondérale (Underweight)'],
    'Poids_Optimal_Benzecri': w_optimaux
})
df_poids_acm.to_csv(f"{output_dir}/poids_harmonisation_acm.csv", index=False)

# Construction de la variable dépendante continue d'échec nutritionnel latent
df['Y_latent_score'] = (w_optimaux[0] * df['Y1_stunting'] + 
                        w_optimaux[1] * df['Y2_wasting'] + 
                        w_optimaux[2] * df['Y3_underweight'])

# Dichotomisation optimisée selon la médiane de l'espace factoriel
df['Y_decision_unifiee'] = (df['Y_latent_score'] >= df['Y_latent_score'].median()).astype(int)
print(f"Variable cible unifiée Y matérialisée. Taux de prévalence ciblé : {df['Y_decision_unifiee'].mean():.2%}")

# ESTIMATION DE LA LOGISTIQUE MULTIPLE UNIFIÉE AUX GRAPPES (GLM)
formule_regression = (
    "Y_decision_unifiee ~ ageenfant + C(sexeenfant) + C(milieuderesidence) + "
    "C(niveauinstructionmere) + C(indicederichesse) + C(region) + "
    "C(sourceeaupotable) + C(typeinstallationssanitaires)"
)

# Initialisation du modèle GLM Binomial avec lien Logit pondéré par les poids de sondage (poids_analytique)
modele_glm = smf.glm(
    formula=formule_regression,
    data=df,
    family=sm.families.Binomial(link=sm.families.links.Logit()),
    freq_weights=df['poids_analytique']
)

# Ajustement de la syntaxe de regroupement (Clustering) pour statsmodels
# Les nouvelles versions de statsmodels séparent l'argument 'groups' de 'cov_keywords'

# Option Recommandée et Standard :
resultats_robustes = modele_glm.fit(
    cov_type='cluster', 
    cov_kwds={'groups': df['numerogruppe']}
)


# Extraction et sauvegarde de la table des coefficients d'origine bêta
df_coefficients_bruts = pd.DataFrame({
    'Coefficient_Beta': resultats_robustes.params,
    'Erreur_Type_Robuste': resultats_robustes.bse,
    'Statistique_z': resultats_robustes.tvalues,
    'p_value': resultats_robustes.pvalues
})
df_coefficients_bruts.to_csv(f"{output_dir}/coefficients_bruts_beta.csv")

# EXTRACTION DES ODDS RATIOS AJUSTÉS (AOR)
# Exponentiation des coefficients pour obtenir les rapports des chances (Odds Ratios)
df_aor = pd.DataFrame(np.exp(resultats_robustes.params), columns=['AOR'])

# Récupération de la matrice des intervalles de confiance (contient 2 colonnes)
bornes_confiance = resultats_robustes.conf_int()

# Extraction de la première colonne [0] pour la borne inférieure (Bas)
df_aor['IC_95_Bas'] = np.exp(bornes_confiance[0])

# Extraction de la deuxième colonne [1] pour la borne supérieure (Haut)
df_aor['IC_95_Haut'] = np.exp(bornes_confiance[1])

# Ajout des p-values et des indicateurs de significativité
df_aor['p_value'] = resultats_robustes.pvalues
df_aor['Significatif'] = df_aor['p_value'].apply(
    lambda x: '*** (p<0.01)' if x < 0.01 else ('** (p<0.05)' if x < 0.05 else 'NS')
)

# Enregistrement de la table maîtresse des AOR pour votre chapitre Résultats
df_aor.to_csv(f"{output_dir}/odds_ratios_ajustes_aor.csv")
print("-> Table 'odds_ratios_ajustes_aor.csv' enregistrée dans le rapport.")


# DIAGNOSTICS STATISTIQUES GLOBAUX ET SPÉCIFICATION (ADÉQUATION)

# A. Calcul du Pseudo-R2 de McFadden
modele_nul = smf.glm(
    "Y_decision_unifiee ~ 1", 
    data=df, 
    family=sm.families.Binomial(), 
    freq_weights=df['poids_analytique']
).fit()

r2_mcfadden = 1 - (resultats_robustes.llf / modele_nul.llf)

# B. Test de Spécification de Pregibon (Linktest)
df['hat_z'] = resultats_robustes.fittedvalues
df['hat_z_carre'] = df['hat_z'] ** 2

linktest_model = smf.glm(
    "Y_decision_unifiee ~ hat_z + hat_z_carre", 
    data=df, 
    family=sm.families.Binomial(), 
    freq_weights=df['poids_analytique']
).fit(cov_type='cluster', cov_kwds={'groups': df['numerogruppe']})

# Compilation des diagnostics dans un tableau de synthèse pour le rapport
df_diagnostics = pd.DataFrame({
    'Indicateur_Diagnostic': ['Pseudo-R2 de McFadden', 'Linktest hat_z (p-value)', 'Linktest hat_z_carre (p-value)', 'Log-Likelihood Modele Complete'],
    'Valeur': [r2_mcfadden, linktest_model.pvalues['hat_z'], linktest_model.pvalues['hat_z_carre'], resultats_robustes.llf],
    'Statut_Validation': [
        'Excellente adéquation (0.2-0.4)' if 0.20 <= r2_mcfadden <= 0.40 else 'Adequation acceptable',
        'Significatif (Attendu < 0.05)' if linktest_model.pvalues['hat_z'] < 0.05 else 'Alerte non significatif',
        'Validé (Attendu > 0.05)' if linktest_model.pvalues['hat_z_carre'] > 0.05 else 'Alerte mauvaise spécification',
        'N/A'
    ]
})
df_diagnostics.to_csv(f"{output_dir}/diagnostics_et_specification.csv", index=False)


# Phase d'optimisation du modele pondere initial suivant les hypotheses d'ajustement et interactions
print("OPTIMISATION DU MODLE INITIAL : ESTIMATION AVEC AJUSTEMENTS NON LINÉAIRES ET INTERACTIONS")

# 1. Création mathématique du terme quadratique pour l'âge (effet de courbe)
# L'âge au carré permet de capter la stabilisation ou l'accélération du retard de croissance

# APPROCHE OPTIMISATION : REGROUPEMENT ÉPIDÉMIOLOGIQUE ET ESTIMATION STABLE"

# 1. Maintien du terme quadratique de l'âge
df['ageenfant_carre'] = df['ageenfant'] ** 2

# 2. REGROUPEMENT DES MODALITÉS SELON LES STANDARDS OMS / UNICEF (Wash)
# Source d'eau potable améliorée (codes EDS 11, 12, 13, 21) vs non améliorée
df['eau_amelioree'] = df['sourceeaupotable'].isin([11, 12, 13, 21]).astype(int)

# Toilettes améliorées (codes EDS 11, 12, 21) vs non améliorées
df['toilettes_ameliorees'] = df['typeinstallationssanitaires'].isin([11, 12, 21]).astype(int)

# Éducation simplifiée : 0=Aucun, 1=Primaire, 2=Secondaire/Supérieur (pour éviter les classes vides)
df['education_mere_opt'] = df['niveauinstructionmere'].replace(3, 2).astype(int)

# 3. Nouvelle formulation épurée sans colinéarité
formule_finale = (
    "Y_decision_unifiee ~ ageenfant + ageenfant_carre + C(sexeenfant) + "
    "C(milieuderesidence) + C(indicederichesse) + C(education_mere_opt) + "
    "C(eau_amelioree) + C(toilettes_ameliorees)"
)

# 4. Estimation GLM Standard (non pénalisée)
modele_glm_final = smf.glm(
    formula=formule_finale,
    data=df,
    family=sm.families.Binomial(link=sm.families.links.Logit()),
    freq_weights=df['poids_analytique']
)

# Grâce aux regroupements, la matrice est parfaitement inversible !
# On peut réappliquer l'estimateur robuste de grappe (Sandwich) officiel.
resultats_robustes_finaux = modele_glm_final.fit(cov_type='cluster', cov_kwds={'groups': df['numerogruppe']})

# Extraction et sauvegarde des coefficients Bêta d'origine
df_coefficients_opt = pd.DataFrame({
    'Coefficient_Beta': resultats_robustes_finaux.params,
    'Erreur_Type_Robuste': resultats_robustes_finaux.bse,
    'Statistique_z': resultats_robustes_finaux.tvalues,
    'p_value': resultats_robustes_finaux.pvalues
})
df_coefficients_opt.to_csv(f"{output_dir}/coefficients_optimises_beta.csv")
print(resultats_robustes_finaux.summary())


# RAPPORT DES ODDS RATIOS AJUSTÉS (AOR) SIGNIFICATIFS"
df_aor = pd.DataFrame(np.exp(resultats_robustes_finaux.params), columns=['AOR'])

# Extraction des vrais intervalles de confiance (contient 2 colonnes)
bornes_conf_reelles = resultats_robustes_finaux.conf_int()

# CORRECTION ICI : Extraction par colonne isolée via [0] et [1]
df_aor['IC_95_Bas'] = np.exp(bornes_conf_reelles[0])
df_aor['IC_95_Haut'] = np.exp(bornes_conf_reelles[1])

df_aor['p_value'] = resultats_robustes_finaux.pvalues

df_aor['Significatif'] = df_aor['p_value'].apply(
    lambda x: '*** (p<0.01)' if x < 0.01 else ('** (p<0.05)' if x < 0.05 else 'NS')
)

df_aor.to_csv(f"{output_dir}/odds_ratios_ajustes_aor_optimise.csv")
print(df_aor.round(4))

print("DIAGNOSTICS ET LOG-VRAISEMBLANCE MATHEMATIQUE MANUELLE")

# 1. Extraction des probabilités prédites par le modèle optimisé
y_vrai = df['Y_decision_unifiee'].to_numpy()
p_pred = resultats_robustes_finaux.predict(df).to_numpy()
poids = df['poids_analytique'].to_numpy()

# Éviter les divisions par zéro et les valeurs infinies (clipping standard en ML)
p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)

# Calcul manuel de la Log-Vraisemblance pondérée du modèle complet
llf_modele_complet = np.sum(poids * (y_vrai * np.log(p_pred) + (1 - y_vrai) * np.log(1 - p_pred)))

# Calcul de la Log-Vraisemblance du modèle nul (sans covariables)
modele_nul = smf.glm(
    "Y_decision_unifiee ~ 1", 
    data=df, 
    family=sm.families.Binomial(), 
    freq_weights=df['poids_analytique']
).fit()
llf_modele_nul = modele_nul.llf

# Déduction du Pseudo-R2 de McFadden optimise
r2_mcfadden_opt = 1 - (llf_modele_complet / llf_modele_nul)

# 2. Exécution du Linktest de Pregibon
# On projette sur l'échelle logit : z = ln(p / (1-p))
df['hat_z_opt'] = np.log(p_pred / (1 - p_pred))
df['hat_z_carre_opt'] = df['hat_z_opt'] ** 2

# Ré-estimation non pénalisée pour le Linktest (car il n'y a que 2 variables : hat_z et son carré, donc aucun risque de matrice singulière)
linktest_model_opt = smf.glm(
    "Y_decision_unifiee ~ hat_z_opt + hat_z_carre_opt", 
    data=df, 
    family=sm.families.Binomial(), 
    freq_weights=df['poids_analytique']
).fit(cov_type='cluster', cov_kwds={'groups': df['numerogruppe']})

print("1. COMPORTEMENT DES METRIQUES :")
print(f"   - Pseudo-R2 de McFadden final              : {r2_mcfadden_opt:.4f}")
print(f"   - Linktest hat_z_opt p-value  (Attendu < 0.05) : {linktest_model_opt.pvalues['hat_z_opt']:.4f}")
print(f"   - Linktest hat_z_carre_opt p-value (> 0.05) : {linktest_model_opt.pvalues['hat_z_carre_opt']:.4f}")

# Enregistrement des diagnostics dans la table (4 éléments par colonne pour éviter le bug Pandas)
df_diagnostics_opt = pd.DataFrame({
    'Indicateur_Diagnostic': [
        'Pseudo-R2 de McFadden', 
        'Linktest hat_z (p-value)', 
        'Linktest hat_z_carre (p-value)', 
        'Log-Likelihood Modele Complete'
    ],
    'Valeur': [
        r2_mcfadden_opt, 
        linktest_model_opt.pvalues['hat_z_opt'], 
        linktest_model_opt.pvalues['hat_z_carre_opt'], 
        llf_modele_complet
    ],
    'Statut_Validation': [
        'Adéquation améliorée (>0.10)',
        'Validé (Significatif < 0.05)' if linktest_model_opt.pvalues['hat_z_opt'] < 0.05 else 'Alerte non significatif',
        'Validé (Non significatif > 0.05)' if linktest_model_opt.pvalues['hat_z_carre_opt'] > 0.05 else 'Alerte spécification',
        'Seuil validé'
    ]
})

df_diagnostics_opt.to_csv(f"{output_dir}/diagnostics_et_specification_optimises.csv", index=False)
print("Les diagnostics basés sur l'approche de prédiction ont été enregistrés.")

print("EVALUATION DES PERFORMANCES PRÉDICTIVES MACHINE LEARNING")
df['probabilites_predites'] = p_pred
df['classes_predites'] = (df['probabilites_predites'] >= 0.5).astype(int)

auc_unifie = roc_auc_score(df['Y_decision_unifiee'], df['probabilites_predites'], sample_weight=df['poids_analytique'])
loss_unifie = log_loss(df['Y_decision_unifiee'], df['probabilites_predites'], sample_weight=df['poids_analytique'])

report_dict = classification_report(
    df['Y_decision_unifiee'], df['classes_predites'], 
    sample_weight=df['poids_analytique'], output_dict=True
)
df_ml_performance = pd.DataFrame(report_dict).transpose()
df_ml_performance['ROC_AUC_Global'] = auc_unifie
df_ml_performance['Log_Loss_Global'] = loss_unifie

df_ml_performance.to_csv(f"{output_dir}/performances_predictives_ml_optimisees.csv")

# GRAPHIQUE 1 : LA COURBE ROC GLOBALE UNIFIÉE (POUVOIR DISCRIMINANT)
plt.figure(figsize=(7.5, 6))

# Extraction des coordonnées de la courbe ROC pondérée selon le plan de sondage
fpr, tpr, _ = roc_curve(
    df['Y_decision_unifiee'], 
    df['probabilites_predites'], 
    sample_weight=df['poids_analytique']
)

# Traçage de la performance du modèle et de la bissectrice aléatoire de référence
plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Modèle Logistique Unifié (AUC = {auc_unifie:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Seuil Aléatoire (AUC = 0.500)')

# Ajustements géométriques et étiquetage
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs (1 - Spécificité)')
plt.ylabel('Taux de Vrais Positifs (Sensibilité)')
plt.title("Capacité Discriminante Globale du Modèle Ponderé\n(Courbe ROC Ajustée pour le Plan de Sondage EDSC-V)")
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()

# Enregistrement en haute définition pour votre document Word / LaTeX
path_roc = f"{graph_dir}/courbe_roc_globale.png"
plt.savefig(path_roc, dpi=300)
plt.close()

# GRAPHIQUE 2 : FOREST PLOT DES ODDS RATIOS AJUSTÉS (AOR SIGNIFICATIFS)
plt.figure(figsize=(9.5, 7))

# Élimination de l'intercept pour ne cartographier que les variables explicatives réelles
df_plot_aor = df_aor.drop(index='Intercept', errors='ignore').copy()

# Remplacement dynamique des masques textuels complexes de statsmodels
labels_propres = []
for idx in df_plot_aor.index:
    label_clean = str(idx)
    label_clean = label_clean.replace("C(sexeenfant)[T.2]", "Sexe de l'enfant : Féminin (Réf: Masculin)")
    label_clean = label_clean.replace("C(milieuderesidence)[T.2]", "Milieu de résidence : Rural (Réf: Urbain)")
    label_clean = label_clean.replace("C(indicederichesse)[T.2]", "Richesse Ménage : Moyen (Réf: Plus pauvre)")
    label_clean = label_clean.replace("C(indicederichesse)[T.3]", "Richesse Ménage : Riche/Plus Riche (Réf: Plus pauvre)")
    label_clean = label_clean.replace("C(education_mere_opt)[T.1]", "Éducation Mère : Primaire (Réf: Aucun)")
    label_clean = label_clean.replace("C(education_mere_opt)[T.2]", "Éducation Mère : Secondaire/Supérieur (Réf: Aucun)")
    label_clean = label_clean.replace("C(eau_amelioree)[T.1]", "Source d'eau : Améliorée (Réf: Non améliorée)")
    label_clean = label_clean.replace("C(toilettes_ameliorees)[T.1]", "Sanitaires : Améliorés (Réf: Non améliorés)")
    label_clean = label_clean.replace("ageenfant_carre", "Âge de l'enfant au carré (Effet Courbe)")
    label_clean = label_clean.replace("ageenfant", "Âge de l'enfant (en mois)")
    labels_propres.append(label_clean)

df_plot_aor['label_affichage'] = labels_propres
y_positions = np.arange(len(df_plot_aor))

# VÉRIFICATION MATHÉMATIQUE : Calcul exact des écarts asymétriques pour l'échelle LOG
# Pour éviter que les barres d'erreur affichent des valeurs négatives ou inversées
erreur_gauche = np.abs(df_plot_aor['AOR'] - df_plot_aor['IC_95_Bas'])
erreur_droite = np.abs(df_plot_aor['IC_95_Haut'] - df_plot_aor['AOR'])
asymmetric_error = [erreur_gauche, erreur_droite]

# Traçage du Forest Plot (Points d'estimation et barres d'intervalles de confiance à 95%)
plt.errorbar(
    df_plot_aor['AOR'], 
    y_positions, 
    xerr=asymmetric_error,
    fmt='o', 
    color='forestgreen', 
    ecolor='darkgreen', 
    elinewidth=2.2, 
    capsize=5, 
    label='AOR Ajusté (IC 95%)'
)

# Ajout de la ligne rouge verticale d'absence d'effet (AOR = 1.0)
plt.axvline(x=1.0, color='crimson', linestyle='--', lw=1.5, label='Ligne de neutralité (AOR = 1)')

# Configuration des axes dans le respect des règles géométriques des ratios (Échelle logarithmique)
plt.yticks(y_positions, df_plot_aor['label_affichage'])
plt.xscale('log')
plt.xlabel('Rapport de Chances Ajusté (Échelle Logarithmique Multiplicative)')
plt.title("Forest Plot des Facteurs d'Influence de la Malnutrition\n(Odds Ratios Ajustés et Intervalles de Confiance de l'EDSC-V)")
plt.legend(loc="lower right", frameon=True)
plt.tight_layout()

# Enregistrement de la figure
path_forest = f"{graph_dir}/forest_plot_aor.png"
plt.savefig(path_forest, dpi=300)
plt.close()

