import pandas as pd
import numpy as np
import xlsxwriter

print("ÉTAPES 1 : CHARGEMENT INITIAL DU FICHIER EDSC-V 2018")
# Lecture brute sans transformation textuelle pour conserver les codes numériques
reader = pd.read_stata("CMKR71FL.DTA", chunksize=None, convert_categoricals=False)
df_raw = reader.copy()

# Dictionnaire de correspondance entre les codes DHS officiels et les noms des variables
dictionnaire_traduction = {
    'v001': 'numerogruppe',
    'v002': 'numeromenage',
    'v005': 'poidsdesondage',
    'v022': 'stratedesondage',
    'v024': 'region',
    'v025': 'milieuderesidence',
    'v008': 'datedelinterview',
    'hw1': 'ageenfant',
    'b4': 'sexeenfant',
    'bord': 'rangdenaissance',
    'b11': 'intervalleintergenesique',
    'm18': 'tailleanaissance',
    'm34': 'dureeallaitement',
    'h11': 'diarrhee',
    'h0': 'vaccinpolio0',
    'h1': 'cartesante',
    'h2': 'vaccinbcg',
    'h3': 'vaccindtp1',
    'h4': 'vaccinpolio1',
    'h5': 'vaccindtp2',
    'h6': 'vaccinpolio2',
    'h7': 'vaccindtp3',
    'h8': 'vaccinpolio3',
    'h9': 'vaccinrougeole1',
    'h10': 'vacciné',
    'v012': 'agemere',
    'v106': 'niveauinstructionmere',
    'v445': 'imcmere',
    'v438': 'hemoglobinemere',
    'v743a': 'prisededecisionmere',
    'v501': 'statutmatrimonialmere',
    'v201': 'nombreenfantsnesvivants',
    'm14': 'nombrevisitesprenatales',
    'v190': 'indicederichesse',
    'v113': 'sourceeaupotable',
    'v116': 'typeinstallationssanitaires',
    'v161': 'typecombustiblecuisine',
    'v136': 'nombremembresmenage',
    'v102': 'lieuderesidence',
    'v101': 'regionecologique',
    'hw5': 'scoreztaillepourage',
    'hw8': 'scorezpoidspourtaille',
    'hw11': 'scorezpoidspourage',
    'hw13': 'indicateurdevalidite'
}

# Filtrage pour isoler uniquement les variables d'étude de notre cadre théorique
colonnes_disponibles = [col for col in dictionnaire_traduction.keys() if col in df_raw.columns]
df_analysis = df_raw[colonnes_disponibles].copy()
print(f"-> Variables chargées. Dimensions de départ : {df_analysis.shape}")


print("ÉTAPE 2 : EXCLUSION DES ANOMALIES ANTHROPOMÉTRIQUES SUIVANT LE PROTOCOLE DHS/OMS)")
# Règle DHS : Conserver uniquement les enfants mesurés validement (hw13 == 0)
if 'hw13' in df_analysis.columns:
    df_analysis = df_analysis[df_analysis['hw13'] == 0]

# Règle DHS/OMS : Exclusion définitive des cas manquants (9999) et des flags aberrants (9998)
for z_col in ['hw5', 'hw8', 'hw11']:
    if z_col in df_analysis.columns:
        df_analysis = df_analysis[~df_analysis[z_col].isin([9998, 9999, np.nan])]
        # Filtrage biologique de sécurité OMS (scores Z valides entre -6.00 et +6.00)
        df_analysis = df_analysis[(df_analysis[z_col] >= -600) & (df_analysis[z_col] <= 600)]

print(f"Après nettoyage du bloc anthropométrique cible on a: {df_analysis.shape} enfants valides.")


print("ÉTAPE 3 : HARMONISATION DES VARIABLES EXPLICATIVES CONTINUES")
# Rétablissement des décimales implicites de l'EDS sans altérer la distribution
if 'v445' in df_analysis.columns: # IMC de la mère
    df_analysis = df_analysis[~df_analysis['v445'].isin([9998, 9999])] # Élimination des non-mesures de l'IMC
    df_analysis['v445'] = df_analysis['v445'] / 100.0

if 'v438' in df_analysis.columns: # Hémoglobine de la mère
    df_analysis = df_analysis[~df_analysis['v438'].isin([998, 999])]
    df_analysis['v438'] = df_analysis['v438'] / 10.0

# Nettoyage et application de la tendance centrale (médiane) pour les variables continues
continues_fields = ['hw1', 'v012', 'v445', 'v438', 'v136', 'b11', 'm14', 'm34', 'bord', 'v201', 'v008']
for col in continues_fields:
    if col in df_analysis.columns:
        # Élimination des codes génériques de non-réponse EDS avant le calcul de la médiane
        df_analysis[col] = df_analysis[col].replace([97, 98, 99, 997, 998, 999], np.nan)
        med_val = df_analysis[col].median()
        df_analysis[col] = df_analysis[col].fillna(med_val)

print("Variables continues nettoyées et ajustées.")

print("ÉTAPE 4 : RECODAGE DES VALEURS MANQUANTES DANS LES COVARIABLES CATÉGORIELLES (MODALITÉ UNIQUE 9)")
# Application de la règle EDS : Regroupement des non-réponses dans une classe "9" (ou "99")
categorical_fields = [
    'b4', 'v025', 'v106', 'v190', 'v024', 'm18', 'h10', 'h11', 
    'v743a', 'v501', 'v113', 'v116', 'v161', 'v102', 'v101',
    'h2', 'h3', 'h4', 'h5', 'h7'
]

dhs_missing_rules = {
    'm15': ([8, 9], 9),      # Poids naissance : 8=Ne sait pas, 9=Manquant -> 9=Manquant/Inconnu
    'm34': ([98, 99], 99),   # Allaitement
    'h10': ([8, 9], 9),      # BCG
    'h11': ([8, 9], 9),      # Diarrhée
    'h1': ([8, 9], 9),      # Vaccine
    'h2': ([8, 9], 9),
    'h3': ([8, 9], 9),
    'h4': ([8, 9], 9),
    'h5': ([8, 9], 9),
    'h6': ([8, 9], 9),      # Vaccin polio 2
    'h7': ([8, 9], 9),      # Vaccin dtp3
    'h8': ([8, 9], 9),      # Vaccin polio 3
    'h9': ([8, 9], 9),      # Vaccin rougeole
    'v743a': ([8, 9], 9),    # Prise de décision
    'v501': ([8, 9], 9),     # Statut matrimonial
    'v113': ([97, 98, 99], 99),  # Source eau
    'v116': ([97, 98, 99], 99),  # Installations sanitaires
    'v161': ([97, 98, 99], 99),  # Combustible cuisine
}

for col in categorical_fields:
    if col in df_analysis.columns:
        if col in dhs_missing_rules:
            bad_codes, target_code = dhs_missing_rules[col]
            df_analysis[col] = df_analysis[col].replace(bad_codes, target_code)
        else:
            df_analysis[col] = df_analysis[col].replace([8, 9, 98, 99], 9)
            
        df_analysis[col] = df_analysis[col].fillna(9)
        df_analysis[col] = df_analysis[col].astype(int)

print("Traitement des valeurs manquantes catégorielles par catégorie dédiée terminé.")

print("ÉTAPE 5 : APPLICATIONS DES RÈGLES DE LA MATRICE DE TRANSITION ICEA")
# Construction des indicateurs unitaires (Seuil OMS < -200 soit -2.00 Écarts-types)
df_analysis['Y1_stunting'] = (df_analysis['hw5'] < -200).astype(int)
df_analysis['Y2_wasting'] = (df_analysis['hw8'] < -200).astype(int)
df_analysis['Y3_underweight'] = (df_analysis['hw11'] < -200).astype(int)

# Transition matricielle combinatoire unifiée vers l'Index Composite de l'Échec Anthropométrique
# Y_ICEA = 1 si au moins une forme de malnutrition est présente, 0 si croissance normale (Témoin)
df_analysis['Y_ICEA'] = ((df_analysis['Y1_stunting'] == 1) | (df_analysis['Y2_wasting'] == 1) | (df_analysis['Y3_underweight'] == 1)).astype(int)

# Création du poids analytique unifié pour le Maximum de Vraisemblance
df_analysis['poids_analytique'] = df_analysis['v005'] / 1000000.0

print("ÉTAPE 6 : COMPILATION DE LA NOMENCLATURE, TYPE DE VARIABLE ET DES MÉTADONNÉES")
meta_records = []
# pour documenter précisément l'onglet métadonnées.
dictionnaire_categories_fr = {
    'v025': "1=Urbain, 2=Rural",
    'b4': "1=Masculin, 2=Feminin",
    'v106': "0=Aucun niveau, 1=Primaire, 2=Secondaire, 3=Superieur",
    'v190': "1=Plus pauvre, 2=Pauvre, 3=Moyen, 4=Riche, 5=Plus riche",
    'v024': "1=Adamaoua, 2=Centre, 3=Douala, 4=Est, 5=ExtremeNord, 6=Littoral, 7=Nord, 8=NordOuest, 9=Ouest, 10=Sud, 11=SudOuest, 12=Yaounde",
    'm18': "1=Tres grand, 2=Plus grand que la moyenne, 3=Moyen, 4=Plus petit que la moyenne, 5=Tres petit, 9=Manquant/Inconnu",
    'v743a': "1=Repondante seule, 2=Repondante et conjoint, 3=Conjoint seul, 4=Quelquun dautre, 9=Inconnu",
    'v501': "0=Jamais mariee, 1=Mariee, 2=Vivant en union, 3=Veuve, 4=Divorcee, 5=Separee, 9=Inconnu",
    'v113': "11=Robinet maison, 12=Robinet cour, 13=Fontaine publique, 21=Forage, 41=Puits ouvert, 43=Source non protegee, 61=Eau de surface, 99=Manquant/Inconnu",
    'v116': "11=Chasse degout, 12=Chasse fosse septique, 21=Latrine amelioree VIP, 22=Latrine ouverte sans dalle, 31=Pas de toilettes/Nature, 99=Manquant/Inconnu",
    'v161': "1=Electricite, 2=Gaz GPL, 7=Charbon de bois, 8=Bois de chauffe, 95=Pas de cuisine, 99=Manquant/Inconnu",
    'v102': "1=Urbain, 2=Rural",
    'v101': "1=Soudanosahelienne, 2=Hautealtitude, 3=Guineennegravifere, 4=Equatorialeforestiere",
    'h0': "0=Non, 1=Date de vaccination sur la carte, 2=Signalé par la mère, 3=Vaccination indiquée sur la carte, 8=Je ne sais pas", 
    'h1': "0=Aucune carte, 1=Oui vu, 2=Oui pas vu, 3=Je n'ai plus de carte",
    'h2': "0=Non, 1=Date de vaccination sur la carte, 2=Signalé par la mère, 3=Vaccination indiquée sur la carte, 8=Je ne sais pas", 
    'h3': "0=Non, 1=Date de vaccination sur la carte, 2=Signalé par la mère, 3=Vaccination indiquée sur la carte, 8=Je ne sais pas", 
    'h4': "0=Non, 1=Date de vaccination sur la carte, 2=Signalé par la mère, 3=Vaccination indiquée sur la carte, 8=Je ne sais pas",
    'h5': "0=Non, 1=Date de vaccination sur la carte, 2=Signalé par la mère, 3=Vaccination indiquée sur la carte, 8=Je ne sais pas", 
    'h6': "0=Non, 1=Date de vaccination sur la carte, 2=Signalé par la mère, 3=Vaccination indiquée sur la carte, 8=Je ne sais pas", 
    'h7': "0=Non, 1=Date de vaccination sur la carte, 2=Signalé par la mère, 3=Vaccination indiquée sur la carte, 8=Je ne sais pas",
    'h8': "0=Non, 1=Date de vaccination sur la carte, 2=Signalé par la mère, 3=Vaccination indiquée sur la carte, 8=Je ne sais pas",
    'h9': "0=Non, 1=Date de vaccination sur la carte, 2=Signalé par la mère, 3=Vaccination indiquée sur la carte, 8=Je ne sais pas",
    'h10': "0=Non, 1=Oui, 8=Je ne sais pas",
    'h11': "0=Pas de diarrhee, 1=Oui (derniere 24h), 2=Oui (2 dernieres semaines), 9=Inconnu"
}

meta_records = []

# Analyse des colonnes pour monter le dictionnaire de l'onglet 2
for original_col in df_analysis.columns:
    # Récupération du nouveau nom en français nettoyé (ou conservation si variable dérivée)
    nom_francais_clean = dictionnaire_traduction.get(original_col, original_col.lower().replace("_", ""))
    
    # Détermination du code DHS d'origine pour la traçabilité épidémiologique
    code_officiel = original_col if original_col in dictionnaire_traduction else "Variable Derivee (Algorithme)"
    
    # Qualification du type statistique
    if original_col in continues_fields or original_col in ['hw5', 'hw8', 'hw11', 'poids_analytique', 'v005']:
        type_stat = "Numerique Continue"
    elif original_col in ['v001', 'v002', 'v022']:
        type_stat = "Identifiant de Sondage"
    else:
        type_stat = "Qualitative / Categorielle"
        
    # Mapping exhaustif des catégories des variables categorielles
    categories_text = dictionnaire_categories_fr.get(original_col, "Echelle continue / Identifiant unique")
    if original_col in ['Y1_stunting', 'Y2_wasting', 'Y3_underweight', 'Y_ICEA']:
        categories_text = "0=Normal (Absence de malnutrition), 1=Echec Anthropometrique (Sous-nutrition active)"
        
    meta_records.append({
        "Variable": nom_francais_clean,
        "CodeDHS": code_officiel,
        "Type": type_stat,
        "ValeursDHS": categories_text
    })

df_metadata_sheet = pd.DataFrame(meta_records)

# Traduction finale des colonnes du jeu de données pour l'onglet 'donnees'
# On applique la nomenclature en français, lowercase, sans caractères spéciaux
df_analysis_clean_names = df_analysis.rename(columns=dictionnaire_traduction)

print("ÉTAPE 7 : EXPORTATION DES DONNEES NETTOYEES VERS LE CLASSEUR EXCEL")
with pd.ExcelWriter("dataset_sous_nutrition_EDSC2018.xlsx", engine='xlsxwriter') as writer:
    # Premier onglet : Données nettoyees
    df_analysis_clean_names.to_excel(writer, sheet_name='donnees', index=False)
    
    # Deuxième onglet : Métadonnées de description des variables
    df_metadata_sheet.to_excel(writer, sheet_name='métadonnées', index=False)
    
    # Bloc de géométrie et d'ajustement dynamique des colonnes pour eviter les coupures de textes
    for sheet_name in ['donnees', 'métadonnées']:
        worksheet = writer.sheets[sheet_name]
        target_df = df_analysis_clean_names if sheet_name == 'donnees' else df_metadata_sheet
        
        for idx, col_name in enumerate(target_df.columns):
            max_len = target_df[col_name].astype(str).map(len).max()
            optimal_width = max(max_len, len(col_name)) + 3
            # Protection contre l'étirement des cellules de texte massives
            worksheet.set_column(idx, idx, min(optimal_width, 65))