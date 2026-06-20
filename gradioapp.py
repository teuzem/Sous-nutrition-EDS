import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr

# Configuration des chemins d'accès aux artefacts de modélisation
MODEL_DIR = "modeles_sauvegardes"
OUTPUT_DIR = "resultats"
path_model = os.path.join(MODEL_DIR, "modele_catboost_sous_nutrition_eds2018.joblib")
path_preprocessor = os.path.join(MODEL_DIR, "column_transformer_api.joblib")
path_features = os.path.join(MODEL_DIR, "co_variables_inference.joblib")
path_taux = os.path.join(MODEL_DIR, "taux_conditionnels_cliniques.joblib")

# Chargement unique des objets en memoire vive
meilleur_modele = joblib.load(path_model)
preprocessor = joblib.load(path_preprocessor)
ordre_exact_features = joblib.load(path_features)
taux_conditionnels = joblib.load(path_taux)

# Fonction interne pour executer le pipeline complet de feature engineering et d'inference
def executer_inference_pipeline(df_input):
    df_input['ageenfant_carre'] = df_input['ageenfant'] ** 2
    df_input['eau_amelioree'] = df_input['sourceeaupotable'].isin([11, 12, 13, 21]).astype(int)
    df_input['toilettes_ameliorees'] = df_input['typeinstallationssanitaires'].isin([11, 12, 21]).astype(int)
    df_input['index_wash_synergie'] = (df_input['eau_amelioree'] * df_input['toilettes_ameliorees']).astype(int)
    df_input['pauvreté_rurale'] = ((df_input['milieuderesidence'] == 2) & (df_input['indicederichesse'].isin([1, 2]))).astype(int)
    df_input['ratio_charge_menage'] = df_input['nombreenfantsnesvivants'] / (df_input['nombremembresmenage'].replace(0, np.nan))
    df_input['ratio_charge_menage'] = df_input['ratio_charge_menage'].fillna(3.0)

    # Re-alignement strict de toutes les colonnes selon l'ordre exact attendu par l'API
    df_aligned = df_input[ordre_exact_features].copy()
    X_transformed = preprocessor.transform(df_aligned)
    
    probabilites_icea = meilleur_modele.predict_proba(X_transformed)[:, 1]
    return probabilites_icea

# Fonction d'inference unitaire pour Gradio
def predir_malnutrition_unitaire(
    ageenfant, sexeenfant, milieuderesidence, indicederichesse, niveauinstructionmere,
    rangdenaissance, hemoglobinemere, intervalleintergenesique, dureeallaitement,
    nombreenfantsnesvivants, nombrevisitesprenatales, nombremembresmenage, imcmere,
    region, vacciné, vaccinbcg, diarrhee, prisededecisionmere, statutmatrimonialmere,
    tailleanaissance, sourceeaupotable, typeinstallationssanitaires, typecombustiblecuisine,
    regionecologique, vaccinpolio0, vaccinpolio1, vaccinpolio2,
    vaccinpolio3, vaccindtp1, vaccindtp2, vaccindtp3, vaccinrougeole1, agemere
):
    dict_sexe = {"Masculin": 1, "Féminin": 2}
    dict_milieu = {"Urbain": 1, "Rural": 2}
    dict_richesse = {"Plus pauvre": 1, "Pauvre": 2, "Moyen": 3, "Riche": 4, "Plus riche": 5}
    dict_educ = {"Aucun niveau": 0, "Primaire": 1, "Secondaire": 2, "Supérieur": 3}
    dict_decision = {"Répondante seule": 1, "Répondante et conjoint": 2, "Conjoint seul": 3, "Autre personne": 4}
    dict_matrimonial = {"Jamais mariée": 0, "Mariée": 1, "Union libre": 2, "Veuve": 3, "Divorcée": 4, "Séparée": 5}
    dict_taille_naiss = {"Très grand": 1, "Plus grand que la moyenne": 2, "Moyen": 3, "Plus petit que la moyenne": 4, "Très petit": 5}
    dict_bool = {"Non": 0, "Oui": 1}
    
    dict_eau = {"Robinet dans la maison": 11, "Robinet dans la cour": 12, "Fontaine publique": 13, "Forage / Puits tubé": 21, "Puits ouvert": 41, "Source non protégée": 43, "Eau de surface (rivière/lac)": 61}
    dict_toilettes = {"Chasse d'eau vers égout": 11, "Chasse d'eau vers fosse septique": 12, "Latrine améliorée VIP": 21, "Latrine ouverte sans dalle": 22, "Pas de toilettes / Nature": 31}
    dict_combustible = {"Électricité": 1, "Gaz GPL": 2, "Charbon de bois": 7, "Bois de chauffe / Paille": 8, "Pas de cuisine": 95}
    dict_eco = {"Soudano-Sahélienne": 1, "Haute Altitude": 2, "Guinéenne Gravifère": 3, "Équatoriale Forestière": 4}

    # Correction de l'extraction de l'identifiant numerique de la region
    id_region = int(region.split('=')[0])

    # CORRECTION CRITIQUE : Integration explicite de agemere dans le dictionnaire d'entree
    data_enfant = {
        'ageenfant': int(ageenfant), 'sexeenfant': dict_sexe[sexeenfant],
        'milieuderesidence': dict_milieu[milieuderesidence], 'indicederichesse': dict_richesse[indicederichesse],
        'niveauinstructionmere': dict_educ[niveauinstructionmere], 'rangdenaissance': int(rangdenaissance),
        'hemoglobinemere': float(hemoglobinemere), 'intervalleintergenesique': int(intervalleintergenesique),
        'dureeallaitement': int(dureeallaitement), 'nombreenfantsnesvivants': int(nombreenfantsnesvivants),
        'nombrevisitesprenatales': int(nombrevisitesprenatales), 'nombremembresmenage': int(nombremembresmenage),
        'imcmere': float(imcmere), 'region': id_region, 'vacciné': dict_bool[vacciné],
        'vaccinbcg': dict_bool[vaccinbcg], 'diarrhee': dict_bool[diarrhee], 'prisededecisionmere': dict_decision[prisededecisionmere],
        'statutmatrimonialmere': dict_matrimonial[statutmatrimonialmere], 'tailleanaissance': dict_taille_naiss[tailleanaissance],
        'sourceeaupotable': dict_eau[sourceeaupotable], 'typeinstallationssanitaires': dict_toilettes[typeinstallationssanitaires],
        'typecombustiblecuisine': dict_combustible[typecombustiblecuisine], 'lieuderesidence': dict_milieu[milieuderesidence],
        'regionecologique': dict_eco[regionecologique], 'vaccinpolio0': dict_bool[vaccinpolio0],
        'vaccinpolio1': dict_bool[vaccinpolio1], 'vaccinpolio2': dict_bool[vaccinpolio2], 'vaccinpolio3': dict_bool[vaccinpolio3],
        'vaccindtp1': dict_bool[vaccindtp1], 'vaccindtp2': dict_bool[vaccindtp2], 'vaccindtp3': dict_bool[vaccindtp3],
        'vaccinrougeole1': dict_bool[vaccinrougeole1], 'agemere': int(agemere)
    }

    df_input = pd.DataFrame([data_enfant])
    p_icea = float(executer_inference_pipeline(df_input)[0])
    
    p_stunting = p_icea * taux_conditionnels['p_y1_sachant_icea']
    p_wasting = p_icea * taux_conditionnels['p_y2_sachant_icea']
    p_underweight = p_icea * taux_conditionnels['p_y3_sachant_icea']
    
    if p_icea >= 0.50:
        statut_texte = "ALERTE : Risque de Malnutrition Eleve"
        couleur_alerte = "rgba(231, 76, 60, 0.15)"
        couleur_texte = "#e74c3c"
    else:
        statut_texte = "NORMAL : Croissance Saine Diagnostiquee"
        couleur_alerte = "rgba(46, 204, 113, 0.15)"
        couleur_texte = "#2ecc71"
        
    html_status = f"""
    <div style="background-color: {couleur_alerte}; border-left: 5px solid {couleur_texte}; padding: 15px; border-radius: 4px; margin-bottom: 10px;">
        <h3 style="color: {couleur_texte}; margin: 0 0 5px 0; font-weight: bold;">{statut_texte}</h3>
        <p style="margin: 0; font-size: 14px; color: #333;">La probabilite globale d'echec anthropometrique (ICEA) est de <b>{p_icea:.2%}</b>.</p>
    </div>
    """
    
    outputs_graphes = {
        "Retard de croissance (Stunting / Chronique)": p_stunting,
        "Amaigrissement aigu (Wasting / Aigu)": p_wasting,
        "Insuffisance ponderale (Underweight / Mixte)": p_underweight
    }
    
    return html_status, outputs_graphes

# Fonction d'inference de masse a partir d'un fichier CSV
def predir_malnutrition_masse(fichier_csv):
    if fichier_csv is None:
        return None, "Veuillez charger un fichier CSV valide."
    
    try:
        df_csv = pd.read_csv(fichier_csv.name)
        
        # Colonnes structurelles obligatoires requises en entree
        colonnes_obligatoires = [
            'ageenfant', 'sexeenfant', 'milieuderesidence', 'indicederichesse', 'niveauinstructionmere',
            'rangdenaissance', 'hemoglobinemere', 'intervalleintergenesique', 'dureeallaitement',
            'nombreenfantsnesvivants', 'nombrevisitesprenatales', 'nombremembresmenage', 'imcmere',
            'region', 'vacciné', 'vaccinbcg', 'diarrhee', 'prisededecisionmere', 'statutmatrimonialmere',
            'tailleanaissance', 'sourceeaupotable', 'typeinstallationssanitaires', 'typecombustiblecuisine',
            'regionecologique', 'vaccinpolio0', 'vaccinpolio1', 'vaccinpolio2', 'vaccinpolio3',
            'vaccindtp1', 'vaccindtp2', 'vaccindtp3', 'vaccinrougeole1', 'agemere'
        ]
        
        manquantes = [col for col in colonnes_obligatoires if col not in df_csv.columns]
        if manquantes:
            return None, f"Erreur : Colonnes manquantes dans le fichier CSV : {manquantes}"
        
        probabilites = executer_inference_pipeline(df_csv)
        
        df_csv['probabilite_icea'] = probabilites
        df_csv['diagnostic_icea'] = (probabilites >= 0.50).astype(int)
        df_csv['prob_retard_croissance_stunting'] = probabilites * taux_conditionnels['p_y1_sachant_icea']
        df_csv['prob_amaigrissement_wasting'] = probabilites * taux_conditionnels['p_y2_sachant_icea']
        df_csv['prob_insuffisance_ponderale_underweight'] = probabilites * taux_conditionnels['p_y3_sachant_icea']
        
        chemin_sortie = os.path.join(OUTPUT_DIR, "predictions_masse_sorties.csv")
        df_csv.to_csv(chemin_sortie, index=False)
        
        total = len(df_csv)
        alertes = int(df_csv['diagnostic_icea'].sum())
        taux_alerte = alertes / total
        
        html_resume = f"""
        <div style="background-color: #f8f9fa; border-left: 5px solid #008080; padding: 15px; border-radius: 4px;">
            <h4 style="color: #008080; margin: 0 0 5px 0; font-weight: bold;">Traitement de masse termine</h4>
            <p style="margin: 0; font-size: 14px; color: #333;">Total enfants analyses : <b>{total}</b></p>
            <p style="margin: 0; font-size: 14px; color: #333;">Cas detectes en alerte malnutrition : <b>{alertes} ({taux_alerte:.2%})</b></p>
        </div>
        """
        
        
        return chemin_sortie, html_resume
    
    except Exception as e:
        return None, f"Erreur lors du traitement du fichier : {str(e)}"
    
# Architecture de l'interface utilisateur web complete Gradio Blocks
with gr.Blocks(title="Plateforme Nutritionnelle Cameroun", theme=gr.themes.Soft(primary_hue="teal", secondary_hue="emerald")) as demo:
    
    gr.Markdown(
        """
        # Plateforme Predictive de la Malnutrition Infantile au Cameroun
        Systeme informatique expert d'aide a la decision nutritionnelle base sur les donnees de l'EDSC-V 2018.
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("Prediction Individuelle (Unitaire)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Formulaire d'evaluation clinique de l'enfant")
                    
                    with gr.Accordion("Caracteristiques Biologiques de l'Enfant", open=True):
                        ageenfant = gr.Slider(minimum=0, maximum=59, value=24, label="Âge de l'enfant (en mois)")
                        sexeenfant = gr.Radio(choices=["Masculin", "Féminin"], value="Masculin", label="Sexe de l'enfant")
                        rangdenaissance = gr.Number(value=1, precision=0, label="Rang de naissance dans la fratrie")
                        intervalleintergenesique = gr.Number(value=0, precision=0, label="Intervalle inter-génésique (en mois, 0 si premier né)")
                        dureeallaitement = gr.Slider(minimum=0, maximum=59, value=12, label="Durée de l'allaitement au sein (en mois)")
                        tailleanaissance = gr.Dropdown(choices=["Très grand", "Plus grand que la moyenne", "Moyen", "Plus petit que la moyenne", "Très petit"], value="Moyen", label="Taille à la naissance perçue par la mère")
                        
                    with gr.Accordion("Profil Anthropometrique et Decisionnel de la Mere", open=False):
                        agemere = gr.Slider(minimum=15, maximum=49, value=28, label="Âge actuel de la mère (en années)")
                        niveauinstructionmere = gr.Dropdown(choices=["Aucun niveau", "Primaire", "Secondaire", "Supérieur"], value="Aucun niveau", label="Niveau d'instruction de la mère")
                        imcmere = gr.Number(value=22.0, label="Indice de Masse Corporelle (IMC) de la mère")
                        hemoglobinemere = gr.Number(value=11.5, label="Taux d'hémoglobine maternel (en g/dl)")
                        nombreenfantsnesvivants = gr.Number(value=2, precision=0, label="Nombre total d'enfants nés de cette mère")
                        nombrevisitesprenatales = gr.Number(value=4, precision=0, label="Nombre de consultations prénatales durant la grossesse")
                        prisededecisionmere = gr.Dropdown(choices=["Répondante seule", "Répondante et conjoint", "Conjoint seul", "Autre personne"], value="Répondante et conjoint", label="Autonomie décisionnelle pour les soins de santé")
                        statutmatrimonialmere = gr.Dropdown(choices=["Jamais mariée", "Mariée", "Union libre", "Veuve", "Divorcée", "Séparée"], value="Mariée", label="Statut matrimonial de la mère")

                    with gr.Accordion("Environnement Socio-Economique et WASH du Menage", open=False):
                        milieuderesidence = gr.Radio(choices=["Urbain", "Rural"], value="Rural", label="Milieu de résidence géographique")
                        indicederichesse = gr.Dropdown(choices=["Plus pauvre", "Pauvre", "Moyen", "Riche", "Plus riche"], value="Plus pauvre", label="Quintile de richesse du ménage")
                        sourceeaupotable = gr.Dropdown(choices=["Robinet dans la maison", "Robinet dans la cour", "Fontaine publique", "Forage / Puits tubé", "Puits ouvert", "Source non protégée", "Eau de surface (rivière/lac)"], value="Puits ouvert", label="Source principale d'eau potable")
                        typeinstallationssanitaires = gr.Dropdown(choices=["Chasse d'eau vers égout", "Chasse d'eau vers fosse septique", "Latrine améliorée VIP", "Latrine ouverte sans dalle", "Pas de toilettes / Nature"], value="Latrine ouverte sans dalle", label="Type d'installations sanitaires (Toilettes)")
                        typecombustiblecuisine = gr.Dropdown(choices=["Électricité", "Gaz GPL", "Charbon de bois", "Bois de chauffe / Paille", "Pas de cuisine"], value="Bois de chauffe / Paille", label="Type de combustible pour la cuisine")
                        nombremembresmenage = gr.Number(value=5, precision=0, label="Nombre total de résidents dans le ménage")
                        region = gr.Dropdown(choices=["1=Adamaoua", "2=Centre", "3=Douala", "4=Est", "5=Extrême-Nord", "6=Littoral", "7=Nord", "8=Nord-Ouest", "9=Ouest", "10=Sud", "11=Sud-Ouest", "12=Yaoundé"], value="5=Extrême-Nord", label="Région administrative du Cameroun")
                        regionecologique = gr.Dropdown(choices=["Soudano-Sahélienne", "Haute Altitude", "Guinéenne Gravifère", "Équatoriale Forestière"], value="Soudano-Sahélienne", label="Zone agro-écologique")

                    with gr.Accordion("Suivi Clinique et Calendrier Vaccinal", open=False):
                        diarrhee = gr.Radio(choices=["Non", "Oui"], value="Non", label="Épisode de diarrhée au cours des 2 dernières semaines")
                        vacciné = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Carnet de vaccination possédé ou vérifié")
                        vaccinbcg = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Vaccin reçu : BCG")
                        vaccinpolio0 = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Vaccin reçu : Polio 0 (Naissance)")
                        vaccinpolio1 = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Vaccin reçu : Polio 1")
                        vaccinpolio2 = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Vaccin reçu : Polio 2")
                        vaccinpolio3 = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Vaccin reçu : Polio 3")
                        vaccindtp1 = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Vaccin reçu : DTP 1")
                        vaccindtp2 = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Vaccin reçu : DTP 2")
                        vaccindtp3 = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Vaccin reçu : DTP 3")
                        vaccinrougeole1 = gr.Radio(choices=["Non", "Oui"], value="Oui", label="Vaccin reçu : Rougeole 1")

                    btn_unitaire = gr.Button("Calculer le diagnostic individuel", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### Resultats de l'analyse individuelle")
                    output_html_status = gr.HTML(label="Statut Decisionnel")
                    output_chart_sous_classes = gr.Label(num_top_classes=3, label="Decomposition du risque par forme clinique (OMS)")

            btn_unitaire.click(
                fn=predir_malnutrition_unitaire,
                inputs=[
                    ageenfant, sexeenfant, milieuderesidence, indicederichesse, niveauinstructionmere,
                    rangdenaissance, hemoglobinemere, intervalleintergenesique, dureeallaitement,
                    nombreenfantsnesvivants, nombrevisitesprenatales, nombremembresmenage, imcmere,
                    region, vacciné, vaccinbcg, diarrhee, prisededecisionmere, statutmatrimonialmere,
                    tailleanaissance, sourceeaupotable, typeinstallationssanitaires, typecombustiblecuisine,
                    regionecologique, vaccinpolio0, vaccinpolio1, vaccinpolio2,
                    vaccinpolio3, vaccindtp1, vaccindtp2, vaccindtp3, vaccinrougeole1, agemere
                ],
                outputs=[output_html_status, output_chart_sous_classes]
            )

        with gr.TabItem("Prediction en Masse (Fichier CSV)"):
            gr.Markdown("### Diagnostic de masse pour plusieurs enfants")
            gr.Markdown("Chargez un fichier tableur au format CSV contenant les dossiers des enfants codés numériquement selon la nomenclature de l'étude.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_file_csv = gr.File(label="Fichier CSV d'entree", file_types=[".csv"])
                    btn_masse = gr.Button("Lancer l'analyse de masse", variant="primary")
                
                with gr.Column(scale=1):
                    output_html_resume = gr.HTML(label="Resume de l'analyse")
                    output_file_csv = gr.File(label="Telecharger les resultats complets (.CSV)")
            
            btn_masse.click(
                fn=predir_malnutrition_masse,
                inputs=[input_file_csv],
                outputs=[output_file_csv, output_html_resume]
            )

if __name__ == "__main__":
    demo.launch(share=False)