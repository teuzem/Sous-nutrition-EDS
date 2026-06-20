import os
import joblib
import pickle
import numpy as np
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialisation de l'application et configuration CORS pour le Frontend
app = FastAPI(
    title="API Prédictive de la Malnutrition Infantile au Cameroun (EDSC-V 2018)",
    description="Moteur d'inférence basé sur CatBoost pour la prédiction de l'ICEA et la marginalisation des phénotypes OMS.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemins d'accès aux artefacts de modélisation
#MODEL_DIR = "modeles_sauvegardes"
#path_model = os.path.join(MODEL_DIR, "modele_catboost_sous_nutrition_eds2018.joblib")
#path_preprocessor = os.path.join(MODEL_DIR, "column_transformer_api.joblib")
#path_features = os.path.join(MODEL_DIR, "co_variables_inference.joblib")
#path_taux = os.path.join(MODEL_DIR, "taux_conditionnels_cliniques.joblib")

#if not all(os.path.exists(p) for p in [path_model, path_preprocessor, path_features, path_taux]):
 #   raise RuntimeError("Erreur critique : Les artefacts de modélisation sont introuvables.")

# Chargement unique en mémoire vive pour optimiser les temps de réponse de l'API
meilleur_modele = joblib.load("C:/Users/Mr Tegou/Desktop/TP regression Multivariee/modeles_sauvegardes/modele_catboost_sous_nutrition_eds2018.joblib")
preprocessor = joblib.load("C:/Users/Mr Tegou/Desktop/TP regression Multivariee/modeles_sauvegardes/column_transformer_api.joblib")
ordre_exact_features = joblib.load("C:/Users/Mr Tegou/Desktop/TP regression Multivariee/modeles_sauvegardes/co_variables_inference.joblib")
taux_conditionnels = joblib.load("C:/Users/Mr Tegou/Desktop/TP regression Multivariee/modeles_sauvegardes/taux_conditionnels_cliniques.joblib")

# Schémas de données typés pour la validation des entrées
class EnfantInputSchema(BaseModel):
    ageenfant: int = Field(..., description="Âge de l'enfant en mois (0 à 59)", example=24)
    sexeenfant: int = Field(..., description="Sexe de l'enfant (1=Masculin, 2=Féminin)", example=1)
    milieuderesidence: int = Field(..., description="Milieu de résidence (1=Urbain, 2=Rural)", example=2)
    indicederichesse: int = Field(..., description="Quintile de richesse (1=Plus pauvre à 5=Plus riche)", example=1)
    niveauinstructionmere: int = Field(..., description="Niveau d'éducation de la mère (0=Aucun à 3=Supérieur)", example=0)
    rangdenaissance: int = Field(..., description="Rang de naissance dans la fratrie", example=2)
    hemoglobinemere: float = Field(..., description="Taux d'hémoglobine maternel en g/dl", example=11.2)
    intervalleintergenesique: int = Field(..., description="Espace inter-génésique en mois", example=36)
    dureeallaitement: int = Field(..., description="Durée totale de l'allaitement en mois", example=12)
    nombreenfantsnesvivants: int = Field(..., description="Nombre total d'enfants nés de cette mère", example=3)
    nombrevisitesprenatales: int = Field(..., description="Nombre de consultations prénatales", example=4)
    nombremembresmenage: int = Field(..., description="Nombre total de résidents dans le ménage", example=6)
    imcmere: float = Field(..., description="Indice de Masse Corporelle de la mère", example=22.4)
    region: int = Field(..., description="Région administrative du Cameroun (1 à 12)", example=5)
    vacciné: int = Field(..., description="Statut vaccinal global (0=Non, 1=Oui)", example=1)
    vaccinbcg: int = Field(..., description="Vaccination BCG (0=Non, 1=Oui)", example=1)
    diarrhee: int = Field(..., description="Épisode de diarrhée récent (0=Non, 1=Oui)", example=0)
    prisededecisionmere: int = Field(..., description="Autonomie de décision de la mère (1 à 4)", example=2)
    statutmatrimonialmere: int = Field(..., description="Statut matrimonial de la mère (0 à 5)", example=1)
    tailleanaissance: int = Field(..., description="Perception de la taille à la naissance (1 à 5)", example=3)
    sourceeaupotable: int = Field(..., description="Code DHS de la source d'eau potable", example=21)
    typeinstallationssanitaires: int = Field(..., description="Code DHS du type de toilettes", example=22)
    typecombustiblecuisine: int = Field(..., description="Code DHS du combustible utilisé", example=8)
    lieuderesidence: int = Field(..., description="Code DHS du type de lieu", example=2)
    regionecologique: int = Field(..., description="Zone écologique du Cameroun (1 à 4)", example=1)
    
    # Prise en charge explicite et exhaustive de l'ensemble du calendrier vaccinal requis
    vaccinpolio0: int = Field(..., description="Vaccination Polio à la naissance (0=Non, 1=Oui)", example=1)
    vaccinpolio1: int = Field(..., description="Vaccination Polio 1 (0=Non, 1=Oui)", example=1)
    vaccinpolio2: int = Field(..., description="Vaccination Polio 2 (0=Non, 1=Oui)", example=1)
    vaccinpolio3: int = Field(..., description="Vaccination Polio 3 (0=Non, 1=Oui)", example=1)
    vaccindtp1: int = Field(..., description="Vaccination DTP 1 (0=Non, 1=Oui)", example=1)
    vaccindtp2: int = Field(..., description="Vaccination DTP 2 (0=Non, 1=Oui)", example=1)
    vaccindtp3: int = Field(..., description="Vaccination DTP 3 (0=Non, 1=Oui)", example=1)
    vaccinrougeole1: int = Field(..., description="Vaccination Rougeole 1 (0=Non, 1=Oui)", example=1)

class PredictionOutputSchema(BaseModel):
    probabilite_icea_unifiee: float = Field(..., description="Risque global d'échec anthropométrique unifié")
    classe_predite_icea: int = Field(..., description="Diagnostic binaire unifié (0=Sain, 1=Alerte Malnutrition)")
    prob_retard_croissance_stunting: float = Field(..., description="Risque marginal de retard de croissance (Chronique)")
    prob_amaigrissement_wasting: float = Field(..., description="Risque marginal d'amaigrissement aigu (Aigu)")
    prob_insuffisance_ponderale_underweight: float = Field(..., description="Risque marginal d'insuffisance pondérale (Mixte)")

# Moteur interne d'inférence et de feature engineering
def executer_pipeline_inference(dictionnaires_enfants: List[dict]) -> List[dict]:
    df_input = pd.DataFrame(dictionnaires_enfants)
    
    # Reconstruction exacte des variables dérivées du Feature Engineering
    df_input['ageenfant_carre'] = df_input['ageenfant'] ** 2
    df_input['eau_amelioree'] = df_input['sourceeaupotable'].isin([11, 12, 13, 21]).astype(int)
    df_input['toilettes_ameliorees'] = df_input['typeinstallationssanitaires'].isin([11, 12, 21]).astype(int)
    df_input['index_wash_synergie'] = (df_input['eau_amelioree'] * df_input['toilettes_ameliorees']).astype(int)
    df_input['pauvreté_rurale'] = ((df_input['milieuderesidence'] == 2) & (df_input['indicederichesse'].isin([1, 2]))).astype(int)
    df_input['ratio_charge_menage'] = df_input['nombreenfantsnesvivants'] / (df_input['nombremembresmenage'].replace(0, np.nan))
    df_input['ratio_charge_menage'] = df_input['ratio_charge_menage'].fillna(3.0)
    
    # Alignement strict de l'ordre des colonnes requis par le ColumnTransformer
    df_aligned = df_input[ordre_exact_features].copy()
    
    # Normalisation et Target Encoding via le préprocesseur de l'API
    X_transformed = preprocessor.transform(df_aligned)
    
    # Inférence des probabilités globales via CatBoost
    probabilites_icea = meilleur_modele.predict_proba(X_transformed)[:, 1]
    classes_icea = (probabilites_icea >= 0.5).astype(int)
    
    responses = []
    for i in range(len(df_input)):
        p_icea = float(probabilites_icea[i])
        
        # Calcul des probabilités marginales conditionnelles pour chaque forme clinique
        p_stunting = p_icea * taux_conditionnels['p_y1_sachant_icea']
        p_wasting = p_icea * taux_conditionnels['p_y2_sachant_icea']
        p_underweight = p_icea * taux_conditionnels['p_y3_sachant_icea']
        
        responses.append({
            "probabilite_icea_unifiee": round(p_icea, 4),
            "classe_predite_icea": int(classes_icea[i]),
            "prob_retard_croissance_stunting": round(p_stunting, 4),
            "prob_amaigrissement_wasting": round(p_wasting, 4),
            "prob_insuffisance_ponderale_underweight": round(p_underweight, 4)
        })
        
    return responses

# Points d'accès pour l'API
@app.get("/")
def check_health():
    return {"status": "healthy", "model_loaded": "CatBoost Classifier Optimisé"}

@app.post("/predict/unitaire", response_model=PredictionOutputSchema)
def predir_enfant_unique(payload: EnfantInputSchema):
    try:
        dict_enfant = payload.model_dump()
        resultat = executer_pipeline_inference([dict_enfant])
        return resultat
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du moteur d'inférence : {str(e)}")

@app.post("/predict/masse", response_model=List[PredictionOutputSchema])
def predir_enfants_en_masse(payload: List[EnfantInputSchema]):
    try:
        liste_enfants = [enfant.model_dump() for enfant in payload]
        resultats = executer_pipeline_inference(liste_enfants)
        return resultats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors du traitement de masse : {str(e)}")
