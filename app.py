"""
NutriScreen Cameroun — Dépistage IA de la Sous-Nutrition Infantile
Application Streamlit refactorisée — Pipeline CatBoost EDS-MICS 2018
Variables, encodages et pipeline 100% alignés sur gradioapp.py
"""
import os, json, base64
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components
from io import BytesIO
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# 0. CONFIGURATION PATHS — identiques à gradioapp.py
OUTPUT_DIR = "resultats"

_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#006D77"/>
  <rect x="26" y="10" width="12" height="44" rx="4" fill="#fff"/>
  <rect x="10" y="26" width="44" height="12" rx="4" fill="#fff"/>
</svg>"""
_favicon_b64 = base64.b64encode(_FAVICON_SVG.encode()).decode()

st.set_page_config(
    page_title="NutriScreen CM — Dépistage IA Sous-Nutrition Infantile",
    page_icon=f"data:image/svg+xml;base64,{_favicon_b64}",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 1. CHARGEMENT DES ARTEFACTS CATBOOST (même logique que gradioapp.py)
@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        model       = joblib.load(os.path.join("modele_catboost_sous_nutrition_eds2018.joblib"))
        preprocessor = joblib.load(os.path.join("column_transformer_api.joblib"))
        features    = joblib.load(os.path.join("co_variables_inference.joblib"))
        taux        = joblib.load(os.path.join("taux_conditionnels_cliniques.joblib"))
        return model, preprocessor, features, taux, None
    except Exception as e:
        return None, None, None, None, str(e)

meilleur_modele, preprocessor, ordre_exact_features, taux_conditionnels, _load_error = load_resources()


# 2. DICTIONNAIRES D'ENCODAGE — identiques à gradioapp.py
DICT_SEXE       = {"Masculin": 1, "Féminin": 2}
DICT_MILIEU     = {"Urbain": 1, "Rural": 2}
DICT_RICHESSE   = {"Plus pauvre": 1, "Pauvre": 2, "Moyen": 3, "Riche": 4, "Plus riche": 5}
DICT_EDUC       = {"Aucun niveau": 0, "Primaire": 1, "Secondaire": 2, "Supérieur": 3}
DICT_DECISION   = {"Répondante seule": 1, "Répondante et conjoint": 2, "Conjoint seul": 3, "Autre personne": 4}
DICT_MATRIMONIAL= {"Jamais mariée": 0, "Mariée": 1, "Union libre": 2, "Veuve": 3, "Divorcée": 4, "Séparée": 5}
DICT_TAILLE     = {"Très grand": 1, "Plus grand que la moyenne": 2, "Moyen": 3,
                   "Plus petit que la moyenne": 4, "Très petit": 5}
DICT_BOOL       = {"Non": 0, "Oui": 1}
DICT_EAU        = {
    "Robinet dans la maison": 11, "Robinet dans la cour": 12,
    "Fontaine publique": 13, "Forage / Puits tubé": 21,
    "Puits ouvert": 41, "Source non protégée": 43,
    "Eau de surface (rivière/lac)": 61
}
DICT_TOILETTES  = {
    "Chasse d'eau vers égout": 11, "Chasse d'eau vers fosse septique": 12,
    "Latrine améliorée VIP": 21, "Latrine ouverte sans dalle": 22,
    "Pas de toilettes / Nature": 31
}
DICT_COMBUSTIBLE= {
    "Électricité": 1, "Gaz GPL": 2, "Charbon de bois": 7,
    "Bois de chauffe / Paille": 8, "Pas de cuisine": 95
}
DICT_ECO        = {
    "Soudano-Sahélienne": 1, "Haute Altitude": 2,
    "Guinéenne Gravifère": 3, "Équatoriale Forestière": 4
}
REGIONS_LIST    = [
    "1=Adamaoua","2=Centre","3=Douala","4=Est","5=Extrême-Nord",
    "6=Littoral","7=Nord","8=Nord-Ouest","9=Ouest","10=Sud","11=Sud-Ouest","12=Yaoundé"
]
REGION_COORDS   = {
    1: {"nom":"Adamaoua","lat":7.33,"lon":13.58},
    2: {"nom":"Centre","lat":4.57,"lon":11.52},
    3: {"nom":"Douala","lat":4.05,"lon":9.70},
    4: {"nom":"Est","lat":4.25,"lon":13.50},
    5: {"nom":"Extrême-Nord","lat":10.60,"lon":14.32},
    6: {"nom":"Littoral","lat":4.60,"lon":10.00},
    7: {"nom":"Nord","lat":9.30,"lon":13.39},
    8: {"nom":"Nord-Ouest","lat":6.07,"lon":10.15},
    9: {"nom":"Ouest","lat":5.48,"lon":10.42},
    10:{"nom":"Sud","lat":2.84,"lon":10.92},
    11:{"nom":"Sud-Ouest","lat":5.00,"lon":9.20},
    12:{"nom":"Yaoundé","lat":3.87,"lon":11.52},
}


# 3. PIPELINE D'INFÉRENCE — identique à gradioapp.py
def executer_inference_pipeline(df_input: pd.DataFrame) -> np.ndarray:
    """Feature engineering + transformation + prédiction CatBoost."""
    df = df_input.copy()
    df['ageenfant_carre']          = df['ageenfant'] ** 2
    df['eau_amelioree']            = df['sourceeaupotable'].isin([11, 12, 13, 21]).astype(int)
    df['toilettes_ameliorees']     = df['typeinstallationssanitaires'].isin([11, 12, 21]).astype(int)
    df['index_wash_synergie']      = (df['eau_amelioree'] * df['toilettes_ameliorees']).astype(int)
    df['pauvreté_rurale']          = ((df['milieuderesidence'] == 2) & (df['indicederichesse'].isin([1, 2]))).astype(int)
    df['ratio_charge_menage']      = df['nombreenfantsnesvivants'] / (df['nombremembresmenage'].replace(0, np.nan))
    df['ratio_charge_menage']      = df['ratio_charge_menage'].fillna(3.0)
    df_aligned    = df[ordre_exact_features].copy()
    X_transformed = preprocessor.transform(df_aligned)
    return meilleur_modele.predict_proba(X_transformed)[:, 1]


def build_row_dict(vals: dict) -> dict:
    """Construit le dictionnaire numérique depuis les valeurs UI (gradioapp.py encodages)."""
    id_region = int(vals['region'].split('=')[0])
    return {
        'ageenfant':                  int(vals['ageenfant']),
        'sexeenfant':                 DICT_SEXE[vals['sexeenfant']],
        'milieuderesidence':          DICT_MILIEU[vals['milieuderesidence']],
        'indicederichesse':           DICT_RICHESSE[vals['indicederichesse']],
        'niveauinstructionmere':      DICT_EDUC[vals['niveauinstructionmere']],
        'rangdenaissance':            int(vals['rangdenaissance']),
        'hemoglobinemere':            float(vals['hemoglobinemere']),
        'intervalleintergenesique':   int(vals['intervalleintergenesique']),
        'dureeallaitement':           int(vals['dureeallaitement']),
        'nombreenfantsnesvivants':    int(vals['nombreenfantsnesvivants']),
        'nombrevisitesprenatales':    int(vals['nombrevisitesprenatales']),
        'nombremembresmenage':        int(vals['nombremembresmenage']),
        'imcmere':                    float(vals['imcmere']),
        'region':                     id_region,
        'vacciné':                    DICT_BOOL[vals['vacciné']],
        'vaccinbcg':                  DICT_BOOL[vals['vaccinbcg']],
        'diarrhee':                   DICT_BOOL[vals['diarrhee']],
        'prisededecisionmere':        DICT_DECISION[vals['prisededecisionmere']],
        'statutmatrimonialmere':      DICT_MATRIMONIAL[vals['statutmatrimonialmere']],
        'tailleanaissance':           DICT_TAILLE[vals['tailleanaissance']],
        'sourceeaupotable':           DICT_EAU[vals['sourceeaupotable']],
        'typeinstallationssanitaires':DICT_TOILETTES[vals['typeinstallationssanitaires']],
        'typecombustiblecuisine':     DICT_COMBUSTIBLE[vals['typecombustiblecuisine']],
        'lieuderesidence':            DICT_MILIEU[vals['milieuderesidence']],
        'regionecologique':           DICT_ECO[vals['regionecologique']],
        'vaccinpolio0':               DICT_BOOL[vals['vaccinpolio0']],
        'vaccinpolio1':               DICT_BOOL[vals['vaccinpolio1']],
        'vaccinpolio2':               DICT_BOOL[vals['vaccinpolio2']],
        'vaccinpolio3':               DICT_BOOL[vals['vaccinpolio3']],
        'vaccindtp1':                 DICT_BOOL[vals['vaccindtp1']],
        'vaccindtp2':                 DICT_BOOL[vals['vaccindtp2']],
        'vaccindtp3':                 DICT_BOOL[vals['vaccindtp3']],
        'vaccinrougeole1':            DICT_BOOL[vals['vaccinrougeole1']],
        'agemere':                    int(vals['agemere']),
    }


COLONNES_OBLIGATOIRES = [
    'ageenfant','sexeenfant','milieuderesidence','indicederichesse','niveauinstructionmere',
    'rangdenaissance','hemoglobinemere','intervalleintergenesique','dureeallaitement',
    'nombreenfantsnesvivants','nombrevisitesprenatales','nombremembresmenage','imcmere',
    'region','vacciné','vaccinbcg','diarrhee','prisededecisionmere','statutmatrimonialmere',
    'tailleanaissance','sourceeaupotable','typeinstallationssanitaires','typecombustiblecuisine',
    'regionecologique','vaccinpolio0','vaccinpolio1','vaccinpolio2','vaccinpolio3',
    'vaccindtp1','vaccindtp2','vaccindtp3','vaccinrougeole1','agemere'
]


# 4. HELPERS VISUELS
def age_txt(m: int) -> str:
    m = int(m)
    a, r = m // 12, m % 12
    if a == 0: return f"{r} mois"
    if r == 0: return f"{a} an{'s' if a > 1 else ''}"
    return f"{a} an{'s' if a > 1 else ''} {r} mois"

def label_richesse(v: int) -> str:
    return {1:"Plus pauvre",2:"Pauvre",3:"Moyen",4:"Riche",5:"Plus riche"}.get(int(v), str(v))

def label_educ(v: int) -> str:
    return {0:"Aucun",1:"Primaire",2:"Secondaire",3:"Supérieur"}.get(int(v), str(v))

def label_milieu(v: int) -> str:
    return {1:"Urbain",2:"Rural"}.get(int(v), str(v))

def label_sexe(v: int) -> str:
    return {1:"Masculin",2:"Féminin"}.get(int(v), str(v))

def label_region(v: int) -> str:
    return REGION_COORDS.get(int(v), {}).get("nom", str(v))

def imc_tag(imc: float) -> tuple:
    if imc < 18.5:   return "Maigreur", "#FEF9C3", "#854D0E"
    elif imc < 25.0: return "Normal",   "#DCFCE7", "#166534"
    elif imc < 30.0: return "Surpoids", "#FEF3C7", "#92400E"
    else:            return "Obésité",  "#FEE2E2", "#991B1B"

# 5. FACTEURS D'INFLUENCE NARRATIFS
def compute_influence_factors(row: dict) -> tuple:
    """Retourne (facteurs_risque, facteurs_protecteurs) sous forme de listes de dicts."""
    risque, protecteur = [], []

    age  = row.get('ageenfant', 0)
    edu  = row.get('niveauinstructionmere', 0)
    ric  = row.get('indicederichesse', 3)
    mil  = row.get('milieuderesidence', 2)
    imc  = row.get('imcmere', 22.0)
    itv  = row.get('intervalleintergenesique', 24)
    hgb  = row.get('hemoglobinemere', 11.5)
    diarhee = row.get('diarrhee', 0)
    rang = row.get('rangdenaissance', 1)
    vps  = row.get('nombrevisitesprenatales', 4)
    eau  = row.get('sourceeaupotable', 41)
    wcs  = row.get('typeinstallationssanitaires', 22)
    comb = row.get('typecombustiblecuisine', 8)
    reg  = row.get('region', 5)
    nb_enf = row.get('nombreenfantsnesvivants', 2)
    durall = row.get('dureeallaitement', 12)

    # Âge enfant
    if 6 <= age <= 23:
        risque.append({"label":"Âge critique", "detail":f"Enfant de {age_txt(age)} — fenêtre de diversification alimentaire la plus vulnérable.", "icon":"🔴"})
    elif age > 36:
        protecteur.append({"label":"Âge favorable", "detail":f"Enfant de {age_txt(age)} — période post-sevrage moins exposée.", "icon":"🟢"})

    # Éducation mère
    if edu >= 2:
        protecteur.append({"label":"Éducation maternelle", "detail":f"Niveau {label_educ(edu)} — corrélé à de meilleures pratiques nutritionnelles.", "icon":"🟢"})
    elif edu == 0:
        risque.append({"label":"Absence de scolarisation", "detail":"Aucune instruction maternelle — facteur de risque nutritionnel majeur confirmé par l'EDS.", "icon":"🔴"})

    # IMC mère
    if imc < 18.5:
        risque.append({"label":"Maigreur maternelle", "detail":f"IMC = {imc:.1f} — insuffisance pondérale maternelle, risque de faible poids à la naissance.", "icon":"🔴"})
    elif 18.5 <= imc <= 25:
        protecteur.append({"label":"IMC maternel optimal", "detail":f"IMC = {imc:.1f} — corpulence normale, bon état nutritionnel maternel.", "icon":"🟢"})

    # Richesse
    if ric <= 2:
        risque.append({"label":"Pauvreté ménage", "detail":f"Quintile '{label_richesse(ric)}' — accès limité à une alimentation équilibrée.", "icon":"🔴"})
    elif ric >= 4:
        protecteur.append({"label":"Aisance financière", "detail":f"Quintile '{label_richesse(ric)}' — meilleur accès à la diversité alimentaire.", "icon":"🟢"})

    # Milieu
    if mil == 2:
        risque.append({"label":"Milieu rural", "detail":"Accès réduit aux services de santé et aux marchés alimentaires.", "icon":"🔴"})
    else:
        protecteur.append({"label":"Milieu urbain", "detail":"Meilleur accès aux soins, à l'alimentation diversifiée et aux structures de santé.", "icon":"🟢"})

    # Hémoglobine mère
    if hgb < 11.0:
        risque.append({"label":"Anémie maternelle", "detail":f"Hémoglobine = {hgb:.1f} g/dl — anémie confirmée, impacte le développement fœtal.", "icon":"🔴"})
    elif hgb >= 12.5:
        protecteur.append({"label":"Bonne hémoglobine", "detail":f"Hémoglobine = {hgb:.1f} g/dl — état hématologique favorable.", "icon":"🟢"})

    # Diarrhée récente
    if diarhee == 1:
        risque.append({"label":"Diarrhée récente", "detail":"Episode dans les 2 dernières semaines — facteur aggravant du statut nutritionnel.", "icon":"🔴"})

    # Intervalle intergénésique
    if 0 < itv < 24:
        risque.append({"label":"Espacement naissance court", "detail":f"Intervalle de {itv} mois — espacement insuffisant, ressources maternelles insuffisamment reconstituées.", "icon":"🔴"})
    elif itv >= 36:
        protecteur.append({"label":"Bon espacement", "detail":f"Intervalle de {itv} mois — rétablissement maternel optimal entre grossesses.", "icon":"🟢"})

    # Rang de naissance
    if rang >= 5:
        risque.append({"label":"Rang de naissance élevé", "detail":f"Rang {rang} — ressources familiales diluées entre nombreux enfants.", "icon":"🔴"})

    # Visites prénatales
    if vps >= 4:
        protecteur.append({"label":"Suivi prénatal correct", "detail":f"{vps} consultations prénatales — recommandation OMS respectée.", "icon":"🟢"})
    elif vps < 2:
        risque.append({"label":"Suivi prénatal insuffisant", "detail":f"Seulement {vps} visite(s) prénatale(s) — accès insuffisant aux soins ante-natals.", "icon":"🔴"})

    # Eau potable
    if eau in [11, 12, 13, 21]:
        protecteur.append({"label":"Eau potable améliorée", "detail":"Accès à une source d'eau sécurisée — réduction du risque d'infections hydro-fécales.", "icon":"🟢"})
    elif eau in [43, 61, 41]:
        risque.append({"label":"Eau non améliorée", "detail":"Eau de surface ou puits non protégé — risque élevé de maladies diarrhéiques.", "icon":"🔴"})

    # Assainissement
    if wcs in [11, 12, 21]:
        protecteur.append({"label":"Assainissement amélioré", "detail":"Toilettes conformes — réduction de la contamination environnementale fécale.", "icon":"🟢"})
    elif wcs in [22, 31]:
        risque.append({"label":"Assainissement précaire", "detail":"Latrines non couvertes ou défécation en plein air — vecteur majeur de contamination.", "icon":"🔴"})

    # Combustible
    if comb in [7, 8]:
        risque.append({"label":"Combustible polluant", "detail":"Bois/charbon — pollution intérieure associée à la malnutrition infantile indirectement.", "icon":"🔴"})
    elif comb in [1, 2]:
        protecteur.append({"label":"Combustible propre", "detail":"Électricité/gaz — faible pollution intérieure, meilleure qualité de vie.", "icon":"🟢"})

    # Région à prévalence élevée
    if reg in [5, 7, 1, 4]:
        risque.append({"label":"Région à forte prévalence", "detail":f"{label_region(reg)} — parmi les zones les plus touchées par la malnutrition infantile au Cameroun (EDS 2018).", "icon":"🔴"})

    # Allaitement
    if durall >= 18:
        protecteur.append({"label":"Allaitement prolongé", "detail":f"{durall} mois d'allaitement — conforme aux recommandations OMS, facteur protecteur établi.", "icon":"🟢"})
    elif durall < 6 and age > 6:
        risque.append({"label":"Allaitement insuffisant", "detail":f"Seulement {durall} mois — allaitement exclusif recommandé au moins 6 mois (OMS).", "icon":"🔴"})

    return risque, protecteur

def local_css(file_name):
    with open(file_name) as f:
        # Streamlit automatically handles wrapper tags if a valid CSS path is parsed
        st.html(f"<style>{f.read()}</style>")

# Load the CSS file
local_css("style.css")


# 6. CSS GLOBAL
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>
""", unsafe_allow_html=True)


# 7. SESSION STATE
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []


# 8. HERO BANNER
n_sess = len(st.session_state["predictions"])

st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">
    <i class="fas fa-stethoscope"></i>&nbsp; Santé Infantile &bull; Intelligence Artificielle &bull; Cameroun EDS 2018
  </div>
  <h1>Dépistage de la <span>Sous-Nutrition Infantile</span><br>au Cameroun</h1>
  <p class="hero-desc">
    Système d'aide au dépistage basé sur un modèle ML <strong>CatBoost</strong> entraîné sur l'Enquête
    Démographique et de Santé (EDS-MICS V 2018). Évaluation de l'Indice Composite d'Échec
    Anthropométrique (ICEA) avec décomposition en Stunting · Wasting · Underweight
    et recommandations d'intervention personnalisées.
  </p>
  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hero-stat-val">CatBoost</div>
      <div class="hero-stat-lbl">Modèle IA</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">ICEA</div>
      <div class="hero-stat-lbl">Score composite</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">33</div>
      <div class="hero-stat-lbl">Variables modèle</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">12</div>
      <div class="hero-stat-lbl">Régions cameroun</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val">{n_sess}</div>
      <div class="hero-stat-lbl">Évaluations session</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ERREUR DE CHARGEMENT MODÈLE
if _load_error:
    st.error(f"""
    **Impossible de charger les artefacts du modèle CatBoost.**

    Vérifiez que le répertoire `{MODEL_DIR}/` contient bien :
    - `modele_catboost_sous_nutrition_eds2018.joblib`
    - `column_transformer_api.joblib`
    - `co_variables_inference.joblib`
    - `taux_conditionnels_cliniques.joblib`

    Erreur technique : `{_load_error}`
    """)
    st.stop()

# 9. ONGLETS PRINCIPAUX
tab1, tab2, tab3 = st.tabs([
    "Prédiction Individuelle",
    "Prédiction en Masse",
    "Historique de Session",
])



# TAB 1 — PRÉDICTION INDIVIDUELLE
with tab1:
    col_form, col_res = st.columns([1, 1], gap="large")

    with col_form:
        # ── BLOC 1 : Enfant
        st.markdown("""
        <div class="form-block">
        <div class="form-block-title"><i class="fas fa-baby"></i> Caractéristiques Biologiques de l'Enfant</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            ageenfant     = st.slider("Âge de l'enfant (mois)", 0, 59, 24)
            rangdenaissance = st.number_input("Rang de naissance", min_value=1, max_value=20, value=2, step=1)
        with c2:
            sexeenfant    = st.radio("Sexe de l'enfant", ["Masculin","Féminin"], horizontal=True)
            dureeallaitement = st.slider("Durée allaitement (mois)", 0, 59, 12)

        c1, c2 = st.columns(2)
        with c1:
            intervalleintergenesique = st.number_input("Intervalle intergénésique (mois, 0=1er né)", min_value=0, max_value=120, value=0, step=1)
        with c2:
            tailleanaissance = st.selectbox("Taille à la naissance (perçue par la mère)",
                list(DICT_TAILLE.keys()), index=2)

        diarrhee = st.radio("Diarrhée dans les 2 dernières semaines", ["Non","Oui"], horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── BLOC 2 : Mère
        st.markdown("""
        <div class="form-block">
        <div class="form-block-title"><i class="fas fa-user-nurse"></i> Profil Anthropométrique & Décisionnel de la Mère</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            agemere       = st.slider("Âge de la mère (années)", 15, 49, 28)
            hemoglobinemere = st.number_input("Hémoglobine maternelle (g/dl)", min_value=4.0, max_value=20.0, value=11.5, step=0.1, format="%.1f")
        with c2:
            imcmere       = st.number_input("IMC de la mère (kg/m²)", min_value=10.0, max_value=60.0, value=22.0, step=0.1, format="%.1f")
            tag_txt, tag_bg, tag_col = imc_tag(imcmere)
            st.markdown(f"""
            <div class="imc-display" style="background:{tag_bg}; border-color:{tag_col}40;">
              <div style="font-size:.65rem;font-weight:800;color:#64748B;letter-spacing:.8px;text-transform:uppercase;">IMC saisi</div>
              <div class="imc-val" style="color:{tag_col};">{imcmere:.1f} kg/m²</div>
              <span class="imc-badge" style="background:{tag_col}20;color:{tag_col};">{tag_txt}</span>
            </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            nombreenfantsnesvivants = st.number_input("Enfants nés vivants (total)", min_value=1, max_value=20, value=2, step=1)
            nombrevisitesprenatales = st.number_input("Visites prénatales", min_value=0, max_value=20, value=4, step=1)
        with c2:
            niveauinstructionmere = st.selectbox("Niveau d'instruction de la mère", list(DICT_EDUC.keys()))
            prisededecisionmere   = st.selectbox("Autonomie décisionnelle (soins)", list(DICT_DECISION.keys()), index=1)

        statutmatrimonialmere = st.selectbox("Statut matrimonial de la mère", list(DICT_MATRIMONIAL.keys()), index=1)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── BLOC 3 : Ménage / Contexte
        st.markdown("""
        <div class="form-block">
        <div class="form-block-title"><i class="fas fa-house-chimney"></i> Environnement Socio-Économique & WASH du Ménage</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            milieuderesidence = st.radio("Milieu de résidence", ["Urbain","Rural"], horizontal=True, index=1)
            indicederichesse  = st.selectbox("Quintile de richesse", list(DICT_RICHESSE.keys()))
        with c2:
            nombremembresmenage = st.number_input("Membres du ménage", min_value=1, max_value=30, value=5, step=1)
            region = st.selectbox("Région administrative", REGIONS_LIST, index=4)

        c1, c2 = st.columns(2)
        with c1:
            sourceeaupotable           = st.selectbox("Source d'eau potable", list(DICT_EAU.keys()), index=4)
            typeinstallationssanitaires = st.selectbox("Type de sanitaires", list(DICT_TOILETTES.keys()), index=3)
        with c2:
            typecombustiblecuisine = st.selectbox("Combustible cuisine", list(DICT_COMBUSTIBLE.keys()), index=3)
            regionecologique       = st.selectbox("Zone agro-écologique", list(DICT_ECO.keys()))

        st.markdown("</div>", unsafe_allow_html=True)

        # ── BLOC 4 : Vaccination
        st.markdown("""
        <div class="form-block">
        <div class="form-block-title"><i class="fas fa-syringe"></i> Suivi Clinique & Calendrier Vaccinal</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            vacciné        = st.radio("Carnet vaccination vérifié", ["Oui","Non"], horizontal=True)
            vaccinbcg      = st.radio("BCG", ["Oui","Non"], horizontal=True)
            vaccinpolio0   = st.radio("Polio 0 (naissance)", ["Oui","Non"], horizontal=True)
            vaccinpolio1   = st.radio("Polio 1", ["Oui","Non"], horizontal=True)
            vaccinpolio2   = st.radio("Polio 2", ["Oui","Non"], horizontal=True)
            vaccinpolio3   = st.radio("Polio 3", ["Oui","Non"], horizontal=True)
        with c2:
            vaccindtp1     = st.radio("DTP 1", ["Oui","Non"], horizontal=True)
            vaccindtp2     = st.radio("DTP 2", ["Oui","Non"], horizontal=True)
            vaccindtp3     = st.radio("DTP 3", ["Oui","Non"], horizontal=True)
            vaccinrougeole1 = st.radio("Rougeole 1", ["Oui","Non"], horizontal=True)

        st.markdown("</div>", unsafe_allow_html=True)

        submit = st.button("Lancer le diagnostic individuel", use_container_width=True)

    # ── COLONNE RÉSULTATS
    with col_res:
        if not submit and not st.session_state["predictions"]:
            st.markdown("""
            <div class="empty-state">
              <i class="fas fa-stethoscope"></i>
              <h3>Aucune évaluation en cours</h3>
              <p>Renseignez le formulaire à gauche<br>et lancez le diagnostic pour obtenir<br>
              l'analyse nutritionnelle complète avec visualisations.</p>
            </div>""", unsafe_allow_html=True)

        if submit:
            ui_vals = {
                'ageenfant': ageenfant, 'sexeenfant': sexeenfant,
                'milieuderesidence': milieuderesidence, 'indicederichesse': indicederichesse,
                'niveauinstructionmere': niveauinstructionmere, 'rangdenaissance': rangdenaissance,
                'hemoglobinemere': hemoglobinemere, 'intervalleintergenesique': intervalleintergenesique,
                'dureeallaitement': dureeallaitement, 'nombreenfantsnesvivants': nombreenfantsnesvivants,
                'nombrevisitesprenatales': nombrevisitesprenatales, 'nombremembresmenage': nombremembresmenage,
                'imcmere': imcmere, 'region': region, 'vacciné': vacciné,
                'vaccinbcg': vaccinbcg, 'diarrhee': diarrhee,
                'prisededecisionmere': prisededecisionmere,
                'statutmatrimonialmere': statutmatrimonialmere,
                'tailleanaissance': tailleanaissance, 'sourceeaupotable': sourceeaupotable,
                'typeinstallationssanitaires': typeinstallationssanitaires,
                'typecombustiblecuisine': typecombustiblecuisine,
                'regionecologique': regionecologique,
                'vaccinpolio0': vaccinpolio0, 'vaccinpolio1': vaccinpolio1,
                'vaccinpolio2': vaccinpolio2, 'vaccinpolio3': vaccinpolio3,
                'vaccindtp1': vaccindtp1, 'vaccindtp2': vaccindtp2,
                'vaccindtp3': vaccindtp3, 'vaccinrougeole1': vaccinrougeole1,
                'agemere': agemere,
            }

            try:
                row_dict = build_row_dict(ui_vals)
                df_input = pd.DataFrame([row_dict])
                p_icea   = float(executer_inference_pipeline(df_input)[0])

                p_stunting    = p_icea * taux_conditionnels['p_y1_sachant_icea']
                p_wasting     = p_icea * taux_conditionnels['p_y2_sachant_icea']
                p_underweight = p_icea * taux_conditionnels['p_y3_sachant_icea']
                is_risk       = p_icea >= 0.50

                # ── Profil rapide
                id_reg = int(region.split('=')[0])
                pm = [
                    ("Région", label_region(id_reg)),
                    ("Âge enfant", age_txt(ageenfant)),
                    ("Sexe", sexeenfant),
                    ("Milieu", milieuderesidence),
                    ("IMC mère", f"{imcmere:.1f} kg/m²"),
                    ("Hémoglobine", f"{hemoglobinemere:.1f} g/dl"),
                    ("Instruction", niveauinstructionmere),
                    ("Richesse", indicederichesse),
                ]
                cells_html = "".join(
                    f'<div class="profile-cell"><div class="profile-cell-lbl">{l}</div>'
                    f'<div class="profile-cell-val">{v}</div></div>'
                    for l, v in pm
                )
                st.markdown(f'<div class="profile-grid">{cells_html}</div>', unsafe_allow_html=True)

                # ── Carte résultat principale
                hdr_cls = "result-hdr-risk" if is_risk else "result-hdr-safe"
                icon    = "triangle-exclamation" if is_risk else "circle-check"
                titre   = "Risque de Malnutrition Élevé (ICEA ≥ 50%)" if is_risk else "Croissance Saine Diagnostiquée"
                sous    = f"Évaluation ICEA — {datetime.now().strftime('%d/%m/%Y à %H:%M')}"

                fill_cls = "gauge-fill-risk" if is_risk else "gauge-fill-safe"
                prob_pct = p_icea * 100

                # Sous-types
                st_cls = "subtype-risk" if p_stunting >= 0.30 else "subtype-safe"
                wa_cls = "subtype-risk" if p_wasting >= 0.15 else "subtype-safe"
                uw_cls = "subtype-risk" if p_underweight >= 0.20 else "subtype-safe"

                st.markdown(f"""
                <div class="result-card">
                  <div class="result-hdr {hdr_cls}">
                    <i class="fas fa-{icon}"></i>
                    <div><h2>{titre}</h2><p>{sous}</p></div>
                  </div>
                  <div class="result-body">
                    <div class="gauge-row">
                      <span>Probabilité ICEA globale</span>
                      <b>{prob_pct:.1f}%</b>
                    </div>
                    <div class="gauge-track">
                      <div class="{fill_cls}" style="width:{min(prob_pct,100):.1f}%"></div>
                    </div>
                    <div class="subtype-row" style="margin-top:16px;">
                      <div class="subtype-card {st_cls}">
                        <span class="subtype-pct">{p_stunting*100:.1f}%</span>
                        <div class="subtype-lbl"> Stunting<br>(Chronique)</div>
                      </div>
                      <div class="subtype-card {wa_cls}">
                        <span class="subtype-pct">{p_wasting*100:.1f}%</span>
                        <div class="subtype-lbl"> Wasting<br>(Aigu)</div>
                      </div>
                      <div class="subtype-card {uw_cls}">
                        <span class="subtype-pct">{p_underweight*100:.1f}%</span>
                        <div class="subtype-lbl"> Underweight<br>(Mixte)</div>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # ── Graphique radar Plotly
                categories   = ['Stunting', 'Wasting', 'Underweight', 'ICEA Globale']
                values_radar = [p_stunting*100, p_wasting*100, p_underweight*100, prob_pct]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=values_radar + [values_radar[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(220,38,38,0.18)' if is_risk else 'rgba(22,163,74,0.18)',
                    line=dict(color='#DC2626' if is_risk else '#16A34A', width=2.5),
                    marker=dict(size=7, color='#DC2626' if is_risk else '#16A34A'),
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%',
                                        gridcolor='#E2E8F0', tickfont=dict(size=10)),
                        angularaxis=dict(gridcolor='#E2E8F0', tickfont=dict(size=11, color='#1A202C')),
                        bgcolor='white',
                    ),
                    showlegend=False,
                    height=280,
                    margin=dict(l=40, r=40, t=30, b=30),
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

                # ── Facteurs d'influence
                risque_factors, protecteur_factors = compute_influence_factors(row_dict)

                st.markdown("<div style='margin-top:4px;'>", unsafe_allow_html=True)

                if risque_factors:
                    html_factors = '<div class="factor-section-title">⚠ Facteurs de risque identifiés</div>'
                    for f in risque_factors:
                        html_factors += f"""
                        <div class="factor-item factor-risk">
                          <span class="factor-icon">{f['icon']}</span>
                          <div>
                            <span class="factor-label">{f['label']}</span>
                            <span class="factor-detail">{f['detail']}</span>
                          </div>
                        </div>"""
                    st.markdown(html_factors, unsafe_allow_html=True)

                if protecteur_factors:
                    html_prot = '<div class="factor-section-title"> Facteurs protecteurs</div>'
                    for f in protecteur_factors:
                        html_prot += f"""
                        <div class="factor-item factor-safe">
                          <span class="factor-icon">{f['icon']}</span>
                          <div>
                            <span class="factor-label">{f['label']}</span>
                            <span class="factor-detail">{f['detail']}</span>
                          </div>
                        </div>"""
                    st.markdown(html_prot, unsafe_allow_html=True)

                # ── Barres d'importance des variables (Plotly)
                feature_importance = {
                    "Âge enfant": min(ageenfant/59*100, 100),
                    "IMC mère": max(0, (22-imcmere)/22*100) if imcmere < 22 else 0,
                    "Hémoglobine": max(0, (12.5-hemoglobinemere)/12.5*100),
                    "Intervalle intergénésique": max(0, (24-intervalleintergenesique)/24*100) if intervalleintergenesique > 0 else 50,
                    "Niveau d'instruction": max(0, (3-DICT_EDUC[niveauinstructionmere])/3*100),
                    "Richesse ménage": max(0, (3-DICT_RICHESSE[indicederichesse])/3*100),
                    "Milieu résidence": 60 if milieuderesidence == "Rural" else 15,
                    "Source d'eau": 70 if DICT_EAU[sourceeaupotable] in [43,61,41] else 15,
                }
                feat_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:7]
                feat_names  = [x[0] for x in feat_sorted]
                feat_vals   = [x[1] for x in feat_sorted]
                feat_colors = ['#DC2626' if v > 40 else '#D97706' if v > 20 else '#16A34A' for v in feat_vals]

                fig_bar = go.Figure(go.Bar(
                    y=feat_names, x=feat_vals,
                    orientation='h',
                    marker=dict(color=feat_colors, line=dict(width=0)),
                    text=[f"{v:.0f}%" for v in feat_vals],
                    textposition='outside', textfont=dict(size=10, color='#374151'),
                ))
                fig_bar.update_layout(
                    title=dict(text="Variables d'influence estimées", font=dict(size=13, color='#0B2447'), x=0),
                    xaxis=dict(title="Contribution au risque (%)", range=[0, 115],
                               ticksuffix='%', gridcolor='#F1F5F9', showgrid=True,
                               zeroline=False, tickfont=dict(size=9)),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=10, color='#374151')),
                    height=280, margin=dict(l=20, r=50, t=40, b=30),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

                # ── Interventions si risque
                if is_risk:
                    intv_html = '<div class="interventions"><h4> Interventions recommandées</h4>'
                    if p_stunting >= 0.30:
                        intv_html += '<div class="intv-item"><h5> Retard de croissance (Stunting) — Malnutrition chronique</h5><p>Diversification alimentaire riche en protéines animales, zinc, vitamines A et D. Supplementation micronutriments. Suivi pédiatrique mensuel sur 12–24 mois. Programme REACH ou équivalent local.</p></div>'
                    if p_wasting >= 0.15:
                        intv_html += '<div class="intv-item"><h5> Amaigrissement aigu (Wasting) — Urgence médicale</h5><p><strong>Référence immédiate</strong> en Centre de Récupération et d\'Education Nutritionnelle (CREN). Aliments Thérapeutiques Prêts à l\'Emploi (ATPE/Plumpy\'Nut). Rehydratation et dépistage infections associées.</p></div>'
                    if p_underweight >= 0.20:
                        intv_html += '<div class="intv-item"><h5> Insuffisance pondérale — Malnutrition mixte</h5><p>Augmentation apport calorique de 20–30% via corps gras sains (huile de palme, arachide). Repas plus fréquents. Surveillance bimensuelle de la courbe poids/âge.</p></div>'
                    intv_html += '</div>'
                    st.markdown(intv_html, unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer">
                  <i class="fas fa-circle-info"></i>
                  <span>Ce résultat est une estimation statistique du modèle CatBoost entraîné sur l'EDS Cameroun 2018.
                  L'ICEA est un indice composite (Stunting + Wasting + Underweight). Il ne constitue pas un diagnostic
                  médical et ne remplace pas l'évaluation d'un professionnel de santé qualifié.</span>
                </div>
                </div>""", unsafe_allow_html=True)

                # Sauvegarde session
                record = {
                    **row_dict,
                    "p_icea": p_icea, "p_stunting": p_stunting,
                    "p_wasting": p_wasting, "p_underweight": p_underweight,
                    "diagnostic": int(is_risk),
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M"),
                    "region_nom": label_region(id_reg),
                }
                st.session_state["predictions"].append(record)

            except Exception as e:
                st.error(f"Erreur pipeline inférence : {e}")
                st.exception(e)

        elif st.session_state["predictions"] and not submit:
            last = st.session_state["predictions"][-1]
            p_icea = last.get("p_icea", 0)
            is_risk = p_icea >= 0.50
            hdr_cls = "result-hdr-risk" if is_risk else "result-hdr-safe"
            icon    = "triangle-exclamation" if is_risk else "circle-check"
            titre   = "Risque de Malnutrition Élevé" if is_risk else "Croissance Saine"
            fill_cls = "gauge-fill-risk" if is_risk else "gauge-fill-safe"
            st.info("⏱ Dernier résultat de session. Relancez une évaluation pour mettre à jour.")
            st.markdown(f"""
            <div class="result-card">
              <div class="result-hdr {hdr_cls}">
                <i class="fas fa-{icon}"></i>
                <div><h2>{titre}</h2><p>Évaluation du {last.get('date','—')}</p></div>
              </div>
              <div class="result-body">
                <div class="gauge-row"><span>ICEA global</span><b>{p_icea*100:.1f}%</b></div>
                <div class="gauge-track">
                  <div class="{fill_cls}" style="width:{min(p_icea*100,100):.1f}%"></div>
                </div>
                <div class="subtype-row" style="margin-top:14px;">
                  <div class="subtype-card {'subtype-risk' if last.get('p_stunting',0)>=.3 else 'subtype-safe'}">
                    <span class="subtype-pct">{last.get('p_stunting',0)*100:.1f}%</span>
                    <div class="subtype-lbl"> Stunting</div>
                  </div>
                  <div class="subtype-card {'subtype-risk' if last.get('p_wasting',0)>=.15 else 'subtype-safe'}">
                    <span class="subtype-pct">{last.get('p_wasting',0)*100:.1f}%</span>
                    <div class="subtype-lbl"> Wasting</div>
                  </div>
                  <div class="subtype-card {'subtype-risk' if last.get('p_underweight',0)>=.20 else 'subtype-safe'}">
                    <span class="subtype-pct">{last.get('p_underweight',0)*100:.1f}%</span>
                    <div class="subtype-lbl"> Underweight</div>
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)


# TAB 2 — PRÉDICTION EN MASSE
with tab2:
    st.markdown("""
    <div style="margin-bottom:16px;">
      <div class="form-block-title" style="font-size:.85rem;color:#0B2447;font-weight:800;margin-bottom:6px;">
        <i class="fas fa-upload" style="color:#006D77;"></i> Diagnostic de masse — Import CSV
      </div>
      <p style="font-size:.9rem;color:#64748B;margin:0;line-height:1.65;">
        Chargez un fichier CSV contenant les dossiers des enfants codés <strong>numériquement</strong>
        selon la nomenclature de l'EDS 2018. Les colonnes doivent correspondre exactement aux variables
        attendues par le modèle CatBoost. Le fichier téléchargeable en bas contient un exemple.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_info = st.columns([1, 1], gap="large")

    with col_up:
        # Téléchargement d'un exemple
        exemple_data = {col: [0]*3 for col in COLONNES_OBLIGATOIRES}
        exemple_data.update({
            'ageenfant': [12, 36, 6],
            'sexeenfant': [1, 2, 1],
            'milieuderesidence': [2, 1, 2],
            'indicederichesse': [1, 3, 2],
            'niveauinstructionmere': [0, 2, 1],
            'rangdenaissance': [3, 1, 2],
            'hemoglobinemere': [10.2, 12.8, 11.0],
            'intervalleintergenesique': [18, 36, 0],
            'dureeallaitement': [6, 24, 18],
            'nombreenfantsnesvivants': [4, 2, 3],
            'nombrevisitesprenatales': [2, 5, 4],
            'nombremembresmenage': [8, 4, 6],
            'imcmere': [17.5, 22.0, 20.0],
            'region': [5, 2, 7],
            'vacciné': [0, 1, 1],
            'vaccinbcg': [1, 1, 1],
            'diarrhee': [1, 0, 0],
            'prisededecisionmere': [3, 2, 1],
            'statutmatrimonialmere': [1, 1, 2],
            'tailleanaissance': [3, 2, 4],
            'sourceeaupotable': [41, 11, 21],
            'typeinstallationssanitaires': [22, 11, 21],
            'typecombustiblecuisine': [8, 2, 7],
            'regionecologique': [1, 2, 1],
            'vaccinpolio0': [0, 1, 1],
            'vaccinpolio1': [0, 1, 1],
            'vaccinpolio2': [0, 1, 0],
            'vaccinpolio3': [0, 1, 0],
            'vaccindtp1': [0, 1, 1],
            'vaccindtp2': [0, 1, 0],
            'vaccindtp3': [0, 0, 0],
            'vaccinrougeole1': [0, 1, 1],
            'agemere': [22, 32, 25],
        })
        df_exemple = pd.DataFrame(exemple_data)
        buf_ex = BytesIO()
        df_exemple.to_csv(buf_ex, index=False)
        st.download_button(
            "Télécharger un fichier CSV exemple",
            data=buf_ex.getvalue(),
            file_name="exemple_inference_catboost.csv",
            mime="text/csv",
        )

        st.markdown("<div style='margin-top:14px;'>", unsafe_allow_html=True)
        fichier_csv = st.file_uploader(
            "Charger le fichier CSV à analyser",
            type=["csv"],
            help="Colonnes numériques — voir nomenclature EDS 2018 dans le fichier exemple"
        )

        if fichier_csv is not None:
            btn_masse = st.button("Lancer l'analyse de masse", use_container_width=True)
        else:
            btn_masse = False
        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class="form-block" style="height:auto;">
          <div class="form-block-title"> Colonnes requises dans le CSV</div>
          <div style="font-size:.8rem;color:#374151;line-height:1.7;columns:2;column-gap:18px;">
            ageenfant · sexeenfant · milieuderesidence · indicederichesse ·
            niveauinstructionmere · rangdenaissance · hemoglobinemere ·
            intervalleintergenesique · dureeallaitement · nombreenfantsnesvivants ·
            nombrevisitesprenatales · nombremembresmenage · imcmere · region ·
            vacciné · vaccinbcg · diarrhee · prisededecisionmere ·
            statutmatrimonialmere · tailleanaissance · sourceeaupotable ·
            typeinstallationssanitaires · typecombustiblecuisine ·
            regionecologique · vaccinpolio0 · vaccinpolio1 · vaccinpolio2 ·
            vaccinpolio3 · vaccindtp1 · vaccindtp2 · vaccindtp3 ·
            vaccinrougeole1 · agemere
          </div>
          <div style="margin-top:12px;font-size:.78rem;color:#94A3B8;border-top:1px solid #E2E8F0;padding-top:10px;">
            ⚠ Toutes les valeurs doivent être <strong>numériques</strong> selon la nomenclature EDS.
            La colonne <code>lieuderesidence</code> sera générée automatiquement depuis <code>milieuderesidence</code>.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Traitement masse
    if btn_masse and fichier_csv is not None:
        try:
            df_csv = pd.read_csv(fichier_csv)
            manquantes = [col for col in COLONNES_OBLIGATOIRES if col not in df_csv.columns]

            if manquantes:
                st.error(f"**Colonnes manquantes :** {', '.join(manquantes)}")
            else:
                with st.spinner("Inférence CatBoost en cours…"):
                    probabilites  = executer_inference_pipeline(df_csv)

                df_csv['probabilite_icea']                       = probabilites
                df_csv['diagnostic_icea']                         = (probabilites >= 0.50).astype(int)
                df_csv['prob_stunting']                           = probabilites * taux_conditionnels['p_y1_sachant_icea']
                df_csv['prob_wasting']                            = probabilites * taux_conditionnels['p_y2_sachant_icea']
                df_csv['prob_underweight']                        = probabilites * taux_conditionnels['p_y3_sachant_icea']
                df_csv['statut_global']                           = df_csv['diagnostic_icea'].map({1:"À risque", 0:"Normal"})
                df_csv['region_nom']                              = df_csv['region'].map(lambda x: label_region(x))

                total    = len(df_csv)
                alertes  = int(df_csv['diagnostic_icea'].sum())
                normaux  = total - alertes
                taux_al  = alertes / total * 100
                moy_icea = df_csv['probabilite_icea'].mean() * 100

                # ── Métriques
                st.markdown(f"""
                <div class="metric-grid" style="margin-top:20px;">
                  <div class="metric-card">
                    <div class="metric-val metric-info">{total:,}</div>
                    <div class="metric-lbl">Enfants analysés</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-val metric-risk">{alertes:,}</div>
                    <div class="metric-lbl">Cas à risque ICEA</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-val metric-safe">{normaux:,}</div>
                    <div class="metric-lbl">Croissance normale</div>
                  </div>
                  <div class="metric-card">
                    <div class="metric-val metric-warn">{taux_al:.1f}%</div>
                    <div class="metric-lbl">Taux de prévalence</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Visualisations Plotly storytelling
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    # Distribution ICEA histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=df_csv['probabilite_icea']*100,
                        nbinsx=30,
                        name='Distribution ICEA',
                        marker=dict(
                            color=[
                                '#DC2626' if v >= 50 else '#F59E0B' if v >= 30 else '#16A34A'
                                for v in (df_csv['probabilite_icea']*100).tolist()
                            ],
                            line=dict(width=0),
                        ),
                        opacity=0.85,
                    ))
                    fig_hist.add_vline(x=50, line=dict(dash='dash', color='#DC2626', width=2),
                                       annotation_text='Seuil risque 50%', annotation_position='top right',
                                       annotation_font=dict(size=10, color='#DC2626'))
                    fig_hist.update_layout(
                        title=dict(text="Distribution des scores ICEA", font=dict(size=13, color='#0B2447'), x=0),
                        xaxis=dict(title='Probabilité ICEA (%)', ticksuffix='%', gridcolor='#F1F5F9'),
                        yaxis=dict(title='Nombre d\'enfants', gridcolor='#F1F5F9', zeroline=False),
                        bargap=0.05, height=280, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=40, b=30),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

                with viz_col2:
                    # Pie chart
                    fig_pie = go.Figure(go.Pie(
                        labels=['À risque', 'Normal'],
                        values=[alertes, normaux],
                        hole=0.58,
                        marker=dict(colors=['#DC2626','#16A34A'], line=dict(color='white', width=3)),
                        textinfo='percent+label',
                        textfont=dict(size=12, color='white'),
                        pull=[0.04, 0],
                    ))
                    fig_pie.update_layout(
                        title=dict(text="Répartition des diagnostics ICEA", font=dict(size=13, color='#0B2447'), x=0),
                        annotations=[dict(text=f"<b>{taux_al:.0f}%</b><br>à risque", x=0.5, y=0.5,
                                          font=dict(size=14, color='#DC2626'), showarrow=False)],
                        height=280, paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=40, b=20), showlegend=False,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

                # ── Analyse par région si disponible
                if 'region' in df_csv.columns:
                    df_reg = df_csv.groupby('region').agg(
                        total=('diagnostic_icea','count'),
                        alertes=('diagnostic_icea','sum'),
                        moy_icea=('probabilite_icea','mean'),
                    ).reset_index()
                    df_reg['taux']     = df_reg['alertes'] / df_reg['total'] * 100
                    df_reg['reg_nom']  = df_reg['region'].map(label_region)
                    df_reg = df_reg.sort_values('taux', ascending=True)

                    fig_reg = go.Figure(go.Bar(
                        y=df_reg['reg_nom'], x=df_reg['taux'],
                        orientation='h',
                        marker=dict(
                            color=df_reg['taux'].apply(lambda v: '#DC2626' if v >= 50 else '#F59E0B' if v >= 30 else '#16A34A'),
                            line=dict(width=0),
                        ),
                        text=df_reg['taux'].apply(lambda v: f'{v:.1f}%'),
                        textposition='outside', textfont=dict(size=10),
                    ))
                    fig_reg.update_layout(
                        title=dict(text="Taux de risque ICEA par région", font=dict(size=13, color='#0B2447'), x=0),
                        xaxis=dict(title='Taux de risque (%)', ticksuffix='%', range=[0,115],
                                   gridcolor='#F1F5F9', zeroline=False),
                        yaxis=dict(tickfont=dict(size=10)),
                        height=max(280, len(df_reg)*30+60),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=60, t=40, b=20), showlegend=False,
                    )
                    st.plotly_chart(fig_reg, use_container_width=True, config={"displayModeBar": False})

                # ── Analyse sous-types
                viz2a, viz2b, viz2c = st.columns(3)
                for col_viz, lbl_st, col_k, seuil, col_color in [
                    (viz2a, "Stunting (Chronique)", "prob_stunting", 0.30, '#7C3AED'),
                    (viz2b, "Wasting (Aigu)",       "prob_wasting",  0.15, '#EA580C'),
                    (viz2c, "Underweight (Mixte)",  "prob_underweight",0.20,'#0369A1'),
                ]:
                    with col_viz:
                        nb_at_risk = (df_csv[col_k] >= seuil).sum()
                        fig_gauge  = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=df_csv[col_k].mean()*100,
                            domain={'x':[0,1],'y':[0,1]},
                            title={'text': lbl_st, 'font': {'size': 11, 'color': '#0B2447'}},
                            number={'suffix':'%', 'font':{'size':18,'color':col_color,'family':'Sora'}},
                            gauge={
                                'axis':{'range':[0,100],'tickwidth':1,'tickcolor':"#E2E8F0"},
                                'bar':{'color':col_color},
                                'bgcolor':'white',
                                'borderwidth':0,
                                'steps':[
                                    {'range':[0,20],'color':'#F0FDF4'},
                                    {'range':[20,50],'color':'#FFFBEB'},
                                    {'range':[50,100],'color':'#FEF2F2'},
                                ],
                            }
                        ))
                        fig_gauge.update_layout(
                            height=180, paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=20, r=20, t=40, b=10),
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
                        st.caption(f"**{nb_at_risk}** cas au-dessus du seuil ({seuil*100:.0f}%)")

                # ── Tableau aperçu
                st.markdown("#### Aperçu des résultats (50 premiers)")
                cols_afficher = ['statut_global','probabilite_icea','prob_stunting','prob_wasting',
                                  'prob_underweight','region_nom','ageenfant','sexeenfant',
                                  'indicederichesse','milieuderesidence']
                cols_disp = [c for c in cols_afficher if c in df_csv.columns]
                df_show = df_csv[cols_disp].head(50).copy()
                pct_cols = ['probabilite_icea','prob_stunting','prob_wasting','prob_underweight']
                for c in pct_cols:
                    if c in df_show.columns:
                        df_show[c] = (df_show[c] * 100).round(1).astype(str) + '%'
                st.dataframe(df_show, use_container_width=True, height=320)

                # ── Téléchargement résultats
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                chemin_sortie = os.path.join(OUTPUT_DIR, "predictions_masse_sorties.csv")
                df_csv.to_csv(chemin_sortie, index=False)

                dl1, dl2 = st.columns(2)
                with dl1:
                    buf_csv_out = BytesIO()
                    df_csv.to_csv(buf_csv_out, index=False)
                    st.download_button(
                        "Télécharger résultats complets (CSV)",
                        data=buf_csv_out.getvalue(),
                        file_name=f"predictions_masse_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                    )
                with dl2:
                    buf_xl_out = BytesIO()
                    df_csv.to_excel(buf_xl_out, index=False, sheet_name="Resultats_ICEA")
                    st.download_button(
                        "Télécharger résultats complets (Excel)",
                        data=buf_xl_out.getvalue(),
                        file_name=f"predictions_masse_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")
            st.exception(e)



# TAB 3 — HISTORIQUE SESSION
with tab3:
    st.markdown("""
    <div class="form-block-title" style="font-size:.85rem;color:#0B2447;font-weight:800;margin-bottom:12px;">
      <i class="fas fa-clock-rotate-left" style="color:#006D77;"></i> Historique des évaluations individuelles
    </div>""", unsafe_allow_html=True)

    n_hist = len(st.session_state["predictions"])
    if n_hist == 0:
        st.markdown("""
        <div class="empty-state">
          <i class="fas fa-folder-open"></i>
          <h3>Aucune évaluation réalisée dans cette session</h3>
          <p>Rendez-vous dans l'onglet <strong>Prédiction Individuelle</strong> pour commencer.<br>
          Les résultats s'accumulent ici au fil de la session.</p>
        </div>""", unsafe_allow_html=True)
    else:
        df_hist = pd.DataFrame(st.session_state["predictions"])

        # KPIs session
        n_risk = int(df_hist['diagnostic'].sum())
        n_ok   = n_hist - n_risk
        moy_icea = df_hist['p_icea'].mean() * 100

        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card">
            <div class="metric-val metric-info">{n_hist}</div>
            <div class="metric-lbl">Évaluations session</div>
          </div>
          <div class="metric-card">
            <div class="metric-val metric-risk">{n_risk}</div>
            <div class="metric-lbl">Cas à risque</div>
          </div>
          <div class="metric-card">
            <div class="metric-val metric-safe">{n_ok}</div>
            <div class="metric-lbl">Cas normaux</div>
          </div>
          <div class="metric-card">
            <div class="metric-val metric-warn">{moy_icea:.1f}%</div>
            <div class="metric-lbl">Score ICEA moyen</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Graphique évolution ICEA dans la session
        if n_hist >= 2:
            fig_evo = go.Figure()
            fig_evo.add_trace(go.Scatter(
                x=list(range(1, n_hist+1)),
                y=df_hist['p_icea']*100,
                mode='lines+markers',
                line=dict(color='#006D77', width=2.5),
                marker=dict(
                    size=10,
                    color=['#DC2626' if d else '#16A34A' for d in df_hist['diagnostic']],
                    line=dict(color='white', width=2),
                ),
                text=df_hist['date'],
                hovertemplate='%{text}<br>ICEA: %{y:.1f}%<extra></extra>',
            ))
            fig_evo.add_hline(y=50, line=dict(dash='dash', color='#DC2626', width=1.5),
                               annotation_text='Seuil 50%', annotation_position='right',
                               annotation_font=dict(size=9, color='#DC2626'))
            fig_evo.update_layout(
                title=dict(text="Évolution du score ICEA dans la session", font=dict(size=13, color='#0B2447'), x=0),
                xaxis=dict(title='Évaluation N°', gridcolor='#F1F5F9', zeroline=False, dtick=1),
                yaxis=dict(title='Score ICEA (%)', ticksuffix='%', range=[0,105], gridcolor='#F1F5F9', zeroline=False),
                height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=40, b=30), showlegend=False,
            )
            st.plotly_chart(fig_evo, use_container_width=True, config={"displayModeBar": False})

        # Tableau historique
        disp_cols = {
            'date':'Date', 'region_nom':'Région',
            'ageenfant':'Âge (mois)', 'agemere':'Âge mère',
            'imcmere':'IMC mère', 'hemoglobinemere':'Hgb (g/dl)',
            'p_icea':'ICEA (%)', 'p_stunting':'Stunting (%)',
            'p_wasting':'Wasting (%)', 'p_underweight':'Underweight (%)',
        }
        df_disp = pd.DataFrame()
        for src, dst in disp_cols.items():
            if src in df_hist.columns:
                if src.startswith('p_'):
                    df_disp[dst] = (df_hist[src]*100).round(1).astype(str)+'%'
                else:
                    df_disp[dst] = df_hist[src]
        df_disp['Statut'] = df_hist['diagnostic'].map({1:'⚠ À risque', 0:'✅ Normal'})

        st.dataframe(df_disp, use_container_width=True, height=320)

        # Exports
        c1, c2, c3 = st.columns(3)
        with c1:
            buf_xl2 = BytesIO()
            df_disp.to_excel(buf_xl2, index=False, sheet_name="Historique")
            st.download_button(" Exporter Excel", buf_xl2.getvalue(),
                               f"historique_{datetime.now().strftime('%Y%m%d')}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c2:
            buf_csv2 = BytesIO()
            df_disp.to_csv(buf_csv2, index=False)
            st.download_button(" Exporter CSV", buf_csv2.getvalue(),
                               f"historique_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        with c3:
            if st.button(" Effacer l'historique"):
                st.session_state["predictions"] = []
                st.rerun()


# FOOTER
st.markdown("""
<div class="app-footer">
  <i class="fas fa-code"></i>
  Développé par <a href="https://github.com/teuzem" target="_blank">NGOUMTSOP TEUZEM Yeiayel</a>
  &nbsp;—&nbsp; NutriScreen Cameroun &nbsp;—&nbsp; EDS-MICS V 2018
  <br>
  Modèle <strong>CatBoost</strong> · Pipeline ICEA (Stunting · Wasting · Underweight) ·
  33 variables · Données : Enquête Démographique et de Santé du Cameroun 2018
</div>
""", unsafe_allow_html=True)
