# Dépistage Sous nutrition Infantile au Cameroun EDS 2018

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Stacking-green)

Application web interactive pour l'évaluation et le dépistage de la Sous nutrition infantile au Cameroun, basée sur un modèle de Machine Learning (Stacking Classifier) entraîné sur les données de l'Enquête Démographique et de Santé (EDSC-V 2018).

## Fonctionnalités

- **Évaluation individuelle** : Renseignez le profil d'un enfant et de sa mère (âge, IMC, éducation, richesse, région, milieu) pour obtenir une prédiction instantanée du risque de sous-nutrition.
- **Explications détaillées** : Le modèle génère une explication complète des facteurs de risque et des facteurs protecteurs identifiés.
- **Rapport PDF** : Téléchargez les résultats d'évaluation au format PDF pour les partager avec des professionnels de santé.
- **Cartographie interactive** : Visualisez la répartition géographique des cas normaux et à risque sur le territoire camerounais.
- **Historique et Suivi** : Conservez l'historique des évaluations de votre session et exportez-le au format Excel.

## 🛠️ Installation et Prérequis

1. **Cloner le dépôt ou extraire les fichiers** dans un répertoire local.
2. **Créer un environnement virtuel** (recommandé) :
   ```bash
   python -m venv Env
   ```
3. **Activer l'environnement virtuel** :
   - Windows : `Env\Scripts\activate`
   - Linux/Mac : `source Env/bin/activate`
4. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## Démarrage de l'Application

Une fois les dépendances installées et l'environnement activé, lancez l'application Streamlit avec la commande suivante :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

## 📂 Structure du Projet

- `app.py` : Script principal de l'application Streamlit.
- `requirements.txt` : Liste des packages Python nécessaires.
- `models/` : Contient les modèles de Machine Learning (`modele_malnutrition_stacking.joblib` et `scaler.joblib`).
- `Dataset_Malnutrition_Cameroun_2018.xlsx` : Jeu de données  d'entrainement.

## ⚠️ Avertissement

Cette application est un outil d'aide basé sur des modèles statistiques et ne remplace en aucun cas un diagnostic médical professionnel. En cas de doute sur l'état nutritionnel d'un enfant, veuillez consulter un professionnel de la santé.

## 👨‍💻 Auteur

Développé par [NGOUMTSOP TEUZEM Yeiayel](https://github.com/teuzem)
Étude de la Malnutrition Infantile — EDS Cameroun 2018
