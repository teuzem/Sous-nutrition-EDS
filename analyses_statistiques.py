# Tests de normalites des variables continues, analyses statistiques et 
# visualisations de la distribution des donnees

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Configuration des parametre plot de design general de graphiques 
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
    }
)

# Creation des dossiers pour sauvegarder les resultats et les graphiques
os.makedirs("Rapports_Statistiques", exist_ok=True)
os.makedirs("Graphiques_Visualisation", exist_ok=True)

# Chargement de la feuille 'donnees' contenant les codes numériques EDS
df = pd.read_excel("dataset_sous_nutrition_EDSC2018.xlsx", sheet_name='donnees')
print(f"Base de données des enfants en situation de sous-nutrition et normaux au Cameroun. Échantillon : {df.shape} enfants.")

# Variables quantitatives discrètes (valeurs entières issues de comptages ou d'intervalles)
vars_discretes = [
    "rangdenaissance", "hemoglobinemere", "intervalleintergenesique", "dureeallaitement", "agemere",
    "nombreenfantsnesvivants", "nombrevisitesprenatales", "nombremembresmenage", "ageenfant"
]

# Variables quantitatives continues (mesures anthropométriques, biologiques ou scores continus)
vars_continues = [
    "imcmere", "ageenfant", "scoreztaillepourage", "scorezpoidspourtaille", 
    "scorezpoidspourage", "agemere"
]

vars_categorielles = [
    "sexeenfant", "milieuderesidence", "niveauinstructionmere", "indicederichesse","region", "vacciné",
    "vaccinbcg", "diarrhee", "prisededecisionmere", "statutmatrimonialmere", "tailleanaissance",
    "sourceeaupotable", "typeinstallationssanitaires", "typecombustiblecuisine", "lieuderesidence",
    "regionecologique", "vaccinpolio0", "vaccinpolio2", "vaccinpolio3", "vaccindtp1", "vaccindtp2", 
    "vaccindtp3", "vaccinpolio1", "vaccinrougeole1"
]


# Definition des labels pour les graphiques de distributions des variables categorielles
libelles_mapping = {
    "sexeenfant": {
        1: "Masculin",
        2: "Féminin"
    },
    "milieuderesidence": {
        1: "Urbain",
        2: "Rural"
    },
    "niveauinstructionmere": {
        0: "Aucun",
        1: "Primaire",
        2: "Secondaire",
        3: "Supérieur"
    },
    "indicederichesse": {
        1: "Plus pauvre",
        2: "Pauvre",
        3: "Moyen",
        4: "Riche",
        5: "Plus riche"
    },
    "region": {
        1: "Adamaoua",
        2: "Centre (sans Yaoundé)",
        3: "Douala",
        4: "Est",
        5: "Extrême-Nord",
        6: "Littoral (sans Douala)",
        7: "Nord",
        8: "Nord-Ouest",
        9: "Ouest",
        10: "Sud",
        11: "Sud-Ouest",
        12: "Yaoundé"
    },
    "tailleanaissance": {
        1: "Très gros",
        2: "Plus gros que la moyenne",
        3: "Moyen",
        4: "Plus petit que la moyenne",
        5: "Très petit"
    },
    "vaccinbcg": {
        0: "Pas vacciné",
        1: "Date sur carnet",
        2: "Rapporté par la mère",
        3: "Marqué sur carnet"
    },
    "diarrhee": {
        0: "Non",
        1: "Oui, au cours des 24 dernières heures",
        2: "Oui, au cours des 2 dernières semaines"
    },
    "prisededecisionmere": {
        1: "Répondante seule",
        2: "Répondante et conjoint",
        3: "Conjoint seul",
        4: "Quelqu'un d'autre",
        5: "Répondante et autre personne"
    },
    "statutmatrimonialmere": {
        0: "Jamais mariée",
        1: "Mariée",
        2: "Union libre / Cohabitation",
        3: "Veuve",
        4: "Divorcée",
        5: "Séparée"
    },
    "sourceeaupotable": {
        11: "Eau courante: dans le logement",
        12: "Eau courante: dans la cour/parcelle",
        13: "Eau courante: robinet public",
        21: "Puits tubé / forage",
        31: "Puits creusé protégé",
        32: "Puits creusé non protégé",
        41: "Source protégée",
        42: "Source non protégée",
        51: "Eau de pluie",
        61: "Camion-citerne",
        62: "Boutique / Charette avec petite citerne",
        71: "Eau de surface (rivière/barrage/lac)",
        91: "Eau en bouteille",
        92: "Eau en sachet (Pure Wata)"
    },
    "typeinstallationssanitaires": {
        11: "Chasse d'eau: branchée à un égout",
        12: "Chasse d'eau: fosse septique",
        13: "Chasse d'eau: latrines à fosse",
        14: "Chasse d'eau: vers un endroit inconnu",
        21: "Latrines à fosse améliorées ventilées (VIP)",
        22: "Latrines à fosse avec dalle",
        23: "Latrines à fosse sans dalle / fosse ouverte",
        31: "Toilettes à compostage",
        41: "Toilettes suspendues / sur pilotis",
        51: "Pas de toilettes / nature / brousse"
    },
    "typecombustiblecuisine": {
        1: "Électricité",
        2: "Gaz de pétrole liquéfié (GPL) / Gaz butane",
        3: "Kérosène / Pétrole lampant",
        4: "Charbon de bois",
        5: "Bois de chauffe",
        6: "Paille / Arbustes / Herbe",
        7: "Cultures / Déchets agricoles",
        8: "Bouse d'animaux",
        95: "Ne cuisine pas à la maison"
    },
    "lieuderesidence": {
        1: "Urbain",
        2: "Rural"
    },
    "regionecologique": {
        1: "Adamaoua",
        2: "Centre",
        3: "Est",
        4: "Extrême-Nord",
        5: "Littoral",
        6: "Nord",
        7: "Nord-Ouest",
        8: "Ouest",
        9: "Sud",
        10: "Sud-Ouest"
    },
    
    "vaccinpolio0": {
        0: "Non", 1: "Date de vaccination sur la carte",
        2: "Signalé par la mère", 3: "Vaccination indiquée sur la carte"
    },
    "cartesante": {
        0: "Aucune carte", 1: "Oui vu", 2: "Oui pas vu", 3: "Je n'ai plus de carte"
    },
    "vaccinbcg": {
        0: "Non", 1: "Date de vaccination sur la carte",
        2: "Signalé par la mère", 3: "Vaccination indiquée sur la carte"
    },
    "vaccindtp1": {
        0: "Non", 1: "Date de vaccination sur la carte",
        2: "Signalé par la mère", 3: "Vaccination indiquée sur la carte"
    },
    "vaccinpolio1": {
        0: "Non", 1: "Date de vaccination sur la carte",
        2: "Signalé par la mère", 3: "Vaccination indiquée sur la carte"
    },
    "vaccindtp2": {
        0: "Non", 1: "Date de vaccination sur la carte",
        2: "Signalé par la mère", 3: "Vaccination indiquée sur la carte"
    },
    "vaccinpolio2": {
        0: "Non", 1: "Date de vaccination sur la carte",
        2: "Signalé par la mère", 3: "Vaccination indiquée sur la carte"
    },
    "vaccindtp3": {
        0: "Non", 1: "Date de vaccination sur la carte",
        2: "Signalé par la mère", 3: "Vaccination indiquée sur la carte"
    },
    "vaccinpolio3": {
        0: "Non", 1: "Date de vaccination sur la carte",
        2: "Signalé par la mère", 3: "Vaccination indiquée sur la carte"
    },
    "vaccinrougeole1": {
        0: "Non", 1: "Date de vaccination sur la carte",
        2: "Signalé par la mère", 3: "Vaccination indiquée sur la carte"
    },
    "vacciné": {
        0: "Non", 1: "Oui"
    },
    "diarrhee": {
        0: "Pas de diarrhée", 1: "Oui (dernières 24h)", 2: "Oui (2 dernières semaines)"
    }
}


# Section Quantitative : Tests de normalité, statistiques descriptives et histogrammes de distributions

rapport_normalite = []
stats_continues = []

for var in vars_continues:
    valeurs = df[var].dropna()

    val_moyenne = valeurs.mean()
    val_mediane = valeurs.median()

    # Approche 1 : Test de normalité de Shapiro-Wilk (Échantillon de 5000 max suivant documentation SciPy)
    if len(valeurs) <= 5000:
        _, p_sw = stats.shapiro(valeurs)
    else:
        _, p_sw = stats.shapiro(valeurs.sample(5000, random_state=42))

    # Approche 2 : Test de D'Agostino-Pearson (Basé sur le Skewness et Kurtosis)
    _, p_dag = stats.normaltest(valeurs)

    forme_dist = (
        "Normale (Symétrique)"
        if (p_sw > 0.05 and p_dag > 0.05)
        else "Non Normale (Asymétrique)"
    )

    rapport_normalite.append(
        {
            "Variable": var,
            "Shapiro_p": p_sw,
            "DAgostino_p": p_dag,
            "Forme": forme_dist,
        }
    )

    # Resume de Statistiques descriptives 
    stats_continues.append(
        {
            "Variable": var,
            "Effectif": len(valeurs),
            "Moyenne": val_moyenne,
            "EcartType": valeurs.std(),
            "Min": valeurs.min(),
            "Q1": valeurs.quantile(0.25),
            "Mediane": val_mediane,
            "Q3": valeurs.quantile(0.75),
            "Max": valeurs.max(),
        }
    )

    seuil_proximite = 1.0
    if abs(val_moyenne - val_mediane) < seuil_proximite:
        titre_axe_x = f"{var}\n(Moyenne ≈ Médiane ≈ {val_mediane:.1f})"
    else:
        titre_axe_x = (
            f"{var}\n(Médiane: {val_mediane:.1f} | Moyenne: {val_moyenne:.1f})"
        )

    # Tracé des graphiques des histogrammes de distributions continues
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        valeurs,
        kde=True,
        stat="density",
        color="#1f77b4",
        alpha=0.5,
        edgecolor="white",
        ax=ax,
    )

    # Ajout des lignes verticales physiques de repère de la moyenne et mediane
    ax.axvline(
        val_mediane,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Médiane: {val_mediane:.1f}",
    )
    ax.axvline(
        val_moyenne,
        color="#2ca02c",
        linestyle="-.",
        linewidth=1.8,
        label=f"Moyenne: {val_moyenne:.1f}",
    )

    # Ajout du titre des graphiques 
    ax.set_title(
        f"Distribution univariée de la variable : {var}",
        fontweight="bold",
        pad=15,
        loc="center",
    )
    fig.text(
        0.5,
        0.91,
        f"Forme de distribution : {forme_dist} | Lignes repères indicateurs tendance centrale",
        ha="center",
        color="gray",
        fontsize=10,
    )

    ax.set_xlabel(titre_axe_x, fontweight="bold", labelpad=8)
    ax.set_ylabel("Densité des observations d'enfants")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(
        f"Graphiques_Visualisation/Distribution_Univ_Continue_{var}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# Sauvegarde des resultats d'analyses statistiques
pd.DataFrame(rapport_normalite).to_csv(
    "Rapports_Statistiques/Tableau_Tests_Normalite.csv", index=False
)
pd.DataFrame(stats_continues).to_csv(
    "Rapports_Statistiques/Tableau_Descriptive_Continue.csv",
    index=False,
)

# SECTION Variables Categorielles: Distributions de frequences

with open(
    "Rapports_Statistiques/Synthese_Descriptive_Categorielles.txt", "w"
) as f_out:

    for var in vars_categorielles:
        # Exclusion des cellules vides ou valeur textuelles "NA"
        colonne_nettoyee = df[var].dropna()
        colonne_nettoyee = colonne_nettoyee[
            ~colonne_nettoyee.astype(str)
            .str.strip()
            .isin(["NA", "na", "", "nan", "NaN"])
        ]

        # Calcul des fréquences et des pourcentages associés a chaque categorie
        counts = colonne_nettoyee.value_counts()
        proportions = colonne_nettoyee.value_counts(normalize=True) * 100

        # DataFrame de structure ordonné par le poids des pourcentages
        df_freq = pd.DataFrame(
            {"Effectif": counts, "Pourcentage": proportions}
        ).reset_index()
        df_freq.columns = ["Code", "Effectif", "Pourcentage"]

        # Application du dictionnaire pour récupérer les libellés des categories
        df_freq["Libelle"] = df_freq["Code"].map(libelles_mapping[var])
        df_freq = df_freq.dropna(subset=["Libelle"])

        f_out.write(f"Repartition Fréquentielle (ordre decroissant) : {var} \n")
        f_out.write(
            df_freq[["Code", "Libelle", "Effectif", "Pourcentage"]].to_string(
                index=False
            )
        )
        f_out.write("\n\n")

        # Graphe univarié de diagramme en bandes de la distribution
        # FIX : On vérifie d'abord si df_freq contient des données
        if df_freq.empty or len(df_freq["Pourcentage"]) == 0:
            print(f"Impossible de générer le graphique pour {var} : les données sont vides.")
            continue  # Saute à la variable suivante dans votre boucle for

        fig, ax = plt.subplots(figsize=(8, 5.2))
        bars = ax.bar(
            df_freq["Libelle"],
            df_freq["Pourcentage"],
            color="#1f77b4",
            alpha=0.8,
            width=0.45,
        )

        # Ajout des étiquettes de pourcentage sur le sommet de chaque barre
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.8,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9.5,
            )

        # Titre des graphiques
        ax.set_title(
            f"Proportion de l'échantillon selon : {var}",
            fontweight="bold",
            pad=15,
            loc="center",
        )
        ax.set_xlabel(
            "Catégories", fontweight="bold"
        )
        ax.set_ylabel("Proportion (%)")
        
        # Calcul du maximum avec une valeur de secours (fallback)
        max_pourcentage = max(df_freq["Pourcentage"]) if not df_freq["Pourcentage"].dropna().empty else 100
        ax.set_ylim(0, max_pourcentage + 7)
        
        plt.xticks(rotation=20, ha="right")

        # Ajustement des marges et sauvegarde en haute définition (300 DPI)
        plt.tight_layout()
        
        # Optionnel : Assurez-vous que le dossier de destination existe
        import os
        os.makedirs("Graphiques_Visualisation", exist_ok=True)
        
        plt.savefig(
            f"Graphiques_Visualisation/Univ_Categorielle_{var}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        

# 1. Configuration des répertoires de destination
output_graphics_dir = "Graphiques_Visualisation"
output_reports_dir = "Rapports_Statistiques"

os.makedirs(output_graphics_dir, exist_ok=True)
os.makedirs(output_reports_dir, exist_ok=True)

# Configuration graphique minimaliste
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.color"] = "#1c85ed"
plt.rcParams["axes.labelcolor"] = "#2c3e50"
plt.rcParams["xtick.color"] = "#1169c0"
plt.rcParams["ytick.color"] = "#4097ed"

# Initialisation d'une liste pour stocker la matrice récapitulative CSV
liste_stats_globales = []

# Ouverture du fichier texte global pour consigner les rapports textuels
with open(
    f"{output_reports_dir}/Rapport_Descriptif_Discretes.txt", "w", encoding="utf-8"
) as f_txt:
    # 4. Traitement séquentiel de chaque variable du vecteur
    for var in vars_discretes:
        if var not in df.columns:
            msg_missing = f"Variable '{var}' introuvable dans le fichier Excel. Passage.\n"
            print(msg_missing.strip())
            f_txt.write(msg_missing)
            continue

        s_var = df[var].dropna()
        if s_var.empty:
            msg_empty = f"La variable '{var}' ne contient aucune donnée valide.\n"
            print(msg_empty.strip())
            f_txt.write(msg_empty)
            continue

        print(f"Traitement de la variable : {var}")

        # Calcul des statistiques descriptives
        moyenne = s_var.mean()
        mediane = s_var.median()
        mode_series = s_var.mode()
        mode_val = mode_series.iloc[0] if not mode_series.empty else np.nan

        variance = s_var.var()
        ecart_type = s_var.std()
        minimum = s_var.min()
        maximum = s_var.max()
        q25 = s_var.quantile(0.25)
        q75 = s_var.quantile(0.75)
        iqr = q75 - q25

        skewness = s_var.skew()
        kurtosis = s_var.kurtosis()

        # Choix du test de normalité selon la taille de l'échantillon
        if len(s_var) <= 5000:
            stat_test, p_val = stats.shapiro(s_var)
            nom_test = "Shapiro-Wilk"
        else:
            stat_test, p_val = stats.normaltest(s_var)
            nom_test = "D'Agostino-Pearson"

        conclusion_normalite = (
            "Non normale" if p_val < 0.05 else "Normale"
        )

        # Écriture des résultats dans le fichier texte (.txt)
        f_txt.write(f"ANALYSE DE LA VARIABLE : {var.upper()}\n")
        f_txt.write("-" * 40 + "\n")
        f_txt.write(
            f"Tendance centrale : Moyenne = {moyenne:.2f} | Médiane = {mediane:.2f} | Mode = {mode_val}\n"
        )
        f_txt.write(
            f"Dispersion : Écart-type = {ecart_type:.2f} | Variance = {variance:.2f} | IQR = {iqr:.2f} [Min: {minimum}, Max: {maximum}]\n"
        )
        f_txt.write(
            f"Forme de la distribution : Skewness = {skewness:.2f} | Kurtosis = {kurtosis:.2f}\n"
        )
        f_txt.write(
            f"Test de normalité ({nom_test}) : Statistique = {stat_test:.4f}, p-value = {p_val:.4e}\n"
        )
        f_txt.write(f"Conclusion : La distribution est {conclusion_normalite}.\n\n")

        # Remplissage du dictionnaire pour le fichier récapitulatif (.csv)
        dict_var = {
            "Variable": var,
            "Effectif_Valide": len(s_var),
            "Moyenne": round(moyenne, 3),
            "Mediane": round(mediane, 3),
            "Mode": round(mode_val, 3) if not pd.isna(mode_val) else np.nan,
            "Ecart_Type": round(ecart_type, 3),
            "Variance": round(variance, 3),
            "Min": minimum,
            "Max": maximum,
            "Q25": round(q25, 3),
            "Q75": round(q75, 3),
            "IQR": round(iqr, 3),
            "Skewness": round(skewness, 3),
            "Kurtosis": round(kurtosis, 3),
            "Nom_Test_Normalite": nom_test,
            "Stat_Test_Normalite": round(stat_test, 4),
            "p_value_Normalite": p_val,
            "Distribution": conclusion_normalite,
        }
        liste_stats_globales.append(dict_var)

        # 5. Boxplot Épuré (Storytelling et Clarté)
        fig, ax = plt.subplots(figsize=(8, 3.5))

        sns.boxplot(
            x=s_var,
            ax=ax,
            color="#ecf0f1",
            width=0.4,
            linewidth=1.5,
            fliersize=3.5,
            flierprops={
                "markerfacecolor": "#e74c3c",
                "markeredgecolor": "none",
                "alpha": 0.6,
            },
        )

        # Ajout discret du repère de la moyenne
        ax.plot(
            moyenne,
            0,
            marker="D",
            markersize=5,
            color="#2c3e50",
            label=f"Moyenne ({moyenne:.1f})",
        )

        # Épuration complète des contours et axes superflus
        ax.set_yticks([])
        sns.despine(left=True, ax=ax)

        # Titre contextuel
        ax.set_title(
            f"Dispersion et indicateurs de centralité : {var}",
            fontweight="bold",
            fontsize=12,
            pad=15,
            loc="left",
        )
        ax.set_xlabel(
            f"Unités / Valeurs ({var})", fontweight="normal", fontsize=10
        )

        # Grille de repère verticale fine
        ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#95a5a6")
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Sauvegarde de l'image
        fig_name = f"{output_graphics_dir}/Boxplot_Discret_{var}.png"
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.close()

# 6. Exportation finale du tableau récapitulatif global au format CSV
df_stats_final = pd.DataFrame(liste_stats_globales)
csv_name = f"{output_reports_dir}/Synthese_Statistiques_Discretes.csv"
df_stats_final.to_csv(csv_name, index=False, encoding="utf-8")
