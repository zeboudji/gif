import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import imageio
import seaborn as sns
import matplotlib.patheffects as path_effects
import plotly.express as px  # Pour les graphiques interactifs

# Appliquer un style moderne
sns.set(style="whitegrid")

# Fonction pour créer et enregistrer le GIF animé
def create_modern_gif(metiers, postes_supplementaires, croissance, graph_type):
    if not (len(metiers) == len(postes_supplementaires) == len(croissance)):
        st.error("Les listes doivent avoir la même longueur.")
        return None

    # Inverser les listes pour que le premier élément soit en bas du graphique
    metiers_reverses = metiers[::-1]
    postes_reverses = postes_supplementaires[::-1]
    croissance_reverses = croissance[::-1]

    # Choisir une palette de couleurs moderne
    palette = sns.color_palette("viridis", len(metiers))

    images = []

    # Nombre de frames pour l'animation
    for i in range(0, 101, 5):
        plt.clf()
        fig, ax = plt.subplots(figsize=(14, 10))

        if graph_type == "Barres horizontales":
            ax.barh(
                metiers_reverses,
                [val * (i / 100) for val in postes_reverses],
                color=palette,
                edgecolor='white',
                alpha=0.8
            )
            ax.set_xlabel("Valeurs", fontsize=14, fontweight='bold')
        elif graph_type == "Barres verticales":
            ax.bar(
                metiers_reverses,
                [val * (i / 100) for val in postes_reverses],
                color=palette,
                edgecolor='white',
                alpha=0.8
            )
            ax.set_ylabel("Valeurs", fontsize=14, fontweight='bold')
            plt.xticks(rotation=90)
        elif graph_type == "Lignes":
            ax.plot(
                metiers_reverses,
                [val * (i / 100) for val in postes_reverses],
                color='blue',
                marker='o'
            )
            ax.set_ylabel("Valeurs", fontsize=14, fontweight='bold')
            plt.xticks(rotation=90)

        ax.set_title("Graphique animé des données fournies", fontsize=18, fontweight='bold')

        # Ajuster les marges
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)

        # Ajouter les labels de croissance
        for index, (val, perc_actuel) in enumerate(zip(
            [val * (i / 100) for val in postes_reverses],
            [perc * (i / 100) for perc in croissance_reverses]
        )):
            perc_display = f"{int(perc_actuel)}%"
            if graph_type == "Barres horizontales":
                ax.text(
                    val + max(postes_supplementaires)*0.01,
                    index,
                    perc_display,
                    va='center',
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5)
                )
            elif graph_type == "Barres verticales":
                ax.text(
                    index,
                    val + max(postes_supplementaires)*0.01,
                    perc_display,
                    ha='center',
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5)
                )
            elif graph_type == "Lignes":
                ax.text(
                    index,
                    val + max(postes_supplementaires)*0.01,
                    perc_display,
                    ha='center',
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5)
                )

        # Sauvegarder l'image dans un buffer en mémoire
        buf = BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', transparent=False)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        images.append(image)
        buf.close()
        plt.close(fig)

    # Convertir les images PIL en tableaux numpy
    frames = [np.array(img) for img in images]

    # Créer le GIF avec imageio dans un buffer en mémoire
    buf_gif = BytesIO()
    durations = [0.05] * len(images)
    imageio.mimsave(buf_gif, frames, format='GIF', duration=durations, loop=1)
    buf_gif.seek(0)
    return buf_gif

# Interface Streamlit
st.set_page_config(page_title="Animation Graphique Personnalisée", layout="wide")
st.title("Animation Graphique Personnalisée")
st.markdown("""
Ce GIF animé montre la progression des données que vous avez fournies.
* **Axe des abscisses** : valeurs numériques.
* **Pourcentage de croissance** : affiché à la fin de chaque barre.

Veuillez télécharger un fichier Excel contenant vos données.
""")

# Uploader de fichier
uploaded_file = st.file_uploader("Veuillez télécharger un fichier Excel avec vos données.", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Lire le fichier Excel
    try:
        df = pd.read_excel(uploaded_file)

        # Afficher un aperçu des données
        st.subheader("Aperçu des données téléchargées")
        st.dataframe(df.head())

        # Vérifier qu'il y a au moins deux colonnes
        if df.shape[1] < 2:
            st.error("Le fichier doit contenir au moins deux colonnes.")
        else:
            # Obtenir la liste des colonnes
            columns = df.columns.tolist()

            # Permettre à l'utilisateur de sélectionner les colonnes
            st.subheader("Sélectionnez les colonnes correspondantes")
            metier_col = st.selectbox("Sélectionnez la colonne pour les libellés", columns)
            postes_col = st.selectbox("Sélectionnez la colonne pour les valeurs numériques", columns)

            # Optionnelle : sélection de la colonne pour la croissance
            croissance_option = st.checkbox("Ajouter une colonne pour les pourcentages de croissance")
            if croissance_option:
                croissance_col = st.selectbox("Sélectionnez la colonne pour les pourcentages de croissance", columns)
            else:
                croissance_col = None

            # Sélection du type de graphique
            st.subheader("Sélectionnez le type de graphique")
            graph_type = st.selectbox("Type de graphique", ["Barres horizontales", "Barres verticales", "Lignes"])

            # Bouton pour générer le graphique
            if st.button("Générer le graphique"):
                # Extraire les données
                metiers = df[metier_col].astype(str).tolist()
                postes_supplementaires = df[postes_col].astype(float).tolist()

                # Si croissance_col est sélectionné
                if croissance_col:
                    croissance = df[croissance_col].astype(float).tolist()
                else:
                    croissance = [0] * len(metiers)  # Par défaut à 0

                # Générer et afficher le GIF
                gif_buffer = create_modern_gif(metiers, postes_supplementaires, croissance, graph_type)
                if gif_buffer:
                    st.image(gif_buffer, caption="Graphique animé", use_column_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Excel: {e}")
else:
    st.info("Veuillez télécharger un fichier Excel pour générer le graphique animé.")
