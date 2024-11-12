import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import imageio
import seaborn as sns
import matplotlib.patheffects as path_effects

# Appliquer un style moderne de Seaborn
sns.set(style="whitegrid")

# Fonction pour créer et enregistrer le GIF animé
def create_modern_gif(metiers, postes_supplementaires, croissance):
    # Vérifier que les listes ont la même longueur
    if not (len(metiers) == len(postes_supplementaires) == len(croissance)):
        st.error("Les listes doivent avoir la même longueur.")
        return None

    # Inverser les listes pour que le premier élément soit en bas du graphique
    metiers_reverses = metiers[::-1]
    postes_reverses = postes_supplementaires[::-1]
    croissance_reverses = croissance[::-1]

    # Choisir une palette de couleurs moderne
    palette = sns.color_palette("viridis", len(metiers))

    # Création de la figure avec une taille suffisante
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, max(postes_supplementaires) * 1.3)
    ax.set_xlabel("Nombre de postes supplémentaires (en milliers)", fontsize=14, fontweight='bold')
    ax.set_title("Graphique animé des données fournies", fontsize=18, fontweight='bold')

    images = []

    # Nombre de frames pour l'animation
    for i in range(0, 101, 2):
        ax.clear()
        ax.barh(
            metiers_reverses,
            [val * (i / 100) for val in postes_reverses],
            color=palette,
            edgecolor='white',
            alpha=0.8
        )
        ax.set_xlim(0, max(postes_supplementaires) * 1.3)
        ax.set_xlabel("Nombre de postes supplémentaires (en milliers)", fontsize=14, fontweight='bold')
        ax.set_title("Graphique animé des données fournies", fontsize=18, fontweight='bold')

        # Ajuster les marges pour éviter que les labels ne soient tronqués
        plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1)

        # Calculer le pourcentage actuel pour chaque métier
        pourcentages_actuels = [perc * (i / 100) for perc in croissance_reverses]

        # Ajouter les labels de croissance
        for index, (val, perc_actuel) in enumerate(zip([val * (i / 100) for val in postes_reverses], pourcentages_actuels)):
            perc_display = f"{int(perc_actuel)}%"

            text = ax.text(
                val + max(postes_supplementaires)*0.01,
                index,
                perc_display,
                va='center',
                fontsize=12,
                fontweight='bold',
                color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5)
            )
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])

        # Sauvegarder l'image dans un buffer en mémoire
        buf = BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', transparent=False)
        buf.seek(0)
        try:
            image = Image.open(buf).convert('RGB')
            images.append(image)
        except Exception as e:
            st.error(f"Erreur lors de l'ouverture de l'image : {e}")
        buf.close()

    plt.close(fig)

    # Convertir les images PIL en tableaux numpy
    try:
        frames = [np.array(img) for img in images]
    except Exception as e:
        st.error(f"Erreur lors de la conversion des images en tableaux numpy : {e}")
        return None

    # Créer le GIF avec imageio dans un buffer en mémoire
    buf_gif = BytesIO()
    try:
        durations = [0.02] * len(images)
        imageio.mimsave(buf_gif, frames, format='GIF', duration=durations, loop=1)
        st.success("GIF créé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de la création du GIF : {e}")
        return None
    buf_gif.seek(0)
    return buf_gif

# Interface Streamlit
st.set_page_config(page_title="Animation Graphique Personnalisée", layout="wide")
st.title("Animation Graphique Personnalisée")
st.markdown("""
Ce GIF animé montre la progression des données que vous avez fournies.
* **Axe des abscisses** : valeurs numériques (par exemple, nombre de postes supplémentaires).
* **Pourcentage de croissance** : affiché à la fin de chaque barre.

Veuillez télécharger un fichier Excel contenant trois colonnes de données numériques ou textuelles.
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

        # Vérifier qu'il y a au moins trois colonnes
        if df.shape[1] < 3:
            st.error("Le fichier doit contenir au moins trois colonnes.")
        else:
            # Obtenir la liste des colonnes
            columns = df.columns.tolist()

            # Permettre à l'utilisateur de sélectionner les colonnes
            st.subheader("Sélectionnez les colonnes correspondantes")
            metier_col = st.selectbox("Sélectionnez la colonne pour les libellés (par exemple, métiers)", columns)
            postes_col = st.selectbox("Sélectionnez la colonne pour les valeurs numériques (par exemple, postes supplémentaires)", columns)
            croissance_col = st.selectbox("Sélectionnez la colonne pour les pourcentages de croissance", columns)

            # Vérifier que les colonnes sélectionnées sont différentes
            if len({metier_col, postes_col, croissance_col}) < 3:
                st.error("Veuillez sélectionner des colonnes différentes pour chaque champ.")
            else:
                # Extraire les données
                metiers = df[metier_col].astype(str).tolist()
                postes_supplementaires = df[postes_col].astype(float).tolist()
                croissance = df[croissance_col].astype(float).tolist()

                # Générer et afficher le GIF
                gif_buffer = create_modern_gif(metiers, postes_supplementaires, croissance)
                if gif_buffer:
                    st.image(gif_buffer, caption="Graphique animé", use_column_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Excel: {e}")
else:
    st.info("Veuillez télécharger un fichier Excel pour générer le graphique animé.")
