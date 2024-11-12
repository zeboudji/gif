import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import imageio
import seaborn as sns

# Appliquer un style moderne avec Seaborn
sns.set_style('whitegrid')  # Style moderne
sns.set_palette('viridis')  # Palette de couleurs modernes

# Fonction pour créer et enregistrer le GIF animé
def create_animated_chart(labels, values, growth=None, chart_type="Barres horizontales"):
    # Vérifier que les listes ont la même longueur
    if not (len(labels) == len(values)):
        st.error("Les listes des labels et des valeurs doivent avoir la même longueur.")
        return None

    if growth is not None and len(growth) != len(labels):
        st.error("La liste de croissance doit avoir la même longueur que les labels.")
        return None

    # Inverser les listes pour certains types de graphiques
    if chart_type == "Barres horizontales":
        labels = labels[::-1]
        values = values[::-1]
        if growth is not None:
            growth = growth[::-1]

    # Choisir une palette de couleurs moderne
    palette = sns.color_palette("viridis", len(labels))

    images = []

    # Création de la figure et des axes en dehors de la boucle
    fig, ax = plt.subplots(figsize=(14, 10))

    # Fixer les limites des axes pour éviter les sauts
    if chart_type == "Barres horizontales":
        ax.set_xlim(0, max(values) * 1.1 if values else 1)
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.set_xlabel("Valeurs", fontsize=14, fontweight='bold')
        bars = ax.barh(labels, [0]*len(values), color=palette, edgecolor='white', alpha=0.8)
    elif chart_type == "Barres verticales":
        ax.set_ylim(0, max(values) * 1.1 if values else 1)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("Valeurs", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        bars = ax.bar(labels, [0]*len(values), color=palette, edgecolor='white', alpha=0.8)
    elif chart_type == "Lignes":
        ax.set_ylim(0, max(values) * 1.1 if values else 1)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("Valeurs", fontsize=14, fontweight='bold')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        line, = ax.plot([], [], color='blue', marker='o', linewidth=2)
    else:
        st.error("Type de graphique non supporté pour cette animation.")
        return None

    ax.set_title("Graphique animé des données fournies", fontsize=18, fontweight='bold')

    # Ajuster les marges
    plt.tight_layout()

    # Préparer les textes pour les labels de croissance
    if growth is not None and chart_type != "Lignes":
        texts = []
        for _ in labels:
            texts.append(ax.text(0, 0, '', fontsize=12, fontweight='bold',
                                 color='black',
                                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5)))

    images = []

    # Nombre de frames pour l'animation
    frames = np.linspace(0, 1, 50)  # Plus de frames pour une animation fluide
    for i in frames:
        current_values = [val * i for val in values]

        if chart_type == "Barres horizontales":
            # Mettre à jour les largeurs des barres
            for bar, val in zip(bars, current_values):
                bar.set_width(val)
            # Mettre à jour les positions des labels de croissance
            if growth is not None:
                for idx, (text, bar, perc) in enumerate(zip(texts, bars, growth)):
                    perc_display = f"{int(perc * i)}%"
                    text.set_position((bar.get_width() + (max(values)*0.01 if values else 0), bar.get_y() + bar.get_height()/2))
                    text.set_text(perc_display)
        elif chart_type == "Barres verticales":
            # Mettre à jour les hauteurs des barres
            for bar, val in zip(bars, current_values):
                bar.set_height(val)
            # Mettre à jour les positions des labels de croissance
            if growth is not None:
                for idx, (text, bar, perc) in enumerate(zip(texts, bars, growth)):
                    perc_display = f"{int(perc * i)}%"
                    text.set_position((bar.get_x() + bar.get_width()/2, bar.get_height() + (max(values)*0.01 if values else 0)))
                    text.set_text(perc_display)
        elif chart_type == "Lignes":
            # Mettre à jour les données de la ligne
            num_points = int(len(values) * i)
            if num_points == 0:
                x_data = []
                y_data = []
            else:
                x_data = range(num_points)
                y_data = values[:num_points]
            line.set_data(x_data, y_data)
            # Pas de labels de croissance pour la ligne pour éviter la surcharge visuelle

        # Enregistrer l'image dans un buffer
        buf = BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', transparent=False)
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        images.append(image)
        buf.close()

    # Ajouter une pause à la fin de l'animation
    pause_duration = 2  # Durée de la pause en secondes
    durations = [0.1] * len(images)
    durations[-1] += pause_duration  # Augmenter la durée de la dernière frame

    plt.close(fig)

    # Convertir les images en frames pour le GIF
    frames = [np.array(img) for img in images]

    # Créer le GIF
    buf_gif = BytesIO()
    imageio.mimsave(buf_gif, frames, format='GIF', duration=durations, loop=0)
    buf_gif.seek(0)
    return buf_gif

# Interface Streamlit
st.set_page_config(page_title="Animation Graphique Personnalisée", layout="wide")
st.title("Animation Graphique Personnalisée")
st.markdown("""
Ce GIF animé montre la progression des données que vous avez fournies.

* **Types de graphiques disponibles** :
    * Barres horizontales
    * Barres verticales
    * Lignes

Veuillez télécharger un fichier Excel ou CSV contenant vos données.
""")

# Uploader de fichier
uploaded_file = st.file_uploader("Veuillez télécharger un fichier Excel ou CSV avec vos données.", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # Lire le fichier Excel ou CSV
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
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
            label_col = st.selectbox("Sélectionnez la colonne pour les libellés", columns)
            value_col = st.selectbox("Sélectionnez la colonne pour les valeurs numériques", columns)

            # Optionnelle : sélection de la colonne pour la croissance
            growth_option = st.checkbox("Ajouter une colonne pour les pourcentages de croissance")
            if growth_option:
                growth_col = st.selectbox("Sélectionnez la colonne pour les pourcentages de croissance", columns)
            else:
                growth_col = None

            # Sélection du type de graphique
            st.subheader("Sélectionnez le type de graphique")
            chart_type = st.selectbox("Type de graphique", ["Barres horizontales", "Barres verticales", "Lignes"])

            # Bouton pour générer le graphique
            if st.button("Générer le graphique"):
                # Extraire les données
                labels = df[label_col].astype(str)
                values = df[value_col]

                # Nettoyer les données en supprimant les lignes avec des valeurs manquantes ou non numériques
                data = pd.DataFrame({label_col: labels, value_col: values})

                # Si growth_col est sélectionné
                if growth_col:
                    growth = df[growth_col]
                    data[growth_col] = growth
                else:
                    growth = None

                # Convertir les colonnes numériques en float, coercer les erreurs et supprimer les NaN
                data[value_col] = pd.to_numeric(data[value_col], errors='coerce')
                if growth_col:
                    data[growth_col] = pd.to_numeric(data[growth_col], errors='coerce')

                # Supprimer les lignes avec des valeurs manquantes
                data = data.dropna()

                # Mettre à jour les listes après nettoyage
                labels = data[label_col].tolist()
                values = data[value_col].tolist()
                if growth_col:
                    growth = data[growth_col].tolist()
                else:
                    growth = None

                # Vérifier que les listes ne sont pas vides
                if not labels or not values:
                    st.error("Aucune donnée valide trouvée après le nettoyage. Veuillez vérifier votre fichier.")
                else:
                    # Générer et afficher le GIF
                    gif_buffer = create_animated_chart(labels, values, growth, chart_type)
                    if gif_buffer:
                        st.image(gif_buffer, caption="Graphique animé", use_column_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
else:
    st.info("Veuillez télécharger un fichier Excel ou CSV pour générer le graphique animé.")
