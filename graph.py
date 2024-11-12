import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import imageio
import seaborn as sns

# Appliquer un style moderne avec Seaborn
sns.set_theme(style='whitegrid')
sns.set_palette('Spectral')

# Fonction pour créer et enregistrer les GIF animés
def create_animated_charts(labels, data_values, chart_type_selection=None, frame_duration=0.15):
    charts = {}
    chart_types = ["Barres horizontales", "Barres verticales", "Lignes", "Zones empilées"]
    selected_chart_types = chart_type_selection
    for chart_type in selected_chart_types:
        gif_buffer = create_animated_chart(labels, data_values, chart_type, frame_duration)
        if gif_buffer:
            charts[chart_type] = gif_buffer
    return charts

# Fonction pour créer un GIF animé pour un type de graphique spécifique
def create_animated_chart(labels, data_values, chart_type="Barres horizontales", frame_duration=0.15):
    num_series = len(data_values)
    num_points = len(labels)

    # Vérifier que toutes les séries ont la même longueur
    for series in data_values:
        if len(series) != num_points:
            st.error("Toutes les séries de données doivent avoir le même nombre de points.")
            return None

    # Palette de couleurs pour les différentes séries
    palette = sns.color_palette("Spectral", num_series)

    images = []

    # Création de la figure et des axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Appliquer un fond moderne
    fig.patch.set_facecolor('#2E3440')
    ax.set_facecolor('#3B4252')

    # Changer la couleur des axes et du texte
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.title.set_color('white')

    max_value = max([max(series) for series in data_values]) * 1.1 if data_values else 1

    if chart_type == "Barres horizontales":
        ax.set_xlim(0, max_value)
        ax.set_ylim(-0.5, num_points - 0.5)
        ax.set_xlabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        bars = []
        for idx, series in enumerate(data_values):
            bars.append(ax.barh(labels, [0]*num_points, color=palette[idx], edgecolor='white', label=f"Série {idx+1}"))
    elif chart_type == "Barres verticales":
        ax.set_ylim(0, max_value)
        ax.set_xlim(-0.5, num_points - 0.5)
        ax.set_ylabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        plt.xticks(rotation=45, ha='right', color='white')
        bars = []
        for idx, series in enumerate(data_values):
            bars.append(ax.bar(labels, [0]*num_points, color=palette[idx], edgecolor='white', label=f"Série {idx+1}"))
    elif chart_type == "Lignes":
        ax.set_ylim(0, max_value)
        ax.set_xlim(-0.5, num_points - 0.5)
        ax.set_ylabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("Labels", fontsize=12, fontweight='bold', color='white')
        plt.xticks(range(num_points), labels, rotation=45, ha='right', color='white')
        lines = []
        for idx in range(num_series):
            line, = ax.plot([], [], color=palette[idx], marker='o', linewidth=2, label=f"Série {idx+1}")
            lines.append(line)
    elif chart_type == "Zones empilées":
        ax.set_ylim(0, max_value)
        ax.set_xlim(-0.5, num_points - 0.5)
        ax.set_ylabel("Valeurs cumulées", fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("Labels", fontsize=12, fontweight='bold', color='white')
        plt.xticks(range(num_points), labels, rotation=45, ha='right', color='white')
    else:
        st.error("Type de graphique non supporté pour cette animation.")
        return None

    ax.set_title(f"Graphique {chart_type}", fontsize=16, fontweight='bold', color='white')
    ax.legend(facecolor='#4C566A', edgecolor='none', labelcolor='white', fontsize=10)
    plt.tight_layout()

    # Animation
    num_frames = 50
    frames = np.linspace(0, 1, num_frames)

    for i in frames:
        if chart_type in ["Barres horizontales", "Barres verticales"]:
            for idx, series in enumerate(data_values):
                current_values = [val * i for val in series]
                if chart_type == "Barres horizontales":
                    for bar, val in zip(bars[idx], current_values):
                        bar.set_width(val)
                elif chart_type == "Barres verticales":
                    for bar, val in zip(bars[idx], current_values):
                        bar.set_height(val)
        elif chart_type == "Lignes":
            for idx, series in enumerate(data_values):
                current_series = [val * i for val in series]
                x_data = np.arange(num_points)
                lines[idx].set_data(x_data, current_series)
        elif chart_type == "Zones empilées":
            cumulative_data = np.zeros(num_points)
            for idx, series in enumerate(data_values):
                current_series = [val * i for val in series]
                ax.fill_between(range(num_points), cumulative_data, cumulative_data + current_series, color=palette[idx], alpha=0.7, label=f"Série {idx+1}")
                cumulative_data += current_series
            ax.legend(facecolor='#4C566A', edgecolor='none', labelcolor='white', fontsize=10)
        else:
            continue

        # Enregistrer l'image dans un buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        image = Image.open(buf).convert('RGBA')
        images.append(image)
        buf.close()

        if chart_type == "Zones empilées":
            ax.collections.clear()

    # Ajouter une pause à la fin de l'animation
    pause_duration = 2
    durations = [frame_duration] * len(images)
    durations[-1] += pause_duration

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
    * Zones empilées

Vous pouvez choisir de générer un ou plusieurs types de graphiques simultanément.

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
            label_col = st.selectbox("Sélectionnez la colonne pour les libellés (axe X)", columns)
            value_cols = st.multiselect("Sélectionnez une ou plusieurs colonnes pour les valeurs numériques (axe Y)", columns)

            if not value_cols:
                st.error("Veuillez sélectionner au moins une colonne pour les valeurs numériques.")
            else:
                # Sélection du type de graphique
                st.subheader("Sélectionnez le(s) type(s) de graphique")
                chart_type_options = ["Barres horizontales", "Barres verticales", "Lignes", "Zones empilées"]
                chart_type_selection = st.multiselect("Sélectionnez le(s) type(s) de graphique", chart_type_options, default=chart_type_options)

                # Ajuster la durée de l'animation
                st.subheader("Ajustez la vitesse de l'animation")
                frame_duration = st.slider("Durée de chaque frame (en secondes)", min_value=0.05, max_value=1.0, value=0.1, step=0.05)

                # Bouton pour générer les graphiques
                if st.button("Générer les graphiques"):
                    # Extraire les données
                    labels = df[label_col].astype(str).tolist()
                    data_values = []

                    for col in value_cols:
                        values = df[col]
                        # Convertir en float et gérer les erreurs
                        values = pd.to_numeric(values, errors='coerce')
                        data_values.append(values.tolist())

                    # Nettoyer les données en supprimant les lignes avec des valeurs manquantes
                    data = pd.DataFrame({label_col: labels})
                    for idx, col in enumerate(value_cols):
                        data[col] = data_values[idx]

                    data = data.dropna()

                    # Mettre à jour les listes après nettoyage
                    labels = data[label_col].tolist()
                    data_values = []
                    for col in value_cols:
                        data_values.append(data[col].tolist())

                    # Vérifier que les listes ne sont pas vides
                    if not labels or not data_values:
                        st.error("Aucune donnée valide trouvée après le nettoyage. Veuillez vérifier votre fichier.")
                    else:
                        # Générer les GIFs pour les types de graphiques sélectionnés
                        charts = create_animated_charts(labels, data_values, chart_type_selection, frame_duration)

                        # Afficher les graphiques
                        st.subheader("Graphiques animés")
                        if charts:
                            cols_per_row = 2  # Nombre de colonnes par ligne
                            rows = [chart_type_selection[i:i + cols_per_row] for i in range(0, len(chart_type_selection), cols_per_row)]
                            for row in rows:
                                cols = st.columns(len(row))
                                for col, chart_type in zip(cols, row):
                                    with col:
                                        st.image(charts[chart_type], caption=f"Graphique {chart_type}", use_column_width=True)
                        else:
                            st.error("Aucun graphique n'a pu être généré avec les données fournies.")
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
else:
    st.info("Veuillez télécharger un fichier Excel ou CSV pour générer les graphiques animés.")
