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
def create_animated_charts(labels, values, growth=None, chart_type_selection=None, frame_duration=0.15):
    charts = {}
    chart_types = ["Barres horizontales", "Barres verticales", "Lignes", "Zones empilées"]
    selected_chart_types = chart_type_selection
    for chart_type in selected_chart_types:
        gif_buffer = create_animated_chart(labels, values, growth, chart_type, frame_duration)
        if gif_buffer:
            charts[chart_type] = gif_buffer
    return charts

# Fonction pour créer un GIF animé pour un type de graphique spécifique
def create_animated_chart(labels, values, growth=None, chart_type="Barres horizontales", frame_duration=0.15):
    # Vérifier que les listes ont la même longueur
    if not (len(labels) == len(values)):
        st.error("Les listes des labels et des valeurs doivent avoir la même longueur.")
        return None

    if growth is not None and len(growth) != len(labels):
        st.error("La liste de croissance doit avoir la même longueur que les labels.")
        return None

    # Inverser les listes pour les barres horizontales
    if chart_type == "Barres horizontales":
        labels = labels[::-1]
        values = values[::-1]
        if growth is not None:
            growth = growth[::-1]

    # Si plusieurs séries de données sont fournies
    multiple_series = isinstance(values[0], list)

    # Choisir une palette de couleurs moderne
    palette = sns.color_palette("Spectral", len(labels) if not multiple_series else len(values))

    images = []

    # Création de la figure et des axes en dehors de la boucle
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

    # Fixer les limites des axes pour éviter les sauts
    if multiple_series:
        max_value = max([max(v) for v in values]) * 1.1 if values else 1
    else:
        max_value = max([v + g if growth else v for v, g in zip(values, growth or [0]*len(values))]) * 1.1 if values else 1

    if chart_type == "Barres horizontales":
        ax.set_xlim(0, max_value)
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.set_xlabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        if multiple_series:
            bars = []
            for idx, series in enumerate(values):
                bar = ax.barh(labels, [0]*len(series), color=palette[idx], edgecolor='white', label=f"Série {idx+1}")
                bars.append(bar)
        else:
            if growth is not None:
                bars_values = ax.barh(labels, [0]*len(values), color=palette, edgecolor='white', label='Valeurs')
                bars_growth = ax.barh(labels, [0]*len(values), left=[0]*len(values), color='lightblue', edgecolor='white', label='Croissance')
            else:
                bars_values = ax.barh(labels, [0]*len(values), color=palette, edgecolor='white')
    elif chart_type == "Barres verticales":
        ax.set_ylim(0, max_value)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        plt.xticks(rotation=45, ha='right', color='white')
        if multiple_series:
            bars = []
            for idx, series in enumerate(values):
                bar = ax.bar(labels, [0]*len(series), color=palette[idx], edgecolor='white', label=f"Série {idx+1}")
                bars.append(bar)
        else:
            if growth is not None:
                bars_values = ax.bar(labels, [0]*len(values), color=palette, edgecolor='white', label='Valeurs')
                bars_growth = ax.bar(labels, [0]*len(values), bottom=[0]*len(values), color='lightblue', edgecolor='white', label='Croissance')
            else:
                bars_values = ax.bar(labels, [0]*len(values), color=palette, edgecolor='white')
    elif chart_type == "Lignes":
        ax.set_ylim(0, max_value)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("Labels", fontsize=12, fontweight='bold', color='white')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', color='white')
        if multiple_series:
            lines = []
            value_texts_list = []
            for idx, series in enumerate(values):
                line, = ax.plot([], [], color=palette[idx], marker='o', linewidth=2, label=f"Série {idx+1}")
                lines.append(line)
                # Préparer les textes pour les valeurs de cette série
                value_texts = [ax.text(0, 0, '', fontsize=10, fontweight='bold', color='white') for _ in range(len(labels))]
                value_texts_list.append(value_texts)
        else:
            line_values, = ax.plot([], [], color='#88C0D0', marker='o', linewidth=3, label='Valeurs')
            if growth is not None:
                line_growth, = ax.plot([], [], color='#D08770', marker='s', linewidth=3, label='Croissance')
            # Préparer les textes pour les valeurs
            value_texts = [ax.text(x, 0, '', fontsize=10, fontweight='bold', color='white') for x in range(len(labels))]
    elif chart_type == "Zones empilées":
        ax.set_ylim(0, max_value)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("Valeurs cumulées", fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("Labels", fontsize=12, fontweight='bold', color='white')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', color='white')
    else:
        st.error("Type de graphique non supporté pour cette animation.")
        return None

    ax.set_title(f"Graphique {chart_type}", fontsize=16, fontweight='bold', color='white')
    if growth is not None or multiple_series:
        ax.legend(facecolor='#4C566A', edgecolor='none', labelcolor='white', fontsize=10)
    plt.tight_layout()

    images = []

    if chart_type == "Lignes":
        x_data = np.arange(len(labels))
        if multiple_series:
            num_series = len(values)
            num_frames = 50
            frames = np.linspace(0, 1, num_frames)
            for i in frames:
                for idx, series in enumerate(values):
                    current_series = [val * i for val in series]
                    lines[idx].set_data(x_data, current_series)
                    # Mettre à jour les textes des valeurs pour cette série
                    value_texts = value_texts_list[idx]
                    for txt in value_texts:
                        txt.set_text('')
                    if current_series:
                        # Afficher la valeur actuelle au dernier point
                        txt = value_texts[-1]
                        txt.set_position((x_data[-1], current_series[-1]))
                        txt.set_text(f"{int(current_series[-1])}")
                        txt.set_fontsize(10)
                        txt.set_fontweight('bold')
                        txt.set_color('white')
                # Enregistrer l'image dans un buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                image = Image.open(buf).convert('RGBA')
                images.append(image)
                buf.close()
        else:
            y_values = np.array(values)
            if growth is not None:
                y_growth = np.array(growth)

            frames_per_segment = 30
            num_segments = len(values) - 1
            num_frames = frames_per_segment * num_segments

            for frame in range(num_frames):
                segment = frame // frames_per_segment
                progress = (frame % frames_per_segment) / frames_per_segment

                current_x = x_data[:segment+1].tolist()
                current_y_values = y_values[:segment+1].tolist()

                if segment < num_segments:
                    next_x = x_data[segment] + progress * (x_data[segment+1] - x_data[segment])
                    next_y_values = y_values[segment] + progress * (y_values[segment+1] - y_values[segment])
                    current_x.append(next_x)
                    current_y_values.append(next_y_values)

                line_values.set_data(current_x, current_y_values)

                if growth is not None:
                    current_y_growth = y_growth[:segment+1].tolist()
                    if segment < num_segments:
                        next_y_growth = y_growth[segment] + progress * (y_growth[segment+1] - y_growth[segment])
                        current_y_growth.append(next_y_growth)
                    line_growth.set_data(current_x, current_y_growth)

                # Mettre à jour les textes des valeurs
                for txt in value_texts:
                    txt.set_text('')
                if current_x:
                    txt = value_texts[segment]
                    txt.set_position((current_x[-1], current_y_values[-1]))
                    txt.set_text(f"{int(current_y_values[-1])}")
                    txt.set_fontsize(10)
                    txt.set_fontweight('bold')
                    txt.set_color('white')

                # Enregistrer l'image dans un buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                image = Image.open(buf).convert('RGBA')
                images.append(image)
                buf.close()
    elif chart_type == "Zones empilées":
        if not multiple_series:
            st.error("Le graphique des zones empilées nécessite plusieurs séries de données.")
            return None
        num_frames = 50
        frames = np.linspace(0, 1, num_frames)
        for i in frames:
            cumulative_data = np.zeros(len(labels))
            fill_collections = []
            for idx, series in enumerate(values):
                current_series = [val * i for val in series]
                collection = ax.fill_between(range(len(labels)), cumulative_data, cumulative_data + current_series, color=palette[idx], alpha=0.7)
                cumulative_data += current_series
                fill_collections.append(collection)
            ax.legend(facecolor='#4C566A', edgecolor='none', labelcolor='white', fontsize=10)
            # Enregistrer l'image dans un buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
            buf.seek(0)
            image = Image.open(buf).convert('RGBA')
            images.append(image)
            buf.close()
            # Supprimer les collections de remplissage
            while fill_collections:
                collection = fill_collections.pop()
                collection.remove()
    else:
        # Pour les graphiques à barres
        num_frames = 50
        frames = np.linspace(0, 1, num_frames)
        if multiple_series:
            for i in frames:
                for idx, series in enumerate(values):
                    current_values = [val * i for val in series]
                    if chart_type == "Barres horizontales":
                        for bar, val in zip(bars[idx], current_values):
                            bar.set_width(val)
                    elif chart_type == "Barres verticales":
                        for bar, val in zip(bars[idx], current_values):
                            bar.set_height(val)
                # Enregistrer l'image dans un buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                image = Image.open(buf).convert('RGBA')
                images.append(image)
                buf.close()
        else:
            value_texts = []
            for _ in labels:
                value_texts.append(ax.text(0, 0, '', fontsize=10, fontweight='bold',
                                           color='white',
                                           bbox=dict(facecolor='#4C566A', alpha=0.6, edgecolor='none', pad=0.5)))
            for i in frames:
                current_values = [val * i for val in values]
                if growth is not None:
                    current_growth = [g * i for g in growth]

                if chart_type == "Barres horizontales":
                    for idx, (bar_value, val) in enumerate(zip(bars_values, current_values)):
                        bar_value.set_width(val)
                    if growth is not None:
                        for idx, (bar_growth, val, gro) in enumerate(zip(bars_growth, current_values, current_growth)):
                            bar_growth.set_width(gro)
                            bar_growth.set_x(val)
                    for idx, (text, bar_value) in enumerate(zip(value_texts, bars_values)):
                        total_width = bar_value.get_width()
                        if growth is not None:
                            total_width += bars_growth[idx].get_width()
                        text.set_position((total_width + max_value*0.01, bar_value.get_y() + bar_value.get_height()/2))
                        text.set_text(f"{int(total_width)}")
                elif chart_type == "Barres verticales":
                    for idx, (bar_value, val) in enumerate(zip(bars_values, current_values)):
                        bar_value.set_height(val)
                    if growth is not None:
                        for idx, (bar_growth, val, gro) in enumerate(zip(bars_growth, current_values, current_growth)):
                            bar_growth.set_height(gro)
                            bar_growth.set_y(val)
                    for idx, (text, bar_value) in enumerate(zip(value_texts, bars_values)):
                        total_height = bar_value.get_height()
                        if growth is not None:
                            total_height += bars_growth[idx].get_height()
                        text.set_position((bar_value.get_x() + bar_value.get_width()/2, total_height + max_value*0.01))
                        text.set_text(f"{int(total_height)}")

                # Enregistrer l'image dans un buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                image = Image.open(buf).convert('RGBA')
                images.append(image)
                buf.close()

    if not images:
        st.error("Aucune image n'a été générée pour le graphique {}.".format(chart_type))
        return None

    # Ajouter une pause à la fin de l'animation
    pause_duration = 2  # Durée de la pause en secondes
    durations = [frame_duration] * len(images)
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
            label_col = st.selectbox("Sélectionnez la colonne pour les libellés", columns)
            value_col = st.selectbox("Sélectionnez la colonne pour les valeurs numériques", columns)

            # Optionnelle : sélection de la colonne pour la troisième dimension
            growth_option = st.checkbox("Ajouter une colonne pour une troisième dimension (ex: croissance)")
            if growth_option:
                growth_col = st.selectbox("Sélectionnez la colonne pour la troisième dimension", columns)
            else:
                growth_col = None

            # Optionnelle : sélection de plusieurs colonnes pour les graphiques en lignes et zones empilées
            multi_series_option = st.checkbox("Sélectionner plusieurs colonnes pour les graphiques en lignes et zones empilées")
            if multi_series_option:
                value_cols = st.multiselect("Sélectionnez les colonnes pour les valeurs numériques", columns)
            else:
                value_cols = [value_col]

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

                if growth_col:
                    growth = df[growth_col]
                    data[growth_col] = growth
                    data[growth_col] = pd.to_numeric(data[growth_col], errors='coerce')
                else:
                    growth = None

                data = data.dropna()

                # Mettre à jour les listes après nettoyage
                labels = data[label_col].tolist()
                if multi_series_option:
                    values = []
                    for col in value_cols:
                        values.append(data[col].tolist())
                else:
                    values = data[value_col].tolist()
                    if growth_col:
                        growth = data[growth_col].tolist()
                    else:
                        growth = None

                # Vérifier que les listes ne sont pas vides
                if not labels or not values:
                    st.error("Aucune donnée valide trouvée après le nettoyage. Veuillez vérifier votre fichier.")
                else:
                    # Générer les GIFs pour les types de graphiques sélectionnés
                    charts = create_animated_charts(labels, values, growth, chart_type_selection, frame_duration)

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
