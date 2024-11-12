import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import imageio
import seaborn as sns
import matplotlib as mpl

# Appliquer un style moderne avec Seaborn
sns.set_theme(style='whitegrid')  # Style moderne
sns.set_palette('Spectral')  # Palette de couleurs modernes

# Fonction pour créer et enregistrer les GIF animés
def create_animated_charts(labels, values, growth=None, chart_type_selection=None, frame_duration=0.15):
    charts = {}
    chart_types = ["Barres horizontales", "Barres verticales", "Lignes", "Camembert"]
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

    # Choisir une palette de couleurs moderne
    palette = sns.color_palette("Spectral", len(labels))

    images = []

    # Création de la figure et des axes en dehors de la boucle
    fig, ax = plt.subplots(figsize=(8, 6))

    # Appliquer un fond moderne
    fig.patch.set_facecolor('#2E3440')  # Couleur de fond de la figure
    ax.set_facecolor('#3B4252')  # Couleur de fond des axes

    # Changer la couleur des axes et du texte
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.title.set_color('white')

    # Fixer les limites des axes pour éviter les sauts
    max_value = max(values) * 1.1 if values else 1

    if chart_type == "Barres horizontales":
        ax.set_xlim(0, max_value)
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.set_xlabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        bars = ax.barh(labels, [0]*len(values), color=palette, edgecolor='white')
    elif chart_type == "Barres verticales":
        ax.set_ylim(0, max_value)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        plt.xticks(rotation=45, ha='right', color='white')
        bars = ax.bar(labels, [0]*len(values), color=palette, edgecolor='white')
    elif chart_type == "Lignes":
        ax.set_ylim(0, max_value)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("Labels", fontsize=12, fontweight='bold', color='white')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right', color='white')
        line, = ax.plot([], [], color='#88C0D0', marker='o', linewidth=3)
        # Préparer les textes pour les valeurs
        value_texts = [ax.text(x, 0, '', fontsize=10, fontweight='bold', color='white') for x in range(len(labels))]
    elif chart_type == "Camembert":
        # Pas d'axes pour un camembert
        ax.axis('equal')
        ax.set_title(f"Graphique {chart_type}", fontsize=16, fontweight='bold', color='white')
        plt.tight_layout()
    else:
        st.error("Type de graphique non supporté pour cette animation.")
        return None

    if chart_type != "Camembert":
        ax.set_title(f"Graphique {chart_type}", fontsize=16, fontweight='bold', color='white')
        # Ajuster les marges
        plt.tight_layout()

    images = []

    if chart_type == "Lignes":
        x_data = np.arange(len(values))
        y_data = np.array(values)

        # Nombre de frames pour l'animation
        frames_per_segment = 30  # Plus de frames pour une animation fluide entre les points
        num_segments = len(values) - 1
        num_frames = frames_per_segment * num_segments

        for frame in range(num_frames):
            # Déterminer le segment actuel
            segment = frame // frames_per_segment
            progress = (frame % frames_per_segment) / frames_per_segment

            # Construire les données jusqu'au point actuel
            current_x = x_data[:segment+1].tolist()
            current_y = y_data[:segment+1].tolist()

            if segment < num_segments:
                # Interpoler le point suivant
                next_x = x_data[segment] + progress * (x_data[segment+1] - x_data[segment])
                next_y = y_data[segment] + progress * (y_data[segment+1] - y_data[segment])
                current_x.append(next_x)
                current_y.append(next_y)

            line.set_data(current_x, current_y)

            # Mettre à jour les textes des valeurs
            for txt in value_texts:
                txt.set_text('')
            if current_x:
                # Afficher la valeur actuelle au dernier point
                txt = value_texts[segment]
                txt.set_position((current_x[-1], current_y[-1]))
                txt.set_text(f"{int(current_y[-1])}")
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
    elif chart_type == "Camembert":
        # Nombre de frames pour l'animation
        num_frames = 50  # Plus de frames pour une animation fluide
        frames = np.linspace(0.01, 1, num_frames)  # Commence à 0.01 pour éviter les fractions nulles

        # Calculer les angles pour chaque valeur
        total = sum(values)
        fractions = [v / total for v in values]

        for i in frames:
            current_fractions = [fraction * i for fraction in fractions]

            # Vérifier que la somme des fractions est supérieure à zéro
            if sum(current_fractions) > 0:
                # Mettre à jour le camembert
                ax.clear()
                # Appliquer un fond moderne
                fig.patch.set_facecolor('#2E3440')
                ax.set_facecolor('#3B4252')
                ax.axis('equal')
                ax.set_title(f"Graphique {chart_type}", fontsize=16, fontweight='bold', color='white')

                # Dessiner le camembert avec les fractions actuelles
                patches, texts = ax.pie(current_fractions, labels=labels, colors=palette, startangle=90, counterclock=False)
                # Changer la couleur des textes
                for text in texts:
                    text.set_color('white')

                # Enregistrer l'image dans un buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
                buf.seek(0)
                image = Image.open(buf).convert('RGBA')
                images.append(image)
                buf.close()
            else:
                # Si la somme est nulle, on saute le dessin du camembert pour cette frame
                continue
    else:
        # Pour les graphiques à barres
        # Nombre de frames pour l'animation
        num_frames = 50  # Augmenter pour une animation plus fluide
        frames = np.linspace(0, 1, num_frames)
        if growth is not None and chart_type != "Lignes":
            texts = []
            for _ in labels:
                texts.append(ax.text(0, 0, '', fontsize=10, fontweight='bold',
                                     color='white',
                                     bbox=dict(facecolor='#4C566A', alpha=0.6, edgecolor='none', pad=0.5)))
        elif chart_type != "Lignes":
            # Pour afficher les valeurs qui s'incrémentent
            value_texts = []
            for _ in labels:
                value_texts.append(ax.text(0, 0, '', fontsize=10, fontweight='bold',
                                           color='white',
                                           bbox=dict(facecolor='#4C566A', alpha=0.6, edgecolor='none', pad=0.5)))
        for i in frames:
            current_values = [val * i for val in values]

            if chart_type == "Barres horizontales":
                # Mettre à jour les largeurs des barres
                for bar, val in zip(bars, current_values):
                    bar.set_width(val)
                # Mettre à jour les positions des labels de croissance ou des valeurs
                if growth is not None:
                    for idx, (text, bar, perc) in enumerate(zip(texts, bars, growth)):
                        perc_display = f"{int(perc * i)}%"
                        text.set_position((bar.get_width() + max_value*0.01, bar.get_y() + bar.get_height()/2))
                        text.set_text(perc_display)
                else:
                    for idx, (text, bar, val) in enumerate(zip(value_texts, bars, current_values)):
                        value_display = f"{int(val)}"
                        text.set_position((bar.get_width() + max_value*0.01, bar.get_y() + bar.get_height()/2))
                        text.set_text(value_display)
            elif chart_type == "Barres verticales":
                # Mettre à jour les hauteurs des barres
                for bar, val in zip(bars, current_values):
                    bar.set_height(val)
                # Mettre à jour les positions des labels de croissance ou des valeurs
                if growth is not None:
                    for idx, (text, bar, perc) in enumerate(zip(texts, bars, growth)):
                        perc_display = f"{int(perc * i)}%"
                        text.set_position((bar.get_x() + bar.get_width()/2, bar.get_height() + max_value*0.01))
                        text.set_text(perc_display)
                else:
                    for idx, (text, bar, val) in enumerate(zip(value_texts, bars, current_values)):
                        value_display = f"{int(val)}"
                        text.set_position((bar.get_x() + bar.get_width()/2, bar.get_height() + max_value*0.01))
                        text.set_text(value_display)

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
    * Camembert

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

            # Optionnelle : sélection de la colonne pour la croissance
            growth_option = st.checkbox("Ajouter une colonne pour les pourcentages de croissance")
            if growth_option:
                growth_col = st.selectbox("Sélectionnez la colonne pour les pourcentages de croissance", columns)
            else:
                growth_col = None

            # Sélection du type de graphique
            st.subheader("Sélectionnez le(s) type(s) de graphique")
            chart_type_options = ["Barres horizontales", "Barres verticales", "Lignes", "Camembert"]
            chart_type_selection = st.multiselect("Sélectionnez le(s) type(s) de graphique", chart_type_options, default=chart_type_options)

            # Ajuster la durée de l'animation
            st.subheader("Ajustez la vitesse de l'animation")
            frame_duration = st.slider("Durée de chaque frame (en secondes)", min_value=0.05, max_value=1.0, value=0.1, step=0.05)

            # Bouton pour générer les graphiques
            if st.button("Générer les graphiques"):
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
