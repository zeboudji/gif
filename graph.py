import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import imageio
import seaborn as sns
from matplotlib.patches import Patch  # Pour créer des légendes personnalisées

# 1. Définir la configuration de la page en premier
st.set_page_config(page_title="🎨 Animation Graphique Personnalisée", layout="wide")

# 2. Appliquer un style moderne avec Seaborn
sns.set_theme(style='whitegrid')

# 3. Ajouter du CSS personnalisé pour rendre les images réactives
st.markdown(
    """
    <style>
    img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 4. Fonction pour créer des frames pour un graphique spécifique
def create_animated_chart_frames(labels, values, growth=None, chart_type="Barres horizontales", title="Titre 1", frame_duration=0.15):
    # Vérifier que les listes ont la même longueur
    if not (len(labels) == len(values)):
        st.error("Les listes des labels et des valeurs doivent avoir la même longueur.")
        return []
    if growth is not None and len(growth) != len(labels):
        st.error("La liste de la colonne 2 doit avoir la même longueur que les labels.")
        return []
    # Vérifier qu'il n'y a pas de valeurs manquantes
    if any(pd.isnull(labels)) or any(pd.isnull(values)) or (growth is not None and any(pd.isnull(growth))):
        st.error("Les données ne doivent pas contenir de valeurs manquantes.")
        return []
    # Pour le graphique Camembert, ignorer 'growth' et avertir l'utilisateur
    if chart_type == "Camembert" and growth is not None:
        st.warning("Le graphique Camembert ne supporte pas la troisième dimension . La colonne 2 sera ignorée.")
        growth = None
    # Choisir une palette de couleurs moderne et robuste
    num_colors = len(labels)
    palette = sns.color_palette("hls", num_colors)  # 'hls' est adapté pour de nombreuses couleurs
    # Vérifier que la palette a le bon nombre de couleurs
    if len(palette) != num_colors:
        st.error(f"La palette de couleurs générée ({len(palette)} couleurs) ne correspond pas au nombre de labels ({num_colors}).")
        return []
    # Création de la figure et des axes avec taille fixe
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)  # Taille et résolution fixes
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
    max_value = max([v + g if growth else v for v, g in zip(values, growth or [0]*len(values))]) * 1.1 if values else 1
    if chart_type == "Barres horizontales":
        ax.set_xlim(0, max_value)
        ax.set_ylim(-0.5, len(labels) - 0.5)
        ax.set_xlabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        if growth is not None:
            bars_values = ax.barh(labels, [0]*len(values), color=palette, edgecolor='white', label='Valeurs 1')
            bars_growth = ax.barh(labels, [0]*len(values), left=[0]*len(values), color='lightblue', edgecolor='white', label='Valeurs 2')
        else:
            bars_values = ax.barh(labels, [0]*len(values), color=palette, edgecolor='white')
    elif chart_type == "Barres verticales":
        ax.set_ylim(0, max_value)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', color='white')
        if growth is not None:
            bars_values = ax.bar(labels, [0]*len(values), color=palette, edgecolor='white', label='Valeurs')
            bars_growth = ax.bar(labels, [0]*len(values), bottom=[0]*len(values), color='lightblue', edgecolor='white', label='Croissance')
        else:
            bars_values = ax.bar(labels, [0]*len(values), color=palette, edgecolor='white')
    elif chart_type == "Lignes":
        # Ajustement des marges pour inclure les annotations
        y_margin = max_value * 0.15
        ax.set_ylim(0, max_value + y_margin)
        ax.set_xlim(-0.5, len(labels) - 0.5)
        ax.set_ylabel("Valeurs", fontsize=12, fontweight='bold', color='white')
        ax.set_xlabel("Labels", fontsize=12, fontweight='bold', color='white')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', color='white')
        line_values, = ax.plot([], [], color='#88C0D0', marker='o', linewidth=3, label='Valeurs')
        if growth is not None:
            line_growth, = ax.plot([], [], color='#D08770', marker='s', linewidth=3, label='Croissance')
        # Préparer les textes pour les valeurs
        value_texts_values = [ax.text(0, 0, '', fontsize=10, fontweight='bold', color='#88C0D0', ha='center') for _ in range(len(labels))]
        if growth is not None:
            value_texts_growth = [ax.text(0, 0, '', fontsize=10, fontweight='bold', color='#D08770', ha='center') for _ in range(len(labels))]
        # Créer une légende correspondante aux lignes
        if growth is not None:
            ax.legend([line_values, line_growth], ['Valeurs', 'Croissance'], facecolor='#4C566A', edgecolor='none', labelcolor='white', fontsize=10)
        else:
            ax.legend([line_values], ['Valeurs'], facecolor='#4C566A', edgecolor='none', labelcolor='white', fontsize=10)
    elif chart_type == "Camembert":
        # Pas d'axes pour un camembert
        ax.axis('equal')
        # Dessiner le camembert une fois pour récupérer les patches
        patches, texts = ax.pie([0]*len(values), labels=labels, colors=palette, startangle=90, counterclock=False)
        for text in texts:
            text.set_color('white')
        # Créer une légende correspondante aux labels
        handles = [Patch(facecolor=palette[i], label=labels[i]) for i in range(len(labels))]
        ax.legend(handles=handles, title='Légende', loc='center left', bbox_to_anchor=(1, 0.5),
                  facecolor='#4C566A', edgecolor='none', labelcolor='white', fontsize=10)
    else:
        st.error("Type de graphique non supporté pour cette animation.")
        return []

    # Titre du graphique sauf pour le camembert (géré différemment)
    if chart_type != "Camembert":
        ax.set_title(title, fontsize=16, fontweight='bold', color='white')
        # Ajuster les marges pour laisser de l'espace à la légende si nécessaire
        fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.25)

    frames = []
    
    if chart_type == "Lignes":
        x_data = np.arange(len(values))
        y_values = np.array(values)
        if growth is not None:
            y_growth = np.array(growth)

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
            current_y_values = y_values[:segment+1].tolist()

            if segment < num_segments:
                # Interpoler le point suivant
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
            # Effacer les textes précédents
            for txt in value_texts_values:
                txt.set_text('')
            if growth is not None:
                for txt in value_texts_growth:
                    txt.set_text('')

            if current_x:
                # Afficher la valeur actuelle au dernier point pour 'Valeurs'
                txt = value_texts_values[segment]
                txt.set_position((current_x[-1], current_y_values[-1] + max_value * 0.05))
                txt.set_text(f"{int(current_y_values[-1])}")
                txt.set_fontsize(10)
                txt.set_fontweight('bold')

                # Afficher la valeur actuelle au dernier point pour 'Croissance' si elle existe
                if growth is not None:
                    txt_growth = value_texts_growth[segment]
                    txt_growth.set_position((current_x[-1], current_y_growth[-1] + max_value * 0.05))
                    txt_growth.set_text(f"{int(current_y_growth[-1])}")
                    txt_growth.set_fontsize(10)
                    txt_growth.set_fontweight('bold')

            # Enregistrer l'image dans un buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=100)
            buf.seek(0)
            image = Image.open(buf).convert('RGBA')
            frames.append(image)
            buf.close()

    elif chart_type == "Camembert":
        try:
            # Nombre de frames pour l'animation
            num_frames = 50  # Plus de frames pour une animation fluide
            frames_values = np.linspace(0.0, 1.0, num_frames)

            fractions_values = [v / total for v in values]

            for i in frames_values:
                current_fractions = [fraction * i for fraction in fractions_values]
                # Mettre à jour le camembert
                ax.clear()
                # Appliquer un fond moderne
                fig.patch.set_facecolor('#2E3440')
                ax.set_facecolor('#3B4252')
                ax.axis('equal')
                ax.set_title(title, fontsize=16, fontweight='bold', color='white')

                # Dessiner le camembert avec les fractions actuelles
                patches, texts = ax.pie(current_fractions, labels=labels, colors=palette, startangle=90, counterclock=False)
                for text in texts:
                    text.set_color('white')

                # Créer une légende correspondante aux labels
                handles = [Patch(facecolor=palette[i], label=labels[i]) for i in range(len(labels))]
                ax.legend(handles=handles, title='Légende', loc='center left', bbox_to_anchor=(1, 0.5),
                          facecolor='#4C566A', edgecolor='none', labelcolor='white', fontsize=10)

                # Enregistrer l'image dans un buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=100)
                buf.seek(0)
                image = Image.open(buf).convert('RGBA')
                frames.append(image)
                buf.close()
        except Exception as e:
            st.error(f"Erreur lors de la création du graphique camembert : {e}")
            return []

    else:
        # Pour les graphiques à barres
        # Nombre de frames pour l'animation
        num_frames = 50  # Augmenter pour une animation plus fluide
        frames = np.linspace(0, 1, num_frames)
        # Préparer les textes pour les valeurs
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
                # Mettre à jour les largeurs des barres
                for idx, (bar_value, val) in enumerate(zip(bars_values, current_values)):
                    bar_value.set_width(val)
                if growth is not None:
                    for idx, (bar_growth, val, gro) in enumerate(zip(bars_growth, current_values, current_growth)):
                        bar_growth.set_width(gro)
                        bar_growth.set_x(val)
                # Mettre à jour les positions des valeurs
                for idx, (text, bar_value) in enumerate(zip(value_texts, bars_values)):
                    total_width = bar_value.get_width()
                    if growth is not None:
                        total_width += bars_growth[idx].get_width()
                    text.set_position((total_width + max_value*0.01, bar_value.get_y() + bar_value.get_height()/2))
                    text.set_text(f"{int(total_width)}")
            elif chart_type == "Barres verticales":
                # Mettre à jour les hauteurs des barres
                for idx, (bar_value, val) in enumerate(zip(bars_values, current_values)):
                    bar_value.set_height(val)
                if growth is not None:
                    for idx, (bar_growth, val, gro) in enumerate(zip(bars_growth, current_values, current_growth)):
                        bar_growth.set_height(gro)
                        bar_growth.set_y(val)
                # Mettre à jour les positions des valeurs
                for idx, (text, bar_value) in enumerate(zip(value_texts, bars_values)):
                    total_height = bar_value.get_height()
                    if growth is not None:
                        total_height += bars_growth[idx].get_height()
                    text.set_position((bar_value.get_x() + bar_value.get_width()/2, total_height + max_value*0.01))
                    text.set_text(f"{int(total_height)}")

            # Enregistrer l'image dans un buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=100)
            buf.seek(0)
            image = Image.open(buf).convert('RGBA')
            frames.append(image)
            buf.close()

    if not frames:
        st.error(f"Aucune image n'a été générée pour le graphique {chart_type}.")
        return []
    
    return frames

# 5. Fonction pour combiner des images horizontalement
from PIL import Image

def combine_images_horizontally(image_list):
    if not image_list:
        return None
    widths, heights = zip(*(i.size for i in image_list))
    total_width = sum(widths)
    max_height = max(heights)
    combined_image = Image.new('RGBA', (total_width, max_height), (0, 0, 0, 0))
    x_offset = 0
    for im in image_list:
        combined_image.paste(im, (x_offset, 0))
        x_offset += im.width
    return combined_image

# 6. Fonction pour créer un seul GIF animé avec plusieurs graphiques
def create_single_animated_gif(labels, values, growth=None, chart_type_selection=None, titles=None, frame_duration=0.15):
    # Dictionnaire pour stocker les frames de chaque graphique
    chart_frames = {}
    selected_chart_types = chart_type_selection
    for chart_type in selected_chart_types:
        # Obtenir le titre correspondant ou utiliser un titre par défaut
        title = titles.get(chart_type, "Titre 1")
        frames = create_animated_chart_frames(labels, values, growth, chart_type, title, frame_duration)
        if frames:
            chart_frames[chart_type] = frames
    
    if not chart_frames:
        st.error("Aucun graphique n'a pu être généré avec les données fournies.")
        return None
    
    # Trouver le nombre maximum de frames
    max_frames = max(len(frames) for frames in chart_frames.values())
    
    # Normaliser le nombre de frames pour chaque graphique
    for chart_type, frames in chart_frames.items():
        if len(frames) < max_frames:
            # Répéter la dernière frame pour atteindre le nombre maximum
            last_frame = frames[-1]
            frames.extend([last_frame] * (max_frames - len(frames)))
            chart_frames[chart_type] = frames
    
    # Combiner les frames
    combined_images = []
    for i in range(max_frames):
        current_images = [chart_frames[chart_type][i] for chart_type in selected_chart_types]
        combined_image = combine_images_horizontally(current_images)
        combined_images.append(combined_image)
    
    # Ajouter une pause à la fin de l'animation
    pause_duration = 2  # Durée de la pause en secondes
    # Convertir pause_duration en nombre de frames
    pause_frames = int(pause_duration / frame_duration)
    combined_images.extend([combined_images[-1]] * pause_frames)
    
    # Convertir les images en frames pour le GIF
    try:
        frames_gif = [np.array(img) for img in combined_images]
        # Créer le GIF
        buf_gif = BytesIO()
        imageio.mimsave(buf_gif, frames_gif, format='GIF', duration=[frame_duration] * len(combined_images), loop=0)
        buf_gif.seek(0)
        return buf_gif
    except Exception as e:
        st.error(f"Erreur lors de la création du GIF unique : {e}")
        return None

# 7. Interface Streamlit
st.title("🎨 Animation Graphique Personnalisée")
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

# 8. Uploader de fichier
uploaded_file = st.file_uploader("📁 Veuillez télécharger un fichier Excel ou CSV avec vos données.", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    # Lire le fichier Excel ou CSV
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            sheet_names = None  # Pas d'onglets pour les CSV
        else:
            # Lire le fichier Excel pour obtenir les noms des onglets
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names

            # Permettre à l'utilisateur de sélectionner un onglet
            st.subheader("🗂️ Sélectionnez l'onglet à utiliser")
            sheet_name = st.selectbox("Choisissez un onglet", sheet_names)

            # Lire le DataFrame à partir de l'onglet sélectionné
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)

        # Afficher un aperçu des données
        st.subheader("🔍 Aperçu des données téléchargées")
        st.dataframe(df.head())

        # Vérifier qu'il y a au moins deux colonnes
        if df.shape[1] < 2:
            st.error("Le fichier doit contenir au moins deux colonnes.")
        else:
            # Obtenir la liste des colonnes
            columns = df.columns.tolist()

            # Permettre à l'utilisateur de sélectionner les colonnes
            st.subheader("📝 Sélectionnez les colonnes correspondantes")
            label_col = st.selectbox("Sélectionnez la colonne pour les **libellés**", columns)
            value_col = st.selectbox("Sélectionnez la colonne pour les **valeurs numériques**", [col for col in columns if col != label_col])

            # Optionnelle : sélection de la colonne pour la troisième dimension
            growth_option = st.checkbox("Ajouter une colonne pour une troisième dimension (ex: croissance)")
            if growth_option:
                # Exclure les colonnes déjà sélectionnées pour éviter les doublons
                available_growth_cols = [col for col in columns if col != label_col and col != value_col]
                if available_growth_cols:
                    growth_col = st.selectbox("Sélectionnez la colonne pour la **troisième dimension**", available_growth_cols)
                else:
                    st.error("Aucune colonne disponible pour la troisième dimension.")
                    growth_col = None
            else:
                growth_col = None

            # Sélection du type de graphique
            st.subheader("📊 Sélectionnez le(s) type(s) de graphique")
            chart_type_options = ["Barres horizontales", "Barres verticales", "Lignes", "Camembert"]
            chart_type_selection = st.multiselect(
                "Sélectionnez le(s) type(s) de graphique",
                chart_type_options,
                default=chart_type_options
            )

            # Si des types de graphiques sont sélectionnés, demander des titres
            titles = {}
            if chart_type_selection:
                st.subheader("🖋️ Entrez les titres des graphiques")
                for i, chart_type in enumerate(chart_type_selection, 1):
                    user_title = st.text_input(f"Entrez le titre pour **{chart_type}**", value=f"Titre {i}")
                    titles[chart_type] = user_title if user_title.strip() else f"Titre {i}"
            else:
                st.info("Veuillez sélectionner au moins un type de graphique.")

            # Ajuster la durée de l'animation
            st.subheader("⏱️ Ajustez la vitesse de l'animation")
            frame_duration = st.slider(
                "Durée de chaque frame (en secondes)",
                min_value=0.05,
                max_value=1.0,
                value=0.1,
                step=0.05
            )

            # Bouton pour générer les graphiques
            if st.button("🎬 Générer les graphiques"):
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
                    # Afficher les longueurs des listes pour débogage (peut être retiré)
                    st.write(f"Nombre de labels : {len(labels)}")
                    st.write(f"Nombre de valeurs : {len(values)}")
                    if growth_col:
                        st.write(f"Nombre de valeurs de croissance : {len(growth)}")

                    # Générer le GIF unique avec les types de graphiques sélectionnés
                    try:
                        charts_gif = create_single_animated_gif(labels, values, growth, chart_type_selection, titles, frame_duration)
                    except Exception as e:
                        st.error(f"Erreur lors de la création des graphiques : {e}")
                        charts_gif = None

                    # Afficher le GIF unique
                    st.subheader("✨ Graphique animé combiné")
                    if charts_gif:
                        st.image(charts_gif, caption="Graphiques Combinés", use_column_width=True)
                    else:
                        st.error("Aucun graphique n'a pu être généré avec les données fournies.")
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
else:
    st.info("Veuillez télécharger un fichier Excel ou CSV pour générer les graphiques animés.")
