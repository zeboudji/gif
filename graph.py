import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import imageio
import seaborn as sns
import matplotlib.patheffects as path_effects  # Import pour les effets de contour

# Appliquer un style moderne de Seaborn
sns.set(style="whitegrid")

# Fonction pour créer et enregistrer le GIF animé
def create_modern_gif(metiers, postes_supplementaires, croissance):
    # Vérifier que les listes ont la même longueur
    if not (len(metiers) == len(postes_supplementaires) == len(croissance)):
        st.error("Les listes doivent avoir la même longueur.")
        return None

    # Inverser les listes pour que le premier métier soit en bas du graphique
    metiers_reverses = metiers[::-1]
    postes_reverses = postes_supplementaires[::-1]
    croissance_reverses = croissance[::-1]
    
    # Choisir une palette de couleurs moderne
    palette = sns.color_palette("viridis", len(metiers))
    
    # Création de la figure avec une taille suffisante
    fig, ax = plt.subplots(figsize=(14, 10))  # Taille augmentée pour plus d'espace
    ax.set_xlim(0, max(postes_supplementaires) * 1.3)  # Augmenter l'espace pour les labels de croissance
    ax.set_xlabel("Nombre de postes supplémentaires (en milliers)", fontsize=14, fontweight='bold')
    ax.set_title("Les métiers en plus forte expansion entre 2019 et 2030", fontsize=18, fontweight='bold')
    
    images = []
    
    # Nombre de frames pour l'animation
    for i in range(0, 101, 2):  # Incrémenter de 2 pour réduire le nombre de frames
        ax.clear()
        ax.barh(
            metiers_reverses, 
            [val * (i / 100) for val in postes_reverses], 
            color=palette,
            edgecolor='white',
            alpha=0.8  # Ajout de transparence pour un effet moderne
        )
        ax.set_xlim(0, max(postes_supplementaires) * 1.3)
        ax.set_xlabel("Nombre de postes supplémentaires (en milliers)", fontsize=14, fontweight='bold')
        ax.set_title("Les métiers en plus forte expansion entre 2019 et 2030", fontsize=18, fontweight='bold')
        
        # Ajuster les marges pour éviter que les labels ne soient tronqués
        plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1)
        
        # Calculer le pourcentage actuel pour chaque métier
        # Ce pourcentage va de 0 à sa valeur finale en fonction de l'avancement i
        pourcentages_actuels = [perc * (i / 100) for perc in croissance_reverses]
        
        # Ajouter les labels de croissance à la fin de chaque barre avec un fond semi-transparent
        for index, (val, perc_actuel) in enumerate(zip([val * (i / 100) for val in postes_reverses], pourcentages_actuels)):
            # Arrondir le pourcentage actuel à l'entier le plus proche
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
            # Ajouter un contour au texte pour le rendre plus visible
            text.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'), path_effects.Normal()])
        
        # Sauvegarder l'image dans un buffer en mémoire
        buf = BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight', transparent=False)
        buf.seek(0)
        try:
            image = Image.open(buf).convert('RGB')  # Convertir en 'RGB'
            images.append(image)
        except Exception as e:
            st.error(f"Erreur lors de l'ouverture de l'image : {e}")
        buf.close()
    
    plt.close(fig)  # Fermer la figure pour libérer de la mémoire
    
    # Convertir les images PIL en tableaux numpy
    try:
        frames = [np.array(img) for img in images]
    except Exception as e:
        st.error(f"Erreur lors de la conversion des images en tableaux numpy : {e}")
        return None
    
    # Créer le GIF avec imageio dans un buffer en mémoire
    buf_gif = BytesIO()
    try:
        # Durée uniforme pour chaque frame : 0.02s
        durations = [0.02] * len(images)
        imageio.mimsave(buf_gif, frames, format='GIF', duration=durations, loop=1)  # loop=1 pour jouer une fois
        st.success("GIF créé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de la création du GIF : {e}")
        return None
    buf_gif.seek(0)
    return buf_gif

# Interface Streamlit
st.set_page_config(page_title="Animation des Métiers en Expansion", layout="wide")
st.title("Animation des Métiers en Expansion (2019-2030)")
st.markdown("""
Ce GIF animé montre la progression des métiers en forte croissance entre 2019 et 2030.
* **Nombre de postes supplémentaires** sur l'axe des abscisses (en milliers).
* **Pourcentage de croissance** affiché à la fin de chaque barre.

Veuillez télécharger un fichier Excel contenant les colonnes suivantes :
- **Metiers** : les noms des métiers.
- **Postes_supplementaires** : le nombre de postes supplémentaires (en milliers).
- **Croissance** : le pourcentage de croissance.

Assurez-vous que les colonnes portent exactement ces noms.
""")

# Uploader de fichier
uploaded_file = st.file_uploader("Veuillez télécharger un fichier Excel au format requis.", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Lire le fichier Excel
    try:
        df = pd.read_excel(uploaded_file)
        # Vérifier que les colonnes nécessaires sont présentes
        required_columns = ['Metiers', 'Postes_supplementaires', 'Croissance']
        if not all(column in df.columns for column in required_columns):
            st.error(f"Le fichier Excel doit contenir les colonnes suivantes: {', '.join(required_columns)}")
        else:
            # Extraire les données
            metiers = df['Metiers'].tolist()
            postes_supplementaires = df['Postes_supplementaires'].tolist()
            croissance = df['Croissance'].tolist()

            # Générer et afficher le GIF
            gif_buffer = create_modern_gif(metiers, postes_supplementaires, croissance)
            if gif_buffer:
                st.image(gif_buffer, caption="Métiers en expansion", use_column_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Excel: {e}")
else:
    st.info("Veuillez télécharger un fichier Excel pour générer le graphique animé.")
