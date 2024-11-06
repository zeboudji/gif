import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import imageio
import seaborn as sns

# Appliquer un style moderne de Seaborn
sns.set(style="whitegrid")

# Fonction pour créer et enregistrer le GIF animé en mémoire avec améliorations modernes
def create_modern_gif():
    metiers = [
        "Ingénieurs de l'informatique",
        "Infirmiers, sages-femmes",
        "Aides-soignants",
        "Cadres commerciaux et technico-commerciaux",
        "Aides à domicile",
        "Ouvriers qualifiés de la manutention",
        "Cadres des services administratifs, comptables et financiers",
        "Ingénieurs et cadres techniques de l'industrie",
        "Cadres du bâtiment et des travaux publics",
        "Ouvriers peu qualifiés de la manutention",
        "Personnels d'études et de recherche",
        "Médecins et assimilés",
        "Techniciens des services administratifs, comptables et financiers",
        "Techniciens et agents de maîtrise de la maintenance",
        "Professions paramédicales"
    ]
    
    postes_supplementaires = [110, 95, 80, 75, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15]
    croissance = [26, 18, 15, 17, 18, 16, 11, 24, 30, 15, 13, 13, 11, 10, 9]
    
    # Vérifier que les listes ont la même longueur
    assert len(metiers) == len(postes_supplementaires) == len(croissance), "Les listes doivent avoir la même longueur."
    
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
            edgecolor='white'
        )
        ax.set_xlim(0, max(postes_supplementaires) * 1.3)
        ax.set_xlabel("Nombre de postes supplémentaires (en milliers)", fontsize=14, fontweight='bold')
        ax.set_title("Les métiers en plus forte expansion entre 2019 et 2030", fontsize=18, fontweight='bold')
        
        # Ajuster les marges pour éviter que les labels ne soient tronqués
        plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1)
        
        # Ajouter les labels de croissance à la fin de chaque barre
        for index, (val, perc) in enumerate(zip([val * (i / 100) for val in postes_reverses], croissance_reverses)):
            ax.text(val + max(postes_supplementaires)*0.01, index, f"{perc}%", va='center', fontsize=12, fontweight='bold', color='black')
        
        # Ajouter des annotations avec des flèches pour une touche moderne
        for index, (val, perc) in enumerate(zip([val * (i / 100) for val in postes_reverses], croissance_reverses)):
            ax.annotate(
                f"{perc}%", 
                xy=(val, index), 
                xytext=(val + max(postes_supplementaires)*0.05, index),
                arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5),
                fontsize=10,
                color='black'
            )
        
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
    
    # Ajouter des copies de la dernière frame pour simuler un temps d'arrêt
    pause_frames = 15  # Nombre de frames de pause
    if images:
        last_frame = images[-1]
        for _ in range(pause_frames):
            images.append(last_frame)
    
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
        # Durée de chaque frame : 0.02s pour les frames d'animation, 0.2s pour les frames de pause
        durations = [0.02]*len(images)
        for i in range(len(images)-pause_frames, len(images)):
            durations[i] = 0.2  # Augmenter la durée des frames de pause
        imageio.mimsave(buf_gif, frames, format='GIF', duration=durations)
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
""")

# Générer et afficher le GIF avec les améliorations modernes
gif_buffer = create_modern_gif()
if gif_buffer:
    st.image(gif_buffer, caption="Métiers en expansion", use_column_width=True)
