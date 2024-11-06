import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
import imageio

# Fonction pour créer et enregistrer le GIF animé en mémoire avec ajustements et options supplémentaires
def create_gif_with_enhancements():
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
    metiers_reverses = metiers[::-1]
    postes_supplementaires_reverses = postes_supplementaires[::-1]
    
    # Création de la figure avec une taille suffisante
    fig, ax = plt.subplots(figsize=(14, 10))  # Taille augmentée
    ax.set_xlim(0, max(postes_supplementaires) * 1.1)
    ax.set_xlabel("Nombre de postes supplémentaires (en milliers)", fontsize=12)
    ax.set_title("Les métiers en plus forte expansion entre 2019 et 2030", fontsize=16)
    
    images = []
    # Réduire le nombre de frames pour optimiser la performance
    for i in range(0, 101, 2):  # Incrémenter de 2
        ax.clear()
        ax.barh(
            metiers_reverses, 
            [val * (i / 100) for val in postes_supplementaires_reverses], 
            color='skyblue'
        )
        ax.set_xlim(0, max(postes_supplementaires) * 1.1)
        ax.set_xlabel("Nombre de postes supplémentaires (en milliers)", fontsize=12)
        ax.set_title("Les métiers en plus forte expansion entre 2019 et 2030", fontsize=16)
        
        # Ajuster les marges pour éviter que les labels ne soient tronqués
        plt.subplots_adjust(left=0.35, right=0.95, top=0.9, bottom=0.1)
        
        # Optionnel : Ajouter des annotations pour chaque barre
        for index, value in enumerate([val * (i / 100) for val in postes_supplementaires_reverses]):
            ax.text(value + max(postes_supplementaires)*0.01, index, f"{value:.1f}", va='center', fontsize=10)
        
        # Sauvegarder l'image dans un buffer en mémoire
        buf = BytesIO()
        plt.savefig(buf, format='PNG', bbox_inches='tight')
        buf.seek(0)
        try:
            image = Image.open(buf).convert('RGB')  # Convertir en 'RGB'
            images.append(image)
            st.write(f"Frame {i}: Image ajoutée avec succès.")
        except Exception as e:
            st.write(f"Frame {i}: Erreur lors de l'ouverture de l'image - {e}")
        buf.close()
    
    plt.close(fig)  # Fermer la figure pour libérer de la mémoire
    
    if not images:
        st.error("Aucune image n'a été générée. Vérifiez la génération des images.")
        return None
    
    # Convertir les images PIL en tableaux numpy
    try:
        frames = [np.array(img) for img in images]
    except Exception as e:
        st.error(f"Erreur lors de la conversion des images en tableaux numpy - {e}")
        return None
    
    # Créer le GIF avec imageio dans un buffer en mémoire
    buf_gif = BytesIO()
    try:
        imageio.mimsave(buf_gif, frames, format='GIF', duration=0.02)  # Durée réduite à 20ms
        st.write("GIF créé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors de la création du GIF - {e}")
        return None
    buf_gif.seek(0)
    return buf_gif

# Streamlit
st.title("Animation des métiers en expansion (2019-2030)")
st.write("Ce GIF animé montre la progression des métiers en forte croissance.")

# Générer et afficher le GIF avec les améliorations et options supplémentaires
gif_buffer = create_gif_with_enhancements()
if gif_buffer:
    st.image(gif_buffer, caption="Métiers en expansion", use_column_width=True)
