import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

# Fonction pour créer et enregistrer le GIF animé en mémoire
def create_gif():
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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, max(postes_supplementaires) * 1.1)
    ax.set_xlabel("Nombre de postes supplémentaires (en milliers)")
    ax.set_title("Les métiers en plus forte expansion entre 2019 et 2030")
    
    images = []
    # Réduire le nombre de frames pour optimiser la performance
    for i in range(0, 101, 2):  # Par exemple, incrémenter de 2
        ax.clear()
        ax.barh(
            metiers_reverses, 
            [val * (i / 100) for val in postes_supplementaires_reverses], 
            color='skyblue'
        )
        ax.set_xlim(0, max(postes_supplementaires) * 1.1)
        ax.set_xlabel("Nombre de postes supplémentaires (en milliers)")
        ax.set_title("Les métiers en plus forte expansion entre 2019 et 2030")
        
        # Sauvegarder l'image dans un buffer en mémoire
        buf = BytesIO()
        plt.savefig(buf, format='PNG')
        buf.seek(0)
        image = Image.open(buf).convert('RGB')  # Convertir en 'RGB' au lieu de 'P'
        images.append(image)
        buf.close()
    
    # Créer le GIF dans un buffer en mémoire
    buf_gif = BytesIO()
    images[0].save(
        buf_gif, 
        save_all=True, 
        append_images=images[1:], 
        duration=50, 
        loop=0
    )
    buf_gif.seek(0)
    return buf_gif

# Streamlit
st.title("Animation des métiers en expansion (2019-2030)")
st.write("Ce GIF animé montre la progression des métiers en forte croissance.")

# Générer et afficher le GIF
gif_buffer = create_gif()
st.image(gif_buffer, caption="Métiers en expansion", use_column_width=True)
