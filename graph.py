import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile
import os

# Fonction pour créer et enregistrer le GIF animé
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
    
    # Utiliser un dossier temporaire pour stocker les images
    with tempfile.TemporaryDirectory() as temp_dir:
        filenames = []
        for i in range(101):
            ax.clear()
            ax.barh(metiers_reverses, [val * (i / 100) for val in postes_supplementaires_reverses], color='skyblue')
            ax.set_xlim(0, max(postes_supplementaires) * 1.1)
            ax.set_xlabel("Nombre de postes supplémentaires (en milliers)")
            ax.set_title("Les métiers en plus forte expansion entre 2019 et 2030")
            filename = os.path.join(temp_dir, f"plot_{i}.png")
            plt.savefig(filename)
            filenames.append(filename)

        # Créer le GIF
        images = [Image.open(f) for f in filenames]
        gif_path = os.path.join(temp_dir, "graphique_anime.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=50, loop=0)
    
    return gif_path

# Streamlit
st.title("Animation des métiers en expansion (2019-2030)")
st.write("Ce GIF animé montre la progression des métiers en forte croissance.")

# Générer et afficher le GIF
gif_path = create_gif()
st.image(gif_path, caption="Métiers en expansion", use_column_width=True)
