import matplotlib.pyplot as plt
import io
from PIL import Image

# Données
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
croissance = [26, 18, 15, 17, 18, 16, 11, 24, 30, 15, 13, 13, 10, 9, 9]

# Tri des données pour un affichage ordonné
metiers_reverses = metiers[::-1]
postes_supplementaires_reverses = postes_supplementaires[::-1]
croissance_reverses = croissance[::-1]

# Création du graphique
plt.figure(figsize=(12, 8))
barres = plt.barh(metiers_reverses, postes_supplementaires_reverses, color='skyblue')

# Ajout des pourcentages de croissance sur les barres
for index, barre in enumerate(barres):
    largeur = barre.get_width()
    plt.text(largeur + 1, barre.get_y() + barre.get_height() / 2,
             f"{croissance_reverses[index]}%", va='center')

# Titres et labels
plt.xlabel("Nombre de postes supplémentaires (en milliers)")
plt.title("Les métiers en plus forte expansion entre 2019 et 2030")

plt.tight_layout()

# Enregistrement du graphique en tant que GIF
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
img.save('graphique.gif', 'GIF')

plt.show()
