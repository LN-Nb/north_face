import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import re
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

from bs4 import BeautifulSoup
import re

data = pd.read_csv('/Users/nicolasbour/Desktop/Projets_IA/Projets_certif/bloc_03_Machine_Learning/03_the_north_face/sample-data.csv')

sample = data #.sample(n=5, random_state=42)

sample_clean = sample.copy()

# Créer un objet BeautifulSoup pour chaque description
html = sample_clean["description"].fillna("").apply(
    lambda texte: BeautifulSoup(texte, "html.parser")
)

def extraire_description(soup):
    """Récupérer le texte avant la première balise <br> ou <b>."""
    morceaux = []

    for element in soup.contents:
        if element.name in ["br", "b"]:
            break

        morceaux.append(str(element))

    texte_html = " ".join(morceaux)
    return BeautifulSoup(texte_html, "html.parser").get_text(" ", strip=True)


sample_clean["description_texte"] = html.apply(extraire_description)

def extraire_details(soup):
    """Récupérer les éléments de la liste Details."""
    titre = soup.find("b", string=re.compile(r"^\s*Details\s*:\s*$", re.I))

    if titre is None:
        return []

    liste = titre.find_next_sibling()

    if liste is None or liste.name != "ul":
        return []

    details = []

    for element in liste.find_all("li"):
        details.append(element.get_text(" ", strip=True))

    return details


sample_clean["details"] = html.apply(extraire_details)

def extraire_valeur(soup, nom):
    """Récupérer le texte qui suit un titre en gras."""
    motif = rf"^\s*{re.escape(nom)}\s*:\s*$"
    titre = soup.find("b", string=re.compile(motif, re.I))

    if titre is None:
        return None

    morceaux = []

    for element in titre.next_siblings:
        if element.name in ["br", "b"]:
            break

        morceaux.append(str(element))

    texte_html = " ".join(morceaux)
    return BeautifulSoup(texte_html, "html.parser").get_text(" ", strip=True)


sample_clean["fabric"] = html.apply(lambda soup: extraire_valeur(soup, "Fabric"))
sample_clean["weight"] = html.apply(lambda soup: extraire_valeur(soup, "Weight"))

def extraire_pays(soup):
    """Récupérer le pays après « Made in »."""
    for texte in soup.stripped_strings:
        if texte.lower().startswith("made in "):
            return texte[8:].strip().rstrip(".")

    return None


sample_clean["pays"] = html.apply(extraire_pays)

sample_clean["details"] = sample_clean["details"].apply(
    lambda liste: " ".join(liste)
)

def nettoyer_texte(texte):
    texte = texte.lower()
    texte = re.sub(r"[^\w\s%/.,-]", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()

for col in ["description_texte", "details", "fabric", "weight"]:
    sample_clean[col] = sample_clean[col].apply(
        lambda texte: nettoyer_texte(texte) if pd.notna(texte) else texte
    )

sample_clean.drop(columns=["description"], inplace=True)

sample_clean.to_csv('/Users/nicolasbour/Desktop/Projets_IA/Projets_certif/bloc_03_Machine_Learning/03_the_north_face/Nico/sample_clean.csv', index=False)