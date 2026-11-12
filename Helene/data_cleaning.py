import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import re
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

from bs4 import BeautifulSoup
import re

data = pd.read_csv(r'C:\Users\LN\Documents\Jedha\Projets\NORTH_FACE\Helene\data.csv')

sample = data #.sample(n=5, random_state=42)

sample_clean = sample.copy()

# Créer un objet BeautifulSoup pour chaque description
html = sample_clean["description"].fillna("").apply(
    lambda text: BeautifulSoup(text, "html.parser")
)

def extract_description(soup):
    """Fetch text before first balise <br> ou <b>."""
    pieces = []

    for element in soup.contents:
        if element.name in ["br", "b"]:
            break

        pieces.append(str(element))

    text_html = " ".join(pieces)
    return BeautifulSoup(text_html, "html.parser").get_text(" ", strip=True)


sample_clean["description_text"] = html.apply(extract_description)

def extract_details(soup):
    """Fetch list details."""
    titre = soup.find("b", string=re.compile(r"^\s*Details\s*:\s*$", re.I))

    if titre is None:
        return []

    list = titre.find_next_sibling()

    if list is None or list.name != "ul":
        return []

    details = []

    for element in list.find_all("li"):
        details.append(element.get_text(" ", strip=True))

    return details


sample_clean["details"] = html.apply(extract_details)

def extract_value(soup, nom):
    """Fetch text after bold titles"""
    motif = rf"^\s*{re.escape(nom)}\s*:\s*$"
    titre = soup.find("b", string=re.compile(motif, re.I))

    if titre is None:
        return None

    morceaux = []

    for element in titre.next_siblings:
        if element.name in ["br", "b"]:
            break

        morceaux.append(str(element))

    text_html = " ".join(morceaux)
    return BeautifulSoup(text_html, "html.parser").get_text(" ", strip=True)


sample_clean["fabric"] = html.apply(lambda soup: extract_value(soup, "Fabric"))
sample_clean["weight"] = html.apply(lambda soup: extract_value(soup, "Weight"))

def extract_country(soup):
    """Fetch country after « Made in »."""
    for text in soup.stripped_strings:
        if text.lower().startswith("made in "):
            return text[8:].strip().rstrip(".")

    return None


sample_clean["pays"] = html.apply(extract_country)

sample_clean["details"] = sample_clean["details"].apply(
    lambda liste: " ".join(liste)
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s%/.,-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

for col in ["description_text", "details", "fabric", "weight"]:
    sample_clean[col] = sample_clean[col].apply(
        lambda text: clean_text(text) if pd.notna(text) else text
    )

sample_clean.drop(columns=["description"], inplace=True)

sample_clean.to_csv(r'C:\Users\LN\Documents\Jedha\Projets\NORTH_FACE\Helene\sample_clean.csv', index=False)