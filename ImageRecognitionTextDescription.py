import os
import base64
import requests
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# Pfad zum Ordner, in dem die descriptions.json liegt
folder_path = r"C:\Users\mschn\Desktop\venv\Test-WebFramework\venv\Ollama\Images"
json_path = os.path.join(folder_path, "descriptions.json")

# Funktion zur Konvertierung eines Bildes in Base64
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

# Funktion, um LLava für eine Beschreibung anzufragen
def query_llava_for_description(base64_image):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llava",
        "prompt": "Imagine the picture was a drawing without background or material. describe only the Structur from the objekt on the images. Hide everything for you except the structure with edges, corners and curves. Use five sentences.pay attention to features such as holes ",
        "stream": False,
        "images": [base64_image]
    }
    print(f"Sende Anfrage an LLava für Bildbeschreibung")
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        print("Erfolgreiche Antwort von LLava erhalten.")
        return response.json()['response']
    else:
        print(f"Fehler bei der Anfrage an LLava: {response.status_code}")
        return f"Error: {response.status_code}"

# JSON mit den Beschreibungen laden
with open(json_path, 'r', encoding='utf-8') as f:
    descriptions = json.load(f)

# Konfiguration für Embeddings und LLM
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")
Settings.llm = Ollama(model="solar", request_timeout=360.0)

# Erstellen des Vektor-Index aus den Beschreibungen
documents = [Document(id=filename, text=description) for filename, description in descriptions.items()]
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

def find_best_match_for_image(image_path: str) -> str:
    # Bild in Base64 umwandeln
    base64_img = convert_image_to_base64(image_path)
    # Bild über LLava beschreiben lassen
    llava_description = query_llava_for_description(base64_img)
    print("LLava-Beschreibung:", llava_description)

    # Query im Index
    response = query_engine.query(llava_description)
    if response.source_nodes:
        best_match = response.source_nodes[0]
        # Format: "filename: beschreibungstext"
        best_filename = best_match.node.text.split(": ", 1)[0]
        return best_filename
    else:
        return "Keine passende Beschreibung gefunden."

# Beispielhafte Verwendung:
image_path = r"C:\Users\mschn\Desktop\venv\Test-WebFramework\venv\Ollama\Images\TestBild\20230530_120729022_iOS.jpg"  # Pfad zum Eingangsbild anpassen
ergebnis = find_best_match_for_image(image_path)
print("Die ähnlichste Beschreibung stammt von:", ergebnis)
