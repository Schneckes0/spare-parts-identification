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
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Schritt 1: CLIP initialisieren
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K").to(device)
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")

# Schritt 2: Bilder indizieren und in Base64 umwandeln
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

def query_llava_for_description(base64_image):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llava",
        "prompt": "Please describe the structure of the object which you can see in the picture. Do not describe the background but only the main object.",
        "stream": False,
        "images": [base64_image]
    }
    print(f"Sende Anfrage an LLava für Bildbeschreibung")
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        print(f"Erfolgreiche Antwort von LLava erhalten.")
        return response.json()['response']
    else:
        print(f"Fehler bei der Anfrage an LLava: {response.status_code}")
        return f"Error: {response.status_code}"

folder_path = "C:/Users/kej9ho/Documents/Bilderkennung/data/train/417.000939"  # Ersetzen Sie dies durch den tatsächlichen Pfad zu Ihrem Bildordner
supported_formats = ('.jpg', '.png')
base64_images = {}
descriptions = {}

print("Indiziere Bilder und wandle sie in Base64 um und beschreibe sie mit LLava...")
for filename in os.listdir(folder_path):
    if filename.lower().endswith(supported_formats):
        image_path = os.path.join(folder_path, filename)
        base64_images[filename] = convert_image_to_base64(image_path)
        description = query_llava_for_description(base64_images[filename])
        descriptions[filename] = f"{filename}: {description}"  # Dateiname in Beschreibung speichern
        print(f"Bild {filename} indiziert und beschrieben.")
        print(description)
        print()

# Schritt 3: Beschreibungen in Vektor-Format konvertieren und in Vektor-Index speichern
print("Konvertiere Bildbeschreibungen in Vektor-Format und speichere im Vektor-Index...")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")  # Verwenden von all-MiniLM-L12-v2 für Text-Embeddings
Settings.llm = Ollama(model="solar", request_timeout=360.0)

# Erstellen der Dokumente im richtigen Format und den Dateinamen als ID verwenden
documents = []
for filename, description in descriptions.items():
    doc_id = filename  # Dateiname als ID verwenden
    documents.append(Document(id=doc_id, text=description))

index = VectorStoreIndex.from_documents(documents)
print("Bildbeschreibungen erfolgreich in Vektor-Index gespeichert.")

# Dynamische Abfragefunktion
def query_image_by_description(description):
    # Ähnlichkeitsabfrage im Index
    query_results = index.as_query_engine().query(description)
    return query_results

# Schritt 4: Chatbot implementieren
class ImageSearchBot:
    def __init__(self):
        self.model = ChatOllama(model="solar")
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for image searching tasks. Use the following context to answer the question.
            If you don't know the answer, just say you don't know. Use no more than three sentences and be concise in your answer. [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )
        self.index = index  # Setzen Sie den zuvor erstellten Index

    def search_image(self, query: str):
        print(f"Suche nach Bild mit der Anfrage: {query}")
        response = query_image_by_description(query)
        if response.source_nodes:
            best_match = response.source_nodes[0]
            score = best_match.score
            description = best_match.node.text  # Beschreibung aus der ersten Antwort extrahieren
            filename = description.split(": ", 1)[0]  # Dateiname aus der Beschreibung extrahieren

            # Überprüfen, ob der Score unter einem bestimmten Schwellenwert liegt
            print()
            print("Score: " + str(score))
            if score > 0.5:  # Anpassen des Schwellenwerts je nach Bedarf
                return f"Ergebnis der Bildsuche: {description}\nDateiname: {filename}"
            else:
                return "Kein passendes Bild gefunden."
        else:
            return "Kein passendes Bild gefunden."

# Schritt 5: Optionales zweites LLM (Llava) nutzen
def query_llava(base64_image, prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llava",
        "prompt": prompt,
        "stream": False,
        "images": [base64_image]
    }
    print(f"Sende Anfrage an LLava mit dem Prompt: {prompt}")
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        print(f"Erfolgreiche Antwort von LLava erhalten.")
        return response.json()['response']
    else:
        print(f"Fehler bei der Anfrage an LLava: {response.status_code}")
        return f"Error: {response.status_code}"

if __name__ == "__main__":
    # Beispiel für die Nutzung des Chatbots
    bot = ImageSearchBot()
    query = "Which picture shows Singapore?"
    result = bot.search_image(query)
    print(result)
    
    # Beispielabfrage nach Bildern basierend auf Beschreibung
    descriptions = ["A photo of Singapore", "A photo of Rome"]
    for description in descriptions:
        print(f"Suche nach Bildern, die {description} zeigen...")
        results = bot.search_image(description)
        print(f"Ergebnis der Abfrage '{description}': {results}")
    
    # Beispielnutzung des zweiten LLM (Llava)
    for filename, base64_image in base64_images.items():
        print(f"Bearbeite Bild {filename} mit LLava...")
        llava_response = query_llava(base64_image, "Please describe the content of this image.")
        print(f"{filename}: {llava_response}")
