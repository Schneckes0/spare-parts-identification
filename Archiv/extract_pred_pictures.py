import os
import shutil

def copy_and_rename_images(base_path, target_folder):
    """
    Kopiert Bilder aus bestimmten Unterordnern in einen Zielordner und benennt sie um.
    """
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Zähler für alle Bilder global initialisieren
    global_image_counter = 1

    # Iteriere durch alle Materialordner
    for material_folder in os.listdir(base_path):
        material_path = os.path.join(base_path, material_folder)
        if not os.path.isdir(material_path):
            continue  # Überspringe Dateien

        # Ordnernamen als Materialnummer verwenden
        material_number = material_folder

        # Suche die relevanten Unterordner
        for subfolder_name in ["Testphotos_Software", "Photos_Software_datafeed"]:
            subfolder_path = os.path.join(material_path, subfolder_name)
            if not os.path.exists(subfolder_path):
                continue  # Überspringe nicht vorhandene Unterordner

            # Kopiere alle Dateien aus dem Unterordner
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                if os.path.isfile(file_path):
                    try:
                        # Neues Bild benennen
                        new_name = f"{material_number}_{global_image_counter}{os.path.splitext(filename)[1]}"
                        target_path = os.path.join(target_folder, new_name)

                        # Datei kopieren
                        shutil.copy(file_path, target_path)
                        print(f"Kopiert: {file_path} -> {target_path}")

                        # Zähler erhöhen
                        global_image_counter += 1
                    except Exception as e:
                        print(f"Fehler beim Kopieren von {file_path}: {e}")

# Basisordner und Zielordner definieren
base_path = r"C:\Users\kej9ho\Documents\Bilderkennung\POC_Bilderkennung\Photos_Noerpel"
target_folder = r"C:\Users\kej9ho\Documents\Bilderkennung\data\pred"

# Funktion aufrufen
copy_and_rename_images(base_path, target_folder)
