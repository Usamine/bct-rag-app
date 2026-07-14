import chromadb
import os

# 1. Connexion à votre base Chroma (ajustez le chemin si votre dossier s'appelle différemment, ex: "./chroma_db")
CHROMA_DATA_PATH = "./data/index/chroma" 

if not os.path.exists(CHROMA_DATA_PATH):
    print(f"❌ Le dossier {CHROMA_DATA_PATH} n'existe pas. Vérifiez le chemin dans votre projet.")
    exit()

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# 2. Lister les collections disponibles
collections = client.list_collections()
print(f"📚 Collections trouvées ({len(collections)}) :")
for col in collections:
    print(f" - {col.name}")

if not collections:
    print("❌ Aucune collection trouvée dans la base de données.")
    exit()

# 3. Inspecter la collection principale
# (Remplacez 'circulars' ou utilisez la première trouvée si votre collection s'appelle autrement)
collection_name = collections[0].name 
collection = client.get_collection(name=collection_name)

# Récupérer toutes les métadonnées pour extraire les noms de fichiers uniques
results = collection.get(include=["metadatas"])

if results and results["metadatas"]:
    # On extrait le nom du fichier (source) de chaque morceau de texte
    files_in_db = set()
    for meta in results["metadatas"]:
        # S'adapte que la clé soit 'source', 'file_name' ou 'path'
        source = meta.get("source") or meta.get("file_name") or meta.get("path") or "Inconnu"
        files_in_db.add(os.path.basename(source))
    
    print(f"\n✅ Total de documents distincts trouvés dans '{collection_name}' : {len(files_in_db)}")
    print("\n📋 Liste des fichiers présents dans la base :")
    for file in sorted(files_in_db):
        # On met en valeur les fichiers de 2026 pour vérifier rapidement
        if "2026" in file:
            print(f" ⭐ [2026] {file}")
        else:
            print(f" 📄 {file}")
else:
    print("\n⚠️ La collection existe mais elle est vide (0 segment de texte).")