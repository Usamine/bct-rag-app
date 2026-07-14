import os
import chromadb

# 1. Définition des chemins basés sur votre architecture
PDF_DIR = "./data/pdfs"
CHROMA_DATA_PATH = "./data/index/chroma"

def get_local_pdfs(directory):
    """Parcourt tous les sous-dossiers et récupère le nom des fichiers PDF."""
    pdf_files = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.add(file)
    return pdf_files

def get_db_pdfs(chroma_path):
    """Se connecte à ChromaDB et extrait les noms de fichiers uniques."""
    if not os.path.exists(chroma_path):
        print(f"❌ La base de données est introuvable au chemin : {chroma_path}")
        return set()

    client = chromadb.PersistentClient(path=chroma_path)
    collections = client.list_collections()
    
    if not collections:
        print("❌ Aucune collection trouvée dans la base.")
        return set()

    # On prend la première collection (par ex: 'bct')
    collection = client.get_collection(name=collections[0].name)
    results = collection.get(include=["metadatas"])
    
    db_files = set()
    if results and results["metadatas"]:
        for meta in results["metadatas"]:
            # Récupère le nom du fichier selon la clé utilisée lors de l'ingestion
            source = meta.get("source") or meta.get("file_name") or meta.get("path") or ""
            if source:
                db_files.add(os.path.basename(source))
                
    return db_files

def main():
    print("🔍 Analyse des fichiers locaux...")
    local_pdfs = get_local_pdfs(PDF_DIR)
    print(f"📁 Total des PDF trouvés dans le dossier 'data/pdfs' : {len(local_pdfs)}")

    print("\n🔍 Analyse de la base de données Chroma...")
    db_pdfs = get_db_pdfs(CHROMA_DATA_PATH)
    print(f"🗄️ Total des PDF trouvés dans ChromaDB : {len(db_pdfs)}")

    # 3. Comparaison mathématique (Ce qui est local MAIS PAS dans la DB)
    missing_pdfs = local_pdfs - db_pdfs

    print("\n" + "="*50)
    if missing_pdfs:
        print(f"⚠️ Alerte : {len(missing_pdfs)} PDF sont absents de la base de données !\n")
        print("📋 Liste des fichiers manquants :")
        for pdf in sorted(missing_pdfs):
            # On cherche dans quel sous-dossier se trouve ce fichier pour vous aider
            for root, dirs, files in os.walk(PDF_DIR):
                if pdf in files:
                    folder = os.path.basename(root)
                    print(f" - [{folder}] {pdf}")
                    break
    else:
        print("✅ Félicitations ! Absolument TOUS vos PDF locaux sont indexés dans la base de données.")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()