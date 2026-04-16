from config import get_qdrant_client
import sys


def cmd_list():
    client = get_qdrant_client()
    collections = client.get_collections().collections
    if not collections:
        print("Aucune collection.")
        return
    for col in collections:
        count = client.count(col.name).count
        print(f"  {col.name} — {count} points")


def cmd_delete(collection_name: str):
    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        print(f"Collection '{collection_name}' introuvable.")
        sys.exit(1)
    client.delete_collection(collection_name)
    print(f"Collection '{collection_name}' supprimée.")


USAGE = """Usage: uv run python manage.py <commande> [args]

Commandes :
  list                        Lister les collections et leur nombre de points
  delete <collection_name>    Supprimer une collection
"""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        cmd_list()
    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Usage: uv run python manage.py delete <collection_name>")
            sys.exit(1)
        cmd_delete(sys.argv[2])
    else:
        print(f"Commande inconnue : '{cmd}'\n{USAGE}")
        sys.exit(1)
