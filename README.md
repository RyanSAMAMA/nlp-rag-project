# nlp-rag-project

Système RAG appliqué aux professions de foi des élections législatives françaises (corpus Archelec / CEVIPOF), scraped depuis Internet Archive.

## Ce que ça fait

- Scrape les documents OCR depuis l'IA, les chunk et les stocke dans Qdrant avec leurs métadonnées (candidat, parti, année, département)
- Propose trois stratégies de retrieval : dense (e5-small), BM25, hybride RRF
- Génère des réponses via GPT-4.1-nano à partir des chunks récupérés
- Évalue les configurations sur un benchmark auto-généré (hit rate, MRR, faithfulness)
- Analyse thématique du corpus sur trois axes : évolution temporelle, gauche/droite, candidates femmes

## Stack

- Qdrant (base vectorielle)
- `intfloat/multilingual-e5-small` pour les embeddings
- OpenAI GPT-4.1-nano pour la génération et le scoring LLM-as-judge
- `chonkie` pour le chunking récursif
- `uv` pour la gestion des dépendances

## Usage

```bash
# Scraper et ingérer (ex. 200 docs par année)
uv run python scrape.py --years 1967 1981 --per-year 200

# Interroger le corpus
uv run python query.py

# Générer le benchmark d'évaluation
uv run python -m evaluate.benchmark_gen

# Lancer l'évaluation sur toutes les configs
uv run python -m evaluate.run_eval

# Analyse thématique
uv run python analyze.py
```

## Variables d'environnement

Fichier `.env` à la racine :

```
QDRANT_API=...
QDRANT_ENDPOINT=...
OPENAI_API_KEY=...
```
