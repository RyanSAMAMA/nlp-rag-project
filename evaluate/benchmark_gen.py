"""
Generate evaluation benchmark from the Archelec Qdrant corpus.
Samples document chunks by year and uses GPT to produce question/answer pairs.
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import COLLECTION_NAME, get_qdrant_client
from evaluate.metrics import _doc_id

load_dotenv()

BENCHMARK_PATH = Path("benchmarks/eval_questions.json")

_GENERATION_PROMPT = """Tu analyses des professions de foi de candidats aux élections législatives françaises.

Voici un extrait d'une profession de foi :
- Candidat : {prenom} {nom}
- Année : {annee}
- Parti : {parti}
- Département : {departement}

Texte :
{chunk}

Génère exactement {n} questions à partir de ce texte. Règles strictes :
1. Ne jamais écrire "le candidat" ou "ce candidat" sans nommer explicitement la personne.
2. Alterner entre deux types :
   - TYPE A (question nommée) : mentionne explicitement "{prenom} {nom}" et "{annee}".
     Exemple : "Quelle est la position de {prenom} {nom} sur l'emploi dans sa profession de foi de {annee} ?"
   - TYPE B (question thématique) : question générale sur un thème politique, sans référence \
à un candidat précis ni à "la profession de foi", utilisable sur tout le corpus.
     Exemple : "Comment les candidats des élections de {annee} justifient-ils leur programme économique ?"
     INTERDIT dans le type B : "la profession de foi", "ce document", "ce texte", "le candidat".
3. Les questions doivent porter sur des thèmes politiques concrets (économie, social, éducation, \
sécurité, agriculture, Europe…).
4. La réponse de référence doit se baser uniquement sur le texte fourni.

Réponds UNIQUEMENT avec un tableau JSON valide, sans markdown :
[
  {{"question": "...", "answer": "...", "theme": "...", "type": "named" | "thematic"}},
  ...
]"""


def _scroll_by_year(client, collection_name: str, annee: str, limit: int) -> list[dict]:
    records, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="annee", match=MatchValue(value=annee))]
        ),
        limit=limit * 3,
        with_payload=True,
        with_vectors=False,
    )
    # garder les chunks long
    rich = [r for r in records if len(r.payload.get("chunk", "")) >= 150]
    random.shuffle(rich)
    return rich[:limit]


def _generate_qs(openai_client: OpenAI, doc: dict, n: int) -> list[dict]:
    chunk = doc.get("chunk", "")[:1500]
    if len(chunk) < 150:
        return []

    prompt = _GENERATION_PROMPT.format(
        prenom=doc.get("titulaire_prenom") or "",
        nom=doc.get("titulaire_nom") or "?",
        annee=doc.get("annee") or "?",
        parti=doc.get("titulaire_soutien") or "?",
        departement=doc.get("departement_nom") or "?",
        chunk=chunk,
        n=n,
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            max_tokens=1024,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()
        # strip
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"    [warn] question generation failed: {e}")
        return []


def generate_benchmark(
    n_docs_per_year: int = 15,
    questions_per_doc: int = 2,
    seed: int = 42,
) -> list[dict]:
    """Sample documents from Qdrant and generate evaluation Q&A pairs."""
    random.seed(seed)
    BENCHMARK_PATH.parent.mkdir(exist_ok=True)

    qdrant = get_qdrant_client()
    openai_client = OpenAI()

    benchmark: list[dict] = []
    q_id = 0

    for annee in ["1967", "1981"]:
        print(f"\nSampling {n_docs_per_year} docs from {annee}…")
        records = _scroll_by_year(qdrant, COLLECTION_NAME, annee, n_docs_per_year)
        print(f"  → {len(records)} chunks kept after filtering")

        for i, rec in enumerate(records):
            payload = rec.payload
            print(
                f"  [{i + 1}/{len(records)}] {payload.get('filename', '?')[:50]}…",
                end=" ",
            )

            qs = _generate_qs(openai_client, payload, questions_per_doc)
            print(f"{len(qs)} questions")

            for q in qs:
                if not q.get("question") or not q.get("answer"):
                    continue
                benchmark.append(
                    {
                        "id": f"q{q_id:03d}",
                        "question": q["question"],
                        "reference_answer": q["answer"],
                        "theme": q.get("theme", ""),
                        "type": q.get("type", "named"),
                        "source_doc_id": _doc_id(payload.get("id", "")),
                        "source_chunk_id": payload.get("id", ""),
                        "annee": payload.get("annee", ""),
                        "parti": payload.get("titulaire_soutien", ""),
                        "departement": payload.get("departement_nom", ""),
                        "candidat_nom": payload.get("titulaire_nom", ""),
                    }
                )
                q_id += 1

    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    print(f"\n{len(benchmark)} questions saved at {BENCHMARK_PATH}")
    return benchmark


if __name__ == "__main__":
    generate_benchmark()
