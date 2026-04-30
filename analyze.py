"""
Analyse du corpus avec la config Hybrid RRF.

Usage:
    uv run python analyze.py
    uv run python analyze.py --study 1
"""
import argparse
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import get_qdrant_client
from corpus.parties import normalize
from retrieving.hybrid_retrieve import retrieve_hybrid
from config import EMBEDDING_MODEL

load_dotenv()

RESULTS_PATH = Path("results/corpus_analysis.json")
MAX_CHUNKS = 25      # chunks fed to GPT per analysis call
CHUNK_SAMPLE_SEED = 42


LEFT_PARTIES  = {"PCF", "PS", "FGDS", "PSU", "LO", "MRG"}
RIGHT_PARTIES = {"RPR", "UDF", "CD", "RI", "Lib."}



def _fetch_all(client):
    records, offset = [], None
    while True:
        batch, offset = client.scroll(
            "documents", limit=500, offset=offset,
            with_payload=True, with_vectors=False,
        )
        records.extend(batch)
        if offset is None:
            break
    return records


def _sample_chunks(records, n, seed=CHUNK_SAMPLE_SEED):
    rng = random.Random(seed)
    return rng.sample(records, min(n, len(records)))


def _build_context(chunks):
    parts = []
    for r in chunks:
        p = r.payload
        party = normalize(p.get("titulaire_soutien") or "")
        year  = p.get("annee", "?")
        dept  = p.get("departement_nom", "?")
        nom   = p.get("titulaire_nom", "")
        text  = p.get("chunk", "").strip()
        parts.append(f"[{year} | {party} | {dept} | {nom}]\n{text}")
    return "\n\n---\n\n".join(parts)


def _ask(client: OpenAI, context: str, question: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": (
                "Tu es un expert en histoire politique française. "
                "Réponds en français, de façon analytique et structurée, "
                "en t'appuyant uniquement sur les extraits fournis. "
                "Cite des exemples précis tirés du texte."
            )},
            {"role": "user", "content": f"Extraits de professions de foi :\n\n{context}\n\n---\n\n{question}"},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()



def study1_temporal(all_records, openai_client):
    print("\n--- étude 1 : évolution temporelle 1967 / 1981 ---")
    results = {}

    themes = {
        "économie et emploi": "Quels sont les principaux thèmes économiques et les propositions sur l'emploi dans ces professions de foi ?",
        "politique sociale":  "Comment les candidats abordent-ils les questions sociales (éducation, santé, logement) dans ces professions de foi ?",
        "institutions et démocratie": "Quelles propositions institutionnelles ou démocratiques les candidats avancent-ils dans ces professions de foi ?",
    }

    for year in ["1967", "1981"]:
        year_records = [r for r in all_records if r.payload.get("annee") == year]
        sample = _sample_chunks(year_records, MAX_CHUNKS)
        context = _build_context(sample)
        results[year] = {}
        for theme, question in themes.items():
            print(f"  [{year}] {theme}...", end=" ", flush=True)
            answer = _ask(openai_client, context, question)
            results[year][theme] = answer
            print("done")

    return results



def study2_left_right(all_records, openai_client):
    print("\n--- étude 2 : gauche / droite ---")
    results = {}

    left_records  = [r for r in all_records if normalize(r.payload.get("titulaire_soutien") or "") in LEFT_PARTIES]
    right_records = [r for r in all_records if normalize(r.payload.get("titulaire_soutien") or "") in RIGHT_PARTIES]

    print(f"  Left chunks: {len(left_records)} | Right chunks: {len(right_records)}")

    questions = {
        "thèmes centraux": "Quels sont les trois ou quatre thèmes les plus récurrents dans ces professions de foi ? Donnez des exemples de formulations typiques.",
        "valeurs et références": "Quelles valeurs, références historiques ou culturelles ces candidats mobilisent-ils pour justifier leurs propositions ?",
        "style rhétorique": "Comment ces candidats structurent-ils leur discours ? Y a-t-il des procédés rhétoriques caractéristiques (appel au peuple, attaque de l'adversaire, promesses chiffrées) ?",
    }

    for side, records in [("gauche", left_records), ("droite", right_records)]:
        sample = _sample_chunks(records, MAX_CHUNKS)
        context = _build_context(sample)
        results[side] = {}
        for theme, question in questions.items():
            print(f"  [{side}] {theme}...", end=" ", flush=True)
            answer = _ask(openai_client, context, question)
            results[side][theme] = answer
            print("done")

    # Comparative synthesis
    print("  [comparaison] synthèse...", end=" ", flush=True)
    left_sample  = _build_context(_sample_chunks(left_records,  12))
    right_sample = _build_context(_sample_chunks(right_records, 12))
    synthesis_context = f"=== GAUCHE ===\n{left_sample}\n\n=== DROITE ===\n{right_sample}"
    results["comparaison"] = _ask(
        openai_client, synthesis_context,
        "En comparant les extraits de gauche et de droite, quelles sont les différences les plus marquantes "
        "en termes de thèmes prioritaires, de valeurs mobilisées et de style discursif ?"
    )
    print("done")
    return results



def study3_female(all_records, openai_client):
    print("\n--- étude 3 : candidates femmes ---")

    female_records = [r for r in all_records if r.payload.get("titulaire_sexe") == "F"]
    male_records   = [r for r in all_records if r.payload.get("titulaire_sexe") == "H"]

    print(f"  Female chunks: {len(female_records)} | Male sample: {MAX_CHUNKS}")

    questions = {
        "thèmes spécifiques": (
            "Quels thèmes ces candidates mettent-elles particulièrement en avant ? "
            "Y a-t-il des sujets (famille, droits des femmes, travail féminin, paix) "
            "plus présents que dans un discours masculin typique ?"
        ),
        "rapport au genre": (
            "Ces candidates évoquent-elles explicitement leur condition de femme candidate, "
            "les droits des femmes, ou la parité politique ?"
        ),
        "partis et positionnement": (
            "Ces candidates appartiennent toutes à des partis de gauche (PCF, PS, LO, PSU). "
            "Comment leur discours reflète-t-il à la fois leur appartenance partisane et leur identité de femme candidate ?"
        ),
    }

    results = {"femmes": {}, "hommes": {}, "comparaison": ""}

    female_sample = _sample_chunks(female_records, MAX_CHUNKS)
    female_context = _build_context(female_sample)
    for theme, question in questions.items():
        print(f"  [femmes] {theme}...", end=" ", flush=True)
        results["femmes"][theme] = _ask(openai_client, female_context, question)
        print("done")

    # Male reference sample (same size, same left parties for fairness)
    left_male = [r for r in male_records
                 if normalize(r.payload.get("titulaire_soutien") or "") in LEFT_PARTIES]
    male_sample = _sample_chunks(left_male, MAX_CHUNKS)
    male_context = _build_context(male_sample)
    print("  [hommes gauche] thèmes...", end=" ", flush=True)
    results["hommes"]["thèmes centraux"] = _ask(
        openai_client, male_context,
        "Quels sont les thèmes principaux de ces professions de foi de candidats hommes de gauche ?"
    )
    print("done")

    # Comparative
    print("  [comparaison] synthèse...", end=" ", flush=True)
    comp_context = f"=== CANDIDATES FEMMES ===\n{female_context}\n\n=== CANDIDATS HOMMES (gauche) ===\n{male_context}"
    results["comparaison"] = _ask(
        openai_client, comp_context,
        "En comparant les professions de foi des candidates femmes et des candidats hommes de gauche, "
        "quelles différences thématiques et rhétoriques observez-vous ? "
        "Les femmes se distinguent-elles par des priorités ou un style particulier ?"
    )
    print("done")
    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=int, choices=[1, 2, 3], default=None)
    args = parser.parse_args()

    client = get_qdrant_client()
    openai_client = OpenAI()

    print("Fetching all corpus chunks...")
    all_records = _fetch_all(client)
    print(f"  {len(all_records)} chunks loaded")

    results = {}
    Path("results").mkdir(exist_ok=True)

    run = [1, 2, 3] if args.study is None else [args.study]

    if 1 in run:
        results["study1_temporal"] = study1_temporal(all_records, openai_client)
    if 2 in run:
        results["study2_left_right"] = study2_left_right(all_records, openai_client)
    if 3 in run:
        results["study3_female"] = study3_female(all_records, openai_client)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
