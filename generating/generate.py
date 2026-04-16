from openai import OpenAI
from retrieving.retrieve import RetrievalResult
from dotenv import load_dotenv
import os

load_dotenv()

MODEL = "gpt-4.1-nano"

SYSTEM_PROMPT = """Tu es un assistant qui répond aux questions en se basant uniquement sur le contexte fourni.
Si la réponse ne se trouve pas dans le contexte, dis-le clairement.
Cite les passages pertinents pour appuyer ta réponse."""

COMPARE_PROMPT = """Tu es un assistant spécialisé en histoire politique française.
On te fournit des extraits de professions de foi de candidats aux élections législatives, organisés par année.
Compare le discours des candidats entre les deux années sur le thème posé.
Structure ta réponse en trois parties :
1. Ce qui se dit en {annee_a}
2. Ce qui se dit en {annee_b}
3. Ce qui a changé entre les deux
Cite des passages précis pour illustrer chaque point."""


def generate(query: str, results: list[RetrievalResult]) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY n'est pas définie dans le .env")

    context = "\n\n".join(
        f"[Candidat: {r.candidat_prenom or ''} {r.candidat_nom or '?'} | Année: {r.annee or '?'} | Parti: {r.parti or '?'} | Département: {r.departement_nom or '?'}]\n{r.chunk}"
        for r in results
    )

    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {query}"},
        ],
    )
    return response.choices[0].message.content


def generate_comparison(query: str, results_a: list[RetrievalResult], results_b: list[RetrievalResult], annee_a: str, annee_b: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY n'est pas définie dans le .env")

    def fmt(results: list[RetrievalResult], annee: str) -> str:
        header = f"{'─'*10} EXTRAITS {annee} {'─'*10}"
        body = "\n\n".join(
            f"[Candidat: {r.candidat_prenom or ''} {r.candidat_nom or '?'} | Parti: {r.parti or '?'} | Département: {r.departement_nom or '?'}]\n{r.chunk}"
            for r in results
        )
        return f"{header}\n{body}"

    context = fmt(results_a, annee_a) + "\n\n" + fmt(results_b, annee_b)
    system = COMPARE_PROMPT.format(annee_a=annee_a, annee_b=annee_b)

    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Contexte :\n{context}\n\nQuestion : {query}"},
        ],
    )
    return response.choices[0].message.content
