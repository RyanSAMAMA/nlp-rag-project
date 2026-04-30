"""Métriques de retrieval et scoring LLM-as-judge. Scores dans [0, 1]."""
import re
from openai import OpenAI


def _doc_id(chunk_id: str) -> str:
    """Strip _PF_XX suffix to get document-level identifier."""
    parts = chunk_id.rsplit("_PF_", 1)
    return parts[0] if len(parts) == 2 else chunk_id


def _is_relevant(retrieved_id: str, source_doc_id: str) -> bool:
    """True if the retrieved chunk belongs to the source document."""
    if not source_doc_id:
        return False
    return _doc_id(retrieved_id) == source_doc_id


def hit(retrieved_filenames: list[str], source_doc_id: str) -> bool:
    """True if any retrieved chunk belongs to the source document."""
    return any(_is_relevant(fn, source_doc_id) for fn in retrieved_filenames)


def reciprocal_rank(retrieved_filenames: list[str], source_doc_id: str) -> float:
    """1/rank of first relevant result, 0 if not found."""
    for i, fn in enumerate(retrieved_filenames):
        if _is_relevant(fn, source_doc_id):
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(retrieved_filenames: list[str], source_doc_id: str) -> float:
    """Fraction of retrieved chunks that belong to the source document."""
    if not retrieved_filenames:
        return 0.0
    return sum(1 for fn in retrieved_filenames if _is_relevant(fn, source_doc_id)) / len(retrieved_filenames)


_FAITHFULNESS_PROMPT = """Évalue la FIDÉLITÉ de cette réponse par rapport au contexte fourni.
Une réponse fidèle ne contient que des informations présentes dans le contexte.

Question : {question}

Contexte (extraits récupérés) :
{context}

Réponse générée :
{answer}

Score entre 0.0 (contient des informations inventées) et 1.0 (entièrement fondée sur le contexte).
Réponds UNIQUEMENT avec un nombre décimal, exemple : 0.85"""

_RELEVANCE_PROMPT = """Évalue si cette réponse répond CORRECTEMENT à la question posée.

Question : {question}

Réponse : {answer}

Score entre 0.0 (hors sujet ou vide) et 1.0 (répond précisément à la question).
Réponds UNIQUEMENT avec un nombre décimal, exemple : 0.85"""


def _parse_score(text: str) -> float:
    text = text.strip()
    match = re.search(r"[0-9]+(?:[.,][0-9]+)?", text)
    if not match:
        return 0.5
    try:
        val = float(match.group().replace(",", "."))
        return max(0.0, min(1.0, val))
    except ValueError:
        return 0.5


def score_faithfulness(client: OpenAI, question: str, context: str, answer: str) -> float:
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            max_tokens=16,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": _FAITHFULNESS_PROMPT.format(
                        question=question,
                        context=context[:2500],
                        answer=answer,
                    ),
                }
            ],
        )
        return _parse_score(resp.choices[0].message.content)
    except Exception:
        return 0.5


def score_relevance(client: OpenAI, question: str, answer: str) -> float:
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            max_tokens=16,
            temperature=0.0,
            messages=[
                {
                    "role": "user",
                    "content": _RELEVANCE_PROMPT.format(question=question, answer=answer),
                }
            ],
        )
        return _parse_score(resp.choices[0].message.content)
    except Exception:
        return 0.5
