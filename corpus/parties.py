"""
Party label normalization for the Archelec corpus.
Maps raw multi-alliance labels to canonical short names.
"""

PARTY_MAP: dict[str, str] = {
    # Gauche
    "Parti communiste français": "PCF",
    "Parti socialiste": "PS",
    "Parti socialiste unifié": "PSU",
    "Fédération de la gauche démocrate et socialiste": "FGDS",
    "Lutte ouvrière": "LO",
    "Mouvement des radicaux de gauche": "MRG",

    # Droite / Centre
    "Rassemblement pour la République": "RPR",
    "Union pour la démocratie française": "UDF",
    "Républicains indépendants": "RI",
    "Centre démocrate": "CD",
    "Mouvement réformateur": "MR",
    "Union des démocrates pour la République": "UDR",

    # Divers
    "indépendant": "Ind.",
    "libéral": "Lib.",
    "aucun parti politique": "Ind.",
}

# Keywords used to assign a canonical label when exact match fails
_KEYWORD_MAP: list[tuple[str, str]] = [
    ("communiste",   "PCF"),
    ("socialiste uni", "PSU"),
    ("socialiste",   "PS"),
    ("fédération",   "FGDS"),
    ("rassemblement pour la r", "RPR"),
    ("union pour la démocratie", "UDF"),
    ("républicains indépendants", "RI"),
    ("radicaux de gauche", "MRG"),
    ("lutte ouvrière", "LO"),
    ("centre démocrate", "CD"),
    ("centre", "CD"),
    ("mouvement réformateur", "MR"),
    ("union des démocrates", "UDR"),
    ("indépendant", "Ind."),
]


def normalize(raw: str) -> str:
    """Return a short canonical party name for a raw label."""
    if not raw or not raw.strip():
        return "N/R"

    # Exact match
    if raw in PARTY_MAP:
        return PARTY_MAP[raw]

    raw_lower = raw.lower()

    # Alliance: "A / B / C" → keep the first recognized component
    parts = [p.strip() for p in raw.split("/")]
    for part in parts:
        if part in PARTY_MAP:
            return PARTY_MAP[part]
        for keyword, label in _KEYWORD_MAP:
            if keyword in part.lower():
                return label

    # Full string keyword scan
    for keyword, label in _KEYWORD_MAP:
        if keyword in raw_lower:
            return label

    return "Autre"
