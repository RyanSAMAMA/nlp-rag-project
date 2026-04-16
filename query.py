from retrieving.retrieve import retrieve
from generating.generate import generate, generate_comparison
from config import COLLECTION_NAME, get_qdrant_client
import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interroge le corpus Archelec",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Exemples :
  uv run python query.py "Question libre"
  uv run python query.py "Question" --annee 1981
  uv run python query.py "Question" --compare 1967 1981
  uv run python query.py "Question" --parti "communiste"
  uv run python query.py "Question" --departement "Loire"
  uv run python query.py "Question" --candidat "Marchais"
  uv run python query.py "Question" --compare 1967 1981 --parti "socialiste"
  uv run python query.py "Question" --annee 1967 --departement "Ain" --parti "communiste"
""",
    )
    parser.add_argument("question", nargs="+", help="La question à poser")
    parser.add_argument("--annee", type=str, default=None, help="Filtre par année (ex: 1967)")
    parser.add_argument("--compare", nargs=2, metavar=("ANNEE_A", "ANNEE_B"),
                        help="Compare deux années (ex: --compare 1967 1981)")
    parser.add_argument("--parti", type=str, default=None,
                        help="Filtre par parti (recherche partielle, ex: 'communiste')")
    parser.add_argument("--departement", type=str, default=None,
                        help="Filtre par département (ex: 'Loire')")
    parser.add_argument("--candidat", type=str, default=None,
                        help="Filtre par nom de candidat (ex: 'Marchais')")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    question = " ".join(args.question)
    client = get_qdrant_client()
    filters = dict(parti=args.parti, departement=args.departement, candidat=args.candidat)

    if args.compare:
        annee_a, annee_b = args.compare
        results_a = retrieve(question, client, annee=annee_a, **filters)
        results_b = retrieve(question, client, annee=annee_b, **filters)
        if not results_a and not results_b:
            print("Aucun document pertinent trouvé.")
            return
        print(generate_comparison(question, results_a, results_b, annee_a, annee_b))
    else:
        results = retrieve(question, client, annee=args.annee, **filters)
        if not results:
            print("Aucun document pertinent trouvé.")
            return
        print(generate(question, results))


if __name__ == "__main__":
    main()
