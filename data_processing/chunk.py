from chonkie import RecursiveChunker, RecursiveRules, RecursiveLevel
from data_processing.convert import ConversionResults
from typing import Union
from chonkie.tokenizer import TokenizerProtocol
from dataclasses import dataclass

rules = RecursiveRules(
    [
        # Headers du plus fort au plus faible
        RecursiveLevel(delimiters=["\n# "], include_delim="next"),
        RecursiveLevel(delimiters=["\n## "], include_delim="next"),
        RecursiveLevel(delimiters=["\n### "], include_delim="next"),
        RecursiveLevel(delimiters=["\n#### "], include_delim="next"),
        # Séparateurs markdown
        RecursiveLevel(delimiters=["\n---\n", "\n___\n"], include_delim="next"),
        # Paragraphes
        RecursiveLevel(delimiters=["\n\n"], include_delim="prev"),
        # Phrases (FR)
        RecursiveLevel(delimiters=[". ", "! ", "? ", "… "], include_delim="prev"),
        # Micro pauses
        RecursiveLevel(delimiters=["; ", ": ", ", "], include_delim="prev"),
        # Mots
        RecursiveLevel(whitespace=True, include_delim="prev"),
        # Fallback tokens
        RecursiveLevel(),
    ]
)


@dataclass
class ChunkResults:
    """
    Dataclass to store the results of the chunking process

    Attributes:
        filename (str): The name of the file that was chunked
        chunking_method (str): The method used to chunk the text
        chunks (list[str]): The list of chunks that were created
    """

    filename: str
    chunking_method: str
    chunks: list[str]


class Chunker:
    """
    Class to chunk a text using the RecursiveChunker class from the chonkie library

    Args:
        conversion_results (ConversionResults): The results of the conversion process
        rules (RecursiveRules): The rules to use for chunking the text
        max_chunk_size (int): The maximum size of each chunk
        min_chars_per_chunk (int): The minimum number of characters per chunk
        tokenizer (Union[str, TokenizerProtocol]): The tokenizer to use for chunking the text
    """

    def __init__(
        self,
        conversion_results: ConversionResults,
        rules: RecursiveRules = rules,
        max_chunk_size: int = 512,
        min_chars_per_chunk: int = 48,
        tokenizer: Union[str, TokenizerProtocol] = "character",
    ):
        self.filename = conversion_results.filename
        self.text = conversion_results.result
        self.rules = rules
        self.max_chunk_size = max_chunk_size
        self.min_chars_per_chunk = min_chars_per_chunk
        self.tokenizer = tokenizer

    def recursive_chunk(self) -> list[str]:
        """
        Chunk the text using the RecursiveChunker class from the chonkie library

        Returns:
            list[str]: The list of chunks that were created
        """
        self.chunking_method = "recursive"
        chunker = RecursiveChunker(
            tokenizer=self.tokenizer,
            rules=self.rules,
            chunk_size=self.max_chunk_size,
            min_characters_per_chunk=self.min_chars_per_chunk,
        )
        chunks = chunker.chunk(self.text)
        return [chunk.text.strip() for chunk in chunks]

    def __call__(self) -> ChunkResults:
        """
        Call the chunking process and return the results

        Returns:
            ChunkResults: The results of the chunking process
        """
        chunks = self.recursive_chunk()
        return ChunkResults(
            filename=self.filename,
            chunking_method=self.chunking_method,
            chunks=chunks,
        )


class FixedSizeChunker:
    """Split plain text into fixed-size character chunks with optional overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 0, min_chars: int = 48):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chars = min_chars

    def chunk(self, text: str, doc_id: str = "") -> ChunkResults:
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        start = 0
        while start < len(text):
            piece = text[start : start + self.chunk_size].strip()
            if len(piece) >= self.min_chars:
                chunks.append(piece)
            start += step
        return ChunkResults(
            filename=doc_id,
            chunking_method=f"fixed_{self.chunk_size}",
            chunks=chunks,
        )
