from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from dataclasses import dataclass
import re
import os


@dataclass
class ConversionResults:
    """
    Dataclass to store the results of the conversion process

    Attributes:
        filename (str): The name of the file that was converted
        conversion_type (str): The type of conversion that was performed
        result (str): The result of the conversion process
    """

    filename: str
    conversion_type: str
    result: str


class DoclingConverter:
    """
    Class to convert a file using the DocumentConverter class from the docling library

    Args:
        pathname (str): The path to the file to be converted
    """

    conversion_method = "docling"

    def __init__(self, pathname: str):
        """
        Initialize the DoclingConverter class with a file

        Args:
            pathname (str): The path to the file to be converted
        """
        self.pathname = pathname
        self.filename = os.path.basename(pathname)

    def get_text(self) -> str:
        """
        Extract text from the file using the DocumentConverter class from the docling library

        Returns:
            str: The extracted text from the file
        """
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        conv_results = doc_converter.convert(self.pathname)
        return conv_results.document.export_to_markdown()

    def delete_useless_line_jumps(self, text: str, separator: str) -> str:
        """
        Supression of useless line jumps

        Args:
            text (str): The text to be processed
            separator (str): The seprator to be used for the line jumps

        Returns:
            str: The text with the useless line jumps removed
        """
        cleaned_text = re.sub(
            f"([a-zà-ÿ]){separator}[\n\r ]+{separator}([a-zà-ÿ])", r"\1 \2", text
        )
        return cleaned_text

    def cleaning(self, text: str) -> str:
        """
        Cleaning of the text by removing useless characters

        Args:
            text (str): The text to be cleaned

        Returns:
            str: The cleaned text
        """
        text = re.sub(r"(\n<!-- image -->\n|\n\d+\n|\n-\n|ʳ ʳ|\n→\n|\n\|\n)", "", text)
        text = re.sub(r"- · ", "    - ", text)
        text = re.sub(r"(- -|- ■ |-  |- ▪ |- Ŷ )", "- ", text)
        text = re.sub(r" → ", " ", text)
        text = re.sub(r"�", ".", text)
        separator = r""
        cleaned_text = self.delete_useless_line_jumps(text, separator)
        return cleaned_text

    def __call__(self) -> ConversionResults:
        result = self.get_text()
        cleaned_result = self.cleaning(result)
        return ConversionResults(
            filename=self.filename,
            conversion_type=self.conversion_method,
            result=cleaned_result,
        )
