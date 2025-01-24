import logging
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.base import ImageRefMode
from docling.chunking import HybridChunker
from docling_core.transforms.chunker import HierarchicalChunker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessingConfig(BaseModel):
    """
    Configuration for the PDFProcessor.

    Attributes:
        pdf_file (str): The name of the PDF file to process.
        main_folder (Path): The main folder where input and output directories are located.
        input_folder (Path): The folder containing the input PDF file.
        output_folder (Path): The folder where processed files will be saved.
        chunker_type (str): Type of chunker to use - "hybrid" or "hierarchical".
        tokenizer (str): Tokenizer model to use for hybrid chunking.
    """
    pdf_file: str
    main_folder: Path = Field(default=Path("data"))
    input_folder: Path = Field(default_factory=lambda: Path("data/input"))
    output_folder: Path = Field(default_factory=lambda: Path("data/output"))
    chunker_type: Literal["hybrid", "hierarchical"] = "hybrid"
    tokenizer: str = "sdadas/st-polish-paraphrase-from-distilroberta"

    @property
    def source(self) -> Path:
        return self.input_folder / self.pdf_file

    @field_validator("input_folder", "output_folder", mode="before")
    def ensure_directory_exists(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

class PDFProcessor:
    """
    A unified class to handle PDF processing, including conversion to Markdown and chunking.
    """
    def __init__(self, config: PDFProcessingConfig):
        """
        Initialize the PDFProcessor class with configuration.

        Args:
            config (PDFProcessingConfig): Configuration object for the processor.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_converter(self, embedded_images=False):
        """
        Set up and return a DocumentConverter instance.

        Args:
            embedded_images (bool): Whether to include embedded images in the conversion.

        Returns:
            DocumentConverter: An instance configured for PDF to Markdown conversion.
        """
        pipeline_options = PdfPipelineOptions()
        if embedded_images:
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def convert_to_markdown(self, embedded_images=False):
        """
        Convert the PDF to Markdown format with optional image embedding.

        Args:
            embedded_images (bool): Whether to include embedded images in the Markdown file.

        Returns:
            Path: The path to the saved Markdown file.
        """
        try:
            converter = self._get_converter(embedded_images=embedded_images)
            result = converter.convert(self.config.source)

            output_filename = f"{Path(self.config.pdf_file).stem}-{'embedded' if embedded_images else 'placeholder'}.md"
            output_path = self.config.output_folder / output_filename

            image_mode = ImageRefMode.EMBEDDED if embedded_images else ImageRefMode.PLACEHOLDER
            result.document.save_as_markdown(output_path, image_mode=image_mode)

            self.logger.info(f"Markdown file saved at: {output_path}")
            return output_path, result.document

        except Exception as e:
            self.logger.error(f"An error occurred during conversion: {e}")
            raise

    def chunk_document(self, doc):
        """
        Chunk the document using the specified chunking method.

        Args:
            doc: The document to chunk.

        Returns:
            Iterator: An iterator over the document chunks.
        """
        if self.config.chunker_type == "hybrid":
            chunker = HybridChunker(tokenizer=self.config.tokenizer)
        else:
            chunker = HierarchicalChunker()

        return chunker.chunk(doc)

    def save_chunks(self, chunks, output_path):
        """
        Save the document chunks to a Markdown file.

        Args:
            chunks: Iterator of document chunks.
            output_path (Path): Path where to save the chunks.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as md_file:
                for idx, chunk in enumerate(chunks):
                    md_file.write(f"\n## Fragment {idx + 1}\n\n")
                    md_file.write(f"### Tekst:\n\n{chunk.text}\n\n")

                    if hasattr(chunk, 'meta') and chunk.meta:
                        metadata = chunk.meta.model_dump()
                        
                        if self.config.chunker_type == "hybrid":
                            filename = metadata.get("origin", {}).get("filename", "Unknown")
                            headings = metadata.get("headings", [])
                            heading_str = headings[0] if headings else ""
                            chunk_path = f"{filename}/{heading_str}" if heading_str else filename
                            metadata["chunkpath"] = chunk_path
                        
                        md_file.write("### Metadane:\n")
                        for key, value in metadata.items():
                            md_file.write(f"- **{key}**: {value}\n")
                    
                    if self.config.chunker_type == "hierarchical" and hasattr(chunk, 'path'):
                        md_file.write(f"\n### Ścieżka: {chunk.path}\n")
                    
                    md_file.write("\n---\n")

            self.logger.info(f"Chunks saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"An error occurred while saving chunks: {e}")
            raise

    def process_document(self, embedded_images=False):
        """
        Process the document: convert to Markdown and create chunks.

        Args:
            embedded_images (bool): Whether to include embedded images in the Markdown file.
        """
        try:
            # First convert the document
            markdown_path, doc = self.convert_to_markdown(embedded_images)
            
            # Then create and save chunks
            chunks_output_path = self.config.output_folder / f"{Path(self.config.pdf_file).stem}-{self.config.chunker_type}-chunked.md"
            chunks = self.chunk_document(doc)
            self.save_chunks(chunks, chunks_output_path)
            
            return markdown_path, chunks_output_path

        except Exception as e:
            self.logger.error(f"An error occurred during document processing: {e}")
            raise

# Usage example
if __name__ == "__main__":
    # Create configuration with hybrid chunking
    config = PDFProcessingConfig(
        pdf_file="IKO.pdf",
        chunker_type="hybrid"
    )
    processor = PDFProcessor(config)

    try:
        # Process document with hybrid chunking and embedded images
        markdown_path, chunks_path = processor.process_document(embedded_images=True)
        print(f"Markdown file (with embedded images) saved at: {markdown_path}")
        print(f"Chunks saved at: {chunks_path}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")