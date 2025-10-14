"""Document handling utilities for various file formats."""

import os
from pathlib import Path
from typing import Tuple, Union
import markdown
from docx import Document
from docx.shared import Inches
import PyPDF2


class DocumentHandler:
    """Handle reading and writing of different document formats."""
    
    SUPPORTED_FORMATS = {'.txt', '.md', '.markdown', '.docx', '.pdf'}
    
    @classmethod
    def read_document(cls, file_path: Union[str, Path]) -> Tuple[str, str]:
        """
        Read document content and return (content, format).
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (content, file_extension)
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = path.suffix.lower()
        
        if file_ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: {cls.SUPPORTED_FORMATS}")
        
        if file_ext == '.txt':
            return cls._read_txt(path), file_ext
        elif file_ext in ['.md', '.markdown']:
            return cls._read_markdown(path), file_ext
        elif file_ext == '.docx':
            return cls._read_docx(path), file_ext
        elif file_ext == '.pdf':
            return cls._read_pdf(path), file_ext
    
    @classmethod
    def write_document(cls, content: str, output_path: Union[str, Path], 
                      original_format: str = '.txt') -> Path:
        """
        Write content to document in specified format.
        
        Args:
            content: Text content to write
            output_path: Path where to save the file
            original_format: Original file format to preserve structure
            
        Returns:
            The actual path where the file was written
        """
        path = Path(output_path)
        
        # Ensure output directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if original_format == '.txt':
            cls._write_txt(content, path)
            return path
        elif original_format in ['.md', '.markdown']:
            cls._write_markdown(content, path)
            return path
        elif original_format == '.docx':
            cls._write_docx(content, path)
            return path
        elif original_format == '.pdf':
            # PDF input -> text output (can't write back to PDF easily)
            actual_path = path.with_suffix('.txt')
            cls._write_txt(content, actual_path)
            return actual_path
        else:
            raise ValueError(f"Cannot write to unsupported format: {original_format}")
    
    @staticmethod
    def _read_txt(path: Path) -> str:
        """Read plain text file."""
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError as e:
            # If UTF-8 decoding fails, show the file content for debugging
            print(f"[ERROR] UTF-8 decoding failed for file: {path}")
            print("[CONTENT] File content (first 500 bytes):")
            try:
                with open(path, 'rb') as f:
                    content = f.read(500)
                    # Safely print binary content
                    try:
                        print(content.decode('utf-8', errors='replace'))
                    except UnicodeEncodeError:
                        print(f"[BINARY] Binary content detected ({len(content)} bytes)")
                    if len(content) == 500:
                        print("... (truncated)")
            except Exception as read_error:
                print(f"[ERROR] Could not read file content: {read_error}")
            raise ValueError(f"File {path} is not valid UTF-8 text. This appears to be a binary file or uses a different encoding. Supported formats: {DocumentHandler.SUPPORTED_FORMATS}") from e
    
    @staticmethod
    def _read_markdown(path: Path) -> str:
        """Read markdown file (return as plain text for translation)."""
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError as e:
            # If UTF-8 decoding fails, show the file content for debugging
            print(f"[ERROR] UTF-8 decoding failed for file: {path}")
            print("[CONTENT] File content (first 500 bytes):")
            try:
                with open(path, 'rb') as f:
                    content = f.read(500)
                    # Safely print binary content
                    try:
                        print(content.decode('utf-8', errors='replace'))
                    except UnicodeEncodeError:
                        print(f"[BINARY] Binary content detected ({len(content)} bytes)")
                    if len(content) == 500:
                        print("... (truncated)")
            except Exception as read_error:
                print(f"[ERROR] Could not read file content: {read_error}")
            raise ValueError(f"File {path} is not valid UTF-8 text. This appears to be a binary file or uses a different encoding. Supported formats: {DocumentHandler.SUPPORTED_FORMATS}") from e
    
    @staticmethod
    def _read_docx(path: Path) -> str:
        """Read DOCX file and extract text content."""
        doc = Document(path)
        content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text)
        
        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    content.append(' | '.join(row_text))
        
        return '\n\n'.join(content)
    
    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Read PDF file and extract text content."""
        content = []
        
        with open(path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    content.append(page_text.strip())
        
        if not content:
            raise ValueError(f"No text could be extracted from PDF: {path}")
        
        return '\n\n'.join(content)
    
    @staticmethod
    def _write_txt(content: str, path: Path) -> None:
        """Write plain text file."""
        path.write_text(content, encoding='utf-8')
    
    @staticmethod
    def _write_markdown(content: str, path: Path) -> None:
        """Write markdown file."""
        # Ensure .md extension
        if path.suffix.lower() not in ['.md', '.markdown']:
            path = path.with_suffix('.md')
        path.write_text(content, encoding='utf-8')
    
    @staticmethod
    def _write_docx(content: str, path: Path) -> None:
        """Write content to DOCX file."""
        # Ensure .docx extension
        if path.suffix.lower() != '.docx':
            path = path.with_suffix('.docx')
            
        doc = Document()
        
        # Split content into paragraphs and add to document
        paragraphs = content.split('\n\n')
        
        for para_text in paragraphs:
            if para_text.strip():
                # Handle potential table-like content (contains |)
                if ' | ' in para_text:
                    # Create table for structured content
                    rows = [row.strip() for row in para_text.split('\n') if ' | ' in row]
                    if rows:
                        cols = rows[0].split(' | ')
                        table = doc.add_table(rows=len(rows), cols=len(cols))
                        table.style = 'Table Grid'
                        
                        for i, row_text in enumerate(rows):
                            cells = row_text.split(' | ')
                            for j, cell_text in enumerate(cells):
                                if j < len(table.rows[i].cells):
                                    table.rows[i].cells[j].text = cell_text.strip()
                else:
                    # Regular paragraph
                    doc.add_paragraph(para_text.strip())
        
        doc.save(path)
    
    @classmethod
    def get_output_filename(cls, input_path: Union[str, Path], step_name: str, 
                           output_dir: Union[str, Path]) -> Path:
        """
        Generate output filename for a pipeline step.
        
        Args:
            input_path: Original input file path
            step_name: Name of the pipeline step
            output_dir: Output directory
            
        Returns:
            Path object for the output file
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        
        # Create document-specific folder: output_dir/document_name/
        document_name = input_path.stem  # filename without extension
        document_dir = output_dir / document_name
        document_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename: step_name.ext (inside document folder)
        ext = input_path.suffix
        
        output_filename = f"{step_name}{ext}"
        return document_dir / output_filename