import os
from abc import ABC, abstractmethod
from pydantic import BaseModel
import extract_msg
from PIL import Image
import base64
from typing import List, Optional
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
# Disable INFO-level logging for the 'extract_msg' module
logging.getLogger("extract_msg").setLevel(logging.WARNING)

_MSG_FILE = [".msg"]
_EXCEL_FILE = [".xls", ".xlsx"]
_WORD_FILE = [".doc", ".docx"]
_TEXT_FILE = [".txt"]
_IMAGE_FILE = [".png", ".jpg", ".jpeg", ".bmp", ".gif"]
_PDF_FILE = [".pdf"]
_TIFF_FILE = [".tif"]


class FileMetadata(BaseModel):
    """
    A Pydantic model for file metadata.
    """
    original_file_path: str
    file_name: str
    base64_file_path: Optional[str]
    parent_file_path: Optional[str]
    mime_type: Optional[str]
    error: Optional[str]


class IFileProcessor(ABC):
    """
    Abstract base class for file processors. Defines the common interface.
    """

    def __init__(self, file_path: str):
        self._logger = logging.getLogger(__name__)
        self._file_path = file_path
        self.tmp_dir = self._create_dir(file_path, "tmp_dir")
        self.CACHE_DIR = os.path.join(os.path.dirname(__file__), "__filecache__")
        self._mime_type = self.get_mime_type()
        
        try:
            self._children_file_metadata = None
            self._file_metadata = self.process()
        finally:
            self._remove_tmp_dir(file_path)
    @property
    def logger(self) -> logging.Logger:
        """
        Get the logger object.

        Returns:
            logging.Logger: The logger object.
        """
        return self._logger

    @property
    def file_path(self) -> str:
        """
        Get the path to the file.

        Returns:
            str: The path to the file.
        """
        return self._file_path

    @property
    def tmp_dir(self) -> str:
        """
        Get the path to the temporary file processing directory.

        Returns:
            str: The path temporary directory.
        """
        return self._tmp_dir
    @tmp_dir.setter
    def tmp_dir(self, value):
        setattr(self, "_tmp_dir", value)
    @property
    def processed_dir(self) -> str:
        """
        Get the path to the processed directory.

        Returns:
            str: The path to the processed directory.
        """
        return self.CACHE_DIR

    @property
    def children_file_metadata(self) -> List[FileMetadata]:
        """
        Get the list containing the file's metadata and that of its children.

        Returns:
            List[FileMetadata]: A list of Pydantic models containing file metadata.
        """
        return self._children_file_metadata

    @property
    def file_metadata(self) -> FileMetadata:
        """
        Get the list containing the file's metadata and that of its children.

        Returns:
            FileMetadata: A Pydantic model containing file metadata.
        """
        return self._file_metadata
    
    def _create_dir(self, original_file_path: str, dir_name: str) -> str:
        """
        Create a directory for storing temporary processing files.

        Args:
            original_file_path (str): The path of the original file.
            dir_name (str | os.PathLike): The name of the directory to create.

        Returns:
            str: The path to the directory.

        Details:
            A directory, whose name is provided via `dir_name`, is created
            in the same directory as `original_file_path`.

        """
        base_dir = os.path.dirname(original_file_path)
        tmp_dir = os.path.join(base_dir, dir_name)
        os.makedirs(tmp_dir, exist_ok=True)
        return tmp_dir

    def _remove_tmp_dir(self, original_file_path: str) -> str:
        """
        Remove the temporary directory.

        Args:
            original_file_path (str): The path of the original file.

        Returns:
            str: The path to the removed temporary directory.
        """
        base_dir = os.path.dirname(original_file_path)
        tmp_dir = os.path.join(base_dir, "tmp_dir")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return tmp_dir

    def _encode_base64(self, file_path: str) -> str:
        """
        Encode a file in base64, cache in _processed_dir, and return its path.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The file_path of the encoded string.
        """

        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)

        base_name_no_extension = os.path.splitext(os.path.basename(file_path))[0]
        base64_file_path = os.path.join(self.CACHE_DIR, base_name_no_extension)

        # Check if the base64 file already exists in the cache
        if os.path.exists(base64_file_path):
            return base64_file_path

        with open(file_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("utf-8")
            with open(base64_file_path, "w") as file:  # writes as string
                file.write(encoded)

        return base64_file_path

    @abstractmethod
    def get_mime_type(self) -> str:
        """
        Get the MIME type of the file.

        Args:
            file_path (str): The path to the file.

        Returns:
            str: The MIME type of the file.
        """
        pass

    @abstractmethod
    def process(self) -> FileMetadata:
        """
        Process the file and return structured data.

        Returns:
            FileMetadata: A list of Pydantic models containing file metadata.
        """
        pass


class MsgFileProcessor(IFileProcessor):
    """
    A class to process MSG files.
    """

    def get_mime_type(self) -> str:
        return "text/plain"

    def process(self) -> FileMetadata:
        """
        Process an MSG file and its attachments.

        Returns:
            FileMetadata: A list of Pydantic models containing file metadata.
        """

        self.logger.info(f"Processing MSG file: {self.file_path}")
        try:
            msg = extract_msg.Message(self.file_path)
        except Exception as e:
            self.logger.exception("Invalid msg extension")
            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=None,
                parent_file_path=None,
                mime_type=None,
                error=str(e),
            )

        text = msg.body

        cleaned_text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
        tmp_file_path = os.path.join(
            self.tmp_dir, os.path.basename(self.file_path) + ".txt"
        )

        with open(tmp_file_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        self._extract_attachments(msg)
        self._process_nested_msgs()
        self._move_files_to_root()
        self._remove_empty_directories()

        base64_file_path = self._encode_base64(tmp_file_path)
        parent_file_metadata = FileMetadata(
            original_file_path=self.file_path,
            file_name=os.path.basename(self.file_path),
            base64_file_path=base64_file_path,
            parent_file_path=None,
            mime_type=self.get_mime_type(),
            error=None,
        )
        results = []

        for filename in os.listdir(self.tmp_dir):
            if filename == os.path.basename(self.file_path) + ".txt":
                continue  # skips the parent file that was already processed
            file_path = os.path.join(self.tmp_dir, filename)
            if os.path.isfile(file_path):
                processor = FileProcessorFactory.get_processor(file_path)
                result = processor.file_metadata
                # set the parent file name to the original file name for each FileMetadata object in result
                result.parent_file_path = self.file_path
                results.append(
                    result
                )

        self._children_file_metadata = results
        return parent_file_metadata

    def _extract_attachments(self, message_object: extract_msg.Message) -> None:
        """
        Extract attachments from an MSG file.

        Args:
            message_object: The MSG file object.
        """
        for attachment in message_object.attachments:
            attachment.save(customPath=self.tmp_dir)

    def _process_nested_msgs(self):
        """
        Recursively processes any nested MSG files found in the processed directory.
        """
        nested_msg = True
        while nested_msg:
            nested_msg = False
            for file in os.listdir(self.tmp_dir):
                if any([file.endswith(msg_suffix) for msg_suffix in _MSG_FILE]):
                    nested_msg = True
                    nested_file_path = os.path.join(self.tmp_dir, file)
                    nested_processor = FileProcessorFactory.get_processor(
                        nested_file_path
                    )
                    nested_results = nested_processor.process()
                    for result in nested_results:
                        result["parent_file_path"] = self.file_path
                    # No need to append to results here, as they are already processed and handled by the nested_processor.process() call

    def _move_files_to_root(self):
        """
        Moves all processed files to the root of the processed directory.
        """
        for root, _, files in os.walk(self.tmp_dir):
            for file in files:
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(self.tmp_dir, file)
                shutil.move(src_file_path, dest_file_path)

    def _remove_empty_directories(self):
        """
        Removes any empty directories within the processed directory.
        """
        for root, dirs, _ in os.walk(self.tmp_dir, topdown=False):
            for directory in dirs:
                dir_path = os.path.join(root, directory)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)


class ExcelFileProcessor(IFileProcessor):
    """
    A class to process Excel files.
    """

    def get_mime_type(self) -> str:
        return "application/pdf"

    def process(self) -> FileMetadata:
        import xlwings as xw

        self.logger.info(f"Processing Excel file: {self.file_path}")
        tmp_file_path = os.path.join(
            self.tmp_dir, os.path.basename(self.file_path) + ".pdf"
        )

        try:
            app = xw.App(visible=False)
            wb = app.books.open(self.file_path)
            wb.to_pdf(tmp_file_path)
            wb.close()
            app.quit()

            base64_file_path = self._encode_base64(tmp_file_path)

            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=base64_file_path,
                parent_file_path=None,
                mime_type=self.get_mime_type(),
                error=None,
            )
        except Exception as e:
            self.logger.error(f"Failed to process Excel file: {e}")
            if app:
                app.quit()

            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=None,
                parent_file_path=None,
                mime_type=None,
                error=str(e),
            )


class WordFileProcessor(IFileProcessor):
    """
    A class to process Word files.
    """

    def get_mime_type(self) -> str:
        return "text/plain"

    def process(self) -> FileMetadata:
        from docx import Document
        from win32com.client import Dispatch

        self.logger.info(f"Processing Word file: {self.file_path}")
        tmp_file_path = os.path.join(
            self.tmp_dir, os.path.basename(self.file_path) + ".txt"
        )

        try:
            if self.file_path.endswith(".docx"):
                doc = Document(self.file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            else:
                word = Dispatch("Word.Application")
                word.Visible = False
                try:
                    doc = word.Documents.Open(self.file_path)
                    text = doc.Content.Text
                    doc.Close()
                except Exception as e:
                    word.Quit()
                    return FileMetadata(
                        original_file_path=self.file_path,
                        file_name=os.path.basename(self.file_path),
                        base64_file_path=None,
                        parent_file_path=None,
                        mime_type=None,
                        error=str(e),
                    )
                word.Quit()

            with open(tmp_file_path, "w", encoding="utf-8") as f:
                f.write(text)

            base64_file_path = self._encode_base64(tmp_file_path)

            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=base64_file_path,
                parent_file_path=None,
                mime_type=self.get_mime_type(),
                error=None,
            )

        except Exception as e:
            self.logger.error(f"Failed to process Word file: {e}")
            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=None,
                parent_file_path=None,
                mime_type=None,
                error=str(e),
            )


class TextFileProcessor(IFileProcessor):
    """
    A class to process text files.
    """

    def get_mime_type(self) -> str:
        return "text/plain"

    def process(self) -> FileMetadata:
        self.logger.info(f"Processing text file: {self.file_path}")
        tmp_file_path = os.path.join(
            self.tmp_dir, os.path.basename(self.file_path)
        )

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(self.file_path, "r", encoding="latin-1") as f:
                    text = f.read()
            except Exception as e:
                return FileMetadata(
                    original_file_path=self.file_path,
                    file_name=os.path.basename(self.file_path),
                    base64_file_path=None,
                    parent_file_path=None,
                    mime_type=None,
                    error=str(e),
                )

        with open(tmp_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        base64_file_path = self._encode_base64(tmp_file_path)

        return FileMetadata(
            original_file_path=self.file_path,
            file_name=os.path.basename(self.file_path),
            base64_file_path=base64_file_path,
            parent_file_path=None,
            mime_type=self.get_mime_type(),
            error=None,
        )


class ImageFileProcessor(IFileProcessor):
    """
    A class to process image files.
    """

    def get_mime_type(self) -> str:
        return "image/png"

    def process(self) -> FileMetadata:
        self.logger.info(f"Processing image file: {self.file_path}")
        try:
            image = Image.open(self.file_path)
            tmp_file_path = os.path.join(
                self.tmp_dir, os.path.basename(self.file_path) + ".processed.png"
            )
            image.save(tmp_file_path)

            base64_file_path = self._encode_base64(tmp_file_path)

            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=base64_file_path,
                parent_file_path=None,
                mime_type=self.get_mime_type(),
                error=None,
            )
        except Exception as e:
            self.logger.error(f"Failed to process image file: {e}")
            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=None,
                parent_file_path=None,
                mime_type=None,
                error=str(e),
            )


class PdfFileProcessor(IFileProcessor):
    """
    A class to process PDF files.
    """

    def get_mime_type(self) -> str:
        return "application/pdf"

    def process(self) -> FileMetadata:
        try:
            self.logger.info(f"Processing PDF file: {self.file_path}")
            tmp_file_path = os.path.join(
                self.tmp_dir, os.path.basename(self.file_path)
            )

            with open(self.file_path, "rb") as src, open(
                tmp_file_path, "wb"
            ) as dst:
                dst.write(src.read())

            base64_file_path = self._encode_base64(tmp_file_path)

            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=base64_file_path,
                parent_file_path=None,
                mime_type=self.get_mime_type(),
                error=None,
            )
        except Exception as e:
            self.logger.error(f"Failed to process PDF file: {e}")
            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=None,
                parent_file_path=None,
                mime_type=None,
                error=str(e),
            )


class TiffFileProcessor(IFileProcessor):
    """
    A class to process TIFF files.
    """

    def get_mime_type(self) -> str:
        return "application/pdf"

    def process(self) -> FileMetadata:
        self.logger.info(f"Processing TIFF file: {self.file_path}")
        tmp_file_path = os.path.join(
            self.tmp_dir, os.path.basename(self.file_path) + ".pdf"
        )

        try:
            image = Image.open(self.file_path)
            image.save(tmp_file_path, "PDF", resolution=100.0)

            base64_file_path = self._encode_base64(tmp_file_path)

            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=base64_file_path,
                parent_file_path=None,
                mime_type=self.get_mime_type(),
                error=None,
            )
        except Exception as e:
            self.logger.error(f"Failed to process TIFF file: {e}")
            return FileMetadata(
                original_file_path=self.file_path,
                file_name=os.path.basename(self.file_path),
                base64_file_path=None,
                parent_file_path=None,
                mime_type=None,
                error=str(e),
            )


class FileProcessorFactory:
    """
    Factory class to create instances of concrete file processors
    based on the file extension.
    """

    @staticmethod
    def get_processor(file_path) -> IFileProcessor:
        """
        Process the file based on its extension and type.

        Returns:
            dict: Information about the processed file, including base64 encoding, MIME type
        """
        # get the extension from the file path
        extension = os.path.splitext(file_path)[1].lower()

        if extension in _MSG_FILE:
            processor = MsgFileProcessor(file_path)
        elif extension in _EXCEL_FILE:
            processor = ExcelFileProcessor(file_path)
        elif extension in _WORD_FILE:
            processor = WordFileProcessor(file_path)
        elif extension in _TEXT_FILE:
            processor = TextFileProcessor(file_path)
        elif extension in _IMAGE_FILE:
            processor = ImageFileProcessor(file_path)
        elif extension in _PDF_FILE:
            processor = PdfFileProcessor(file_path)
        elif extension in _TIFF_FILE:
            processor = TiffFileProcessor(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        return processor


class File:
    def __init__(self, file_path: str):
        self._file_path = file_path
        # self._cache_path = os.path.join(self.CACHE_DIR, os.path.basename(file_path))
        self.processor = FileProcessorFactory.get_processor(file_path)
        self._file_metadata = self.processor.file_metadata
        self._children_file_metadata = self.processor.children_file_metadata

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def file_metadata(self) -> FileMetadata:
        return self._file_metadata

    @property
    def children_file_metadata(self) -> List[FileMetadata]:
        return self._children_file_metadata

    @property
    def processor(self):
        return self._processor

    @processor.setter
    def processor(self, value):
        setattr(self, "_processor", value)
    
    @property
    def base64(self) -> Optional[str]:
        """
        Get the base64 encoded string of the file.

        Returns:
            str: The base64 encoded string.
        """
        if self._file_metadata.base64_file_path:
            with open(
                self._file_metadata.base64_file_path, "r"
            ) as file:  # reads as string
                return file.read()
        return None
    
    def reprocess_file(self) -> FileMetadata:
        """
        Reprocess the file and return structured data.

        Returns:
            FileMetadata: A list of Pydantic models containing file metadata.
        """
        self.processor = FileProcessorFactory.get_processor(self._file_path)
        self._file_metadata = self.processor.file_metadata
        self._children_file_metadata = self.processor.children_file_metadata
        return self._file_metadata


    def delete_file(self) -> None:
        """
        Delete the processed file.
        """
        if self._file_metadata.base64_file_path:
            os.remove(self._file_metadata.base64_file_path)
        return None
    
    def delete_processed_dir(self) -> None:
        """
        Delete the processed directory.
        """
        shutil.rmtree(self.processor.processed_dir)
        return None

def clear_cache():
    """
    Clear the cache directory.
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "__filecache__")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)
        logging.info("Cache cleared.")
