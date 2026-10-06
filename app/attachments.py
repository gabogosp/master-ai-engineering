from io import BytesIO

import structlog
from docx import Document
from fastapi import UploadFile
from pypdf import PdfReader

logger = structlog.get_logger(__name__)


class UnsupportedAttachmentError(Exception):
    """El adjunto no es un PDF ni un documento Word."""


def _extract_pdf_text(raw: bytes) -> str:
    reader = PdfReader(BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def _extract_docx_text(raw: bytes) -> str:
    document = Document(BytesIO(raw))
    return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()


async def extract_text(upload_file: UploadFile) -> str:
    """Extrae texto localmente de un PDF o Word (Camino B de la consigna).

    No se sube el archivo al LLM: se extrae el texto acá y se concatena al
    transcript como texto plano. Esto mantiene el wrapper agnóstico de
    proveedor -- el fallback openai<->anthropic que construimos sigue
    funcionando igual con o sin adjuntos. El camino A (Files API) ataría
    el adjunto a un proveedor específico y rompería esa propiedad.
    """
    filename = upload_file.filename or ""
    raw = await upload_file.read()
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        text = _extract_pdf_text(raw)
    elif lower_name.endswith(".docx"):
        text = _extract_docx_text(raw)
    else:
        raise UnsupportedAttachmentError(
            f"Formato no soportado: '{filename}'. Solo se aceptan .pdf y .docx."
        )

    logger.info("attachment_extracted", filename=filename, chars=len(text))
    return text


def build_transcript_with_attachments(
    transcript: str, attachment_texts: list[tuple[str, str]]
) -> str:
    """Concatena el texto extraído de cada adjunto al transcript, con un
    separador por archivo, tal como pide la consigna."""
    parts = [transcript]
    for filename, text in attachment_texts:
        parts.append(f"--- attachment: {filename} ---\n{text}")
    return "\n\n".join(parts)
