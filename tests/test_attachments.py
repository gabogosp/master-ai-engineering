from io import BytesIO
from unittest.mock import patch

from fastapi import UploadFile

import app.attachments as attachments_module
from app.attachments import extract_text


def _fake_upload(filename: str) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(b"contenido irrelevante -- se mockea la extraccion"))


async def test_extract_text_passes_through_when_under_the_cap():
    with patch.object(attachments_module, "_extract_pdf_text", return_value="x" * 100):
        text = await extract_text(_fake_upload("propuesta.pdf"))
    assert len(text) == 100


async def test_extract_text_truncates_at_max_attachment_chars():
    long_text = "x" * 70_000  # por encima del default de 60_000
    with patch.object(attachments_module, "_extract_pdf_text", return_value=long_text):
        text = await extract_text(_fake_upload("propuesta.pdf"))
    assert len(text) == 60_000


async def test_extract_text_respects_a_configured_cap():
    with patch.object(attachments_module.settings, "max_attachment_chars", 50):
        with patch.object(attachments_module, "_extract_pdf_text", return_value="x" * 200):
            text = await extract_text(_fake_upload("propuesta.pdf"))
    assert len(text) == 50
