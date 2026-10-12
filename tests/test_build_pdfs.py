from evals.stress.fixtures.build_pdfs import _build_pdf, _verify_extracted


def test_build_pdf_produces_extractable_text_close_to_the_target(tmp_path):
    """No pedimos igualdad exacta -- fpdf2 no garantiza una relación 1:1
    entre caracteres escritos y lo que pypdf extrae de vuelta (ver
    docstring del script). Alcanza con estar en el entorno correcto."""
    target_kb = 1
    output_path = tmp_path / "attach_1kb.pdf"

    written_chars, file_bytes = _build_pdf(target_kb, output_path)

    assert output_path.exists()
    assert written_chars >= target_kb * 1024
    assert file_bytes > 0

    extracted_chars = _verify_extracted(output_path)
    # Dentro de +-20% del objetivo -- generoso a propósito, el punto es
    # detectar una regresión grosera (ej. el PDF queda vacío), no calibrar
    # al carácter.
    assert extracted_chars > target_kb * 1024 * 0.8


def test_larger_target_produces_a_larger_pdf(tmp_path):
    """Monotonía básica: pedir más texto no debería dar un archivo más
    chico -- atraparía, por ejemplo, un bug donde el loop de repetición
    del párrafo no corta bien."""
    _, bytes_1kb = _build_pdf(1, tmp_path / "a.pdf")
    _, bytes_10kb = _build_pdf(10, tmp_path / "b.pdf")
    assert bytes_10kb > bytes_1kb
