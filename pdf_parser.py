import fitz  # PyMuPDF
import pdfplumber
import re
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

# Import utils for cleaning, chunking, and equation extraction
import utils

# Optional: OCR fallback
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None

logging.basicConfig(level=logging.INFO)


def extract_text_with_pdfplumber(pdf_path: Union[str, Path]) -> str:
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logging.warning(f"pdfplumber failed on {pdf_path}: {e}")
    return text


def extract_text_with_pymupdf(pdf_path: Union[str, Path]) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    return text


def ocr_image_page(page_image) -> str:
    if pytesseract is None:
        raise ImportError("pytesseract is not installed. Please install it for OCR support.")
    return pytesseract.image_to_string(page_image)


def extract_metadata(pdf_path: Union[str, Path]) -> Dict[str, Optional[str]]:
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    return {
        "title": meta.get("title"),
        "author": meta.get("author"),
        "subject": meta.get("subject"),
        "keywords": meta.get("keywords"),
        "creationDate": meta.get("creationDate"),
        "modDate": meta.get("modDate"),
        "producer": meta.get("producer"),
    }


def extract_sections(text: str) -> List[Dict[str, str]]:
    # Improved: catch common scientific headings
    section_pattern = re.compile(
        r'(?P<heading>^((Abstract|Introduction|Related Work|Background|Methodology|Methods|Experiments|Results|Discussion|Conclusion|References|Acknowledgements|Appendix)[^\n]*)\n+)(?P<body>(.*?))(?=^((Abstract|Introduction|Related Work|Background|Methodology|Methods|Experiments|Results|Discussion|Conclusion|References|Acknowledgements|Appendix)[^\n]*)\n+|\Z)',
        re.MULTILINE | re.DOTALL | re.IGNORECASE
    )
    sections = [
        {"heading": m.group("heading").strip(), "body": m.group("body").strip()}
        for m in section_pattern.finditer(text)
    ]
    return sections


def extract_tables(pdf_path: Union[str, Path]) -> List[Dict[str, object]]:
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    # Structured table (list of lists)
                    structured = [row for row in table if row]
                    # Markdown table
                    if structured:
                        header = structured[0]
                        md = '| ' + ' | '.join(header) + ' |\n'
                        md += '| ' + ' | '.join(['---'] * len(header)) + ' |\n'
                        for row in structured[1:]:
                            md += '| ' + ' | '.join(row) + ' |\n'
                    else:
                        md = ''
                    tables.append({
                        'structured': structured,
                        'markdown': md
                    })
    except Exception as e:
        logging.warning(f"Table extraction failed on {pdf_path}: {e}")
    return tables


def extract_figures_and_captions(pdf_path: Union[str, Path], text: str, output_dir: Union[str, Path] = "figures") -> List[Dict[str, str]]:
    figures = []
    doc = fitz.open(pdf_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(doc):
        images = page.get_images(full=True)
        for j, img in enumerate(images):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            output_path = Path(output_dir) / f"page{i+1}_img{j+1}.png"
            if pix.n < 5:  # grayscale or RGB
                pix.save(output_path)
            else:  # CMYK
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.save(output_path)
                pix1 = None
            # Try to find a caption in the text
            caption_pattern = re.compile(rf'(Figure|Fig\\.) ?{j+1}[:.\s-]+(.{{0,200}})', re.IGNORECASE)
            caption_match = caption_pattern.search(text)
            caption = caption_match.group(2).strip() if caption_match else ''
            figures.append({
                'path': str(output_path),
                'caption': caption
            })
    return figures


def parse_pdf(
    pdf_path: Union[str, Path],
    extract_figs=True,
    extract_eqs=True,
    extract_tbls=True,
    extract_secs=True,
    enable_ocr=False,
    return_raw_text=False
) -> Dict:
    pdf_path = Path(pdf_path)
    logging.info(f"Parsing {pdf_path.name}")

    text = extract_text_with_pdfplumber(pdf_path) or extract_text_with_pymupdf(pdf_path)
    text = utils.clean_text(text)

    if enable_ocr and not text.strip() and pytesseract:
        logging.info("Falling back to OCR")
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += ocr_image_page(img)
        text = utils.clean_text(text)

    output = {
        "metadata": extract_metadata(pdf_path),
        "raw_text": text if return_raw_text else None,
    }

    if extract_secs:
        output["sections"] = extract_sections(text)
    if extract_eqs:
        output["equations"] = utils.extract_equations(text)
    if extract_tbls:
        output["tables"] = extract_tables(pdf_path)
    if extract_figs:
        output["figures"] = extract_figures_and_captions(pdf_path, text)

    # Add chunked text for downstream embedding
    output["chunks"] = utils.intelligent_chunking(text)

    return output


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR fallback")
    parser.add_argument("--raw", action="store_true", help="Return raw text")
    args = parser.parse_args()

    parsed = parse_pdf(args.pdf_path, enable_ocr=args.ocr, return_raw_text=args.raw)
    print(json.dumps(parsed, indent=2))