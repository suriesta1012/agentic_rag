# document_processor.py
"""
PDF loading + chunking, with two boundary edge cases handled that the
naive "split each page independently" approach gets wrong:

1. Page-boundary bridging
   ~~~~~~~~~~~~~~~~~~~~~~~~
   `PyPDFLoader` returns one `Document` per page. If you hand those to
   `RecursiveCharacterTextSplitter.split_documents()` directly (the naive
   approach), the splitter can never cross a page boundary -- it splits
   *within* each page's Document independently. A sentence or paragraph
   that happens to straddle a page break (extremely common in real PDFs)
   gets torn in half: the tail ends up alone at the start of the next
   page's first chunk, divorced from the sentence it belongs to, which
   hurts both retrieval precision (the fragment is a poor semantic match
   for a query about that sentence) and generation quality (the LLM sees
   an incomplete thought). We fix this by concatenating all page text
   into one continuous stream *before* splitting, so the splitter's
   sentence/paragraph-aware separators can flow across what used to be a
   page edge -- then re-attribute each resulting chunk back to a page
   number via its character offset, so citations still work.

2. Orphan / tiny fragment chunks
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   The last piece of a paragraph before a natural break (or a stray
   header/footer/page-number line) can end up as its own chunk of only
   a few characters. These low-signal fragments are pure noise in a
   vector index -- they rarely match a real query, and when they do
   (e.g. a header line matching on keyword overlap) they crowd out a
   genuinely relevant chunk in top_k. We merge any chunk shorter than
   `MIN_CHUNK_CHARS` into its predecessor rather than keeping it standalone.
"""

import re
from typing import List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS


# Separators ordered from coarsest to finest so the splitter tries to
# break at natural boundaries first (double newline = paragraph, single
# newline = line, sentence-ending punctuation, then words).
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# Joins page N's text to page N+1's text. A paragraph break (not a hard
# page-break marker) so the splitter treats it exactly like any other
# paragraph boundary and is free to pull a straddling sentence to
# whichever side it best fits, instead of being forced to cut there.
_PAGE_JOINER = "\n\n"


def _clean_text(text: str) -> str:
    """Remove hyphenated line-breaks and normalise whitespace."""
    # Re-join words broken across lines (e.g. "infor-\nmation" -> "information")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple blank lines to one paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _page_for_offset(offset: int, page_ranges: List[Tuple[int, int, Optional[int]]]) -> Optional[int]:
    """Find which page's [start, end) range an offset falls in."""
    for start, end, page_num in page_ranges:
        if start <= offset < end:
            return page_num
    # Offset landed in a page-joiner gap (or past the end) -- attribute to
    # the nearest preceding page rather than losing the citation entirely.
    for start, end, page_num in reversed(page_ranges):
        if offset >= start:
            return page_num
    return page_ranges[0][2] if page_ranges else None


def _split_with_offsets(full_text: str) -> List[Tuple[str, int]]:
    """
    Run the coarse-then-fine two-pass split and recover an approximate
    character offset for every resulting chunk (text splitters don't
    return offsets natively, so we recover them by searching forward
    from a rolling cursor -- cheap and accurate enough for page
    attribution, which only needs to land in the right ballpark).
    """
    coarse_splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=CHUNK_SIZE * 4,       # generous first-pass window
        chunk_overlap=CHUNK_OVERLAP * 2,
        length_function=len,
    )
    fine_splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    fine_chunks: List[str] = []
    for coarse_chunk in coarse_splitter.split_text(full_text):
        fine_chunks.extend(fine_splitter.split_text(coarse_chunk))

    chunks_with_offsets: List[Tuple[str, int]] = []
    search_from = 0
    for chunk_text in fine_chunks:
        # Search with a small backward slack to account for overlap
        # between consecutive chunks, then fall back to a full scan,
        # then to the last known position if the text genuinely can't
        # be found (e.g. was altered by a separator quirk).
        idx = full_text.find(chunk_text, max(0, search_from - CHUNK_OVERLAP - 50))
        if idx == -1:
            idx = full_text.find(chunk_text)
        if idx == -1:
            idx = search_from
        chunks_with_offsets.append((chunk_text, idx))
        search_from = idx + max(len(chunk_text) - CHUNK_OVERLAP, 1)

    return chunks_with_offsets


def _merge_orphan_chunks(
    chunks_with_pages: List[Tuple[str, Optional[int]]],
    min_chars: int,
) -> List[Tuple[str, Optional[int]]]:
    """
    Merge any chunk shorter than `min_chars` into the previous chunk
    (keeping the previous chunk's page attribution). A short chunk with
    no predecessor (i.e. it's first) is instead merged *forward* into
    the next chunk, so a document that opens with a stray title line
    doesn't ship a near-empty first chunk either.
    """
    merged: List[List] = []  # list of [text, page] (mutable for in-place merge)
    for text, page in chunks_with_pages:
        text = text.strip()
        if not text:
            continue
        if len(text) < min_chars and merged:
            merged[-1][0] = f"{merged[-1][0]} {text}".strip()
        else:
            merged.append([text, page])

    # Handle a too-short *first* chunk (nothing to merge backward into
    # at the time it was seen) by folding it forward now.
    if len(merged) >= 2 and len(merged[0][0]) < min_chars:
        merged[1][0] = f"{merged[0][0]} {merged[1][0]}".strip()
        merged.pop(0)

    return [(text, page) for text, page in merged]


def load_and_chunk_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF and return a list of well-formed Document chunks.

    Raises:
        ValueError: if the PDF contains no extractable text, or if
                    CHUNK_OVERLAP/CHUNK_SIZE are misconfigured.
    """
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than "
            f"CHUNK_SIZE ({CHUNK_SIZE}); otherwise chunk boundaries never "
            f"advance and splitting can loop or produce degenerate chunks."
        )

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    if not pages:
        raise ValueError("PDF appears to be empty or could not be parsed.")

    # Clean raw text on every page before chunking
    for page in pages:
        page.page_content = _clean_text(page.page_content)

    # Filter out pages with no usable content (scanned pages, cover images, etc.)
    pages = [p for p in pages if len(p.page_content.strip()) > 50]

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned/image-based."
        )

    # --- Bridge page boundaries -------------------------------------------
    # Concatenate all pages into one continuous text stream so the splitter
    # can flow sentences across what used to be a hard page-Document edge,
    # while tracking each page's [start, end) offset range in that stream
    # so we can still attribute every chunk to a source page afterward.
    text_parts: List[str] = []
    page_ranges: List[Tuple[int, int, Optional[int]]] = []
    cursor = 0
    for page in pages:
        content = page.page_content
        start = cursor
        text_parts.append(content)
        cursor += len(content)
        page_ranges.append((start, cursor, page.metadata.get("page")))
        text_parts.append(_PAGE_JOINER)
        cursor += len(_PAGE_JOINER)
    full_text = "".join(text_parts)

    chunks_with_offsets = _split_with_offsets(full_text)

    chunks_with_pages = [
        (text, _page_for_offset(offset, page_ranges))
        for text, offset in chunks_with_offsets
    ]

    # --- Merge orphan / tiny fragment chunks --------------------------------
    chunks_with_pages = _merge_orphan_chunks(chunks_with_pages, MIN_CHUNK_CHARS)

    # --- Build final Document objects with citation metadata ---------------
    chunks: List[Document] = []
    for i, (text, page_num) in enumerate(chunks_with_pages):
        chunks.append(
            Document(
                page_content=text,
                metadata={
                    "chunk_index": i,
                    "source": file_path,
                    "page": page_num,
                },
            )
        )

    return chunks
