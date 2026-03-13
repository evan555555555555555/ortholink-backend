"""
OrthoLink Translator
gpt-4o translation with regulatory term preservation.
PRD: NOT Google Translate (loses legal precision).
Large texts are chunked (max 6,000 words) to stay under 30k TPM.
"""

import logging
import time
from typing import Optional

from app.tools.llm import chat_completion

logger = logging.getLogger(__name__)

MAX_WORDS_PER_CHUNK = 6000
SLEEP_BETWEEN_CHUNKS_SEC = 1.0
RETRY_AFTER_429_SEC = 60
MAX_RETRIES_429 = 1

# Regulatory terms that should be preserved across translations
PRESERVED_TERMS = [
    "CE Mark", "510(k)", "PMA", "De Novo",
    "IVD", "IVDR", "MDR", "MDD",
    "QMS", "QMSR", "GMP", "ISO 13485",
    "IEC 62304", "ISO 14971", "IEC 60601",
    "CDSCO", "PMDA", "TGA", "ANVISA",
    "NMPA", "MFDS", "SFDA", "Swissmedic",
    "COFEPRIS", "Roszdravnadzor",
    "Class I", "Class II", "Class IIa", "Class IIb", "Class III",
    "Notified Body", "Authorized Representative",
    "Technical File", "Design Dossier",
    "Clinical Evaluation Report", "CER",
    "Declaration of Conformity", "DoC",
    "Certificate of Free Sale", "CFS",
    "UDI", "EUDAMED", "GUDID",
]


UKRAINE_LEGAL_SYSTEM_PROMPT = (
    "You are a legal translator specializing in medical device regulations. "
    "Translate the following Ukrainian regulatory text to English. "
    "Preserve all Article numbers, Clause numbers, and legal terminology "
    "exactly. Do not paraphrase. Do not summarize. Translate verbatim. "
    "Preserve the hierarchical structure: Part > Chapter > Article > Clause."
)


def _split_into_translation_chunks(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    """
    Split text into chunks of at most max_words. Split at paragraph (\\n\\n)
    boundaries; if a paragraph exceeds max_words, split at sentence (". ") boundaries.
    """
    if not text or not text.strip():
        return []
    text = text.strip()
    word_count = len(text.split())
    if word_count <= max_words:
        return [text]

    chunks: list[str] = []
    # First split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if para_words > max_words:
            # Flush current chunk
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_words = 0
            # Split long paragraph at sentence boundaries
            sentences = [s.strip() for s in para.split(". ") if s.strip()]
            for sent in sentences:
                sent_words = len(sent.split())
                if current_words + sent_words > max_words and current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_words = 0
                current.append(sent + "." if not sent.endswith(".") else sent)
                current_words += sent_words
            continue

        if current_words + para_words > max_words and current:
            chunks.append("\n\n".join(current))
            current = []
            current_words = 0
        current.append(para)
        current_words += para_words

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _translate_one_chunk_ukrainian(chunk: str) -> str:
    """Translate a single chunk with Ukrainian legal prompt. Raises on failure."""
    return chat_completion(
        system_prompt=UKRAINE_LEGAL_SYSTEM_PROMPT,
        user_prompt=chunk,
        temperature=0.1,
        max_tokens=8192,
    )


def translate_ukrainian_regulatory_to_english(text: str) -> dict:
    """
    Translate Ukrainian regulatory text to English with exact legal structure preservation.
    Chunks at 6,000 words to stay under 30k TPM. Retries once on 429 after 60s.
    """
    try:
        chunks = _split_into_translation_chunks(text, max_words=MAX_WORDS_PER_CHUNK)
        if not chunks:
            return {
                "translated_text": "",
                "source_language": "uk",
                "target_language": "en",
                "country": "UA",
                "success": False,
                "error": "No text to translate",
            }
        translated_parts: list[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(SLEEP_BETWEEN_CHUNKS_SEC)
            last_error: Optional[Exception] = None
            for attempt in range(MAX_RETRIES_429 + 1):
                try:
                    part = _translate_one_chunk_ukrainian(chunk)
                    translated_parts.append(part)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if "429" in err_str or "rate" in err_str or "too many" in err_str:
                        if attempt < MAX_RETRIES_429:
                            logger.warning(f"Translation rate limited, waiting {RETRY_AFTER_429_SEC}s then retry")
                            time.sleep(RETRY_AFTER_429_SEC)
                        else:
                            raise
                    else:
                        raise
            if last_error is not None:
                raise last_error
        combined = "\n\n".join(translated_parts)
        return {
            "translated_text": combined,
            "source_language": "uk",
            "target_language": "en",
            "country": "UA",
            "success": True,
        }
    except Exception as e:
        logger.error(f"Ukrainian regulatory translation failed: {e}")
        return {
            "translated_text": "",
            "source_language": "uk",
            "target_language": "en",
            "country": "UA",
            "success": False,
            "error": str(e),
        }


def _translate_one_chunk_generic(
    chunk: str,
    system_prompt: str,
    source_language: str,
    country: str,
) -> str:
    """Translate one chunk with generic legal prompt. Raises on failure."""
    user_prompt = (
        f"Translate the following {source_language} regulatory text from {country} to English. "
        f"This is a medical device regulation.\n\n{chunk}"
    )
    return chat_completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=8192,
    )


def translate_to_english(
    text: str,
    source_language: str,
    country: str,
    preserve_terms: Optional[list[str]] = None,
) -> dict:
    """
    Translate regulatory text to English using gpt-4o.
    Chunks at 6,000 words for large texts (e.g. India PDF). Retries once on 429.
    Returns dict with translated text and metadata.
    """
    if source_language.lower() == "uk":
        return translate_ukrainian_regulatory_to_english(text)

    terms_to_preserve = preserve_terms or PRESERVED_TERMS
    system_prompt = (
        "You are a professional legal/regulatory translator specializing in medical device "
        "regulations. You translate regulatory texts with extreme precision, preserving legal "
        "meaning and regulatory terminology. You MUST:\n"
        "1. Preserve all regulatory terms, acronyms, and proper nouns exactly as they appear\n"
        "2. Maintain the legal structure (articles, sections, clauses) of the original\n"
        "3. Use standard English regulatory terminology\n"
        "4. Never paraphrase legal requirements — translate literally\n"
        "5. Keep article/section numbers unchanged\n"
        f"6. These specific terms MUST be preserved if encountered: {', '.join(terms_to_preserve[:20])}"
    )

    word_count = len(text.split())
    if word_count <= MAX_WORDS_PER_CHUNK:
        try:
            translated = _translate_one_chunk_generic(text, system_prompt, source_language, country)
            return {
                "translated_text": translated,
                "source_language": source_language,
                "target_language": "en",
                "country": country,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                "translated_text": "",
                "source_language": source_language,
                "target_language": "en",
                "country": country,
                "success": False,
                "error": str(e),
            }

    # Chunked path for large text (e.g. India 211-page PDF)
    try:
        chunks = _split_into_translation_chunks(text, max_words=MAX_WORDS_PER_CHUNK)
        translated_parts: list[str] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(SLEEP_BETWEEN_CHUNKS_SEC)
            last_error: Optional[Exception] = None
            for attempt in range(MAX_RETRIES_429 + 1):
                try:
                    part = _translate_one_chunk_generic(
                        chunk, system_prompt, source_language, country
                    )
                    translated_parts.append(part)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    err_str = str(e).lower()
                    if "429" in err_str or "rate" in err_str or "too many" in err_str:
                        if attempt < MAX_RETRIES_429:
                            logger.warning(
                                f"Translation rate limited, waiting {RETRY_AFTER_429_SEC}s then retry"
                            )
                            time.sleep(RETRY_AFTER_429_SEC)
                        else:
                            raise
                    else:
                        raise
            if last_error is not None:
                raise last_error
        combined = "\n\n".join(translated_parts)
        return {
            "translated_text": combined,
            "source_language": source_language,
            "target_language": "en",
            "country": country,
            "success": True,
        }
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return {
            "translated_text": "",
            "source_language": source_language,
            "target_language": "en",
            "country": country,
            "success": False,
            "error": str(e),
        }


def detect_language(text: str) -> str:
    """
    Detect the language of a text sample.
    Returns ISO 639-1 language code.
    """
    system_prompt = (
        "You are a language detection tool. Given a text, respond with ONLY the "
        "ISO 639-1 two-letter language code (e.g., 'en', 'uk', 'ja', 'pt', 'de'). "
        "Nothing else."
    )

    try:
        result = chat_completion(
            system_prompt=system_prompt,
            user_prompt=text[:500],  # Only need a sample
            temperature=0.0,
            max_tokens=10,
        )
        return result.strip().lower()[:2]
    except Exception:
        return "en"  # Default to English
