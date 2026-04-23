import { CHUNK_CHAR_LIMIT } from "./config.js";

// Shape of a processed text chunk ready for S3 upload and Bedrock KB ingestion.
export interface Chunk {
  chunkId: string;
  source: string;        // full law name, e.g. "Employment Act 2007"
  chapter: string;       // e.g. "Part III — Termination of Contract"
  section: string;       // e.g. "Section 40"
  title: string;         // e.g. "Termination of employment"
  text: string;          // body text for this chunk
  startPage: number;     // PDF page where this section header appears
  pageImageKey?: string; // S3 key for the page PDF, used in the citation carousel
  // Classification used by the advanced-RAG retriever to exclude
  // low-signal chunks before rerank:
  //   - "body"        — substantive legal text (retained)
  //   - "toc"         — table-of-contents / arrangement-of-sections
  //                     listings (dropped)
  //   - "preamble"    — cover page / pre-Section 1 OCR noise (dropped)
  //   - "short-title" — Section 1's "This Act may be cited as …" one-
  //                     liner. Low signal, high match-rate, so dropped.
  //   - "definitions" — Section 2's flat Interpretation list. Matches
  //                     almost any query because it contains every
  //                     defined term, so dropped for non-definitional
  //                     retrieval. Re-enable per-query if we add a
  //                     "what does X mean" intent.
  chunkType: "body" | "toc" | "preamble" | "short-title" | "definitions";
  // "statute" for chunks derived from Act/Constitution PDFs (default), "faq"
  // for crawled Q&A content (SheriaPlex, KenyaLaw summaries). Bedrock KB uses
  // this to route queries: the FAQAgent filters on corpus="faq" while the
  // statute specialists filter on source=<Act name> (corpus=statute implied).
  corpus?: "statute" | "faq";
  category?: string;     // FAQ-only: SheriaPlex category (e.g. "Employment")
  url?: string;          // FAQ-only: original Q&A URL for attribution
}

// ── Noise filtering ───────────────────────────────────────────────────────────

/**
 * Returns true for OCR noise lines that should be excluded from chunk text:
 * bare page numbers, "Laws of Kenya" running headers, act name headers, etc.
 * Blank lines are kept because they mark paragraph boundaries.
 */
export function isNoise(line: string): boolean {
  const t = line.trim();
  if (!t) return false;                                            // keep blank lines
  if (/^\d+$/.test(t)) return true;                               // bare page number
  if (/^—\s*\d+\s*—$/.test(t)) return true;                      // — 42 —
  if (/^laws of kenya$/i.test(t)) return true;
  if (/^no\.\s*\d+\s+of\s+\d{4}$/i.test(t)) return true;        // No. 6 of 2012
  // Running header pattern: "Constitution of Kenya 2010", "Employment Act, 2007   21"
  if (/^(constitution|employment act|land act)[,\s]*(of kenya)?[,\s]*\d{4}/i.test(t)) return true;
  return false;
}

// ── Chunk splitting ───────────────────────────────────────────────────────────

/**
 * Splits body text into segments no larger than CHUNK_CHAR_LIMIT (~500 tokens).
 * Breaks at paragraph boundaries (double newlines) to preserve readability.
 * Returns a single-element array if the text fits within the limit.
 */
export function splitToSegments(text: string): string[] {
  if (text.length <= CHUNK_CHAR_LIMIT) return [text];

  const segments: string[] = [];
  const paragraphs = text.split(/\n{2,}/);
  let current = "";

  for (const para of paragraphs) {
    if ((current + "\n\n" + para).length > CHUNK_CHAR_LIMIT && current) {
      segments.push(current.trim());
      current = para;
    } else {
      current = current ? current + "\n\n" + para : para;
    }
  }
  if (current.trim()) segments.push(current.trim());
  return segments;
}

// ── Slug generation ───────────────────────────────────────────────────────────

/** Converts a string to a URL/S3-safe slug (lowercase, hyphens, no special chars). */
export function slugify(str: string): string {
  return str.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

// ── Chapter/Part label normalization ─────────────────────────────────────────

// The Constitution uses English word-form chapter numbers (ONE, TWO, …).
// We convert them to numerals so chunk IDs are consistent ("Chapter 4", not "Chapter FOUR").
const WORD_TO_NUM: Record<string, string> = {
  ONE: "1", TWO: "2", THREE: "3", FOUR: "4", FIVE: "5",
  SIX: "6", SEVEN: "7", EIGHT: "8", NINE: "9", TEN: "10",
  ELEVEN: "11", TWELVE: "12", THIRTEEN: "13", FOURTEEN: "14",
  FIFTEEN: "15", SIXTEEN: "16", SEVENTEEN: "17", EIGHTEEN: "18",
};

/**
 * Normalizes a raw Constitution chapter heading to "Chapter N — Title".
 * Handles both word-form ("CHAPTER FOUR — THE BILL OF RIGHTS") and
 * numeral form ("CHAPTER 4 — THE BILL OF RIGHTS").
 */
export function normalizeChapterNum(raw: string): string {
  const clean = raw.replace(/\s+/g, " ").trim();
  const dashIdx = clean.search(/[—–-]/);
  const numPart = (dashIdx >= 0 ? clean.slice(0, dashIdx) : clean)
    .replace(/^CHAPTER\s+/i, "").trim().toUpperCase();
  const title = dashIdx >= 0 ? clean.slice(dashIdx + 1).trim() : undefined;
  const num = WORD_TO_NUM[numPart] ?? numPart;
  return title ? `Chapter ${num} — ${title}` : `Chapter ${num}`;
}

/**
 * Normalizes a raw Act part heading to title-case "Part" prefix.
 * e.g. "PART III — TERMINATION OF CONTRACT" → "Part III — TERMINATION OF CONTRACT"
 */
export function normalizePart(raw: string): string {
  return raw.replace(/\s+/g, " ").trim().replace(/^PART /i, "Part ");
}
