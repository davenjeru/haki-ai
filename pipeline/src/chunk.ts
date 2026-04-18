import { CHUNK_CHAR_LIMIT, LawConfig } from "./config.js";

export interface PageSpan {
  pageNum: number;
  startChar: number; // byte offset of this page's start in the joined batch string
}

export interface Chunk {
  chunkId: string;
  source: string;
  chapter: string;
  section: string;
  title: string;
  text: string;
  startPage: number;     // PDF page this section header was found on
  pageImageKey?: string; // S3 key for the extracted page PDF; set by run.ts
}

interface Node {
  chapter: string;
  section: string;
  title: string;
  lines: string[];
  pageNum: number;
}

export function chunkText(fullText: string, law: LawConfig, pageSpans: PageSpan[]): Chunk[] {
  const nodes =
    law.structure === "constitution"
      ? parseConstitution(fullText, pageSpans)
      : parseAct(fullText, pageSpans);

  const chunks: Chunk[] = [];

  for (const node of nodes) {
    const body = node.lines.join("\n").trim();
    if (body.length < 50) continue;

    const segments = splitToSegments(body);

    segments.forEach((text, i) => {
      const suffix = segments.length > 1 ? `-${i + 1}` : "";
      const base = slugify(`${law.shortId}-${node.chapter}-${node.section}`);
      chunks.push({
        chunkId: `${base}${suffix}`,
        source: law.name,
        chapter: node.chapter,
        section: node.section,
        title: node.title,
        text,
        startPage: node.pageNum,
      });
    });
  }

  return chunks;
}

// ── Constitution of Kenya: CHAPTER headers + Article numbers ─────────────────

function parseConstitution(text: string, pageSpans: PageSpan[]): Node[] {
  const nodes: Node[] = [];
  let currentChapter = "Preamble";
  let currentSection = "Preamble";  // fix #7: pre-section text gets flushed
  let currentTitle = "";
  let currentPage = pageSpans[0]?.pageNum ?? 1;
  let buffer: string[] = [];
  let charOffset = 0;

  const flush = () => {
    if (currentSection && buffer.length) {
      nodes.push({
        chapter: currentChapter,
        section: currentSection,
        title: currentTitle,
        lines: buffer,
        pageNum: currentPage,
      });
    }
    buffer = [];
  };

  for (const line of text.split("\n")) {
    // fix #6: skip running page headers and bare page numbers
    if (isNoise(line)) {
      charOffset += line.length + 1;
      continue;
    }

    // fix #4 + #5: capture the full title after the dash, normalize word-form to numeral
    const chapterMatch = line.match(
      /^(CHAPTER\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|SEVENTEEN|EIGHTEEN|\d+))(?:\s*[—–-]\s*(.+))?\s*$/i
    );
    // fix #1 + #2: \s+ instead of \s{1,4}, [A-Za-z] instead of [A-Z]
    const articleMatch = line.match(/^(\d{1,3})\.\s+([A-Za-z][^\n]{2,80})\s*$/);

    if (chapterMatch) {
      flush();
      const num = normalizeChapterNum(chapterMatch[1]);   // fix #4
      const title = chapterMatch[2]?.trim();              // fix #5
      currentChapter = title ? `${num} — ${title}` : num;
    } else if (articleMatch) {
      flush();
      currentPage = pageForOffset(charOffset, pageSpans);
      currentSection = `Article ${articleMatch[1]}`;
      currentTitle = articleMatch[2].trim().replace(/\.$/, "");
    } else {
      buffer.push(line);
    }
    charOffset += line.length + 1;
  }
  flush();
  return nodes;
}

// ── Acts (Employment, Land): PART headers + Section numbers ──────────────────

function parseAct(text: string, pageSpans: PageSpan[]): Node[] {
  const nodes: Node[] = [];
  let currentPart = "Preliminary";
  let currentSection = "Preliminary"; // fix #7: pre-section text gets flushed
  let currentTitle = "";
  let currentPage = pageSpans[0]?.pageNum ?? 1;
  let buffer: string[] = [];
  let charOffset = 0;

  const flush = () => {
    if (currentSection && buffer.length) {
      nodes.push({
        chapter: currentPart,
        section: currentSection,
        title: currentTitle,
        lines: buffer,
        pageNum: currentPage,
      });
    }
    buffer = [];
  };

  for (const line of text.split("\n")) {
    // fix #6: skip running page headers and bare page numbers
    if (isNoise(line)) {
      charOffset += line.length + 1;
      continue;
    }

    // fix #3: drop [A-Z]+ alternative (too broad); fix #5: capture title after dash
    const partMatch = line.match(
      /^(PART\s+(?:[IVXLCDM]+|\d+))(?:\s*[—–-]\s*(.+))?\s*$/i
    );
    // fix #1 + #2: \s+ instead of \s{1,4}, [A-Za-z] instead of [A-Z]
    const sectionMatch = line.match(/^(\d{1,3})\.\s+([A-Za-z][^\n]{2,80})\s*$/);

    if (partMatch) {
      flush();
      const part = normalizePart(partMatch[1]);           // fix #4 (roman → kept as-is, already correct)
      const title = partMatch[2]?.trim();                 // fix #5
      currentPart = title ? `${part} — ${title}` : part;
    } else if (sectionMatch) {
      flush();
      currentPage = pageForOffset(charOffset, pageSpans);
      currentSection = `Section ${sectionMatch[1]}`;
      currentTitle = sectionMatch[2].trim().replace(/\.$/, "");
    } else {
      buffer.push(line);
    }
    charOffset += line.length + 1;
  }
  flush();
  return nodes;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// filter out OCR noise — running headers and bare page numbers
export function isNoise(line: string): boolean {
  const t = line.trim();
  if (!t) return false; // keep blank lines (paragraph breaks)
  if (/^\d+$/.test(t)) return true;                          // bare page number
  if (/^—\s*\d+\s*—$/.test(t)) return true;                 // — 42 —
  if (/^laws of kenya$/i.test(t)) return true;
  if (/^no\.\s*\d+\s+of\s+\d{4}$/i.test(t)) return true;   // No. 6 of 2012
  // running header: "Constitution of Kenya, 2010" or "Constitution of Kenya, 2010    21"
  if (/^(constitution|employment act|land act)[,\s]*(of kenya)?[,\s]*\d{4}/i.test(t)) return true;
  return false;
}

function pageForOffset(offset: number, spans: PageSpan[]): number {
  let page = spans[0]?.pageNum ?? 1;
  for (const s of spans) {
    if (offset >= s.startChar) page = s.pageNum;
    else break;
  }
  return page;
}

// fix #4: convert word-form chapter numbers to "Chapter N"
const WORD_TO_NUM: Record<string, string> = {
  ONE: "1", TWO: "2", THREE: "3", FOUR: "4", FIVE: "5",
  SIX: "6", SEVEN: "7", EIGHT: "8", NINE: "9", TEN: "10",
  ELEVEN: "11", TWELVE: "12", THIRTEEN: "13", FOURTEEN: "14",
  FIFTEEN: "15", SIXTEEN: "16", SEVENTEEN: "17", EIGHTEEN: "18",
};

export function normalizeChapterNum(raw: string): string {
  const clean = raw.replace(/\s+/g, " ").trim();
  // Split on the first dash/em-dash separator to separate number word from title
  const dashIdx = clean.search(/[—–-]/);
  const numPart = (dashIdx >= 0 ? clean.slice(0, dashIdx) : clean)
    .replace(/^CHAPTER\s+/i, "").trim().toUpperCase();
  const title = dashIdx >= 0 ? clean.slice(dashIdx + 1).trim() : undefined;
  const num = WORD_TO_NUM[numPart] ?? numPart;
  return title ? `Chapter ${num} — ${title}` : `Chapter ${num}`;
}

export function normalizePart(raw: string): string {
  return raw.replace(/\s+/g, " ").trim().replace(/^PART /i, "Part ");
}

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

export function slugify(str: string): string {
  return str.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}
