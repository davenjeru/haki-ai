/**
 * Assembles Chunk[] from Haiku LLM extraction results.
 *
 * Algorithm:
 *   Walk pages in order. For each page:
 *     - Update the current chapter/part label if a new one starts on this page.
 *     - If no section headers are present, append all lines to the open node.
 *     - If section headers are present, flush lines up to each header into the
 *       previous node, then open a new node for each section found.
 *   After all pages, flush the final open node.
 *   Convert each node to one or more Chunks by joining its lines and splitting
 *   at paragraph boundaries if the text exceeds CHUNK_CHAR_LIMIT.
 */
import type { Chunk } from "./chunk.js";
import {
  splitToSegments,
  slugify,
  normalizeChapterNum,
  normalizePart,
  isNoise,
} from "./chunk.js";
import type { LawConfig } from "./config.js";
import type { PageExtraction } from "./llm-extract.js";

interface SectionNode {
  chapter: string;
  section: string;
  title: string;
  lines: string[];
  startPage: number;
  // Marked true whenever any page contributing lines to this node was
  // flagged as a table-of-contents page by the Haiku extractor, OR when
  // the section title itself matches a TOC heuristic ("Arrangement of
  // Sections", "Table of Contents"). Propagated to Chunk.chunkType.
  isToc: boolean;
}

// Matches "Arrangement of Sections" / "Arrangement of Articles" / "Table of
// Contents" (case-insensitive). Used as a structural fallback when Haiku
// misses an obvious TOC page (e.g. when the page text contains real section
// numbers that look like body text to the model).
function hasTocHeading(text: string | null | undefined): boolean {
  if (!text) return false;
  return /arrangement of (sections|articles)|table of contents/i.test(text);
}

function formatSectionLabel(
  number: string,
  structure: "constitution" | "act"
): string {
  return structure === "constitution"
    ? `Article ${number}`
    : `Section ${number}`;
}

function normalizeGroupLabel(
  raw: string,
  structure: "constitution" | "act"
): string {
  return structure === "constitution"
    ? normalizeChapterNum(raw)
    : normalizePart(raw);
}

export function assembleChunks(
  law: LawConfig,
  pageTexts: Map<number, string>,
  extractions: PageExtraction[]
): Chunk[] {
  const sorted = [...extractions].sort((a, b) => a.pageNum - b.pageNum);

  let currentChapter = "Preliminary";
  let currentNode: SectionNode | null = null;
  const nodes: SectionNode[] = [];

  const flush = () => {
    if (currentNode && currentNode.lines.length > 0) {
      nodes.push(currentNode);
    }
    currentNode = null;
  };

  for (const extraction of sorted) {
    const { pageNum, chapterOrPart, sections, isToc } = extraction;
    const rawLines = (pageTexts.get(pageNum) ?? "")
      .split("\n")
      .filter((line) => !isNoise(line));

    // Update chapter/part if a new one begins on this page
    if (chapterOrPart) {
      currentChapter = normalizeGroupLabel(chapterOrPart, law.structure);
    }

    const pageIsToc = isToc === true || hasTocHeading(chapterOrPart);

    if (sections.length === 0) {
      // Entire page is body text for the current section
      if (currentNode === null) {
        // Pre-section preamble — create a synthetic node
        currentNode = {
          chapter: currentChapter,
          section: "Preamble",
          title: "",
          lines: [...rawLines],
          startPage: pageNum,
          isToc: pageIsToc,
        };
      } else {
        currentNode.lines.push(...rawLines);
        if (pageIsToc) currentNode.isToc = true;
      }
      continue;
    }

    // Page has one or more section headers
    let lastBodyEnd = 0;

    for (const hit of sections) {
      // Lines from lastBodyEnd up to (but not including) the header line
      // belong to the previous section. The header line itself is skipped.
      // bodyStartLine points to the first body line, so headerLine = bodyStartLine - 1.
      const headerLine = Math.max(lastBodyEnd, hit.bodyStartLine - 1);
      const linesBeforeHeader = rawLines.slice(lastBodyEnd, headerLine);

      if (currentNode !== null) {
        currentNode.lines.push(...linesBeforeHeader);
        if (pageIsToc) currentNode.isToc = true;
      }

      flush();

      currentNode = {
        chapter: currentChapter,
        section: formatSectionLabel(hit.number, law.structure),
        title: hit.title,
        lines: rawLines.slice(hit.bodyStartLine),
        startPage: pageNum,
        isToc: pageIsToc || hasTocHeading(hit.title),
      };

      lastBodyEnd = hit.bodyStartLine;
    }

    // Lines after the last section header's body start are already in currentNode.lines
    // (sliced above). Subsequent pages will continue appending to currentNode.
  }

  flush();

  // Convert nodes → Chunk[]
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
        startPage: node.startPage,
        pageImageKey: `page-images/${law.shortId}/page-${node.startPage}.pdf`,
        chunkType: node.isToc ? "toc" : "body",
      });
    });
  }

  return chunks;
}
