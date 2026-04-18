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
    const { pageNum, chapterOrPart, sections } = extraction;
    const rawLines = (pageTexts.get(pageNum) ?? "")
      .split("\n")
      .filter((line) => !isNoise(line));

    // Update chapter/part if a new one begins on this page
    if (chapterOrPart) {
      currentChapter = normalizeGroupLabel(chapterOrPart, law.structure);
    }

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
        };
      } else {
        currentNode.lines.push(...rawLines);
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
      }

      flush();

      currentNode = {
        chapter: currentChapter,
        section: formatSectionLabel(hit.number, law.structure),
        title: hit.title,
        lines: rawLines.slice(hit.bodyStartLine),
        startPage: pageNum,
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
      });
    });
  }

  return chunks;
}
