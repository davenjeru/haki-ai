import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import {
  GetObjectCommand,
  PutObjectCommand,
} from "@aws-sdk/client-s3";
import {
  AWS_REGION,
  BEDROCK_MODEL_ID,
  PAGE_EXTRACTIONS_PREFIX,
  S3_BUCKET,
} from "./config.js";
import type { LawConfig } from "./config.js";
import { s3 } from "./s3-client.js";

// Always point Bedrock at real AWS — LocalStack doesn't support it.
// Explicitly set endpoint so AWS_ENDPOINT_URL (used for LocalStack S3) is ignored.
const bedrockClient = new BedrockRuntimeClient({
  region: AWS_REGION,
  endpoint: `https://bedrock-runtime.${AWS_REGION}.amazonaws.com`,
});

export interface SectionHit {
  number: string;        // e.g. "22", "12A", "Preamble"
  title: string;         // e.g. "Enforcement of Bill of Rights"
  bodyStartLine: number; // 0-indexed line within the page text where body begins
}

export interface PageExtraction {
  pageNum: number;
  chapterOrPart: string | null; // null if no new chapter/part starts on this page
  sections: SectionHit[];       // empty if page is pure body text
  // True if this page is a table of contents / arrangement-of-sections listing
  // rather than substantive legal text. Pages flagged here propagate a
  // `chunkType: "toc"` metadata attribute so retrieval can filter them out.
  isToc?: boolean;
}

// ── S3 extraction cache ───────────────────────────────────────────────────────

function extractionKey(shortId: string, pageNum: number): string {
  return `${PAGE_EXTRACTIONS_PREFIX}${shortId}/page-${pageNum}.json`;
}

async function getCachedExtraction(
  shortId: string,
  pageNum: number
): Promise<PageExtraction | null> {
  try {
    const cmd = new GetObjectCommand({
      Bucket: S3_BUCKET,
      Key: extractionKey(shortId, pageNum),
    });
    const response = await s3.send(cmd);
    const body = (await response.Body?.transformToString("utf-8")) ?? "";
    return JSON.parse(body) as PageExtraction;
  } catch {
    return null;
  }
}

async function saveExtraction(
  shortId: string,
  extraction: PageExtraction
): Promise<void> {
  await s3.send(
    new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: extractionKey(shortId, extraction.pageNum),
      Body: JSON.stringify(extraction),
      ContentType: "application/json",
    })
  );
}

// ── Prompt ────────────────────────────────────────────────────────────────────

function buildPrompt(
  lawName: string,
  structure: "constitution" | "act",
  pageNum: number,
  pageText: string
): string {
  const sectionWord = structure === "constitution" ? "Article" : "Section";
  const groupWord = structure === "constitution" ? "Chapter" : "Part";

  return `You are a legal document parser for Kenyan law. Your task is to identify structural headings in a single page of OCR-extracted text.

LAW: ${lawName}
STRUCTURE TYPE: ${structure} (headings use ${groupWord} and ${sectionWord})
PAGE NUMBER: ${pageNum}

RAW PAGE TEXT (each line is on its own line, starting from line 0):
${pageText}

---

Identify the following from the page text above:

1. CHAPTER/PART HEADER: If a new ${groupWord} begins on this page, extract its full heading text exactly as it appears (e.g. "CHAPTER FOUR — THE BILL OF RIGHTS" or "PART III — TERMINATION OF CONTRACT"). If no new ${groupWord} starts on this page, return null.

2. SECTION/ARTICLE HEADERS: List every new ${sectionWord} that begins on this page. For each one provide:
   - "number": the section/article number as a string (may include letters, e.g. "12A", "22")
   - "title": the title text that follows the number on the same line (strip trailing periods)
   - "bodyStartLine": the 0-indexed line number of the FIRST line of body text for this section (i.e. the line AFTER the heading line)

3. TABLE OF CONTENTS FLAG: Set "isToc" to true if this page is a table of contents / arrangement of sections / arrangement of articles (a listing of section numbers and titles with no substantive body text — typically a few pages at the start of each Act or Chapter). Otherwise omit the field or set it to false.

Return ONLY valid JSON matching this exact schema — no markdown fences, no explanation:

{
  "chapterOrPart": "<string or null>",
  "sections": [
    { "number": "<string>", "title": "<string>", "bodyStartLine": <integer> }
  ],
  "isToc": <boolean, optional>
}

Rules:
- If a line looks like a running header (e.g. "Laws of Kenya", "Constitution of Kenya 2010", "The Employment Act, 2007", a bare page number like "42", or "— 42 —"), ignore it entirely.
- If a section heading spans two lines (title wraps), use only the first line for "title" and set "bodyStartLine" to the line after the wrapped continuation.
- Section numbers are strings — preserve alphanumeric forms like "12A".
- If no sections start on this page, return an empty array for "sections".
- "bodyStartLine" must be a non-negative integer. If a section header is the last line on the page, set "bodyStartLine" to the total number of lines on the page (body continues on next page).`;
}

// ── Response parsing ──────────────────────────────────────────────────────────

function parseHaikuResponse(rawText: string, pageNum: number): PageExtraction {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawText);
  } catch {
    // Haiku sometimes wraps in markdown fences despite instructions — strip and retry
    const stripped = rawText
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/, "");
    try {
      parsed = JSON.parse(stripped);
    } catch {
      console.warn(
        `\n  [page ${pageNum}] Could not parse Haiku JSON, using empty extraction.`
      );
      return { pageNum, chapterOrPart: null, sections: [] };
    }
  }

  if (
    typeof parsed !== "object" ||
    parsed === null ||
    !("sections" in parsed)
  ) {
    console.warn(
      `\n  [page ${pageNum}] Haiku response missing 'sections' key.`
    );
    return { pageNum, chapterOrPart: null, sections: [] };
  }

  const obj = parsed as Record<string, unknown>;
  const chapterOrPart =
    typeof obj.chapterOrPart === "string" ? obj.chapterOrPart : null;
  const isToc = typeof obj.isToc === "boolean" ? obj.isToc : false;

  const sections: SectionHit[] = [];
  if (Array.isArray(obj.sections)) {
    for (const s of obj.sections) {
      if (
        typeof s === "object" &&
        s !== null &&
        typeof (s as Record<string, unknown>).number === "string" &&
        typeof (s as Record<string, unknown>).title === "string" &&
        typeof (s as Record<string, unknown>).bodyStartLine === "number"
      ) {
        sections.push({
          number: String((s as Record<string, unknown>).number),
          title: String((s as Record<string, unknown>).title),
          bodyStartLine: Math.max(
            0,
            Math.floor(
              Number((s as Record<string, unknown>).bodyStartLine)
            )
          ),
        });
      }
    }
  }

  return { pageNum, chapterOrPart, sections, isToc };
}

// ── Retry helpers ─────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isThrottling(err: unknown): boolean {
  if (err instanceof Error) {
    return (
      err.name === "ThrottlingException" ||
      err.message.toLowerCase().includes("too many requests")
    );
  }
  return false;
}

function retryDelay(attempt: number, throttling: boolean): number {
  const base = throttling ? 8000 : 1000;
  const expo = base * 2 ** attempt;
  const jitter = Math.random() * base; // spread concurrent retries
  return expo + jitter;
}

// ── Public API ────────────────────────────────────────────────────────────────

export async function extractPageStructure(
  law: LawConfig,
  pageNum: number,
  pageText: string
): Promise<PageExtraction> {
  // Return cached extraction if it exists (supports resume after interruption)
  const cached = await getCachedExtraction(law.shortId, pageNum);
  if (cached) return cached;

  const prompt = buildPrompt(law.name, law.structure, pageNum, pageText);
  const body = JSON.stringify({
    anthropic_version: "bedrock-2023-05-31",
    max_tokens: 1024,
    messages: [{ role: "user", content: prompt }],
  });

  const MAX_RETRIES = 5;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const cmd = new InvokeModelCommand({
        modelId: BEDROCK_MODEL_ID,
        contentType: "application/json",
        accept: "application/json",
        body: Buffer.from(body),
      });

      const response = await bedrockClient.send(cmd);
      const responseBody = JSON.parse(
        Buffer.from(response.body).toString("utf-8")
      );
      const rawText: string = responseBody.content[0].text.trim();
      const extraction = parseHaikuResponse(rawText, pageNum);

      // Persist to S3 so re-runs skip this page
      await saveExtraction(law.shortId, extraction);

      return extraction;
    } catch (err) {
      if (attempt === MAX_RETRIES) {
        console.warn(
          `\n  [page ${pageNum}] Haiku failed after ${MAX_RETRIES + 1} attempts, using empty extraction. Error: ${String(err)}`
        );
        return { pageNum, chapterOrPart: null, sections: [], isToc: false };
      }

      const throttling = isThrottling(err);
      const delay = retryDelay(attempt, throttling);
      if (throttling) {
        process.stdout.write(
          `\n  [page ${pageNum}] Throttled — waiting ${(delay / 1000).toFixed(1)}s before retry ${attempt + 1}/${MAX_RETRIES}...`
        );
      }
      await sleep(delay);
    }
  }

  return { pageNum, chapterOrPart: null, sections: [], isToc: false };
}
