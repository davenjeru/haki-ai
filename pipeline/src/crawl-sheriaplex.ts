/**
 * SheriaPlex forum crawler (Phase 4a).
 *
 * Fetches Q&A threads from https://www.sheriaplex.com/forum/, extracts the
 * question title + body and the lawyer answers, and writes them to S3 as
 * faq-chunks/{slug}.txt + .metadata.json entries tagged `corpus="faq"`.
 *
 * The backend `FAQAgent` filters retrieval on `corpus="faq"` so these
 * chunks show up when users ask procedural or scenario questions that
 * don't map cleanly onto a single statute.
 *
 * Usage
 * -----
 * # Discover + crawl up to 40 threads and upload to S3:
 *   npm run crawl
 *
 * # Dry-run (write JSON to /tmp/sheriaplex-dryrun/, no S3 calls):
 *   npm run crawl -- --dry-run
 *
 * # Limit the crawl (useful for testing selectors after a site change):
 *   npm run crawl -- --limit 5
 *
 * Politeness: the crawler issues sequential requests with a 1.5s delay
 * between fetches, identifies itself with a clear UA string, and hard
 * limits to `--limit` pages per run so an accidental re-run doesn't
 * hammer the origin.
 *
 * Schema of each written chunk:
 *   body text:
 *     Question: {title}
 *     {question_body}
 *
 *     Lawyer answers:
 *     {author}: {answer_body}
 *     ---
 *     {author}: {answer_body}
 *
 *   metadata.json (Bedrock KB sidecar):
 *     {
 *       corpus: "faq",
 *       source: "SheriaPlex",
 *       section: {title},
 *       category: {derived from topic},
 *       url: {canonical URL},
 *       chunkId: {slug},
 *       chunkType: "body"
 *     }
 */

import { PutObjectCommand } from "@aws-sdk/client-s3";
import { mkdir, writeFile } from "fs/promises";
import * as path from "path";
import { FAQ_CHUNKS_PREFIX, S3_BUCKET } from "./config.js";
import { s3 } from "./s3-client.js";
import { slugify } from "./chunk.js";

// Lazy import so `npm run dev` (the PDF pipeline) doesn't pay the cheerio
// parse cost. cheerio is a devDependency under the pipeline package.
import { load as loadHtml } from "cheerio";

// ── Config ────────────────────────────────────────────────────────────────────

const SHERIAPLEX_ROOT = "https://www.sheriaplex.com";
const FORUM_INDEX = `${SHERIAPLEX_ROOT}/forum/`;
const USER_AGENT =
  "HakiAI-Crawler/1.0 (educational legal-aid chatbot; +https://github.com/)";
const REQUEST_DELAY_MS = 1500;
const DEFAULT_LIMIT = 40;

// ── CLI ───────────────────────────────────────────────────────────────────────

interface Args {
  limit: number;
  dryRun: boolean;
  outDir: string;
}

function parseArgs(argv: string[]): Args {
  const readFlag = (name: string): string | undefined => {
    const i = argv.indexOf(name);
    return i >= 0 ? argv[i + 1] : undefined;
  };
  const limit = Number(readFlag("--limit")) || DEFAULT_LIMIT;
  const dryRun = argv.includes("--dry-run");
  const outDir = readFlag("--out-dir") ?? "/tmp/sheriaplex-dryrun";
  return { limit, dryRun, outDir };
}

// ── HTTP helpers ──────────────────────────────────────────────────────────────

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url, {
    headers: { "User-Agent": USER_AGENT, Accept: "text/html" },
  });
  if (!response.ok) {
    throw new Error(`GET ${url} → ${response.status} ${response.statusText}`);
  }
  return await response.text();
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Discovery: extract forum thread URLs from the index page ──────────────────

export function extractForumLinks(indexHtml: string): string[] {
  const $ = loadHtml(indexHtml);
  const hrefs = new Set<string>();
  $("a[href]").each((_, el) => {
    const href = $(el).attr("href") ?? "";
    const abs = href.startsWith("http")
      ? href
      : href.startsWith("/")
        ? `${SHERIAPLEX_ROOT}${href}`
        : null;
    if (!abs) return;
    // Only accept thread pages. Forum listing has /forum/{id}-{slug} — the
    // numeric prefix avoids false-positives on category / paging links.
    const match = abs.match(/^https:\/\/www\.sheriaplex\.com\/forum\/(\d+)-[a-z0-9-]+/i);
    if (match) hrefs.add(abs.split("?")[0].replace(/\/$/, ""));
  });
  return Array.from(hrefs);
}

// ── Extraction: pull fields out of a single thread page ──────────────────────

export interface ForumThread {
  url: string;
  id: string;
  slug: string;
  title: string;
  question: string;
  answers: { author: string; body: string }[];
  relatedTopics: string[];
}

export function extractThread(html: string, url: string): ForumThread | null {
  const $ = loadHtml(html);
  const title = $("h2.mb-4").first().text().trim();
  if (!title) return null;

  const question = $(".details-container").first().text().trim();
  const answers: { author: string; body: string }[] = [];
  $(".comment-box").each((_, el) => {
    const author = $(el).find(".comment-author").first().text().trim();
    // <br> → \n so the body retains its original paragraph breaks.
    const raw = $(el).find(".comment-content").first();
    raw.find("br").replaceWith("\n");
    const body = raw.text().replace(/\s+\n/g, "\n").trim();
    if (body) answers.push({ author: author || "Anonymous", body });
  });

  const related: string[] = [];
  $(".related-questions-list a[href]").each((_, el) => {
    related.push($(el).text().trim());
  });

  const m = url.match(/\/forum\/(\d+)-(.*?)$/);
  const id = m?.[1] ?? slugify(title).slice(0, 8);
  const slug = m?.[2] ?? slugify(title);

  return { url, id, slug, title, question, answers, relatedTopics: related };
}

// ── Categorisation (heuristic) ────────────────────────────────────────────────

const CATEGORY_KEYWORDS: Array<[string, RegExp]> = [
  ["employment", /\b(employ|termination|dismissal|probation|wage|salary|redundan|contract of service|notice|leave|sick|maternity|overtime|labour)\b/i],
  ["land", /\b(land|tenant|landlord|eviction|lease|rent|property|title deed|easement|adverse possession|succession)\b/i],
  ["constitution", /\b(constitution|bill of rights|devolution|public participation|fundamental right|citizen|judiciary|parliament|gender equality)\b/i],
  ["family", /\b(marriage|divorce|custody|inheritance|will|adoption|cohabitation)\b/i],
  ["criminal", /\b(crime|criminal|bail|bond|police|arrest|prosecution|plea bargain|cybercrime|defamation)\b/i],
  ["contract", /\b(contract|breach|liquidated damages|capacity|oral contract|online contract)\b/i],
];

export function categorise(title: string, body: string): string {
  const text = `${title} ${body}`.toLowerCase();
  for (const [cat, re] of CATEGORY_KEYWORDS) {
    if (re.test(text)) return cat;
  }
  return "general";
}

// ── Rendering ─────────────────────────────────────────────────────────────────

function renderChunkText(thread: ForumThread): string {
  const answers = thread.answers
    .map((a) => `${a.author}: ${a.body}`)
    .join("\n\n---\n\n");
  return [
    `Question: ${thread.title}`,
    thread.question ? `\n${thread.question}` : "",
    "",
    "Lawyer answers:",
    answers || "(no answers yet)",
  ].join("\n");
}

function buildMetadata(thread: ForumThread): Record<string, string> {
  const category = categorise(thread.title, thread.question);
  return {
    corpus: "faq",
    source: "SheriaPlex",
    section: thread.title,
    category,
    url: thread.url,
    chunkId: `sheriaplex-${thread.id}-${thread.slug}`,
    chunkType: "body",
  };
}

// ── Writers (S3 vs. dry-run filesystem) ──────────────────────────────────────

async function writeToS3(chunkId: string, text: string, metadata: Record<string, string>): Promise<void> {
  const key = `${FAQ_CHUNKS_PREFIX}${chunkId}.txt`;
  await s3.send(new PutObjectCommand({
    Bucket: S3_BUCKET,
    Key: key,
    Body: text,
    ContentType: "text/plain; charset=utf-8",
  }));
  await s3.send(new PutObjectCommand({
    Bucket: S3_BUCKET,
    Key: `${key}.metadata.json`,
    Body: JSON.stringify({ metadataAttributes: metadata }, null, 2),
    ContentType: "application/json",
  }));
}

async function writeToDisk(
  outDir: string,
  chunkId: string,
  text: string,
  metadata: Record<string, string>,
): Promise<void> {
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, `${chunkId}.txt`), text, "utf-8");
  await writeFile(
    path.join(outDir, `${chunkId}.txt.metadata.json`),
    JSON.stringify({ metadataAttributes: metadata }, null, 2),
    "utf-8",
  );
}

// ── Orchestrator ──────────────────────────────────────────────────────────────

async function crawlOne(url: string, args: Args): Promise<boolean> {
  try {
    const html = await fetchText(url);
    const thread = extractThread(html, url);
    if (!thread) {
      console.warn(`  [skip] ${url} — missing title`);
      return false;
    }
    const text = renderChunkText(thread);
    const metadata = buildMetadata(thread);
    const chunkId = metadata["chunkId"];
    if (args.dryRun) {
      await writeToDisk(args.outDir, chunkId, text, metadata);
      console.log(`  [dry] ${chunkId} → ${args.outDir}/`);
    } else {
      await writeToS3(chunkId, text, metadata);
      console.log(`  [s3 ] s3://${S3_BUCKET}/${FAQ_CHUNKS_PREFIX}${chunkId}.txt`);
    }
    return true;
  } catch (err) {
    console.warn(`  [err] ${url} — ${err instanceof Error ? err.message : err}`);
    return false;
  }
}

export async function main(argv: string[] = process.argv.slice(2)): Promise<void> {
  const args = parseArgs(argv);
  console.log(
    `SheriaPlex crawler — limit=${args.limit} dryRun=${args.dryRun}` +
      (args.dryRun ? ` outDir=${args.outDir}` : ` bucket=${S3_BUCKET}`),
  );

  console.log(`Discovering threads from ${FORUM_INDEX} …`);
  const index = await fetchText(FORUM_INDEX);
  const links = extractForumLinks(index);
  console.log(`  found ${links.length} thread URLs on the index`);

  const selected = links.slice(0, args.limit);
  let ok = 0;
  for (let i = 0; i < selected.length; i++) {
    console.log(`(${i + 1}/${selected.length}) ${selected[i]}`);
    if (await crawlOne(selected[i], args)) ok++;
    await delay(REQUEST_DELAY_MS);
  }

  console.log(`\nDone. ${ok}/${selected.length} threads ingested.`);
}

const isMain = (() => {
  try {
    // process.argv[1] is the executed file. tsx resolves it to an absolute
    // path; we check by suffix to avoid a url-vs-path mismatch.
    return (process.argv[1] ?? "").endsWith("crawl-sheriaplex.ts");
  } catch {
    return false;
  }
})();
if (isMain) {
  main().catch((err) => {
    console.error(err);
    process.exitCode = 1;
  });
}
