import {
  GetObjectCommand,
  HeadObjectCommand,
  ListObjectsV2Command,
  PutObjectCommand,
} from "@aws-sdk/client-s3";
import { mkdir, writeFile, readFile, readdir, rm } from "fs/promises";
import { existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import {
  BEDROCK_CONCURRENCY,
  CHUNKS_PREFIX,
  PAGE_TEXT_PREFIX,
  S3_BUCKET,
  LAWS,
} from "./config.js";
import { extractPageStructure } from "./llm-extract.js";
import type { PageExtraction } from "./llm-extract.js";
import { assembleChunks } from "./assemble.js";
import { uploadChunks } from "./upload.js";
import type { LawConfig } from "./config.js";
import type { Chunk } from "./chunk.js";
import { s3 } from "./s3-client.js";

async function main(): Promise<void> {
  console.log("Haki AI — LLM-assisted chunking pipeline\n");

  for (const law of LAWS) {
    console.log(`\nProcessing: ${law.name} (${law.shortId})`);
    await processLaw(law);
  }

  console.log("\nAll done.");
  console.log("Next: trigger Bedrock KB ingestion job to index the new chunks.");
}

main().catch((err) => {
  console.error("\nFatal error:", err);
  process.exit(1);
});

// ── Temp dir (resume support) ─────────────────────────────────────────────────
// Each chunk is written as {chunkId}.json when assembled.
// After a successful S3 upload, {chunkId}.done is written.
// On restart: load .json files, skip those with a matching .done file.
// Temp dir is deleted after the law's .complete marker is written to S3.

const __dirname = dirname(fileURLToPath(import.meta.url));
const TEMP_BASE = join(__dirname, "..", ".tmp-chunks");

function lawTempDir(shortId: string): string {
  return join(TEMP_BASE, shortId);
}

async function saveChunksToTemp(chunks: Chunk[], shortId: string): Promise<void> {
  const dir = lawTempDir(shortId);
  await mkdir(dir, { recursive: true });
  await Promise.all(
    chunks.map((chunk) =>
      writeFile(join(dir, `${chunk.chunkId}.json`), JSON.stringify(chunk))
    )
  );
  console.log(`  Saved ${chunks.length} chunks to ${dir}`);
}

// Returns pending chunks (skipping already-uploaded ones), or null if no temp dir exists.
async function loadPendingChunks(
  shortId: string
): Promise<{ pending: Chunk[]; total: number } | null> {
  const dir = lawTempDir(shortId);
  if (!existsSync(dir)) return null;

  const files = await readdir(dir);
  const jsonFiles = files.filter((f) => f.endsWith(".json"));
  if (jsonFiles.length === 0) return null;

  const pending: Chunk[] = [];
  let skipped = 0;

  for (const file of jsonFiles) {
    const chunkId = file.slice(0, -5); // strip .json
    if (existsSync(join(dir, `${chunkId}.done`))) {
      skipped++;
      continue;
    }
    const data = await readFile(join(dir, file), "utf-8");
    pending.push(JSON.parse(data) as Chunk);
  }

  return { pending, total: jsonFiles.length };
}

async function markChunkDone(shortId: string, chunkId: string): Promise<void> {
  await writeFile(join(lawTempDir(shortId), `${chunkId}.done`), "");
}

async function cleanTempDir(shortId: string): Promise<void> {
  const dir = lawTempDir(shortId);
  if (existsSync(dir)) await rm(dir, { recursive: true });
}

// ── Completion marker ─────────────────────────────────────────────────────────
// Written to S3 after all chunks for a law are successfully uploaded.
// Re-runs check for this marker and skip the law if it exists.
// To re-process a law: delete the marker and its chunks from S3.
//   aws s3 rm s3://haki-ai-data/processed-chunks/constitution-2010/.complete
//   aws s3 rm s3://haki-ai-data/processed-chunks/ --recursive --exclude "*" --include "constitution-2010-*"

function completionMarkerKey(shortId: string): string {
  return `${CHUNKS_PREFIX}${shortId}/.complete`;
}

async function isLawComplete(shortId: string): Promise<boolean> {
  try {
    await s3.send(
      new HeadObjectCommand({
        Bucket: S3_BUCKET,
        Key: completionMarkerKey(shortId),
      })
    );
    return true;
  } catch {
    return false;
  }
}

async function markLawComplete(shortId: string): Promise<void> {
  await s3.send(
    new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: completionMarkerKey(shortId),
      Body: new Date().toISOString(),
      ContentType: "text/plain",
    })
  );
}

// ── Main per-law processing ───────────────────────────────────────────────────

async function processLaw(law: LawConfig): Promise<void> {
  if (await isLawComplete(law.shortId)) {
    console.log(
      `  Already complete — skipping. To re-process, delete:\n` +
      `    s3://${S3_BUCKET}/${completionMarkerKey(law.shortId)}\n` +
      `    s3://${S3_BUCKET}/${CHUNKS_PREFIX}${law.shortId}-*`
    );
    return;
  }

  // Check for an in-progress temp dir from a previous interrupted run
  const resumed = await loadPendingChunks(law.shortId);

  let pendingChunks: Chunk[];
  let totalChunks: number;

  if (resumed !== null) {
    // Resume: skip assembly and re-extraction entirely
    console.log(
      `  Resuming from ${lawTempDir(law.shortId)}\n` +
      `  ${resumed.total - resumed.pending.length}/${resumed.total} chunks already uploaded — ${resumed.pending.length} remaining`
    );
    pendingChunks = resumed.pending;
    totalChunks = resumed.total;
  } else {
    // Fresh run: extract → assemble → save to temp dir
    const pageKeys = await listPageTextKeys(law.shortId);
    if (pageKeys.length === 0) {
      console.warn(`  No page text files found for ${law.shortId}. Run 'npm run dev' first.`);
      return;
    }
    console.log(`  Found ${pageKeys.length} pages in S3`);

    console.log("  Fetching page texts from S3...");
    const pageTexts = await fetchAllPageTexts(pageKeys);

    console.log(`  Extracting structure with Haiku (concurrency=${BEDROCK_CONCURRENCY})...`);
    const pageNums = [...pageTexts.keys()].sort((a, b) => a - b);
    const extractions = await extractAllPages(law, pageNums, pageTexts);

    const cachedCount = extractions.filter(
      (e) => e.sections.length > 0 || e.chapterOrPart !== null
    ).length;
    console.log(
      `  Extractions complete (${cachedCount}/${pageNums.length} pages had structural headers)`
    );

    const allChunks = assembleChunks(law, pageTexts, extractions);
    console.log(`  Assembled ${allChunks.length} chunks`);

    if (allChunks.length === 0) {
      console.warn("  No chunks to upload — check Haiku output for this law.");
      return;
    }

    await saveChunksToTemp(allChunks, law.shortId);
    pendingChunks = allChunks;
    totalChunks = allChunks.length;
  }

  await uploadChunks(
    pendingChunks,
    (chunkId) => markChunkDone(law.shortId, chunkId),
    totalChunks
  );

  await markLawComplete(law.shortId);
  await cleanTempDir(law.shortId);
  console.log(`  Done: ${law.shortId}`);
}

// ── S3 helpers ────────────────────────────────────────────────────────────────

async function listPageTextKeys(shortId: string): Promise<string[]> {
  const prefix = `${PAGE_TEXT_PREFIX}${shortId}/`;
  const keys: string[] = [];
  let continuationToken: string | undefined;

  do {
    const cmd = new ListObjectsV2Command({
      Bucket: S3_BUCKET,
      Prefix: prefix,
      ContinuationToken: continuationToken,
    });
    const response = await s3.send(cmd);
    for (const obj of response.Contents ?? []) {
      if (obj.Key) keys.push(obj.Key);
    }
    continuationToken = response.NextContinuationToken;
  } while (continuationToken);

  keys.sort((a, b) => extractPageNum(a) - extractPageNum(b));
  return keys;
}

function extractPageNum(key: string): number {
  const match = key.match(/page-(\d+)\.txt$/);
  return match ? parseInt(match[1], 10) : 0;
}

async function fetchAllPageTexts(
  keys: string[]
): Promise<Map<number, string>> {
  const map = new Map<number, string>();
  await Promise.all(
    keys.map(async (key) => {
      const pageNum = extractPageNum(key);
      const cmd = new GetObjectCommand({ Bucket: S3_BUCKET, Key: key });
      const response = await s3.send(cmd);
      const body = (await response.Body?.transformToString("utf-8")) ?? "";
      map.set(pageNum, body);
    })
  );
  return map;
}

function printProgress(resolved: number, total: number): void {
  const pct = Math.round((resolved / total) * 100);
  const bar = "█".repeat(Math.floor(pct / 5)) + "░".repeat(20 - Math.floor(pct / 5));
  process.stdout.write(`\r  [${bar}] ${pct}% — ${resolved}/${total} pages`);
}

async function extractAllPages(
  law: LawConfig,
  pageNums: number[],
  pageTexts: Map<number, string>
): Promise<PageExtraction[]> {
  const results: PageExtraction[] = [];
  const queue = [...pageNums];
  let active = 0;
  let resolved = 0;
  const total = pageNums.length;

  return new Promise((resolve, reject) => {
    const dispatch = () => {
      while (active < BEDROCK_CONCURRENCY && queue.length > 0) {
        const pageNum = queue.shift()!;
        active++;
        const text = pageTexts.get(pageNum) ?? "";

        printProgress(resolved, total);

        extractPageStructure(law, pageNum, text)
          .then((extraction) => {
            results.push(extraction);
            resolved++;
            active--;

            if (resolved === total) {
              process.stdout.write(
                `\r  LLM extraction: ${resolved}/${total} pages done.\n`
              );
              results.sort((a, b) => a.pageNum - b.pageNum);
              resolve(results);
            } else {
              printProgress(resolved, total);
              dispatch();
            }
          })
          .catch(reject);
      }
    };

    dispatch();
  });
}
