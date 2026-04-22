import { PutObjectCommand } from "@aws-sdk/client-s3";
import { readFile } from "fs/promises";
import { CHUNKS_PREFIX, PAGE_IMAGES_PREFIX, PAGE_TEXT_PREFIX, RAW_PREFIX, S3_BUCKET } from "./config.js";
import { Chunk } from "./chunk.js";
import { s3 } from "./s3-client.js";

/** Uploads the original law PDF to raw-laws/{filename}. */
export async function uploadRawPdf(filepath: string, filename: string): Promise<void> {
  const body = await readFile(filepath);
  await s3.send(new PutObjectCommand({
    Bucket: S3_BUCKET,
    Key: `${RAW_PREFIX}${filename}`,
    Body: body,
    ContentType: "application/pdf",
  }));
  console.log(`    uploaded: s3://${S3_BUCKET}/${RAW_PREFIX}${filename}`);
}

/** Uploads a single-page PDF to page-images/{shortId}/page-N.pdf. Returns the S3 key. */
export async function uploadPagePdf(
  pdfBytes: Uint8Array,
  lawShortId: string,
  pageNum: number
): Promise<string> {
  const key = `${PAGE_IMAGES_PREFIX}${lawShortId}/page-${pageNum}.pdf`;
  await s3.send(new PutObjectCommand({
    Bucket: S3_BUCKET,
    Key: key,
    Body: Buffer.from(pdfBytes),
    ContentType: "application/pdf",
  }));
  return key;
}

/** Uploads raw OCR text to page-text/{shortId}/page-N.txt. Returns the S3 key. */
export async function uploadPageText(
  text: string,
  lawShortId: string,
  pageNum: number
): Promise<string> {
  const key = `${PAGE_TEXT_PREFIX}${lawShortId}/page-${pageNum}.txt`;
  await s3.send(new PutObjectCommand({
    Bucket: S3_BUCKET,
    Key: key,
    Body: text,
    ContentType: "text/plain; charset=utf-8",
  }));
  return key;
}

const UPLOAD_CONCURRENCY = 20;

/**
 * Uploads processed chunks to S3 in batches of UPLOAD_CONCURRENCY.
 * Each chunk produces two S3 objects:
 *   - {chunkId}.txt              — the body text, read by Bedrock KB
 *   - {chunkId}.txt.metadata.json — Bedrock KB sidecar with citation attributes
 *
 * @param onChunkUploaded - called after each successful upload (used to write .done markers)
 * @param totalForProgress - total chunk count for the progress bar (may differ from chunks.length on resume)
 */
export async function uploadChunks(
  chunks: Chunk[],
  onChunkUploaded?: (chunkId: string) => Promise<void>,
  totalForProgress?: number
): Promise<void> {
  let done = 0;
  const total = totalForProgress ?? chunks.length;

  const uploadOne = async (chunk: Chunk): Promise<void> => {
    const key = `${CHUNKS_PREFIX}${chunk.chunkId}.txt`;
    await s3.send(new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: key,
      Body: chunk.text,
      ContentType: "text/plain; charset=utf-8",
    }));
    // Bedrock KB rejects empty-string metadata values — it fails the whole
    // chunk with "attribute values are acceptable data types (strings,
    // numbers, Booleans, or String Lists)". Preamble chunks legitimately
    // have an empty `title`, so we omit any key whose value is missing
    // or empty. All fields are treated as optional at the sidecar level;
    // citation rendering already handles absent fields gracefully.
    const metadataAttributes: Record<string, string> = {};
    const maybePut = (key: string, value: string | null | undefined) => {
      if (typeof value === "string" && value.trim() !== "") {
        metadataAttributes[key] = value;
      }
    };
    maybePut("source", chunk.source);
    maybePut("chapter", chunk.chapter);
    maybePut("section", chunk.section);
    maybePut("title", chunk.title);
    maybePut("chunkId", chunk.chunkId);
    maybePut("pageImageKey", chunk.pageImageKey);
    maybePut("category", chunk.category);
    maybePut("url", chunk.url);
    // chunkType is always "body" or "toc" — never empty, so we set it
    // directly rather than via maybePut. Filtering in advanced RAG excludes
    // chunkType="toc" before the reranker sees it.
    metadataAttributes["chunkType"] = chunk.chunkType ?? "body";
    // corpus defaults to "statute" so existing Act/Constitution chunks keep
    // their current behaviour; Phase 4a SheriaPlex + KenyaLaw crawls set it
    // to "faq" and the FAQAgent filters accordingly.
    metadataAttributes["corpus"] = chunk.corpus ?? "statute";

    await s3.send(new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: `${key}.metadata.json`,
      Body: JSON.stringify({ metadataAttributes }, null, 2),
      ContentType: "application/json",
    }));
    await onChunkUploaded?.(chunk.chunkId);
    done++;
    const pct = Math.round((done / total) * 100);
    const bar = "█".repeat(Math.floor(pct / 5)) + "░".repeat(20 - Math.floor(pct / 5));
    process.stdout.write(`\r  uploading chunks: [${bar}] ${pct}% — ${done}/${total}`);
  };

  for (let i = 0; i < chunks.length; i += UPLOAD_CONCURRENCY) {
    await Promise.all(chunks.slice(i, i + UPLOAD_CONCURRENCY).map(uploadOne));
  }

  process.stdout.write(`\r  uploading chunks: [${"█".repeat(20)}] 100% — ${total}/${total}\n`);
}
