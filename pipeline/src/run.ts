/**
 * Script 1 of 2: Page extraction pipeline.
 *
 * For each law PDF in data/raw/:
 *   - Extracts raw text via LiteParse (OCR), processed in 10-page batches.
 *   - Extracts each page as a single-page PDF using pdf-lib (no rendering — fast).
 *   - Uploads page text  → page-text/{shortId}/page-N.txt    (input for chunking)
 *   - Uploads page PDFs  → page-images/{shortId}/page-N.pdf  (citation carousel)
 *   - Uploads the full PDF → raw-laws/{filename}
 *
 * Run with: npm run dev          (real AWS S3)
 *           npm run dev:local    (LocalStack)
 */
import { LiteParse } from "@llamaindex/liteparse";
import { PDFDocument } from "pdf-lib";
import { existsSync, readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { LAWS } from "./config.js";
import { uploadRawPdf, uploadPagePdf, uploadPageText } from "./upload.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(__dirname, "..", "..", "data", "raw");

// LiteParse parses in batches. 10 pages balances OCR speed vs. memory usage.
const PAGES_PER_BATCH = 10;

console.log("Haki AI — page extraction pipeline\n");

for (const law of LAWS) {
  const pdfPath = join(DATA_DIR, law.filename);

  if (!existsSync(pdfPath)) {
    console.warn(`SKIP ${law.filename} — not found in data/raw/`);
    continue;
  }

  console.log(`Processing: ${law.name}`);

  // Load PDF once — pdf-lib gives us total page count and fast page extraction
  const pdfBytes = readFileSync(pdfPath);
  const srcPdf = await PDFDocument.load(pdfBytes);
  const totalPages = srcPdf.getPageCount();
  console.log(`  ${totalPages} pages`);

  let batchStart = 1;

  while (batchStart <= totalPages) {
    const batchEnd = Math.min(batchStart + PAGES_PER_BATCH - 1, totalPages);
    const targetPages = `${batchStart}-${batchEnd}`;

    process.stdout.write(`  pages ${batchStart}–${batchEnd}: parsing... `);

    // targetPages limits OCR to this batch's range, avoiding full-document reprocessing.
    const parser = new LiteParse({ ocrEnabled: true, targetPages });
    const result = await parser.parse(pdfPath, /* quiet */ true);

    for (const parsedPage of result.pages) {
      const { pageNum, text } = parsedPage;

      // Single-page PDF — extracted with pdf-lib (no rendering, just copy)
      const dest = await PDFDocument.create();
      const [extractedPage] = await dest.copyPages(srcPdf, [pageNum - 1]);
      dest.addPage(extractedPage);
      const pageBytes = await dest.save();
      await uploadPagePdf(pageBytes, law.shortId, pageNum);

      // Raw text from LiteParse — this is what we'll study to build chunking
      await uploadPageText(text, law.shortId, pageNum);
    }

    console.log(`${result.pages.length} pages done.`);
    batchStart = batchEnd + 1;
  }

  process.stdout.write("  uploading raw PDF... ");
  await uploadRawPdf(pdfPath, law.filename);
  console.log("done.\n");
}

console.log("All done.");
console.log("Next: review s3://haki-ai-data/page-text/ to build chunking strategy.");
