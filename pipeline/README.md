# Haki AI — PDF Pipeline

Two scripts that turn raw Kenyan law PDFs into Bedrock Knowledge Base chunks.

## Run order

```bash
npm run dev    # Script 1 — extract pages
npm run chunk  # Script 2 — LLM chunking
```

Then trigger a Bedrock KB ingestion job to embed and index.

---

## Script 1 — Page extraction (`src/run.ts`)

```mermaid
flowchart TD
    A([data/raw/*.pdf]) --> B[pdf-lib\ngetPageCount]
    B --> C{10-page batch loop}
    C --> D[LiteParse\nOCR extraction]
    D --> E[result.pages]
    E --> F[pdf-lib\ncopyPages — single-page PDF]
    E --> G[raw text per page]
    F --> H[(S3\npage-images/shortId/page-n.pdf)]
    G --> I[(S3\npage-text/shortId/page-n.txt)]
    C -- next batch --> C
    C -- done --> J[upload raw PDF]
    J --> K[(S3\nraw-laws/filename.pdf)]
```

---

## Script 2 — LLM-assisted chunking (`src/chunk-laws.ts`)

```mermaid
flowchart TD
    Start([npm run chunk]) --> Complete{law .complete\nmarker in S3?}
    Complete -- yes --> Skip([skip law])
    Complete -- no --> List[list page-text keys from S3]

    List --> Fetch[fetch all page texts\nfrom S3 in parallel]
    Fetch --> Cache{cached extraction\nin S3?}

    Cache -- yes\npage-extractions/shortId/page-n.json --> Assemble
    Cache -- no --> Haiku

    Haiku[Claude Haiku\nvia Bedrock InvokeModel\nconcurrency=2] --> Retry{success?}
    Retry -- ThrottlingException\nexponential backoff + jitter --> Haiku
    Retry -- success --> SaveCache[(S3\npage-extractions/\nshortId/page-n.json)]
    Retry -- failed after 5 retries --> EmptyExtraction[empty extraction\npage text kept\nunder previous section]

    SaveCache --> Assemble
    EmptyExtraction --> Assemble

    Assemble[assembleChunks\naccumulate lines per section\nacross page boundaries]
    Assemble --> Split[splitToSegments\nmax 2000 chars\nsplit on paragraphs]
    Split --> Upload[uploadChunks]

    Upload --> TXT[(S3\nprocessed-chunks/\nchunkId.txt)]
    Upload --> Meta[(S3\nprocessed-chunks/\nchunkId.txt.metadata.json)]
    Upload --> Marker[(S3\nprocessed-chunks/\nshortId/.complete)]

    Marker --> Next([trigger Bedrock KB\ningestion job])
```

---

## S3 layout

| Prefix | Contents | Written by |
|--------|----------|------------|
| `raw-laws/` | Original law PDFs | `run.ts` |
| `page-images/` | Single-page PDFs — used in citation carousel | `run.ts` |
| `page-text/` | Raw OCR text per page — input to chunking | `run.ts` |
| `page-extractions/` | Cached Haiku JSON — enables resume after throttle | `chunk-laws.ts` |
| `processed-chunks/` | `.txt` + `.txt.metadata.json` pairs for Bedrock KB | `chunk-laws.ts` |

---

## Chunk metadata sidecar

Each chunk gets a `.txt.metadata.json` sidecar in Bedrock KB native format:

```json
{
  "metadataAttributes": {
    "source":       "Employment Act 2007",
    "chapter":      "Part III — Termination of Contract",
    "section":      "Section 40",
    "title":        "Termination of employment",
    "chunkId":      "employment-act-2007-part-iii-section-40",
    "pageImageKey": "page-images/employment-act-2007/page-40.pdf"
  }
}
```

`pageImageKey` is used by the frontend citation carousel to render the source page.

---

## Re-processing a law

```bash
LAW=constitution-2010

# Remove completion marker and chunks
aws s3 rm s3://haki-ai-data/processed-chunks/${LAW}/.complete
aws s3 rm s3://haki-ai-data/processed-chunks/ --recursive \
  --exclude "*" --include "${LAW}-*"

# Remove cached Haiku extractions (only needed to force re-extraction)
aws s3 rm s3://haki-ai-data/page-extractions/${LAW}/ --recursive
```
