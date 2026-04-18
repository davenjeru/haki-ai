export const S3_BUCKET = process.env.S3_BUCKET ?? "haki-ai-data";
export const AWS_REGION = process.env.AWS_REGION ?? "us-east-1";
export const BEDROCK_MODEL_ID =
  process.env.BEDROCK_MODEL_ID ?? "us.anthropic.claude-haiku-4-5-20251001-v1:0";
export const BEDROCK_CONCURRENCY = 2;
export const PAGE_EXTRACTIONS_PREFIX = "page-extractions/";
export const RAW_PREFIX = "raw-laws/";
export const CHUNKS_PREFIX = "processed-chunks/";
export const PAGE_IMAGES_PREFIX = "page-images/";
export const PAGE_TEXT_PREFIX = "page-text/";

// ~500 tokens at ~4 chars/token
export const CHUNK_CHAR_LIMIT = 2000;

export type LawStructure = "constitution" | "act";

export interface LawConfig {
  name: string;       // full citation name used in metadata
  shortId: string;    // slug used in chunk IDs and S3 keys
  filename: string;   // filename inside data/raw/
  structure: LawStructure;
}

export const LAWS: LawConfig[] = [
  {
    name: "Constitution of Kenya 2010",
    shortId: "constitution-2010",
    filename: "constitution-2010.pdf",
    structure: "constitution",
  },
  {
    name: "Employment Act 2007",
    shortId: "employment-act-2007",
    filename: "employment-act-2007.pdf",
    structure: "act",
  },
  {
    name: "Land Act 2012",
    shortId: "land-act-2012",
    filename: "land-act-2012.pdf",
    structure: "act",
  },
];
