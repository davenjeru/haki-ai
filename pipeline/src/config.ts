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
// Phase 4a — FAQ corpus (SheriaPlex Q&A + KenyaLaw case summaries).
// Lives under the same bucket as statute chunks but under a distinct prefix
// so the Bedrock KB data source can ingest both side-by-side; retrieval is
// separated by the corpus=faq metadata attribute.
export const FAQ_CHUNKS_PREFIX = "faq-chunks/";

// ~500 tokens at ~4 chars/token
export const CHUNK_CHAR_LIMIT = 2000;

export type LawStructure = "constitution" | "act";

export interface LawConfig {
  name: string;       // full citation name used in metadata
  shortId: string;    // slug used in chunk IDs and S3 keys
  filename: string;   // filename inside data/raw/
  structure: LawStructure;
}

// Optional comma-separated allowlist of shortIds, e.g.
//   LAW_IDS=landlord-and-tenant-act-cap-301,law-of-contract-act
// Useful for smoke-testing new corpus additions before paying LiteParse
// OCR costs on the big statutes (Penal Code, Criminal Procedure Code).
const _LAW_IDS_FILTER: string[] | null = (() => {
  const raw = (process.env.LAW_IDS ?? "").trim();
  if (!raw) return null;
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
})();

const _ALL_LAWS: LawConfig[] = [
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
  // ── Corpus expansion (Tier 1: high query volume, low overlap) ──────────────
  {
    name: "Law of Contract Act",
    shortId: "law-of-contract-act",
    filename: "law-of-contract-act.pdf",
    structure: "act",
  },
  {
    name: "Marriage Act 2014",
    shortId: "marriage-act-2014",
    filename: "marriage-act-2022.pdf",
    structure: "act",
  },
  {
    name: "Penal Code (Cap. 63)",
    shortId: "penal-code-cap-63",
    filename: "penal-code-cap-63.pdf",
    structure: "act",
  },
  {
    name: "Criminal Procedure Code (Cap. 75)",
    shortId: "criminal-procedure-code-cap-75",
    filename: "criminal-procedure-code-cap-75.pdf",
    structure: "act",
  },
  {
    name: "Children Act 2022",
    shortId: "children-act-2022",
    filename: "children-act-2022.pdf",
    structure: "act",
  },
  {
    name: "Landlord and Tenant Act (Cap. 301)",
    shortId: "landlord-and-tenant-act-cap-301",
    filename: "landlord-and-tenant-act-cap-301.pdf",
    structure: "act",
  },
  {
    name: "Consumer Protection Act 2012",
    shortId: "consumer-protection-act-2012",
    filename: "consumer-protection-act-2012.pdf",
    structure: "act",
  },
  {
    name: "Sexual Offences Act 2006",
    shortId: "sexual-offences-act-2006",
    filename: "sexual-offences-act-2006.pdf",
    structure: "act",
  },
];

export const LAWS: LawConfig[] = _LAW_IDS_FILTER
  ? _ALL_LAWS.filter((l) => _LAW_IDS_FILTER!.includes(l.shortId))
  : _ALL_LAWS;
