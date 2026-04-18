/** Matches Comprehend + Lambda contract described in CLAUDE.md */
export type DetectedLanguage = 'english' | 'swahili' | 'mixed'

export interface Citation {
  /** Act or instrument name, e.g. "Employment Act 2007" */
  source: string
  chapter?: string
  section?: string
  title?: string
  /** Presigned S3 URL for the source PDF page image */
  pageImageUrl?: string
}

/** Expected Lambda JSON shape: { response, citations, language, blocked } */
export interface ChatResponse {
  response: string
  citations: Citation[]
  language: DetectedLanguage
  blocked: boolean
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  language?: DetectedLanguage
  blocked?: boolean
  error?: string
}
