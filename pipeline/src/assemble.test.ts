/**
 * Unit tests for the law chunker (`assembleChunks`).
 *
 * Run with: npx tsx src/assemble.test.ts
 *
 * These are the regression tests for the boundary-bleed bug documented
 * in `swahili_retrieval_ulizallama.plan.md` §5b (i). Before the fix,
 * opening Section N's node eagerly captured `rawLines.slice(bodyStartLine)`
 * — so Section 1's chunk body contained the full Section 2 body too, and
 * a "Section 1" metadata tag ended up on a chunk that embedded like a
 * flat definitions list. The tests below guard against that regression
 * and verify the new `chunkType` classifications.
 */

import assert from "node:assert/strict";
import { assembleChunks } from "./assemble.js";
import type { PageExtraction } from "./llm-extract.js";
import type { LawConfig } from "./config.js";

const LAW: LawConfig = {
  name: "Test Act 2024",
  shortId: "test-act-2024",
  structure: "act",
  filename: "test-act-2024.pdf",
};

// ── Boundary bleed: two sections on the same page should not overlap ─────

{
  // A single page containing "PART I" heading, Section 1 body, then
  // Section 2 body. The page looks like this (0-indexed line numbers
  // in comments):
  // Section bodies are padded above the 50-char minimum enforced by
  // `assembleChunks` so both chunks survive the body-length filter.
  const pageText = [
    "PART I – PRELIMINARY PROVISIONS",                     // 0 — chapter heading
    "1. Short title and commencement",                      // 1 — section 1 header
    "This Act may be cited as the Test Act, 2024 and shall come into operation on gazettement.", // 2 — section 1 body
    "2. Interpretation",                                    // 3 — section 2 header
    "In this Act, unless the context otherwise requires—",  // 4 — section 2 body
    "\"minor\" means any person under eighteen years of age.", // 5 — section 2 body
    "\"tenant\" has the meaning ascribed to it in section 5.", // 6
  ].join("\n");

  const pageTexts = new Map<number, string>([[10, pageText]]);
  const extractions: PageExtraction[] = [
    {
      pageNum: 10,
      chapterOrPart: "PART I – PRELIMINARY PROVISIONS",
      sections: [
        { number: "1", title: "Short title and commencement", bodyStartLine: 2 },
        { number: "2", title: "Interpretation", bodyStartLine: 4 },
      ],
    },
  ];

  const chunks = assembleChunks(LAW, pageTexts, extractions);
  assert.equal(
    chunks.length,
    2,
    `expected 2 chunks (Section 1 + Section 2), got ${chunks.length}`
  );

  const s1 = chunks.find((c) => c.section === "Section 1");
  const s2 = chunks.find((c) => c.section === "Section 2");
  assert.ok(s1, "Section 1 chunk missing");
  assert.ok(s2, "Section 2 chunk missing");

  // Regression: Section 1's body must NOT contain Section 2's header or
  // any of its definition lines. Before the fix, s1!.text was the full
  // remainder of the page starting at bodyStartLine=2.
  assert.ok(
    !s1!.text.includes("2. Interpretation"),
    `Section 1 leaked the Section 2 header:\n${s1!.text}`
  );
  assert.ok(
    !s1!.text.includes("means any person under 18"),
    `Section 1 leaked a Section 2 definition:\n${s1!.text}`
  );
  assert.ok(
    s1!.text.includes("This Act may be cited"),
    `Section 1 dropped its own body:\n${s1!.text}`
  );

  // Section 2 should start with its body, not with Section 1's short title.
  assert.ok(
    s2!.text.startsWith("In this Act") || s2!.text.startsWith("\""),
    `Section 2 body has leading noise:\n${s2!.text.slice(0, 80)}`
  );
  assert.ok(
    !s2!.text.includes("This Act may be cited"),
    `Section 2 leaked Section 1's short title:\n${s2!.text}`
  );

  console.log("pass: boundary-bleed fix (two sections on same page)");
}

// ── chunkType classification: preamble / short-title / definitions ───────

{
  // Minimal pre-section preamble page followed by a page with Sections 1,
  // 2, 3. We expect four chunks whose chunkType reflects their role.
  const preamblePage = [
    "THE REPUBLIC OF KENYA",
    "THE TEST ACT, 2024",
    "ARRANGEMENT WHATEVER",
    "(Preliminary cover text exceeds fifty characters to pass the chunk threshold.)",
  ].join("\n");
  const bodyPage = [
    "PART I – PRELIMINARY PROVISIONS",
    "1. Short title",
    "This Act may be cited as the Test Act, 2024. It takes effect on gazettement.",
    "2. Interpretation",
    "In this Act—",
    "\"court\" means the High Court or any subordinate court.",
    "\"Minister\" means the Cabinet Secretary responsible for the subject matter.",
    "3. Scope",
    "This Act applies to every natural and juridical person in Kenya.",
  ].join("\n");

  const pageTexts = new Map<number, string>([
    [1, preamblePage],
    [10, bodyPage],
  ]);
  const extractions: PageExtraction[] = [
    { pageNum: 1, chapterOrPart: null, sections: [] },
    {
      pageNum: 10,
      chapterOrPart: "PART I – PRELIMINARY PROVISIONS",
      sections: [
        { number: "1", title: "Short title", bodyStartLine: 2 },
        { number: "2", title: "Interpretation", bodyStartLine: 4 },
        { number: "3", title: "Scope", bodyStartLine: 8 },
      ],
    },
  ];

  const chunks = assembleChunks(LAW, pageTexts, extractions);
  const bySection = new Map(chunks.map((c) => [c.section, c]));

  const preamble = bySection.get("Preamble");
  assert.ok(preamble, "Preamble chunk missing");
  assert.equal(
    preamble!.chunkType,
    "preamble",
    `Preamble should be classified as 'preamble', got ${preamble!.chunkType}`
  );

  const s1 = bySection.get("Section 1")!;
  assert.equal(
    s1.chunkType,
    "short-title",
    `Section 1 should be 'short-title', got ${s1.chunkType}`
  );

  const s2 = bySection.get("Section 2")!;
  assert.equal(
    s2.chunkType,
    "definitions",
    `Section 2 should be 'definitions', got ${s2.chunkType}`
  );

  const s3 = bySection.get("Section 3")!;
  assert.equal(
    s3.chunkType,
    "body",
    `Section 3 should be 'body', got ${s3.chunkType}`
  );

  console.log("pass: chunkType classification (preamble / short-title / definitions / body)");
}

// ── Multi-page section: continuation pages must not duplicate content ────

{
  // Section 5's body spans pages 20 and 21. Page 21 has no section
  // headers, so its entire content should flow into Section 5 — with
  // no duplication of page 20's lines.
  const page20 = [
    "5. Forms of tenure",
    "(1) There shall be the following forms of land tenure—",
    "(a) freehold;",
    "(b) leasehold;",
  ].join("\n");
  const page21 = [
    "(c) customary land rights, where consistent with the Constitution.",
    "(2) There shall be equal recognition of all tenure systems.",
  ].join("\n");

  const pageTexts = new Map<number, string>([
    [20, page20],
    [21, page21],
  ]);
  const extractions: PageExtraction[] = [
    {
      pageNum: 20,
      chapterOrPart: null,
      sections: [{ number: "5", title: "Forms of tenure", bodyStartLine: 1 }],
    },
    { pageNum: 21, chapterOrPart: null, sections: [] },
  ];

  const chunks = assembleChunks(LAW, pageTexts, extractions);
  assert.equal(chunks.length, 1, `expected 1 Section 5 chunk, got ${chunks.length}`);
  const s5 = chunks[0];
  assert.equal(s5.section, "Section 5");

  // Content from both pages should be present exactly once.
  const freeholdCount = (s5.text.match(/freehold/g) ?? []).length;
  const customaryCount = (s5.text.match(/customary/g) ?? []).length;
  assert.equal(freeholdCount, 1, `"freehold" duplicated ${freeholdCount}x`);
  assert.equal(customaryCount, 1, `"customary" duplicated ${customaryCount}x`);

  console.log("pass: multi-page section concatenation without duplication");
}

console.log("\nAll assemble.test.ts assertions passed.");
