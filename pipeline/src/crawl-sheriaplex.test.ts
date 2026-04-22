/**
 * Lightweight unit tests for the SheriaPlex crawler.
 *
 * Run with: npx tsx src/crawl-sheriaplex.test.ts
 *
 * No test framework — the pipeline package deliberately stays lean. The
 * assertions here are mostly smoke tests: they verify that extraction
 * still works against a frozen snapshot of the HTML so selector
 * regressions (a SheriaPlex redesign, for instance) are caught the next
 * time CI runs the pipeline's typecheck + tests.
 */

import assert from "node:assert/strict";
import { categorise, extractForumLinks, extractThread } from "./crawl-sheriaplex.js";

const THREAD_HTML = `
<!doctype html>
<html>
<body>
  <main>
    <h2 class="mb-4">Test for Employment Termination in Kenya</h2>
    <div class="details-container">
      In Kenya, the Employment Act 2007 provides the framework for fair vs. unfair termination.
    </div>
    <div class="comment-box">
      <span class="comment-author">Bauni Kithinji Advocates</span> said:
      <div class="comment-content">
        Substantive justification and procedural fairness are both required under Sections 41, 43 and 45 of the Act.
        <br />Employers bear the burden of proof.
      </div>
    </div>
    <div class="related-questions-list">
      <a href="/forum/15-what-constitutes-unfair-dismissal-from-work-in-kenya">Unfair dismissal</a>
    </div>
  </main>
</body>
</html>
`;

const INDEX_HTML = `
<!doctype html>
<html><body>
  <a href="/forum/476-test-for-employment-termination-in-kenya">t1</a>
  <a href="https://www.sheriaplex.com/forum/63-what-are-my-rights-if-i-m-unfairly-dismissed-from-my-job">t2</a>
  <a href="/forum/">no-number</a>
  <a href="/kenya-acts/1463-section-49">not-forum</a>
  <a href="https://evil.com/forum/99-spam">not-sheriaplex</a>
</body></html>
`;

let passed = 0;
let failed = 0;

function it(name: string, body: () => void): void {
  try {
    body();
    console.log(`  [ok] ${name}`);
    passed++;
  } catch (err) {
    console.error(`  [FAIL] ${name}: ${err instanceof Error ? err.message : err}`);
    failed++;
  }
}

console.log("extractThread");
it("pulls title, question, and answer text", () => {
  const url = "https://www.sheriaplex.com/forum/476-test-for-employment-termination-in-kenya";
  const thread = extractThread(THREAD_HTML, url);
  assert.ok(thread);
  assert.equal(thread.title, "Test for Employment Termination in Kenya");
  assert.equal(thread.id, "476");
  assert.equal(thread.slug, "test-for-employment-termination-in-kenya");
  assert.match(thread.question, /Employment Act 2007/);
  assert.equal(thread.answers.length, 1);
  assert.equal(thread.answers[0].author, "Bauni Kithinji Advocates");
  assert.match(thread.answers[0].body, /Sections 41, 43 and 45/);
  assert.match(thread.answers[0].body, /burden of proof/);
});

it("returns null when title is missing", () => {
  const result = extractThread("<html><body>no h2</body></html>", "https://x/forum/1-x");
  assert.equal(result, null);
});

console.log("extractForumLinks");
it("keeps only /forum/{id}-slug URLs on the sheriaplex origin", () => {
  const links = extractForumLinks(INDEX_HTML);
  assert.equal(links.length, 2);
  assert.ok(links.some((l) => l.endsWith("476-test-for-employment-termination-in-kenya")));
  assert.ok(links.some((l) => l.endsWith("63-what-are-my-rights-if-i-m-unfairly-dismissed-from-my-job")));
});

console.log("categorise");
it("maps termination questions to employment", () => {
  assert.equal(categorise("Test for Employment Termination in Kenya", ""), "employment");
});
it("maps land disputes to land", () => {
  assert.equal(categorise("Tenant eviction procedure", ""), "land");
});
it("maps bill of rights to constitution", () => {
  assert.equal(categorise("What does the Bill of Rights say about fair hearing?", ""), "constitution");
});
it("falls back to general when no keywords match", () => {
  assert.equal(categorise("Something unrelated to statutes", ""), "general");
});

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
