// Dependency-free DOM substitute: check the HTTP consumer, safe text rendering,
// Markdown, and tab storage. This is not a visual or real-browser test.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const fixture = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const source = fs.readFileSync(process.argv[3], 'utf8');
class Node {
  constructor(tag = 'div') {
    this.tagName = tag; this.children = []; this.handlers = {};
    this.value = ''; this.hidden = false; this.dataset = {};
    this.classList = { add() {}, remove() {}, toggle() {} };
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return (this.text || '') + this.children.map((item) => item.textContent).join('\n'); }
  set innerHTML(_) { throw new Error('Untrusted content must use textContent'); }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.text = ''; this.children = items; }
  addEventListener(name, callback) { this.handlers[name] = callback; }
  setAttribute() {}
  focus() {}
  scrollIntoView() {}
}
const stored = new Map([['research-desk.run.v1', JSON.stringify({state: {correction_count: 1}})]]);
function boot() {
  const nodes = new Map();
  const context = vm.createContext({
    console, setTimeout, clearTimeout,
    sessionStorage: {getItem: (key) => stored.get(key), setItem: (key, value) => stored.set(key, value), removeItem: (key) => stored.delete(key)},
    document: {getElementById: (id) => {
      if (!nodes.has(id)) nodes.set(id, new Node());
      return nodes.get(id);
    }, createElement: (tag) => new Node(tag), querySelectorAll: () => [], body: new Node('body')},
    fetch: async () => ({ok: true, json: async () => ({ready: true, max_upload_mb: 4})}),
    matchMedia: () => ({matches: false}),
  });
  vm.runInContext(source, context);
  return {context, nodes, evaluate: (code) => vm.runInContext(code, context)};
}
const page = boot();
assert.equal(page.evaluate('run'), null, 'old schema must not restore');
page.context.messages = fixture.messages;
page.evaluate('messages.forEach(acceptMessage)');
assert.equal(page.evaluate('run.state.status'), fixture.status);
const body = page.nodes.get('report-content').textContent;
assert.match(body, /Paper explanation — report draft/);
assert.match(body, /Final critical review/);
assert.match(body, /Remaining uncertainties/);
assert.match(body, /Source coverage/);
assert.match(body, /not proof of understanding/);
assert.match(body, /not supplied/);
assert.equal(page.nodes.get('run-counts').textContent.includes('correction'), false);
const markdown = page.evaluate('markdownReport()');
assert.match(markdown, /Paper explanation — report draft/);
assert.match(markdown, /Referenced source passages/);
assert.match(markdown, /E0001 · PDF page 1/);
assert.match(markdown, /Our original study compares two sorting methods/);
if (fixture.status === 'complete') {
  assert.match(markdown, /Final critical review/);
  if (fixture.target) {
    assert.match(body, new RegExp(fixture.target.toUpperCase() + ' / unsupported comparison'));
    assert.match(markdown, new RegExp(fixture.target.toUpperCase() + ' concern'));
    assert.match(body, /The comparison goes beyond the reported experiment/);
    if (fixture.target === 'report') {
      assert.match(body, /Affected draft statement: Original finding/);
      assert.match(body, /has not been rewritten or corrected/);
      assert.match(markdown, /Affected draft statement/);
    }
  } else assert.match(body, /limited inspection/i);
} else if (fixture.status === 'incomplete') {
  assert.match(body, /did not receive a completed final critical review/);
  assert.match(markdown, /did not receive a completed final critical review/);
} else {
  assert.match(body, /Awaiting the final reviewer/);
  assert.match(markdown, /did not receive a completed final critical review/);
}
if (fixture.revision) {
  assert.match(body, /Optional targeted revision/);
  assert.match(markdown, /Optional targeted revision/);
  assert.match(markdown, /not reviewed again/);
  assert.match(body, /The comparison goes beyond the reported experiment/);
}
assert.match(page.nodes.get('run-metrics').textContent, /Run metrics/);
assert.ok(page.nodes.get('cancel-run').handlers.click);
page.evaluate('$("access-token").value = "private-browser-access-token"');
page.evaluate('save()');
assert.equal(page.evaluate('sessionStorage.getItem(storageKey)').includes('private-browser-access-token'), false);
const fresh = boot();
assert.ok(fresh.evaluate('run.state.report_draft.findings.length'));
assert.equal(fresh.evaluate('run.state.status'), fixture.status === 'running' ? 'incomplete' : fixture.status);
assert.match(fresh.nodes.get('report-content').textContent, /Paper explanation — report draft/);
if (fixture.status === 'running') {
  assert.match(fresh.evaluate('markdownReport()'), /did not receive a completed final critical review/);
}
// Document/model markup is displayed as literal text, never interpreted as HTML.
page.evaluate('run.state.report_draft.findings[0].statement = "<img src=x onerror=alert(1)>"; renderReport()');
assert.match(page.nodes.get('report-content').textContent, /<img src=x onerror=alert\(1\)>/);
console.log('HTTP/UI contract, draft/review rendering, Markdown, safe text, and restore passed.');
