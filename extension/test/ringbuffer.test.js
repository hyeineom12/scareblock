/**
 * 링버퍼 이분 탐색 검증. 브라우저 없이 도는 유일한 부분이다.
 *   node extension/test/ringbuffer.test.js
 *
 * 랩어라운드 뒤에도 논리 순서가 시간 오름차순이어야 이분 탐색이 성립한다.
 * 그게 깨지면 화면이 과거·미래로 튄다.
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');

// 브라우저처럼 window를 전역 객체 자체로 둔다.
// 확장은 `window.SB = ...` 로 전역 SB를 만들므로 이래야 bare SB가 잡힌다.
global.window = global;
global.document = {
  createElement: () => ({
    width: 0, height: 0,
    getContext: () => ({ drawImage() {} }),
  }),
};

eval(fs.readFileSync(path.join(__dirname, '../src/ringbuffer.js'), 'utf8'));
const { RingBuffer } = window.SB;

let pass = 0;
const ok = (name, fn) => {
  try { fn(); console.log(`  ✓ ${name}`); pass++; }
  catch (e) { console.error(`  ✗ ${name}\n    ${e.message}`); process.exitCode = 1; }
};

const make = (slots = 10) => new RingBuffer({ width: 4, height: 4, slots });
const fill = (r, n, step = 0.1) => {
  for (let i = 0; i < n; i++) r.push({}, +(i * step).toFixed(4));
  return r;
};

console.log('링버퍼');

ok('빈 버퍼는 null', () => {
  assert.strictEqual(make().findAtOrBefore(1.0), null);
});

ok('가득 차기 전 — 정확한 슬롯을 찾는다', () => {
  const r = fill(make(10), 5);            // 0.0 0.1 0.2 0.3 0.4
  assert.strictEqual(r.findAtOrBefore(0.25).t, 0.2);
  assert.strictEqual(r.findAtOrBefore(0.4).t, 0.4);
});

ok('모든 값보다 이른 target은 null', () => {
  const r = fill(make(10), 5);
  assert.strictEqual(r.findAtOrBefore(-1), null);
});

ok('모든 값보다 늦은 target은 가장 새것', () => {
  const r = fill(make(10), 5);
  assert.strictEqual(r.findAtOrBefore(99).t, 0.4);
});

ok('랩어라운드 뒤에도 순서가 유지된다', () => {
  const r = fill(make(10), 25);           // 0.0 … 2.4, 마지막 10개만 남음
  assert.strictEqual(r.filled, 10);
  assert.strictEqual(r.at(0).t, 1.5);     // 가장 오래된 것
  assert.strictEqual(r.at(9).t, 2.4);     // 가장 새것
  assert.strictEqual(r.findAtOrBefore(2.05).t, 2.0);
  assert.strictEqual(r.findAtOrBefore(1.5).t, 1.5);
  assert.strictEqual(r.findAtOrBefore(1.4), null);  // 이미 밀려난 시각
});

ok('3초 지연 조회 — 실제 설정과 같은 규모', () => {
  const SLOTS = 130, FPS = 40;
  const r = make(SLOTS);
  for (let i = 0; i < 500; i++) r.push({}, +(i / FPS).toFixed(4));
  const now = 499 / FPS;                  // 12.475s
  const s = r.findAtOrBefore(now - 3.0);
  assert.ok(s, '3초 전 프레임이 있어야 한다');
  const gap = now - s.t;
  assert.ok(gap >= 3.0 && gap < 3.0 + 1 / FPS,
    `지연이 3.0~${(3 + 1 / FPS).toFixed(3)}s 사이여야 하는데 ${gap.toFixed(4)}`);
});

ok('clear 뒤에는 아무것도 없다', () => {
  const r = fill(make(10), 25);
  r.clear();
  assert.strictEqual(r.filled, 0);
  assert.strictEqual(r.findAtOrBefore(99), null);
});

ok('이분 탐색이 선형 탐색과 같은 답을 낸다', () => {
  const r = make(64);
  for (let i = 0; i < 300; i++) r.push({}, +(i * 0.025 + Math.random() * 0.004).toFixed(5));
  for (let k = 0; k < 200; k++) {
    const target = Math.random() * 8;
    let brute = null;
    for (let i = 0; i < r.filled; i++) {
      const s = r.at(i);
      if (s.t >= 0 && s.t <= target && (!brute || s.t > brute.t)) brute = s;
    }
    const fast = r.findAtOrBefore(target);
    assert.strictEqual(fast?.t ?? null, brute?.t ?? null,
      `target=${target} 에서 불일치`);
  }
});

console.log(`\n${pass}/8 통과`);
