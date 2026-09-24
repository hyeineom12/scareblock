/**
 * 고정 크기 표본 수집기.
 *
 * 렌더 루프 안에서 쓰므로 **재생 중에 할당하지 않는다.** 미리 잡은
 * Float64Array를 링으로 돌려 쓰고, 정렬은 보고할 때만 한다.
 * 계측이 계측 대상을 바꾸는 함정을 두 번 겪었다 (poc/README.md 참고).
 */
window.SB = window.SB || {};

SB.Samples = class {
  constructor(capacity = 4096) {
    this.buf = new Float64Array(capacity);
    this.seen = 0;
  }

  push(v) {
    this.buf[this.seen % this.buf.length] = v;
    this.seen++;
  }

  get count() {
    return Math.min(this.seen, this.buf.length);
  }

  /** 선형 보간 없는 최근접 순위 백분위수. p는 0~100. */
  percentile(p) {
    const n = this.count;
    if (!n) return null;
    const sorted = Float64Array.prototype.slice.call(this.buf, 0, n).sort();
    const i = Math.min(n - 1, Math.max(0, Math.ceil(p / 100 * n) - 1));
    return sorted[i];
  }

  summary() {
    const n = this.count;
    if (!n) return null;
    return {
      n,
      p50: +this.percentile(50).toFixed(2),
      p90: +this.percentile(90).toFixed(2),
      max: +this.percentile(100).toFixed(2),
    };
  }

  reset() {
    this.seen = 0;
  }
};
