/**
 * 프레임 링버퍼.
 *
 * 재생 중에는 새로 할당하지 않는다. 슬롯은 mediaTime 순으로 채워지므로
 * 조회를 이분 탐색으로 한다 — PoC는 매 프레임 전체를 훑었고, 슬롯이
 * 130개로 늘면 그 비용이 무시할 수 없다 (#5 리뷰).
 */
window.SB = window.SB || {};

SB.RingBuffer = class {
  constructor({ width, height, slots }) {
    this.w = width;
    this.h = height;
    this.slots = slots;
    this.buf = Array.from({ length: slots }, () => {
      const c = document.createElement('canvas');
      c.width = width;
      c.height = height;
      return { canvas: c, ctx: c.getContext('2d', { alpha: false }), t: -1 };
    });
    this.head = 0;      // 다음에 쓸 자리
    this.count = 0;     // 채워진 개수 (slots 까지)
  }

  /** 현재 head 자리에 그리고 한 칸 전진한다. */
  push(source, mediaTime) {
    const s = this.buf[this.head];
    s.ctx.drawImage(source, 0, 0, this.w, this.h);
    s.t = mediaTime;
    this.head = (this.head + 1) % this.slots;
    if (this.count < this.slots) this.count++;
    return s;
  }

  /** 시크 등으로 시간 순서가 깨졌을 때 전부 버린다. */
  clear() {
    for (const s of this.buf) s.t = -1;
    this.head = 0;
    this.count = 0;
  }

  /** 논리적 i번째(0 = 가장 오래된 것) 슬롯. */
  at(i) {
    const start = this.count < this.slots ? 0 : this.head;
    return this.buf[(start + i) % this.slots];
  }

  /**
   * mediaTime이 target 이하인 것 중 가장 새로운 슬롯.
   * 논리 순서가 시간 오름차순이라 이분 탐색이 성립한다. O(log n).
   */
  findAtOrBefore(target) {
    let lo = 0;
    let hi = this.count - 1;
    let found = null;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      const s = this.at(mid);
      if (s.t >= 0 && s.t <= target) {
        found = s;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return found;
  }

  get filled() {
    return this.count;
  }

  get estimatedMB() {
    return +(this.slots * this.w * this.h * 4 / 1048576).toFixed(1);
  }
};
