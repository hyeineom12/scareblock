/**
 * 가짜 탐지기 — M1 전용.
 *
 * 인터페이스 계약의 **생산자**다. 02단계의 규칙 기반 탐지기가 이 자리에
 * 그대로 들어온다. 재생기는 어느 쪽이 붙었는지 알 필요가 없다.
 *
 * M1이 확인하려는 것은 "탐지가 맞는가"가 아니라
 * **"계약대로 넘기면 도달 전에 가려지는가"** 다.
 */
window.SB = window.SB || {};

SB.FakeDetector = class {
  /** @param {number} everySec 몇 초마다 가짜 트리거를 낼지 */
  constructor(video, player, everySec = 10) {
    this.video = video;
    this.player = player;
    this.everySec = everySec;
    this.timer = null;
    this.n = 0;
  }

  start() {
    this.timer = setInterval(() => {
      if (this.video.paused) return;
      const at = this.video.currentTime;      // 라이브 시각으로 예약한다
      const cat = ['jumpscare', 'blood', 'siren'][this.n++ % 3];
      this.player.addTriggers([
        { time: at, category: cat, confidence: 0.99, duration: 1.5, source: 'fake' },
      ]);
      SB.log(
        `가짜 트리거 ${cat} t=${at.toFixed(2)}s → ` +
        `${SB.config.DELAY_SEC}s 뒤 도달 전 블러`
      );
    }, this.everySec * 1000);
  }

  stop() {
    clearInterval(this.timer);
  }
};
