/**
 * 재생기 — 지연 재생과 개입.
 *
 * 인터페이스 계약의 **소비자**다. 탐지기가 누구든 아래 배열만 받는다.
 *
 *   [{ time, category, confidence, duration, source }]
 *
 * time은 **라이브 mediaTime**이다. 화면은 그보다 DELAY_SEC 뒤를 보여주므로,
 * 표시 시각이 그 구간에 들어올 때 블러가 걸린다. 탐지와 개입 사이의
 * 그 간격이 이 시스템의 존재 이유다.
 */
window.SB = window.SB || {};

SB.Player = class {
  constructor(video, cfg = SB.config) {
    this.video = video;
    this.cfg = cfg;
    this.triggers = [];
    this.running = false;

    // 계측 — M1 판정은 렌더끊김·디코더드롭 두 축으로 한다 (README M1 참고)
    this.stats = {
      missedFrames: 0,   // rVFC 콜백이 아예 보지 못한 원본 프레임 (사용자는 못 느낌)
      renderStalls: 0,   // 출력이 STALL_MS 넘게 멈춘 횟수
      renderFrames: 0,
      decoderDropsAtStart: 0,
      seeks: 0,
      startedAt: 0,
    };
    this._lastPresented = -1;
    this._lastCap = -1;
    this._lastRenderAt = 0;
    this._lastIncidentAt = 0;   // 마지막 끊김·디코더드롭 시각
    this._hudAt = 0;            // HUD·품질조회 스로틀
    this._drops = 0;            // 마지막으로 읽은 디코더드롭
    this._hudLine1 = '';
    this._hudLine2 = '';
    this._gainNow = 1;
  }

  /** 계약 소비 — 탐지기가 부른다. */
  addTriggers(list) {
    for (const t of list) {
      if (typeof t.time !== 'number' || typeof t.duration !== 'number') {
        console.warn('[scareblock] 계약에 맞지 않는 트리거', t);
        continue;
      }
      this.triggers.push(t);
    }
    if (this.triggers.length > 500) this.triggers.splice(0, this.triggers.length - 500);
  }

  triggerAt(mediaTime) {
    return this.triggers.find(
      (t) => mediaTime >= t.time && mediaTime <= t.time + t.duration
    ) || null;
  }

  start() {
    const { video, cfg } = this;
    const slots = Math.ceil(cfg.DELAY_SEC * cfg.CAPTURE_FPS) + 10;
    this.ring = new SB.RingBuffer({ width: cfg.BUF_W, height: cfg.BUF_H, slots });

    const player = video.closest('#movie_player') || video.parentElement;
    const out = document.createElement('canvas');
    out.width = cfg.BUF_W;
    out.height = cfg.BUF_H;
    out.style.cssText =
      'position:absolute;inset:0;width:100%;height:100%;z-index:1;background:#000';
    player.appendChild(out);
    this.out = out;
    this.octx = out.getContext('2d', { alpha: false });

    this._prevOpacity = video.style.opacity;
    video.style.opacity = '0';

    this._setupAudio();

    this.stats.decoderDropsAtStart = this._decoderDrops();
    this.stats.startedAt = performance.now();
    this.running = true;

    video.requestVideoFrameCallback(this._onFrame);
    requestAnimationFrame(this._render);
    SB.log(
      `지연 재생 시작 — ${cfg.DELAY_SEC}s / ${cfg.CAPTURE_FPS}fps / ` +
      `${cfg.BUF_W}x${cfg.BUF_H} / ${slots}슬롯 ≈ ${this.ring.estimatedMB}MB`
    );
  }

  /**
   * 오디오 그래프. 노드는 재사용하되 **연결은 매번 다시 세운다** —
   * stop()이 만든 직결이 남아 영상만 지연되던 버그가 있었다 (#5 리뷰 🔴1).
   */
  _setupAudio() {
    if (!SB._audio) {
      const ac = new AudioContext();
      const src = ac.createMediaElementSource(this.video);
      SB._audio = {
        ac,
        src,
        delay: ac.createDelay(10),
        gain: ac.createGain(),
        analyser: Object.assign(ac.createAnalyser(), { fftSize: 1024 }),
      };
    }
    const { ac, src, delay, gain, analyser } = SB._audio;
    try {
      src.disconnect();
      delay.disconnect();
      gain.disconnect();
    } catch { /* 아직 안 걸림 */ }

    src.connect(analyser);          // 지연 전 — 탐지기가 미래를 먼저 듣는다
    src.connect(delay);
    delay.connect(gain);
    gain.connect(ac.destination);   // 사용자에게는 지연된 소리만
    delay.delayTime.value = this.cfg.DELAY_SEC;
    gain.gain.value = 1;
    ac.resume();
    this.audio = SB._audio;
  }

  _decoderDrops() {
    return this.video.getVideoPlaybackQuality?.().droppedVideoFrames ?? 0;
  }

  _onFrame = (_now, meta) => {
    if (!this.running) return;
    const t = meta.mediaTime;

    if (this._lastPresented >= 0 && meta.presentedFrames > this._lastPresented + 1) {
      this.stats.missedFrames += meta.presentedFrames - this._lastPresented - 1;
    }
    this._lastPresented = meta.presentedFrames;

    // 뒤로 시크하면 버퍼의 시간 순서가 깨져 이분 탐색이 성립하지 않는다. 비운다.
    if (t < this._lastCap) {
      this.ring.clear();
      this.stats.seeks++;
      this._lastCap = -1;
    }

    if (this._lastCap < 0 || t - this._lastCap >= 1 / this.cfg.CAPTURE_FPS - 0.002) {
      this.ring.push(this.video, t);
      this._lastCap = t;
    }
    this.video.requestVideoFrameCallback(this._onFrame);
  };

  _render = () => {
    if (!this.running) return;
    const { octx, cfg } = this;
    const now = performance.now();
    if (this._lastRenderAt && now - this._lastRenderAt > cfg.STALL_MS) {
      this.stats.renderStalls++;
      this._lastIncidentAt = now;
    }
    this._lastRenderAt = now;
    this.stats.renderFrames++;

    // getVideoPlaybackQuality()와 HUD 문자열 생성은 프레임마다 할 일이 아니다.
    // 4Hz로 낮춘다 — 프레임당 비용이 끊김의 원인이 될 수 있다.
    if (now - this._hudAt > 250) {
      this._hudAt = now;
      const d = this._decoderDrops() - this.stats.decoderDropsAtStart;
      if (d > this._drops) this._lastIncidentAt = now;
      this._drops = d;
      this._composeHud(d, now);
    }

    const slot = this.ring.findAtOrBefore(this.video.currentTime - cfg.DELAY_SEC);
    if (slot) {
      const hit = this.triggerAt(slot.t);
      octx.filter = hit ? `blur(${cfg.BLUR_PX}px)` : 'none';
      octx.drawImage(slot.canvas, 0, 0, cfg.BUF_W, cfg.BUF_H);
      octx.filter = 'none';
      const want = hit ? cfg.FADE_DB : 1;            // 음량 페이드다운
      if (want !== this._gainNow) {
        this._gainNow = want;
        if (this.audio?.gain) this.audio.gain.gain.value = want;
      }
      if (hit) this._drawBadge(hit);
    } else {
      octx.fillStyle = '#000';
      octx.fillRect(0, 0, cfg.BUF_W, cfg.BUF_H);
      octx.fillStyle = '#fff';
      octx.font = '20px system-ui, sans-serif';
      octx.fillText('안전 버퍼 채우는 중…', 28, 52);
    }
    this._drawHud();
    requestAnimationFrame(this._render);
  };

  _drawBadge(hit) {
    const { octx, cfg } = this;
    octx.fillStyle = 'rgba(0,0,0,.72)';
    octx.fillRect(cfg.BUF_W / 2 - 150, cfg.BUF_H / 2 - 26, 300, 52);
    octx.fillStyle = '#fff';
    octx.font = '22px system-ui, sans-serif';
    octx.textAlign = 'center';
    octx.fillText(`${hit.category} 가림`, cfg.BUF_W / 2, cfg.BUF_H / 2 + 8);
    octx.textAlign = 'left';
  }

  /** 4Hz로만 부른다. 문자열 생성과 품질 조회를 프레임에서 뺀다. */
  _composeHud(drops, now) {
    const { cfg, stats } = this;
    const sec = (now - stats.startedAt) / 1000;
    this._hudLine1 =
      `지연 ${cfg.DELAY_SEC}s · 버퍼 ${this.ring.filled}/${this.ring.slots} · ` +
      `트리거 ${this.triggers.length} · 시크 ${stats.seeks}`;
    this._hudLine2 =
      `무드롭 ${this.cleanStreakSec(now).toFixed(0)}s · 끊김 ${stats.renderStalls} · ` +
      `드롭 ${drops} · 누락 ${stats.missedFrames} · ` +
      `${(stats.renderFrames / sec).toFixed(0)}fps`;
  }

  /** 마지막 끊김·드롭 이후 흐른 시간. M1의 「60초 이상 유지」가 이것이다. */
  cleanStreakSec(now = performance.now()) {
    return (now - Math.max(this.stats.startedAt, this._lastIncidentAt)) / 1000;
  }

  _drawHud() {
    const { octx, cfg } = this;
    octx.fillStyle = 'rgba(0,0,0,.6)';
    octx.fillRect(8, cfg.BUF_H - 52, 520, 44);
    octx.fillStyle = '#0f0';
    octx.font = '13px ui-monospace, monospace';
    octx.fillText(this._hudLine1, 14, cfg.BUF_H - 34);
    octx.fillText(this._hudLine2, 14, cfg.BUF_H - 16);
  }

  report() {
    const sec = (performance.now() - this.stats.startedAt) / 1000;
    return {
      해상도: `${this.cfg.BUF_W}x${this.cfg.BUF_H}@${this.cfg.CAPTURE_FPS}fps`,
      지연: this.cfg.DELAY_SEC,
      버퍼적재: `${this.ring.filled}/${this.ring.slots}`,
      메모리추정MB: this.ring.estimatedMB,
      경과초: +sec.toFixed(1),
      적재누락프레임: this.stats.missedFrames,
      렌더끊김: this.stats.renderStalls,
      디코더드롭: this._decoderDrops() - this.stats.decoderDropsAtStart,
      렌더fps: +(this.stats.renderFrames / sec).toFixed(1),
      시크: this.stats.seeks,
      트리거수: this.triggers.length,
      // 누적은 오래 돌수록 커진다. 비율로도 낸다.
      끊김_분당: +(this.stats.renderStalls / (sec / 60)).toFixed(2),
      누락_초당: +(this.stats.missedFrames / sec).toFixed(1),
      // M1 판정 — 「60초 이상 유지」는 누적 0이 아니라 연속 무드롭이다
      연속무드롭초: +this.cleanStreakSec().toFixed(1),
      M1통과: this.cleanStreakSec() >= 60,
    };
  }

  stop() {
    this.running = false;
    this.out?.remove();
    this.video.style.opacity = this._prevOpacity ?? '';
    const a = SB._audio;
    if (a) {
      try {
        a.delay.disconnect();
        a.gain.disconnect();
        a.src.disconnect();
        a.src.connect(a.ac.destination);
      } catch { /* 이미 끊김 */ }
    }
    SB.log('원복 완료');
  }
};
