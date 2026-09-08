/**
 * scareblock 00단계 PoC — 3초 지연 재생
 *
 * 유튜브 영상 페이지에서 콘솔에 통째로 붙여넣으면 동작한다.
 * 영상을 DELAY_SEC 만큼 늦게 보여주고, 그 사이 라이브(지연 전) 오디오에서
 * 갑툭튀 후보를 찾아 사용자가 그 장면에 도달하기 전에 블러를 건다.
 *
 *   scareblockDemo.stop()          원복
 *   scareblockDemo.blurNow(1.5)    수동 블러 — 지금 소리 나는 장면을 도달 전에 가린다
 *   scareblockDemo.stats()         버퍼·탐지 상태
 *
 * PoC 한계 (01단계에서 해결) — 캔버스로 다시 그리므로 전체화면·자막·화질설정이
 * 동작하지 않고, 해상도가 BUF_W×BUF_H로 고정된다. 시크·일시정지는 대응하지 않는다.
 */
(() => {
  'use strict';

  const DELAY_SEC   = 3.0;   // 지연 버퍼 길이
  const CAPTURE_FPS = 20;    // 링버퍼 적재 속도 (메모리와 맞바꾼다)
  const BUF_W = 640, BUF_H = 360;
  const BLUR_PX = 28;
  const DETECT = true;       // 오디오 급등 탐지 (조악한 규칙 — 02단계에서 제대로)

  if (window.scareblockDemo) { window.scareblockDemo.stop(); }

  const video = document.querySelector('video.html5-main-video') || document.querySelector('video');
  if (!video) { console.error('[scareblock] video 엘리먼트를 못 찾았다'); return; }
  const player = video.closest('#movie_player') || video.parentElement;

  // ── 링버퍼: 재생 중에는 새로 할당하지 않는다 ────────────────────────────
  const SLOTS = Math.ceil(DELAY_SEC * CAPTURE_FPS) + 10;
  const ring = Array.from({ length: SLOTS }, () => {
    const c = document.createElement('canvas');
    c.width = BUF_W; c.height = BUF_H;
    return { canvas: c, ctx: c.getContext('2d', { alpha: false }), t: -1 };
  });
  let head = 0, lastCap = -1;

  // ── 출력 캔버스 ─────────────────────────────────────────────────────────
  const out = document.createElement('canvas');
  out.width = BUF_W; out.height = BUF_H;
  out.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;z-index:1;background:#000';
  const octx = out.getContext('2d', { alpha: false });
  player.appendChild(out);
  const prevOpacity = video.style.opacity;
  video.style.opacity = '0';

  // ── 오디오: 라이브 탭(탐지용) + 지연 경로(재생용) ───────────────────────
  // createMediaElementSource는 엘리먼트당 한 번만 가능해서 전역에 재사용한다.
  if (!window.__sbAudio) {
    const ac = new AudioContext();
    const src = ac.createMediaElementSource(video);
    const delay = ac.createDelay(10);
    const analyser = ac.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);           // 지연 전 — 미래를 먼저 듣는다
    src.connect(delay);
    delay.connect(ac.destination);   // 사용자에게는 지연된 소리만
    window.__sbAudio = { ac, src, delay, analyser };
  }
  const { ac, src, delay, analyser } = window.__sbAudio;
  delay.delayTime.value = DELAY_SEC;
  ac.resume();

  // ── 블러 구간: 라이브 시각 기준으로 예약하고, 표시 시각으로 판정한다 ────
  const blurs = [];
  const isBlurred = (t) => blurs.some((b) => t >= b.from && t <= b.to);
  const scheduleBlur = (at, dur) => {
    blurs.push({ from: at, to: at + dur });
    if (blurs.length > 200) blurs.shift();
  };

  let running = true, detections = 0;

  // ── 프레임 적재 ─────────────────────────────────────────────────────────
  function onFrame(_now, meta) {
    if (!running) return;
    const t = meta.mediaTime;
    if (lastCap < 0 || t - lastCap >= 1 / CAPTURE_FPS - 0.002) {
      const s = ring[head];
      s.ctx.drawImage(video, 0, 0, BUF_W, BUF_H);
      s.t = t;
      head = (head + 1) % SLOTS;
      lastCap = t;
    }
    video.requestVideoFrameCallback(onFrame);
  }
  video.requestVideoFrameCallback(onFrame);

  // ── 지연 렌더 ───────────────────────────────────────────────────────────
  function render() {
    if (!running) return;
    const target = video.currentTime - DELAY_SEC;
    let best = null;
    for (const s of ring) {
      if (s.t >= 0 && s.t <= target && (!best || s.t > best.t)) best = s;
    }

    if (best) {
      octx.filter = isBlurred(best.t) ? `blur(${BLUR_PX}px)` : 'none';
      octx.drawImage(best.canvas, 0, 0, BUF_W, BUF_H);
      octx.filter = 'none';
    } else {
      octx.fillStyle = '#000';
      octx.fillRect(0, 0, BUF_W, BUF_H);
      octx.fillStyle = '#fff';
      octx.font = '16px system-ui, sans-serif';
      octx.fillText('안전 버퍼 채우는 중…', 20, 40);
    }

    // HUD
    const filled = ring.filter((s) => s.t >= 0).length;
    octx.fillStyle = 'rgba(0,0,0,.6)';
    octx.fillRect(8, BUF_H - 34, 300, 26);
    octx.fillStyle = '#0f0';
    octx.font = '13px ui-monospace, monospace';
    octx.fillText(
      `지연 ${DELAY_SEC}s · 버퍼 ${filled}/${SLOTS} · 탐지 ${detections}`,
      14, BUF_H - 16
    );

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);

  // ── 조악한 갑툭튀 규칙: 조용하다가 갑자기 커지면 ────────────────────────
  const wave = new Float32Array(analyser.fftSize);
  const hist = [];
  let cooldown = 0, detTimer = null;

  function rms() {
    analyser.getFloatTimeDomainData(wave);
    let s = 0;
    for (let i = 0; i < wave.length; i++) s += wave[i] * wave[i];
    return Math.sqrt(s / wave.length);
  }

  if (DETECT) {
    detTimer = setInterval(() => {
      if (!running || video.paused) return;
      const cur = rms();
      hist.push(cur);
      if (hist.length > 20) hist.shift();          // 최근 약 1초
      if (hist.length < 20) return;

      const prev = hist.slice(0, -1).sort((a, b) => a - b);
      const median = prev[Math.floor(prev.length / 2)];
      const now = performance.now();

      // 직전 1초가 조용했고(정적), 지금 급등했으면 후보
      if (now > cooldown && median < 0.02 && cur > Math.max(0.05, median * 5)) {
        cooldown = now + 2000;
        detections++;
        scheduleBlur(video.currentTime, 1.5);
        console.log(
          `[scareblock] 갑툭튀 후보 t=${video.currentTime.toFixed(2)}s ` +
          `rms=${cur.toFixed(3)} (직전 중앙값 ${median.toFixed(3)}) → ${DELAY_SEC}s 뒤 도달 전 블러`
        );
      }
    }, 50);
  }

  window.scareblockDemo = {
    stop() {
      running = false;
      if (detTimer) clearInterval(detTimer);
      out.remove();
      video.style.opacity = prevOpacity;
      try { delay.disconnect(); src.connect(ac.destination); } catch (e) { /* 이미 끊김 */ }
      console.log('[scareblock] 원복 완료');
    },
    blurNow(dur = 1.5) { scheduleBlur(video.currentTime, dur); },
    stats() {
      return {
        지연: DELAY_SEC,
        버퍼적재: ring.filter((s) => s.t >= 0).length + '/' + SLOTS,
        메모리추정MB: +(SLOTS * BUF_W * BUF_H * 4 / 1048576).toFixed(1),
        탐지수: detections,
        현재재생: +video.currentTime.toFixed(2),
      };
    },
  };

  console.log(
    `[scareblock] 지연 재생 시작 — ${DELAY_SEC}s / ${CAPTURE_FPS}fps / ${BUF_W}x${BUF_H}\n` +
    `링버퍼 ${SLOTS}슬롯 ≈ ${(SLOTS * BUF_W * BUF_H * 4 / 1048576).toFixed(0)}MB\n` +
    `stop: scareblockDemo.stop()  ·  수동 블러: scareblockDemo.blurNow()`
  );
})();
