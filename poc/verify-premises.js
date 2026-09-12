/**
 * scareblock 00단계 — 기술 전제 콘솔 검증 3종
 *
 * DRM 여부 · 픽셀 접근 · 오디오 접근을 한 번에 재고 결과를 표로 찍는다.
 * 영상이 재생 중인 탭의 콘솔에 붙여넣는다.
 *
 * 주의 — 오디오 검사는 createMediaElementSource를 호출하므로 그 페이지의
 * 오디오 경로가 WebAudio로 바뀐다. 원복하려면 새로고침한다.
 */
(async () => {
  'use strict';

  const v = document.querySelector('video.html5-main-video') || document.querySelector('video');
  if (!v) { console.error('[verify] video 엘리먼트 없음'); return; }
  if (v.readyState < 2) { console.error('[verify] 영상이 아직 로드되지 않았다. 재생 후 다시 실행한다'); return; }

  const R = {};

  // ── 1. DRM (EME) ────────────────────────────────────────────────────────
  // mediaKeys가 붙어 있으면 EME로 보호된 재생이다.
  R.DRM = {
    항목: 'DRM (EME)',
    결과: v.mediaKeys ? '보호됨' : '없음',
    상세: v.mediaKeys
      ? `keySystem=${v.mediaKeys.keySystem ?? '알 수 없음'}`
      : 'mediaKeys 없음 — 평문 재생',
  };

  // ── 2. 픽셀 접근 ────────────────────────────────────────────────────────
  const c = document.createElement('canvas');
  c.width = 320; c.height = 180;
  const cx = c.getContext('2d', { willReadFrequently: true });
  let drawOk = false, taint = null, nonBlack = 0;
  try {
    cx.drawImage(v, 0, 0, c.width, c.height);
    drawOk = true;
  } catch (e) { taint = `drawImage ${e.name}`; }
  if (drawOk) {
    try { c.toDataURL(); } catch (e) { taint = `toDataURL ${e.name}`; }
    if (!taint) {
      const d = cx.getImageData(0, 0, c.width, c.height).data;
      const step = 4000;
      let total = 0;
      for (let i = 0; i < d.length; i += step) { total++; if (d[i] + d[i + 1] + d[i + 2] > 30) nonBlack++; }
      R.픽셀 = { 항목: '픽셀 접근', 결과: nonBlack > total * 0.5 ? 'OK' : '검은 화면', 상세: `비검정 샘플 ${nonBlack}/${total}` };
    }
  }
  if (taint) R.픽셀 = { 항목: '픽셀 접근', 결과: '차단', 상세: taint };

  // ── 3. 오디오 접근 ──────────────────────────────────────────────────────
  // 엘리먼트당 한 번만 가능하므로 전역에 재사용한다.
  try {
    if (!window.__sbVerifyAudio) {
      const ac = new AudioContext();
      const src = ac.createMediaElementSource(v);
      const an = ac.createAnalyser();
      an.fftSize = 1024;
      src.connect(an);
      src.connect(ac.destination);   // 소리는 계속 들리게 둔다
      window.__sbVerifyAudio = { ac, src, an };
    }
    const { ac, an } = window.__sbVerifyAudio;
    await ac.resume();

    const buf = new Float32Array(an.fftSize);
    const rms = () => { an.getFloatTimeDomainData(buf); let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i]; return Math.sqrt(s / buf.length); };

    const samples = [];
    for (let i = 0; i < 25; i++) { samples.push(rms()); await new Promise((r) => setTimeout(r, 80)); }
    const nz = samples.filter((x) => x > 0).length;
    const max = Math.max(...samples);

    R.오디오 = {
      항목: '오디오 접근',
      결과: nz > 0 && max > 1e-4 ? 'OK' : '무음',
      상세: `0이 아닌 샘플 ${nz}/25 · 최대 RMS ${max.toFixed(4)}` + (v.muted || v.volume === 0 ? ' ⚠️ 음소거 상태' : ''),
    };
  } catch (e) {
    R.오디오 = { 항목: '오디오 접근', 결과: '차단', 상세: `${e.name}: ${e.message}` };
  }

  // ── 출력 ────────────────────────────────────────────────────────────────
  console.table(Object.values(R));

  const 판정 =
    R.DRM.결과 === '보호됨' ? 'DRM 보호 콘텐츠 — 범위 선언대로 대상 외'
    : R.픽셀.결과 === 'OK' && R.오디오.결과 === 'OK' ? '전제 3종 통과 — 대상 콘텐츠'
    : '접근 일부 차단 — 상세 확인 필요';

  console.log(
    `\n[verify] ${판정}\n` +
    `url        ${location.href}\n` +
    `해상도      ${v.videoWidth}x${v.videoHeight} · readyState=${v.readyState}\n` +
    `DRM        ${R.DRM.결과} (${R.DRM.상세})\n` +
    `픽셀       ${R.픽셀.결과} (${R.픽셀.상세})\n` +
    `오디오      ${R.오디오.결과} (${R.오디오.상세})\n` +
    `측정        ${new Date().toISOString()}\n`
  );

  window.__sbVerifyResult = { 판정, ...R, url: location.href, at: new Date().toISOString() };
  return 판정;
})();
