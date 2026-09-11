/** 조립. 유튜브가 SPA라 영상이 바뀌어도 따라붙어야 한다. */
window.SB = window.SB || {};

(() => {
  let player = null;
  let detector = null;

  function findVideo() {
    return document.querySelector('video.html5-main-video') ||
           document.querySelector('#movie_player video');
  }

  function attach(video) {
    if (player?.running) return;
    if (!video || video.readyState < 2) return;

    player = new SB.Player(video);
    player.start();
    detector = new SB.FakeDetector(video, player);   // 02단계에서 교체된다
    detector.start();

    window.scareblock = {
      stats: () => player.report(),
      stop: () => { detector.stop(); player.stop(); player = null; },
      blurNow: (dur = 1.5) => player.addTriggers([{
        time: video.currentTime, category: 'manual',
        confidence: 1, duration: dur, source: 'manual',
      }]),
      player: () => player,
    };
    SB.log('준비됨 — scareblock.stats() / scareblock.stop() / scareblock.blurNow()');
  }

  function boot() {
    const v = findVideo();
    if (!v) return;
    if (v.readyState >= 2) attach(v);
    else v.addEventListener('loadeddata', () => attach(v), { once: true });
  }

  // 유튜브 SPA 네비게이션
  document.addEventListener('yt-navigate-finish', () => {
    if (player?.running) { detector?.stop(); player.stop(); player = null; }
    setTimeout(boot, 800);
  });

  boot();
})();
