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
    // 콘텐츠 스크립트는 매치되는 URL로 **문서가 로드될 때만** 주입된다. /watch로만
    // 매치하면 홈·검색에서 영상을 클릭하는 SPA 경로에는 주입 자체가 없어 확장이
    // 없는 것과 같다. 그래서 youtube.com 전체로 매치하고 여기서 걸러낸다.
    if (location.pathname !== '/watch') return;
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
