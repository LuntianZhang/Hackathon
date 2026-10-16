() => {
  // Mount once per page; Gradio swaps the image element after each GPU render.
  if (window.food3dDragCleanup) window.food3dDragCleanup();
  const abort = new AbortController();
  let active = true, epoch = null, desired = {yaw: 0, pitch: 0, zoom: 1};
  let down = null, seq = 0, sentSeq = 0, inflight = false, sentAt = 0;
  let lastSent = '', suppressUntil = 0, settleUntil = 0;
  const input = id => document.querySelector(`#${id} textarea, #${id} input`);
  const scene = () => document.getElementById('food3d_scene');
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const fingerprint = () => JSON.stringify(desired);
  const onSceneImage = e => scene()?.contains(e.target) && e.target.closest('img');
  function send(now) {
    const field = input('food3d_gesture');
    if (!field || !epoch || inflight || fingerprint() === lastSent || now-sentAt < 130) return;
    sentSeq = ++seq;
    const payload = JSON.stringify({epoch, seq: sentSeq, ...desired});
    const proto = field.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, 'value').set.call(field, payload);
    inflight = true;
    sentAt = now;
    lastSent = fingerprint();
    field.dispatchEvent(new Event('input', {bubbles: true}));
  }
  function tick(now) {
    if (!active) return;
    const field = input('food3d_camera_control');
    if (field?.value) {
      try {
        const value = JSON.parse(field.value);
        if (value.epoch !== epoch) {
          epoch = value.epoch;
          desired = {yaw: value.yaw, pitch: value.pitch, zoom: value.zoom};
          down = null; seq = value.ack; sentSeq = value.ack; inflight = false;
          lastSent = fingerprint();
        } else if (inflight && value.ack >= sentSeq) {
          inflight = false;
          // Wait for Gradio's image update before allowing a semantic click.
          settleUntil = now + 180;
        }
      } catch (_) {}
    }
    // A failed response must not leave the view permanently blocked.
    if (inflight && now-sentAt > 5000) { inflight = false; lastSent = ''; }
    send(now);
    const root = scene();
    if (root) {
      root.classList.toggle('food3d_rotating', Boolean(down?.moved));
      root.classList.toggle('food3d_waiting', inflight);
      for (const img of root.querySelectorAll('img')) img.draggable = false;
    }
    requestAnimationFrame(tick);
  }
  const listen = (event, handler, extra = {}) => document.addEventListener(event, handler,
    {capture: true, signal: abort.signal, ...extra});
  listen('pointerdown', e => {
    if (!onSceneImage(e) || e.button !== 0 || !epoch || !e.isPrimary) return;
    const rect = e.target.closest('img').getBoundingClientRect();
    down = {id: e.pointerId, x: e.clientX, y: e.clientY, width: rect.width, height: rect.height,
      start: {...desired}, moved: false};
  });
  listen('pointermove', e => {
    if (!down || down.id !== e.pointerId) return;
    const dx = e.clientX-down.x, dy = e.clientY-down.y;
    if (Math.hypot(dx, dy) > 5) down.moved = true;
    if (!down.moved) return;
    e.preventDefault();
    desired.yaw = ((down.start.yaw - 120*dx/down.width + 540) % 360)-180;
    desired.pitch = clamp(down.start.pitch - 90*dy/down.height, -75, 75);
  }, {passive: false});
  function end(e) {
    if (!down || down.id !== e.pointerId) return;
    if (down.moved) suppressUntil = performance.now()+500;
    down = null;
  }
  listen('pointerup', end);
  listen('pointercancel', end);
  listen('dragstart', e => {if (onSceneImage(e)) e.preventDefault();});
  listen('click', e => {
    if (onSceneImage(e) && (inflight || down?.moved || performance.now() < Math.max(suppressUntil, settleUntil))) {
      e.preventDefault(); e.stopImmediatePropagation();
    }
  });
  listen('wheel', e => {
    if (!onSceneImage(e) || !epoch) return;
    e.preventDefault();
    const delta = e.deltaY*(e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
    desired.zoom = clamp(desired.zoom*Math.exp(clamp(delta, -300, 300)*.0015), .5, 2.5);
    suppressUntil = performance.now()+300;
  }, {passive: false});
  window.food3dDragCleanup = () => {active = false; abort.abort();};
  requestAnimationFrame(tick);
}
