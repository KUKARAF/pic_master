// Generic Tinder-style card-stack engine, shared by every swipe-based suggestion
// stream (faces, sets, categories, tags). Kept in its own file rather than app.js
// so this feature never has to touch the same lines app.js's other concurrent
// work does. Feature-specific bits (ranking, endpoints, card copy) are supplied
// per page via the `config` object passed to initSwipeStack — this file only
// knows about stack positioning, drag/keyboard gestures, and buffer bookkeeping.
//
// Swipe direction is vertical only: up = confirm, down = reject ("no match").
// Cards peek horizontally (stacked left-to-right, full width) purely for the
// visual "stack of photos" look — that's unrelated to the decide gesture.
window.initSwipeStack = function (config) {
  const BUFFER_SIZE = 10;
  const VISIBLE_PEEK = 4;
  const DECIDE_THRESHOLD = 90; // px vertical drag distance that commits a decision
  const CARD_WIDTH_FRACTION = 0.72; // each card's width, as a fraction of the stack's own width

  const stackEl = document.getElementById(config.stackElId);
  const confirmBtn = document.getElementById(config.confirmBtnId);
  const rejectBtn = document.getElementById(config.rejectBtnId);
  if (!stackEl) return;

  let queue = (config.initialCards || []).slice(0, BUFFER_SIZE);
  const known = new Set(queue.map(c => c.ref)); // queued or already-decided refs
  let fetching = false;
  let dragging = null; // {ref, startX, startY, dx, dy, el}

  function cardEl(card, index) {
    const el = document.createElement('div');
    el.className = 'swipe-card';
    el.dataset.ref = card.ref;

    const wrapWidth = stackEl.clientWidth || stackEl.getBoundingClientRect().width;
    const cardWidth = wrapWidth * CARD_WIDTH_FRACTION;
    const maxOffset = wrapWidth - cardWidth;
    const step = VISIBLE_PEEK > 1 ? maxOffset / (VISIBLE_PEEK - 1) : 0;
    const scale = 1 - index * 0.03;
    el.style.width = cardWidth + 'px';
    el.style.zIndex = String(BUFFER_SIZE - index);
    el.style.transform = `translateX(${index * step}px) scale(${scale})`;
    el.style.opacity = index < VISIBLE_PEEK ? '1' : '0';

    const { imageUrl, questionHtml, scoreText, metaHtml, originalUrl } = config.renderCard(card);
    const imgTag = `<img src="${imageUrl}" alt="suggestion">`;
    el.innerHTML = `
      ${originalUrl
        ? `<a class="swipe-card-img-link" href="${originalUrl}" target="_blank" rel="noopener">${imgTag}</a>`
        : imgTag}
      <div class="swipe-card-body">
        <div class="swipe-card-question">${questionHtml}</div>
        <div class="swipe-card-score">${scoreText}</div>
        ${metaHtml ? `<div class="swipe-card-meta">${metaHtml}</div>` : ''}
      </div>
      <div class="swipe-card-stamp stamp-yes">SAME</div>
      <div class="swipe-card-stamp stamp-no">NOT</div>
    `;
    if (index === 0) bindTopCardGestures(el, card);
    return el;
  }

  function renderEmptyState() {
    if (typeof config.emptyStateHtml === 'function') {
      config.emptyStateHtml(stackEl);
    } else {
      stackEl.innerHTML = config.emptyStateHtml || '<div class="swipe-empty">No more suggestions right now.</div>';
    }
  }

  function render() {
    stackEl.innerHTML = '';
    if (queue.length === 0) {
      renderEmptyState();
      return;
    }
    const visible = queue.slice(0, Math.max(VISIBLE_PEEK, 1));
    // Append back-to-front so the first (top) card ends up last in DOM order / on top visually.
    for (let i = visible.length - 1; i >= 0; i--) {
      stackEl.appendChild(cardEl(visible[i], i));
    }
    maybeFetchMore();
  }

  function topCard() {
    return queue[0] || null;
  }

  async function maybeFetchMore(biasKey, biasAction) {
    if (fetching || queue.length >= BUFFER_SIZE) return;
    fetching = true;
    try {
      const need = BUFFER_SIZE - queue.length;
      const url = config.fetchMoreUrl(known, biasKey, biasAction, need);
      if (!url) return;
      const resp = await fetch(url);
      if (!resp.ok) return;
      const data = await resp.json();
      for (const card of data.cards || []) {
        if (known.has(card.ref)) continue;
        known.add(card.ref);
        queue.push(card);
      }
    } catch (e) {
      // Silent — the buffer just stays smaller until the next swipe retries the fetch.
    } finally {
      fetching = false;
    }
  }

  async function decide(action) {
    const card = topCard();
    if (!card) return;
    const el = stackEl.querySelector(`.swipe-card[data-ref="${CSS.escape(card.ref)}"]`);
    if (el) flyOut(el, action);
    queue.shift();

    const req = action === 'confirm'
      ? (config.onConfirm ? config.onConfirm(card) : null)
      : (config.onReject ? config.onReject(card) : null);
    if (req) {
      fetch(req.url, {
        method: req.method || 'POST',
        headers: req.body ? { 'Content-Type': 'application/json' } : undefined,
        body: req.body ? JSON.stringify(req.body) : undefined,
      }).catch(() => {});
    }

    const biasKey = config.biasKeyFor ? config.biasKeyFor(card) : card.ref;
    maybeFetchMore(biasKey, action);
    setTimeout(render, el ? 180 : 0);
  }

  function flyOut(el, action) {
    el.classList.add('dragging');
    const dy = action === 'confirm' ? -700 : 700;
    const rot = action === 'confirm' ? -6 : 6;
    el.style.transition = 'transform .25s ease, opacity .25s ease';
    el.style.transform = `translate(0, ${dy}px) rotate(${rot}deg)`;
    el.style.opacity = '0';
  }

  function bindTopCardGestures(el, card) {
    const stampYes = el.querySelector('.stamp-yes');
    const stampNo = el.querySelector('.stamp-no');

    function pointerDown(e) {
      const point = e.touches ? e.touches[0] : e;
      dragging = { ref: card.ref, startX: point.clientX, startY: point.clientY, dx: 0, dy: 0, el };
      el.classList.add('dragging');
      window.addEventListener('mousemove', pointerMove);
      window.addEventListener('touchmove', pointerMove, { passive: false });
      window.addEventListener('mouseup', pointerUp);
      window.addEventListener('touchend', pointerUp);
    }

    function pointerMove(e) {
      if (!dragging) return;
      e.preventDefault && e.preventDefault();
      const point = e.touches ? e.touches[0] : e;
      dragging.dx = point.clientX - dragging.startX;
      dragging.dy = point.clientY - dragging.startY;
      const rot = dragging.dy / 18;
      el.style.transform = `translate(${dragging.dx}px, ${dragging.dy}px) rotate(${rot}deg)`;
      stampYes.style.opacity = String(Math.max(0, Math.min(1, -dragging.dy / DECIDE_THRESHOLD)));
      stampNo.style.opacity = String(Math.max(0, Math.min(1, dragging.dy / DECIDE_THRESHOLD)));
    }

    function pointerUp() {
      window.removeEventListener('mousemove', pointerMove);
      window.removeEventListener('touchmove', pointerMove);
      window.removeEventListener('mouseup', pointerUp);
      window.removeEventListener('touchend', pointerUp);
      if (!dragging) return;
      const { dy } = dragging;
      dragging = null;
      if (Math.abs(dy) < DECIDE_THRESHOLD) {
        el.classList.remove('dragging');
        el.style.transform = '';
        stampYes.style.opacity = '0';
        stampNo.style.opacity = '0';
        return;
      }
      decide(dy < 0 ? 'confirm' : 'reject');
    }

    el.addEventListener('mousedown', pointerDown);
    el.addEventListener('touchstart', pointerDown, { passive: true });
  }

  if (confirmBtn) confirmBtn.addEventListener('click', () => decide('confirm'));
  if (rejectBtn) rejectBtn.addEventListener('click', () => decide('reject'));

  document.addEventListener('keydown', (e) => {
    if (!topCard()) return;
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      decide('confirm');
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      decide('reject');
    }
  });

  render();

  return { refill: () => maybeFetchMore() };
};
