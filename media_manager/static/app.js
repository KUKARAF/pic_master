/* app.js — nav widgets (all pages) + photo detail page tag/set/face management */

(function () {
  'use strict';

  /* ------------------------------------------------------------------ */
  /* Toasts — app-wide non-blocking notifications, replacing native      */
  /* showToast() for status/success/error messages. window.showToast is      */
  /* available to every page's inline scripts (app.js loads before them).*/
  /* ------------------------------------------------------------------ */
  window.showToast = function (message, type) {
    // type: 'info' (default) | 'success' | 'error'. When omitted, guess from
    // the message so the many "Failed to …" call sites colour themselves.
    if (!type) type = /\b(fail|error)/i.test(String(message)) ? 'error' : 'info';

    var container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      container.className = 'toast-container';
      document.body.appendChild(container);
    }

    var toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.setAttribute('role', type === 'error' ? 'alert' : 'status');

    var msg = document.createElement('span');
    msg.className = 'toast-msg';
    msg.textContent = String(message);
    toast.appendChild(msg);

    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'toast-close';
    closeBtn.setAttribute('aria-label', 'Dismiss');
    closeBtn.textContent = '×';
    toast.appendChild(closeBtn);

    container.appendChild(toast);
    requestAnimationFrame(function () { toast.classList.add('show'); });

    var timer = null;
    function dismiss() {
      if (timer) { clearTimeout(timer); timer = null; }
      toast.classList.remove('show');
      var removed = false;
      function drop() { if (!removed) { removed = true; toast.remove(); } }
      toast.addEventListener('transitionend', drop, { once: true });
      setTimeout(drop, 400);   // fallback if the transition never fires
    }
    closeBtn.addEventListener('click', dismiss);
    timer = setTimeout(dismiss, type === 'error' ? 6000 : 3500);
    return toast;
  };

  /* Generic dropdown toggle, used by both the tags and warning-bell nav buttons */
  function wireDropdown(btnId, menuId) {
    var btn = document.getElementById(btnId);
    var menu = document.getElementById(menuId);
    if (!btn || !menu) return;
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      menu.classList.toggle('open');
    });
    document.addEventListener('click', function () {
      menu.classList.remove('open');
    });
  }
  wireDropdown('warn-dropdown-btn', 'warn-dropdown-menu');
  wireDropdown('bulk-actions-btn', 'bulk-actions-menu');

  /* ⚡ Bulk actions menu: each button POSTs its data-start, then polls
     data-status and shows done/total (+ matched) until the job finishes — mirrors
     the body-index build poll. On page load each row also checks its status once,
     so a job started elsewhere keeps updating and idle rows show "N pending". */
  (function () {
    function fmt(d, countKey) {
      if (d.error) return 'error: ' + d.error;
      var s = (d.done || 0).toLocaleString() + '/' + (d.total || 0).toLocaleString();
      if (countKey && d[countKey] != null) s += ' · ' + d[countKey] + ' matched';
      // Optional server-provided time-left estimate (e.g. the imdb index load).
      if (d.eta_seconds != null) {
        var e = Math.round(d.eta_seconds);
        s += ' · ~' + (e < 90 ? e + 's' : Math.floor(e / 60) + 'm ' + (e % 60) + 's') + ' left';
      }
      return s;
    }
    document.querySelectorAll('.bulk-action-btn').forEach(function (btn) {
      var statusEl = btn.parentElement.querySelector('[data-role="status"]');
      var countKey = btn.dataset.count || null;

      function done() { btn.disabled = false; delete btn.dataset.busy; }

      function poll() {
        fetch(btn.dataset.status)
          .then(function (r) { return r.json(); })
          .then(function (d) {
            if (d.running) { statusEl.textContent = fmt(d, countKey); setTimeout(poll, 1500); return; }
            statusEl.textContent = d.error ? ('error: ' + d.error) : ('done · ' + fmt(d, countKey));
            done();
          })
          .catch(function (err) { statusEl.textContent = 'poll failed: ' + err.message; done(); });
      }

      btn.addEventListener('click', function (e) {
        e.stopPropagation();  // keep the dropdown open while it runs
        if (btn.dataset.busy) return;
        btn.dataset.busy = '1';
        btn.disabled = true;
        statusEl.textContent = 'starting…';
        fetch(btn.dataset.start, { method: 'POST' })
          .then(function (r) { return r.json(); })
          .then(function (d) {
            if (d.started === false) { statusEl.textContent = d.message || 'already running'; done(); return; }
            if (!d.total) { statusEl.textContent = 'nothing to do'; done(); return; }
            poll();
          })
          .catch(function (err) { statusEl.textContent = 'failed: ' + err.message; done(); });
      });

      // On load: resume polling if a job is already running, else surface a pending count.
      fetch(btn.dataset.status)
        .then(function (r) { return r.json(); })
        .then(function (d) {
          if (d.running) { btn.dataset.busy = '1'; btn.disabled = true; poll(); }
          else if (d.pending) { statusEl.textContent = d.pending + ' pending'; }
        })
        .catch(function () { /* status is best-effort on load */ });
    });
  })();

  /* Mobile hamburger — toggles the collapsed nav links (see .nav-links in CSS). */
  (function () {
    var navToggle = document.getElementById('nav-toggle');
    var navLinks = document.getElementById('nav-links');
    if (!navToggle || !navLinks) return;
    navToggle.addEventListener('click', function (e) {
      e.stopPropagation();
      var open = navLinks.classList.toggle('open');
      navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    document.addEventListener('click', function (e) {
      if (navLinks.contains(e.target) || e.target === navToggle) return;
      navLinks.classList.remove('open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  })();

  /* Generic modal — used by the set picker and face-naming modal */
  var modalOverlay = document.getElementById('modal-overlay');
  var modalBox = document.getElementById('modal-box');

  function openModal(titleText, buildFn) {
    if (!modalOverlay || !modalBox) return;
    modalBox.innerHTML = '';
    var title = document.createElement('div');
    title.className = 'modal-title';
    var titleSpan = document.createElement('span');
    titleSpan.textContent = titleText;
    var closeBtn = document.createElement('button');
    closeBtn.className = 'modal-close-btn';
    closeBtn.type = 'button';
    closeBtn.textContent = '×';
    closeBtn.addEventListener('click', closeModal);
    title.appendChild(titleSpan);
    title.appendChild(closeBtn);
    modalBox.appendChild(title);
    buildFn(modalBox);
    modalOverlay.style.display = 'flex';
  }

  function closeModal() {
    if (!modalOverlay) return;
    modalOverlay.style.display = 'none';
    modalBox.innerHTML = '';
  }

  if (modalOverlay) {
    modalOverlay.addEventListener('click', function (e) {
      if (e.target === modalOverlay) closeModal();
    });
  }

  // Exposed globally so per-page inline <script> blocks (sets.html, set_detail.html,
  // faces.html, ...) can reuse the same modal instead of rebuilding one.
  window.openModal = openModal;
  window.closeModal = closeModal;

  // Programmatically seed the watch-queue arrow-key browsing session (see the
  // queue-nav IIFE further down, and base.html's click-capture path which is
  // the only other writer of this sessionStorage key) — lets a page build a
  // queue from an arbitrary list of ids instead of only from clicking a card
  // in a data-queue-source grid.
  window.setPhotoQueue = function (ids, label, cursor) {
    sessionStorage.setItem('photoQueue', JSON.stringify({ ids: ids, cursor: cursor || 0, label: label }));
  };

  // "Custom action" — a declarative, one-shot action shown as a button at the
  // very top of the photo page's sidebar (see the renderer IIFE further down).
  // Declarative (not a callback) because it has to survive a real page
  // navigation via sessionStorage, not just an in-memory closure. `request`
  // is a plain {method, url, body} fetch spec; `successMessage` may use
  // {{key}} placeholders resolved against the JSON response. `count` and
  // `targetLabel` (e.g. 42 and "Vacation 2022") back the cancel-confirmation
  // and the tab-close warning below — both spell out "N photos will NOT be
  // added to X" rather than a generic message.
  window.setPhotoCustomAction = function (action) {
    sessionStorage.setItem('photoCustomAction', JSON.stringify(action));
  };

  // Warn before closing the tab, reloading, or navigating away entirely
  // (following a link to another site) while a custom action is still
  // pending — this was previously silent, so an abandoned "add 42 photos to
  // X" banner just stuck around across every photo view in the tab forever,
  // with no way to tell it was still live short of clicking it. Browsers
  // don't allow customizing beforeunload's dialog text (a security
  // restriction, not an oversight here) — this only gets the browser's own
  // generic "leave site?" prompt. The custom-worded "N photos will NOT be
  // added to X, proceed?" confirmation lives on the banner's own Cancel
  // button instead (see the renderer IIFE further down), for in-app
  // dismissal rather than actually leaving.
  //
  // NOT shown for Left/Right (or the queue grid overlay's) browsing between
  // photos that are themselves part of the pending action's own queue — that
  // IS the review flow this action exists for (see set_detail.html's "Add
  // folder" comment), not abandoning it, and since every photo view here is a
  // real page navigation (not a SPA route change), skipping this check would
  // otherwise mean a "leave site?" prompt on every single step through the
  // very queue you're supposed to browse. The queue-nav code sets
  // photoQueueNavigating right before calling location.href for exactly this
  // reason; it's cleared again the moment the next page's script runs (below)
  // so it only ever suppresses the one navigation it was set for.
  if (sessionStorage.getItem('photoQueueNavigating')) sessionStorage.removeItem('photoQueueNavigating');
  window.addEventListener('beforeunload', function (e) {
    if (!sessionStorage.getItem('photoCustomAction')) return;
    if (sessionStorage.getItem('photoQueueNavigating')) return;
    e.preventDefault();
    e.returnValue = '';
  });

  // Shared with the queue-thumbnail-grid overlay further down (Space on the
  // photo page) — a plain outer-scope flag rather than event-registration-
  // order/stopImmediatePropagation tricks, so the existing Left/Right queue,
  // "/" palette, and F/Esc handlers below can each just check it and no-op
  // while the grid is open, instead of the grid needing to race them.
  let photoGridOpen = false;

  // Exposed globally so any per-page inline <script> (including swipe-core.js
  // config blocks) can escape user-controlled strings before dropping them into
  // innerHTML, without each page reimplementing this.
  window.escapeHtml = function (s) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : s;
    return d.innerHTML;
  };

  // Shared swipe-card metadata line — every swipe suggestion (faces/sets/
  // categories/tags) attaches the same filename/tags/sets/category/people
  // fields server-side (see web.py's _attach_file_meta / _enrich_rows), so this
  // formats them identically instead of each page's config reimplementing it.
  window.swipeCardMeta = function (card) {
    const parts = [];
    if (card.filename) parts.push(escapeHtml(card.filename));
    if (card.people && card.people.length) parts.push('with ' + card.people.map(function (p) {
      return escapeHtml(p.name) + (p.age != null ? ' (' + Math.round(p.age) + ')' : '');
    }).join(', '));
    if (card.categories && card.categories.length) parts.push('categories: ' + card.categories.map(function (c) { return escapeHtml(c.name); }).join(', '));
    if (card.sets && card.sets.length) parts.push('in ' + card.sets.map(function (s) {
      const studioText = s.studio ? ' — ' + escapeHtml(s.studio) : '';
      const peopleTags = (s.people || []).map(function (p) {
        return ' @' + escapeHtml(p.name) + (p.age != null ? Math.round(p.age) : '');
      }).join('');
      return escapeHtml(s.name) + studioText + peopleTags;
    }).join(', '));
    if (card.tags && card.tags.length) parts.push('tags: ' + card.tags.map(escapeHtml).join(', '));
    return parts.join(' · ');
  };

  // Shared "keep searching until X are found" buffer-size slider wiring —
  // live-updates the displayed number while dragging, only actually resizes
  // the swipe-core.js stack's buffer (and kicks off a refill fetch) once the
  // slider settles on `change`, so dragging through the range doesn't fire a
  // fetch per tick. Same behavior find_person.html and set_detail.html both
  // want; extracted here instead of each page reimplementing it. `stack` is
  // whatever initSwipeStack(...) returned (has a setBufferSize method).
  window.wireBufferSizeSlider = function (stack, sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const value = document.getElementById(valueId);
    if (!slider || !value) return;
    slider.addEventListener('input', function () {
      value.textContent = slider.value;
    });
    slider.addEventListener('change', function () {
      stack.setBufferSize(parseInt(slider.value, 10));
    });
  };

  // A plain <input type="range"> mirroring its live value into a label span
  // on every drag tick — the one piece genuinely identical across every
  // threshold slider in the app (find_person.html's expand-search slider,
  // set_detail.html's expand-search slider), even though what happens on
  // the *button* click next to each of those sliders differs per page (one
  // fetches a one-shot results grid, the other just re-launches the swipe
  // stream at the new threshold) — so only this shared sliver is extracted,
  // not the whole panel around it. `formatFn` defaults to 2-decimal fixed,
  // matching every current caller's own formatting.
  window.wireLiveLabelSlider = function (sliderEl, labelEl, formatFn) {
    if (!sliderEl || !labelEl) return;
    const format = formatFn || function (v) { return parseFloat(v).toFixed(2); };
    sliderEl.addEventListener('input', function () {
      labelEl.textContent = format(sliderEl.value);
    });
  };

  // The "expand similar search" one-shot results grid pattern find_person.html
  // uses: fetch candidates at a slider-chosen threshold once, render a grid of
  // cards each with a single confirm action, remove a card from the grid once
  // its action succeeds. Parameterized so any future "search looser, then
  // pick individual matches to confirm" feature can reuse it instead of
  // hand-rolling the fetch/render/remove wiring again — set_detail.html's own
  // expand-search doesn't use this (it re-launches its swipe stream instead
  // of showing a one-shot grid), so it isn't force-fit here.
  //
  // opts = {
  //   sliderId, sliderValueId, expandBtnId, statusId, gridId,
  //   fetchUrl(threshold) -> url,               // GET, expects {results: [...]}
  //   cardHtml(item) -> string,                  // full card markup, must include
  //                                               // a `.expand-grid-use-btn` with
  //                                               // data-item-ref set to item.ref
  //   confirmRequest(item) -> {url, method, body} // fired when that button is clicked
  //   noResultsText(threshold) -> string,
  // }
  window.wireExpandSearchGrid = function (opts) {
    const slider = document.getElementById(opts.sliderId);
    const sliderValue = document.getElementById(opts.sliderValueId);
    const expandBtn = document.getElementById(opts.expandBtnId);
    const status = document.getElementById(opts.statusId);
    const grid = document.getElementById(opts.gridId);
    if (!expandBtn || !grid) return;

    wireLiveLabelSlider(slider, sliderValue);

    expandBtn.addEventListener('click', function () {
      const threshold = slider ? slider.value : 0.3;
      status.style.display = 'block';
      status.textContent = 'Searching…';
      fetch(opts.fetchUrl(threshold))
        .then(function (r) { if (!r.ok) throw new Error('Request failed: ' + r.status); return r.json(); })
        .then(function (data) {
          const results = data.results || [];
          if (!results.length) {
            status.textContent = opts.noResultsText(threshold);
            return;
          }
          status.style.display = 'none';
          grid.innerHTML = '';
          results.forEach(function (item) {
            const wrap = document.createElement('div');
            wrap.innerHTML = opts.cardHtml(item);
            const card = wrap.firstElementChild;
            grid.appendChild(card);
            const btn = card.querySelector('.expand-grid-use-btn');
            if (!btn) return;
            btn.addEventListener('click', function () {
              btn.disabled = true;
              const req = opts.confirmRequest(item);
              fetch(req.url, {
                method: req.method || 'POST',
                headers: req.body ? { 'Content-Type': 'application/json' } : undefined,
                body: req.body ? JSON.stringify(req.body) : undefined,
              })
                .then(function (r) { if (!r.ok) throw new Error('Request failed'); return r.json(); })
                .then(function () { card.remove(); })
                .catch(function (err) {
                  btn.disabled = false;
                  showToast('Failed: ' + err.message);
                });
            });
          });
        })
        .catch(function (err) {
          status.textContent = 'Search error: ' + err.message;
        });
    });
  };

  /* Render a heart button from a count: "♥ N" when favorited (count > 0), "♡"
     when 0. Keeps the count in data-fav-count. Shared so JS-built markup can
     seed the same look. */
  window.renderHeart = function (btn, count) {
    count = count || 0;
    btn.dataset.favCount = count;
    btn.classList.toggle('is-favorite', count > 0);
    btn.textContent = count > 0 ? '♥ ' + count : '♡';
  };

  /* Shared favorite-heart COUNTER — left-click bumps +1, right-click -1 (floored
     server-side at 0). POSTs {delta} to `endpoint`, reads the new {count} back,
     and re-renders. Used by every heart button across the app so the behavior is
     identical everywhere. */
  window.wireHeartButton = function (btn, endpoint) {
    function bump(delta) {
      if (btn.disabled) return;
      btn.disabled = true;
      fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delta: delta }),
      })
        .then(function (r) {
          if (!r.ok) throw new Error('Request failed: ' + r.status);
          return r.json();
        })
        .then(function (data) {
          window.renderHeart(btn, data.count);
          btn.disabled = false;
        })
        .catch(function (err) {
          btn.disabled = false;
          showToast('Failed to update favorite: ' + err.message);
        });
    }
    btn.addEventListener('click', function (e) { e.preventDefault(); e.stopPropagation(); bump(1); });
    btn.addEventListener('contextmenu', function (e) { e.preventDefault(); e.stopPropagation(); bump(-1); });
  };

  /* Warning bell — populated on every page load, independent of photo context */
  var warnBadge = document.getElementById('warn-badge');
  var warnList = document.getElementById('warn-list');
  var warnMarkAllBtn = document.getElementById('warn-mark-all-btn');

  function renderWarnings(items, unreadCount) {
    if (unreadCount > 0) {
      warnBadge.textContent = String(unreadCount);
      warnBadge.style.display = 'inline-block';
    } else {
      warnBadge.style.display = 'none';
    }
    warnMarkAllBtn.style.display = items.length ? 'block' : 'none';
    warnList.innerHTML = '';
    if (!items.length) {
      var empty = document.createElement('div');
      empty.className = 'warn-empty';
      empty.textContent = 'No warnings.';
      warnList.appendChild(empty);
      return;
    }
    items.forEach(function (item) {
      var row = document.createElement('div');
      row.className = 'warn-item';

      var body = document.createElement('div');
      body.className = 'warn-body';
      var path = document.createElement('div');
      path.className = 'path';
      path.textContent = item.path;
      var message = document.createElement('div');
      message.className = 'message';
      message.textContent = item.message;
      body.appendChild(path);
      body.appendChild(message);

      var btn = document.createElement('button');
      btn.className = 'warn-read-btn';
      btn.type = 'button';
      btn.title = 'Mark read';
      btn.textContent = '✓';
      btn.addEventListener('click', function () {
        fetch('/api/errors/' + item.id + '/read', { method: 'POST' })
          .then(function () { loadWarnings(); })
          .catch(function () {});
      });

      row.appendChild(body);
      row.appendChild(btn);
      warnList.appendChild(row);
    });
  }

  function loadWarnings() {
    if (!warnBadge || !warnList) return;
    fetch('/api/errors?unread_only=true&limit=20')
      .then(function (r) { return r.json(); })
      .then(function (data) { renderWarnings(data.items, data.unread_count); })
      .catch(function () {});
  }

  if (warnMarkAllBtn) {
    warnMarkAllBtn.addEventListener('click', function () {
      fetch('/api/errors/read-all', { method: 'POST' })
        .then(function () { loadWarnings(); })
        .catch(function () {});
    });
  }

  loadWarnings();

  /* ------------------------------------------------------------------ */
  /* Media worker status — nav badge + "outsourced" toasts.              */
  /* Polls /api/worker/status every 5s. `since` tracks the max recent    */
  /* task id we've toasted so we only toast NEW dispatches; the first    */
  /* successful load seeds `since` from the backlog WITHOUT toasting.    */
  /* A failed fetch (older backend / transient) just skips the cycle.    */
  /* ------------------------------------------------------------------ */
  (function initWorkerStatus() {
    var badge = document.getElementById('worker-badge');
    if (!badge) return;
    var label = badge.querySelector('.worker-label');
    var since = 0;      // max recent id we've accounted for
    var seeded = false; // set on the first successful poll
    var OP_LABELS = {
      detect_faces: 'face match',
      embed_bbox: 'face embed',
      embed_image: 'CLIP embed',
      embed_text: 'text embed',
      detect_objects: 'object detect',
    };

    function setBadge(cls, text, title) {
      badge.className = 'worker-badge ' + cls;
      label.textContent = text;
      badge.title = title || '';
      badge.style.display = 'inline-flex';
    }

    function render(data) {
      if (!data.enabled) {         // no worker configured — stay out of the way
        badge.style.display = 'none';
        return;
      }
      var addr = data.address ? String(data.address).slice(0, 8) : '';
      // Label is just a wrench to keep the navbar narrow; the dot colour + title tooltip
      // convey connected/offline.
      if (data.connected) {
        setBadge('is-online', '🔧', addr ? 'Worker ' + addr + ' connected' : 'Worker connected');
      } else {
        setBadge('is-offline', '🔧',
                 addr ? 'Worker ' + addr + ' (unreachable)' : 'Worker unreachable');
      }
    }

    function handleRecent(recent) {
      // Seed on the FIRST successful poll — even when the buffer is empty — so
      // the first real dispatch after page load toasts instead of being
      // swallowed as backlog.
      if (!seeded) {
        (recent || []).forEach(function (t) { if (t.id > since) since = t.id; });
        seeded = true;
        return;
      }
      if (!recent || !recent.length) return;
      var fresh = recent.filter(function (t) { return t.id > since; });
      recent.forEach(function (t) { if (t.id > since) since = t.id; });
      if (!fresh.length) return;
      if (fresh.length > 3) {
        window.showToast('Outsourced ' + fresh.length + ' tasks to worker', 'info');
      } else {
        fresh.forEach(function (t) {
          var op = OP_LABELS[t.op] || t.op;
          window.showToast('Outsourced ' + op + ' → ' + t.name, 'info');
        });
      }
    }

    var lastPollErr = null;
    function poll() {
      var url = '/api/worker/status' + (seeded ? '?since=' + since : '');
      fetch(url)
        .then(function (r) { if (!r.ok) throw new Error('status ' + r.status); return r.json(); })
        .then(function (data) { lastPollErr = null; render(data); handleRecent(data.recent); })
        .catch(function (e) {
          // 404 = older backend without the endpoint (expected, stay quiet). Any
          // other error (e.g. a 500 from the backend raising on a corrupt
          // worker.json) is real — surface it once per distinct message rather
          // than swallowing it, but don't spam the console every 5s.
          var msg = e && e.message ? e.message : String(e);
          if (msg.indexOf('404') === -1 && msg !== lastPollErr) {
            lastPollErr = msg;
            if (window.console && console.warn) console.warn('worker status poll failed:', msg);
          }
        });
    }

    poll();
    setInterval(poll, 5000);
  })();

  /* Shared "pick or create a set" flow — keyboard-first, same shape everywhere it's
     used (photo page's "Add to set", the search page's bulk "Add to set"): one big
     autofocused input backed by a native <datalist> of existing sets; Enter either
     resolves to a matching existing set or, if nothing matches, opens a second
     keyboard-first step asking for the new set's (optional) studio before creating
     it. `onResolved(set)` is called with the final set object either way — it does
     NOT assign anything to a file itself, callers decide what resolving a set means
     for them (assign to the current photo, or become the bulk-add target). */
  function createSet(name, studio) {
    return fetch('/api/sets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name, studio: studio || null }),
    }).then(function (r) {
      if (!r.ok) throw new Error('Request failed: ' + r.status);
      return r.json();
    });
  }

  /* A plain <input> backed by a native <datalist> — the shared shape both the
     studio and person fields in openNewSetDetailsModal use. Populates from
     `listUrl`, calling `labelFn` per item for the datalist option text.
     `opts.value`, if given, pre-fills the input (e.g. the rename-set modal
     editing an existing studio, vs. the new-set modal always starting blank). */
  function wireDatalistField(box, opts) {
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = opts.placeholder;
    input.setAttribute('list', opts.datalistId);
    input.autocomplete = 'off';
    input.style.width = '100%';
    input.style.fontSize = '1.2rem';
    input.style.padding = '10px 12px';
    input.style.marginBottom = '8px';
    if (opts.value) input.value = opts.value;
    box.appendChild(input);

    const datalist = document.createElement('datalist');
    datalist.id = opts.datalistId;
    box.appendChild(datalist);

    const status = document.createElement('div');
    status.className = 'modal-empty';
    status.style.marginBottom = '10px';
    status.textContent = 'Loading ' + opts.noun + 's…';
    box.appendChild(status);

    fetch(opts.listUrl)
      .then(function (r) { return r.json(); })
      .then(function (items) {
        datalist.innerHTML = '';
        items.forEach(function (item) {
          const opt = document.createElement('option');
          opt.value = opts.labelFn(item);
          datalist.appendChild(opt);
        });
        status.textContent = items.length
          ? 'Type to search ' + items.length + ' existing ' + opts.noun + '(s), or a new one.'
          : 'No ' + opts.noun + 's yet — type one, or leave blank.';
      })
      .catch(function () {
        status.textContent = 'Failed to load ' + opts.noun + 's.';
      });

    return input;
  }
  // Exposed so per-page inline <script> blocks (e.g. set_detail.html's
  // rename-set modal) can reuse the same studio/person suggestion field
  // instead of falling back to a plain, suggestion-less <input>.
  window.wireDatalistField = wireDatalistField;

  function openNewSetDetailsModal(setName, onResolved) {
    openModal('Details for "' + setName + '"', function (box) {
      const studioLabel = document.createElement('div');
      studioLabel.className = 'sub';
      studioLabel.style.marginBottom = '4px';
      studioLabel.textContent = 'Studio';
      box.appendChild(studioLabel);
      const studioInput = wireDatalistField(box, {
        placeholder: 'Studio (optional)…',
        datalistId: 'studio-prompt-datalist',
        listUrl: '/api/studios',
        labelFn: function (s) { return s.name; },
        noun: 'studio',
      });

      const personLabel = document.createElement('div');
      personLabel.className = 'sub';
      personLabel.style.margin = '10px 0 4px';
      personLabel.textContent = 'People (optional)';
      box.appendChild(personLabel);
      const personInput = wireDatalistField(box, {
        placeholder: 'Add a person…',
        datalistId: 'person-prompt-datalist',
        listUrl: '/api/identities',
        labelFn: function (i) { return i.name; },
        noun: 'person',
      });

      const addPersonBtn = document.createElement('button');
      addPersonBtn.type = 'button';
      addPersonBtn.className = 'btn-similar';
      addPersonBtn.style.marginBottom = '8px';
      addPersonBtn.textContent = '+ Add';
      box.appendChild(addPersonBtn);

      const personChipList = document.createElement('div');
      personChipList.style.display = 'flex';
      personChipList.style.flexWrap = 'wrap';
      personChipList.style.gap = '6px';
      personChipList.style.marginBottom = '10px';
      box.appendChild(personChipList);

      // Nothing here is persisted until the whole modal is submitted — this
      // is a local accumulation, POSTed in a loop only from submit() (unlike
      // person.html's alias editor, which persists each add/remove
      // immediately since it's editing an already-existing identity).
      const pendingPeople = [];

      function renderPeopleChips() {
        personChipList.innerHTML = '';
        pendingPeople.forEach(function (name, idx) {
          const chip = document.createElement('span');
          chip.className = 'tag-removable';
          chip.style.display = 'inline-flex';
          chip.textContent = name;
          const rm = document.createElement('button');
          rm.type = 'button';
          rm.className = 'rm';
          rm.title = 'Remove';
          rm.textContent = '×';
          rm.addEventListener('click', function () {
            pendingPeople.splice(idx, 1);
            renderPeopleChips();
          });
          chip.appendChild(rm);
          personChipList.appendChild(chip);
        });
      }

      function addPerson() {
        const name = personInput.value.trim();
        if (!name) return;
        const exists = pendingPeople.some(function (p) {
          return p.toLowerCase() === name.toLowerCase();
        });
        personInput.value = '';
        if (exists) return;
        pendingPeople.push(name);
        renderPeopleChips();
        personInput.focus();
      }
      addPersonBtn.addEventListener('click', addPerson);

      const createBtn = document.createElement('button');
      createBtn.type = 'button';
      createBtn.className = 'btn-similar btn-primary';
      createBtn.textContent = 'Create set';
      box.appendChild(createBtn);

      let submitted = false;
      function submit() {
        if (submitted) return;
        submitted = true;
        studioInput.disabled = true;
        personInput.disabled = true;
        addPersonBtn.disabled = true;
        createBtn.disabled = true;
        createSet(setName, studioInput.value.trim() || null)
          .then(function (data) {
            const failures = [];
            // Sequential, not Promise.all — the backend uses one shared
            // SQLite connection per database with no locking, so firing
            // these all at once from separate request threads corrupts
            // cursor state (mirrors search.html's bulk-add-to-set).
            return pendingPeople.reduce(function (chain, personName) {
              return chain.then(function () {
                return fetch('/api/identities/' + encodeURIComponent(personName) + '/assign-set', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ set_id: data.id }),
                }).then(function (r) {
                  if (!r.ok) failures.push(personName);
                }).catch(function () {
                  failures.push(personName);
                });
              });
            }, Promise.resolve()).then(function () {
              // The set itself is already created successfully at this point —
              // a failure to link one or more people shouldn't lose that or
              // block the caller, just surface it and move on.
              if (failures.length) {
                showToast('Set created, but failed to link: ' + failures.join(', '));
              }
              return data;
            });
          })
          .then(function (data) { closeModal(); onResolved(data); })
          .catch(function (err) {
            submitted = false;
            studioInput.disabled = false;
            personInput.disabled = false;
            addPersonBtn.disabled = false;
            createBtn.disabled = false;
            showToast('Failed to create set: ' + err.message);
          });
      }

      createBtn.addEventListener('click', submit);

      studioInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); submit(); }
        else if (e.key === 'Escape') { e.preventDefault(); studioInput.value = ''; submit(); }
      });

      personInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
          e.preventDefault();
          // Nothing typed here yet — Enter means "done", not "add a blank
          // person", so it confirms creation the same as Enter in the studio
          // field does, instead of silently no-op'ing via addPerson's own
          // empty-value guard.
          if (personInput.value.trim()) addPerson();
          else submit();
        } else if (e.key === 'Escape') { e.preventDefault(); personInput.value = ''; }
      });

      setTimeout(function () { studioInput.focus(); }, 0);
    });
  }

  /* Fuzzy subsequence match: every character of `query` must appear in `text`
     in order (case-insensitive) but not necessarily contiguously. Returns
     null when it's not a match at all; otherwise a score where lower is
     better (an earlier, tighter match ranks first), for stable sorting. */
  function fuzzyScore(query, text) {
    if (!query) return 0;
    const q = query.toLowerCase();
    const t = (text || '').toLowerCase();
    let searchFrom = 0, firstIndex = -1, lastIndex = -1;
    for (let qi = 0; qi < q.length; qi++) {
      const idx = t.indexOf(q[qi], searchFrom);
      if (idx === -1) return null;
      if (firstIndex === -1) firstIndex = idx;
      lastIndex = idx;
      searchFrom = idx + 1;
    }
    return firstIndex + (lastIndex - firstIndex) * 0.1;
  }

  // Shared, invalidated cache for the handful of "fetch the whole list" GET
  // endpoints hit repeatedly from client-side pickers (openEntitySearchModal
  // below) and the search-palette preload further down — keyed by URL rather
  // than by a notion of "entity type", since two different features can
  // legitimately want two different endpoints for what's conceptually the
  // same entity (e.g. the tag picker wants /api/vocab's full YOLO-World +
  // confirmed vocabulary, the palette wants /api/tags' confirmed-with-counts
  // list) — this only has to guarantee each cached URL reflects its own most
  // recent successful GET, not unify two different responses into one.
  //
  // Invalidation: rather than hand-patching every one of the many places that
  // create/rename/delete a set/category/tag/identity (already gone wrong once
  // — the palette preload below used to just never invalidate at all), any
  // successful non-GET request to /api/... clears every cached GET
  // unconditionally. Slightly coarser than per-endpoint precision, but
  // correctness-by-construction beats an enumerated list that's one miss away
  // from reintroducing the exact staleness bug this replaces. Refetching
  // these lists is cheap; staying stale isn't.
  var GET_CACHE_TTL_MS = 5 * 60 * 1000;
  var getCache = {}; // url -> {promise, expiresAt}

  // The in-memory cache above only helps within a single page's lifetime —
  // this app navigates via full page loads, so every fresh /sets, /photo/*,
  // etc. load starts with an empty getCache and pays the full round-trip
  // again the first time something opens the set picker. localStorage
  // persists across that reload: LOCAL_CACHE_PREFIX + url holds the last
  // successful response body, returned INSTANTLY (no network wait) while a
  // real fetch still runs in the background to refresh it for next time
  // (stale-while-revalidate) — so the very slow part (network latency /
  // server-side query cost) only ever blocks the very first use ever, not
  // every page load after.
  var LOCAL_CACHE_PREFIX = 'mm_get_cache:';

  function cachedGetJson(url) {
    var entry = getCache[url];
    if (entry && entry.expiresAt > Date.now()) return entry.promise;

    var freshPromise = fetch(url).then(function (r) {
      if (!r.ok) throw new Error('Request failed: ' + r.status);
      return r.json();
    }).then(function (data) {
      try { localStorage.setItem(LOCAL_CACHE_PREFIX + url, JSON.stringify(data)); } catch (e) {}
      return data;
    });

    var promise = freshPromise;
    try {
      var cachedRaw = localStorage.getItem(LOCAL_CACHE_PREFIX + url);
      if (cachedRaw) {
        promise = Promise.resolve(JSON.parse(cachedRaw));
        freshPromise.catch(function () {}); // background refresh — its failure must never surface via this call
      }
    } catch (e) {
      // Corrupt localStorage entry, or JSON.parse failed — fall back to the real fetch.
    }

    getCache[url] = { promise: promise, expiresAt: Date.now() + GET_CACHE_TTL_MS };
    promise.catch(function () { delete getCache[url]; }); // never cache a failed fetch
    return promise;
  }

  (function () {
    var origFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      var method = ((init && init.method) || 'GET').toUpperCase();
      var url = typeof input === 'string' ? input : (input && input.url) || '';
      return origFetch(input, init).then(function (response) {
        if (response.ok && method !== 'GET' && url.indexOf('/api/') === 0) {
          getCache = {};
          // Same coarse "clear everything" invalidation as the in-memory
          // cache above, applied to its localStorage-backed counterpart —
          // otherwise a set/category/tag/identity created or renamed just
          // now would keep showing the pre-mutation snapshot on next page
          // load until its background refresh happened to complete first.
          try {
            for (var i = localStorage.length - 1; i >= 0; i--) {
              var key = localStorage.key(i);
              if (key && key.indexOf(LOCAL_CACHE_PREFIX) === 0) localStorage.removeItem(key);
            }
          } catch (e) {}
        }
        return response;
      });
    };
  })();

  // Great-circle distance in km between two lat/lon points — mirrors the server's
  // _haversine (web.py). Used by the location picker to show how far each named
  // location is from the current photo's EXIF GPS (window.MEDIA_FILE_GPS).
  function haversineKm(lat1, lon1, lat2, lon2) {
    var toRad = function (d) { return d * Math.PI / 180; };
    var r = 6371;
    var dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
    var a = Math.sin(dLat / 2) * Math.sin(dLat / 2)
      + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return 2 * r * Math.asin(Math.sqrt(a));
  }

  // Per-type behavior for openEntitySearchModal — the one place that knows how
  // to list/label/create each kind of entity. `image` is only set for
  // 'identity' (a face-crop thumbnail); every other type renders text-only.
  const ENTITY_TYPE_CONFIGS = {
    set: {
      title: 'Choose a set',
      placeholder: 'Search or create a set…',
      // Recognizes a capitalized word that matches a known person (e.g.
      // "Joe") as you type it, pulling it out of the text box into its own
      // removable chip so the rest of what you type (e.g. "trip 2022") keeps
      // filtering set names/studios as usual, now narrowed to sets linked to
      // that person too. See openEntitySearchModal's personAware handling.
      personAware: true,
      // light=true: text-only fields (id/name/studio/aliases) — skips the
      // per-set age/people attach (_attach_set_people), which is real query
      // work this picker never displays. See api_list_sets.
      fetchAll: function () { return cachedGetJson('/api/sets?light=true'); },
      label: function (s) { return s.name; },
      // Matching includes aliases (an alias is never shown as the primary
      // label — only name is — but typing one still has to surface the set).
      matchText: function (s) {
        return [s.name, s.studio || '', (s.aliases || []).join(' ')].filter(Boolean).join(' ');
      },
      secondary: function (s) {
        // No age/gender here (light picker fetch omits it, see api_list_sets) —
        // just names, which is all personAware's filter needs anyway.
        const peopleText = s.people && s.people.length ? s.people.map(function (p) { return '@' + p.name; }).join(' ') : '';
        const aliasText = s.aliases && s.aliases.length ? 'aka ' + s.aliases.join(', ') : '';
        return [s.studio || '', peopleText, aliasText].filter(Boolean).join(' — ');
      },
      image: function (s) { return s.thumb_id != null ? '/thumb/' + s.thumb_id : null; },
      imageSquare: true,
      // Matches today's exact chained behavior: a new set's studio is its own
      // keyboard-first step, not folded into this one.
      createFn: function (typedName) {
        return new Promise(function (resolve) { openNewSetDetailsModal(typedName, resolve); });
      },
    },
    category: {
      title: 'Set category',
      placeholder: 'Search or create a category…',
      fetchAll: function () { return cachedGetJson('/api/categories'); },
      label: function (c) { return c.name; },
      secondary: function (c) { return (c.image_count || 0) + ' photo(s)'; },
      image: null,
      createFn: function (typedName) {
        return fetch('/api/categories', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: typedName }),
        }).then(function (r) {
          if (!r.ok) throw new Error('Request failed: ' + r.status);
          return r.json();
        });
      },
    },
    location: {
      title: 'Set location',
      placeholder: 'Search or create a location…',
      fetchAll: function () { return cachedGetJson('/api/locations'); },
      label: function (l) { return l.name; },
      secondary: function (l) {
        if (l.gps_lat == null) {
          return (l.file_count || 0) + ' photo(s)';
        }
        var coords = l.gps_lat.toFixed(4) + ', ' + l.gps_lon.toFixed(4);
        // On the photo page, if this photo has EXIF GPS, lead with how far this
        // location is from it — so you can pick the nearest named place at a glance.
        var g = window.MEDIA_FILE_GPS;
        if (g) {
          var km = haversineKm(g.lat, g.lon, l.gps_lat, l.gps_lon);
          return '📍 ' + (km < 10 ? km.toFixed(1) : Math.round(km)) + ' km · ' + coords;
        }
        return coords;
      },
      image: null,
      createFn: function (typedName) {
        return fetch('/api/locations', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: typedName }),
        }).then(function (r) {
          if (!r.ok) throw new Error('Request failed: ' + r.status);
          return r.json();
        });
      },
    },
    tag: {
      title: 'Search or create a tag',
      placeholder: 'Search or type a tag…',
      // Vocab is already the merged "every known label" list (YOLO-World's
      // search terms + manual.db's confirmed tags) — exactly the right
      // candidate pool for this picker, hint-only entries included.
      fetchAll: function () {
        return cachedGetJson('/api/vocab')
          .then(function (data) { return (data.vocab || []).map(function (name) { return { name: name }; }); });
      },
      label: function (t) { return t.name; },
      secondary: null,
      image: null,
      createFn: null, // any typed text is a legal tag — nothing to create ahead of time
    },
    identity: {
      title: 'Choose a person',
      placeholder: 'Search or type a name…',
      fetchAll: function () { return cachedGetJson('/api/identities'); },
      label: function (i) { return i.name; },
      secondary: function (i) { return i.count + ' photo(s)'; },
      image: function (i) { return i.face_id != null ? '/face-crop/manual:' + i.face_id : null; },
      createFn: null, // naming is just picking a string; the caller decides what "use" means
    },
  };

  /* The one consolidated "search existing or create new" picker — every
     entity type (set/category/tag/identity) goes through this, replacing
     three previously separate, near-duplicate flows. Native <datalist>
     can't render an image per option (needed for a face-crop thumbnail next
     to an identity match), so results are a custom-rendered, keyboard-
     navigable list instead of a datalist.

     A "＋ Create '<query>'" row is always appended (not only when nothing
     matches) whenever the input has text, so a fuzzy match against an
     unrelated existing entity never silently steals a name you meant to
     create fresh — you always have an explicit, visible way to create even
     when something similar already exists.

     options = {
       type,                 // 'set' | 'category' | 'tag' | 'identity'
       title,                 // optional override of the per-type default
       excludeIds,             // single id or array — filtered out of results
       previewImage,            // optional url shown above the input
       allowEmpty,              // adds a "Save without a name" button (identity naming)
       extraSuggestion(resolve, box),  // optional extra one-click option (identity's embedding-match suggestion)
       onResolved(entity),      // entity = {..., name} for an existing pick, or
                                 // {name, isNew:true} for a typed one (isNew only
                                 // ever reaches here for types with no createFn —
                                 // set/category resolve to the real created object)
     } */
  function openEntitySearchModal(options) {
    const config = ENTITY_TYPE_CONFIGS[options.type];
    if (!config) throw new Error('Unknown entity type: ' + options.type);
    const excludeIds = options.excludeIds == null ? []
      : Array.isArray(options.excludeIds) ? options.excludeIds : [options.excludeIds];

    openModal(options.title || config.title, function (box) {
      if (options.previewImage) {
        const preview = document.createElement('img');
        preview.src = options.previewImage;
        preview.width = 80;
        preview.height = 80;
        preview.style.borderRadius = '6px';
        preview.style.display = 'block';
        preview.style.marginBottom = '10px';
        box.appendChild(preview);
      }

      // Input + an inline "＋ Create" button (replaces the old create-as-a-list-row):
      // creating a new entity lives right in the search box now.
      const inputRow = document.createElement('div');
      inputRow.style.cssText = 'display:flex;gap:6px;align-items:stretch;margin-bottom:8px;';
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = config.placeholder;
      input.autocomplete = 'off';
      input.style.flex = '1';
      inputRow.appendChild(input);
      const createBtn = document.createElement('button');
      createBtn.type = 'button';
      createBtn.className = 'btn-similar';
      createBtn.style.cssText = 'white-space:nowrap;display:none;';
      createBtn.textContent = '＋ Create';
      createBtn.addEventListener('click', function () {
        const q = input.value.trim();
        if (q) resolveEntry({ kind: 'create', text: q });
      });
      inputRow.appendChild(createBtn);
      box.appendChild(inputRow);

      // Create is offered only with no active person filter (a person chip means
      // "pick from Joe's sets", not "make an unrelated new set").
      function updateCreateBtn() {
        const q = input.value.trim();
        createBtn.style.display = (q && !personChips.length) ? '' : 'none';
        createBtn.title = q ? ('Create "' + q + '"') : '';
      }

      // Person chips — only wired up for types that opt in (currently just
      // 'set'). identityNames is populated once, below, alongside the entity
      // list itself; personChips holds the confirmed (exact, canonical-case)
      // identity names currently filtering the results.
      let identityNames = null;
      const personChips = [];
      let personChipsEl = null;
      if (config.personAware) {
        personChipsEl = document.createElement('div');
        personChipsEl.className = 'palette-chips';
        personChipsEl.style.marginBottom = '8px';
        box.appendChild(personChipsEl);
      }
      // Live, clickable person-name suggestions that pop up as you type (replaces
      // the old "type a name + Space to auto-chip" magic — see renderNameSuggestions).
      let nameSuggestEl = null;
      if (config.personAware) {
        nameSuggestEl = document.createElement('div');
        nameSuggestEl.className = 'modal-name-suggest';
        box.appendChild(nameSuggestEl);
      }

      function renderPersonChips() {
        if (!personChipsEl) return;
        personChipsEl.innerHTML = '';
        personChips.forEach(function (nameValue, i) {
          const chip = document.createElement('span');
          chip.className = 'palette-chip palette-chip-face';
          const tag = document.createElement('span');
          tag.className = 'palette-chip-tag';
          tag.textContent = 'FACE';
          const val = document.createElement('span');
          val.textContent = nameValue;
          const x = document.createElement('span');
          x.className = 'palette-chip-x';
          x.textContent = '×';
          chip.appendChild(tag);
          chip.appendChild(val);
          chip.appendChild(x);
          chip.title = 'Remove this person filter';
          chip.addEventListener('click', function () {
            personChips.splice(i, 1);
            renderPersonChips();
            renderNameSuggestions();
            applyFilter();
          });
          personChipsEl.appendChild(chip);
        });
      }

      // The word currently being typed (last whitespace-delimited token) — what
      // the name suggestions match against.
      function currentNameToken() {
        const parts = input.value.split(/\s+/);
        return parts[parts.length - 1] || '';
      }

      // Add `canonical` as a person filter, consuming the token being typed, then
      // refresh chips/suggestions/results. Shared by clicking a suggestion and
      // Enter-picking the top one.
      function addPersonChip(canonical) {
        if (personChips.indexOf(canonical) === -1) personChips.push(canonical);
        const parts = input.value.split(/\s+/);
        parts.pop();                                   // drop the just-typed token
        input.value = parts.join(' ') + (parts.length ? ' ' : '');
        renderPersonChips();
        renderNameSuggestions();
        applyFilter();
        input.focus();
      }

      // Grid of known people whose names match what's being typed — click one to
      // add it as a filter (like the search palette). Shown only while typing a
      // token that matches at least one not-yet-selected person.
      function renderNameSuggestions() {
        if (!nameSuggestEl) return;
        nameSuggestEl.innerHTML = '';
        if (!identityNames || !identityNames.length) return;
        const token = currentNameToken();
        if (!token) return;
        const matches = identityNames
          .filter(function (n) { return personChips.indexOf(n) === -1; })
          .map(function (n) { return { name: n, score: fuzzyScore(token, n) }; })
          .filter(function (x) { return x.score !== null; })
          .sort(function (a, b) { return a.score - b.score; })
          .slice(0, 12);
        if (!matches.length) return;
        matches.forEach(function (m) {
          const chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'modal-name-suggest-chip';
          chip.textContent = m.name;
          chip.title = 'Filter to sets with ' + m.name;
          // mousedown (not click) so it fires before the input's blur, and prevent
          // the input from losing focus.
          chip.addEventListener('mousedown', function (e) { e.preventDefault(); addPersonChip(m.name); });
          nameSuggestEl.appendChild(chip);
        });
      }

      const status = document.createElement('div');
      status.className = 'modal-empty';
      status.textContent = 'Loading…';
      box.appendChild(status);

      const list = document.createElement('div');
      list.className = 'modal-list';
      box.appendChild(list);

      const extraBox = document.createElement('div');
      box.appendChild(extraBox);

      // Suggested sets surfaced INSIDE the picker (only when opened from a photo
      // page, which passes suggestForFileId). CLIP-centroid ranked matches shown
      // above the search list as one-click picks — clicking one assigns it, the
      // same as picking a search result. Absent for bulk-add-from-search callers.
      let suggestWrap = null;
      let suggestSets = [];
      if (options.suggestForFileId != null && options.type === 'set') {
        suggestWrap = document.createElement('div');
        suggestWrap.className = 'modal-suggest';
        box.insertBefore(suggestWrap, list);
        fetch('/api/files/' + options.suggestForFileId + '/suggested-sets?limit=6')
          .then(function (r) { return r.ok ? r.json() : { results: [] }; })
          .then(function (data) {
            if (resolved) return;
            suggestSets = (data.results || []).filter(function (s) { return excludeIds.indexOf(s.id) === -1; });
            renderSuggestions();
          })
          .catch(function () { /* suggestions are best-effort; search still works */ });
      }

      // The smart suggestions are NOT filtered by the person chips / text (that
      // would often empty them) — they're RE-RANKED: a set matching the active
      // person chips (and then the typed text) floats to the top, so selecting
      // "Joe" honors the filter by surfacing Joe's sets first. Re-run from
      // applyFilter on every keystroke / chip change.
      function suggestionSortKey(set) {
        const people = (set.people || []).map(function (p) { return p.name.toLowerCase(); });
        const chipMatches = personChips.filter(function (c) { return people.indexOf(c.toLowerCase()) !== -1; }).length;
        const q = input.value.trim();
        let textMiss = 0, textScore = 0;
        if (q) {
          const s = fuzzyScore(q, [set.name, set.studio || '', people.join(' ')].join(' '));
          if (s === null) { textMiss = 1; textScore = Infinity; } else { textScore = s; }
        }
        return { chipMatches: chipMatches, textMiss: textMiss, textScore: textScore, clip: -(set.score || 0) };
      }

      function renderSuggestions() {
        if (!suggestWrap) return;
        suggestWrap.innerHTML = '';
        if (!suggestSets.length || resolved) return;
        const ordered = suggestSets.slice().sort(function (a, b) {
          const ka = suggestionSortKey(a), kb = suggestionSortKey(b);
          return (kb.chipMatches - ka.chipMatches)
            || (ka.textMiss - kb.textMiss)
            || (ka.textScore - kb.textScore)
            || (ka.clip - kb.clip);
        });
        const heading = document.createElement('div');
        heading.className = 'modal-suggest-title';
        heading.textContent = '✨ Suggested sets';
        suggestWrap.appendChild(heading);
        ordered.forEach(function (set) {
          const row = document.createElement('div');
          row.className = 'modal-list-item';
          const text = document.createElement('div');
          const label = document.createElement('div');
          label.textContent = set.name;
          text.appendChild(label);
          const metaLine = buildSetMetaLine(set);
          if (metaLine) { const sub = document.createElement('div'); sub.className = 'sub'; sub.appendChild(metaLine); text.appendChild(sub); }
          row.appendChild(text);
          if (set.score != null) {
            const scoreSpan = document.createElement('span');
            scoreSpan.className = 'score-badge';
            scoreSpan.style.position = 'static';
            scoreSpan.style.marginLeft = 'auto';
            scoreSpan.textContent = Number(set.score).toFixed(2);
            row.appendChild(scoreSpan);
          }
          // Representative cover thumbnail, same square style as set search rows.
          if (set.thumb_id != null) {
            const thumb = document.createElement('img');
            thumb.className = 'modal-list-item-thumb square';
            thumb.src = '/thumb/' + set.thumb_id;
            row.appendChild(thumb);
          }
          row.addEventListener('click', function () { resolveWith(set); });
          suggestWrap.appendChild(row);
        });
      }

      let allItems = [];
      let visible = []; // [{kind:'existing', item} | {kind:'create', text}]
      let highlighted = -1;
      let resolved = false;

      function resolveWith(entity) {
        if (resolved) return;
        resolved = true;
        input.disabled = true;
        closeModal();
        options.onResolved(entity);
      }

      function resolveEntry(entry) {
        if (resolved) return;
        if (entry.kind === 'create' && config.createFn) {
          resolved = true;
          input.disabled = true;
          config.createFn(entry.text)
            .then(function (created) { closeModal(); options.onResolved(created); })
            .catch(function (err) {
              resolved = false;
              input.disabled = false;
              showToast('Failed to create: ' + err.message);
            });
        } else if (entry.kind === 'create') {
          resolveWith({ name: entry.text, isNew: true });
        } else {
          resolveWith(entry.item);
        }
      }

      function updateHighlight() {
        Array.from(list.children).forEach(function (row, i) {
          row.classList.toggle('is-highlighted', i === highlighted);
        });
      }

      function renderList() {
        list.innerHTML = '';
        visible.forEach(function (entry, i) {
          const row = document.createElement('div');
          row.className = 'modal-list-item' + (i === highlighted ? ' is-highlighted' : '');
          if (entry.kind === 'create') {
            row.textContent = '＋ Create "' + entry.text + '"';
          } else {
            const item = entry.item;
            const text = document.createElement('div');
            const label = document.createElement('div');
            label.textContent = config.label(item);
            text.appendChild(label);
            const secondary = config.secondary ? config.secondary(item) : '';
            if (secondary) {
              const sub = document.createElement('div');
              sub.className = 'sub';
              sub.textContent = secondary;
              text.appendChild(sub);
            }
            row.appendChild(text);
            const imgUrl = config.image ? config.image(item) : null;
            if (imgUrl) {
              const thumb = document.createElement('img');
              thumb.className = 'modal-list-item-thumb' + (config.imageSquare ? ' square' : '');
              thumb.src = imgUrl;
              row.appendChild(thumb);
            }
          }
          row.addEventListener('click', function () { resolveEntry(entry); });
          row.addEventListener('mouseenter', function () { highlighted = i; updateHighlight(); });
          list.appendChild(row);
        });
      }

      function applyFilter() {
        const query = input.value.trim();
        let pool = allItems;
        if (personChips.length) {
          pool = pool.filter(function (item) {
            const people = item.people || [];
            return personChips.every(function (chipName) {
              return people.some(function (p) { return p.name.toLowerCase() === chipName.toLowerCase(); });
            });
          });
        }
        let matched;
        if (!query) {
          matched = pool.slice(0, 50);
        } else {
          matched = pool
            .map(function (item) { return { item: item, score: fuzzyScore(query, config.matchText ? config.matchText(item) : config.label(item)) }; })
            .filter(function (x) { return x.score !== null; })
            .sort(function (a, b) { return a.score - b.score; })
            .slice(0, 50)
            .map(function (x) { return x.item; });
        }
        visible = matched.map(function (item) { return { kind: 'existing', item: item }; });
        highlighted = visible.length ? 0 : -1;
        renderList();
        updateCreateBtn();       // create now lives in the input row, not the list
        renderSuggestions();     // re-rank the smart suggestions for the new filter
      }

      input.addEventListener('input', function () {
        renderNameSuggestions();
        applyFilter();
      });

      input.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          if (highlighted < visible.length - 1) { highlighted++; updateHighlight(); }
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          if (highlighted > 0) { highlighted--; updateHighlight(); }
        } else if (e.key === 'Enter') {
          e.preventDefault();
          if (highlighted >= 0 && visible[highlighted]) {
            resolveEntry(visible[highlighted]);
          } else {
            // Nothing matched/highlighted — Enter creates, same as the ＋ Create button.
            const q = input.value.trim();
            if (q && !personChips.length) resolveEntry({ kind: 'create', text: q });
          }
        } else if (e.key === 'Escape') {
          e.preventDefault();
          closeModal();
        }
      });

      if (options.allowEmpty) {
        const noNameBtn = document.createElement('button');
        noNameBtn.type = 'button';
        noNameBtn.className = 'btn-similar';
        noNameBtn.style.fontSize = '0.85em';
        noNameBtn.style.marginTop = '8px';
        noNameBtn.textContent = 'Save without a name';
        noNameBtn.title = 'Confirm this is a distinct person without naming them yet — rename anytime later.';
        noNameBtn.addEventListener('click', function () { resolveWith({ name: null, isNew: true }); });
        box.appendChild(noNameBtn);
      }

      Promise.all([
        config.fetchAll(),
        config.personAware ? cachedGetJson('/api/identities') : Promise.resolve([]),
      ])
        .then(function (results) {
          let items = results[0];
          if (excludeIds.length) {
            items = items.filter(function (item) { return excludeIds.indexOf(item.id) === -1; });
          }
          allItems = items;
          identityNames = results[1].map(function (i) { return i.name; });
          status.textContent = items.length
            ? 'Type to search ' + items.length + ' — Enter to pick, or create a new one.'
            : 'Nothing yet — type a name and press Enter to create one.';
          applyFilter();
        })
        .catch(function () {
          status.textContent = 'Failed to load — you can still type a name and press Enter.';
        });

      if (options.extraSuggestion) options.extraSuggestion(resolveWith, extraBox);

      setTimeout(function () { input.focus(); }, 0);
    });
  }

  window.openEntitySearchModal = openEntitySearchModal;

  function openSetSearchModal(onResolved, excludeSetId, opts) {
    openEntitySearchModal(Object.assign(
      { type: 'set', excludeIds: excludeSetId, onResolved: onResolved },
      opts || {}
    ));
  }

  window.openSetSearchModal = openSetSearchModal;

  /* Shown when a set's studio matches an existing one under a different exact
     spelling only (studios have no id to merge — just reusing the existing
     spelling so it groups on /studios instead of fragmenting). Declining
     keeps whatever the user actually typed; either way the caller re-sends
     with confirm_studio_merge so this check isn't repeated forever. */
  function confirmStudioSpelling(typedStudio, existing) {
    const wantsExisting = confirm(
      'A studio named "' + existing.studio + '" already exists (you typed "' + typedStudio + '") ' +
      'across ' + existing.set_count + ' set(s).\n\nUse "' + existing.studio + '" instead, so this set groups with them?'
    );
    return wantsExisting ? existing.studio : typedStudio;
  }

  /* Used by both the "+ Create set" modal (openNewSetDetailsModal) and the
     plain create-set form on sets.html. POST /api/sets 409s when the studio
     matches an existing one under a different spelling — see
     confirmStudioSpelling. */
  function createSet(name, studio, confirmStudioMerge) {
    return fetch('/api/sets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name, studio: studio || null, confirm_studio_merge: !!confirmStudioMerge }),
    }).then(function (r) {
      if (r.status === 409) {
        return r.json().then(function (data) {
          const finalStudio = confirmStudioSpelling(studio, data.existing_studio);
          return createSet(name, finalStudio, true);
        });
      }
      if (!r.ok) throw new Error('Request failed: ' + r.status);
      return r.json();
    });
  }
  window.createSet = createSet;

  /* Shared by both rename-set modals (sets.html's per-card editor and
     set_detail.html's page-level one). PUT /api/sets/{id} 409s either with
     the colliding set (asks to merge into it) or, once that's clear, with a
     differently-spelled existing studio (asks to reuse its spelling) — the
     caller only ever sees the final resolved result (or an Error if the user
     declined a name-merge; declining a studio-spelling prompt just keeps
     what was typed and continues). */
  window.renameSetWithMergePrompt = function (setId, name, studio) {
    function attempt(confirmMerge, confirmStudioMerge, studioOverride) {
      const effectiveStudio = studioOverride !== undefined ? studioOverride : studio;
      return fetch('/api/sets/' + setId, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name, studio: effectiveStudio,
          confirm_merge: !!confirmMerge, confirm_studio_merge: !!confirmStudioMerge,
        }),
      }).then(function (r) {
        if (r.status === 409) {
          return r.json().then(function (data) {
            if (data.conflict === 'studio') {
              const finalStudio = confirmStudioSpelling(effectiveStudio, data.existing_studio);
              return attempt(confirmMerge, true, finalStudio);
            }
            const existing = data.existing_set;
            const wantsMerge = confirm(
              'A set named "' + existing.name + '" already exists' +
              (existing.studio ? ' (studio: ' + existing.studio + ')' : '') +
              ' with ' + existing.image_count + ' photo(s).\n\nMerge this set into it?'
            );
            if (!wantsMerge) throw new Error('canceled — a set with that name already exists');
            return attempt(true, confirmStudioMerge, effectiveStudio);
          });
        }
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      });
    }
    return attempt(false, false);
  };

  /* Same shape as renameSetWithMergePrompt but for identities: PUT
     /api/identities/{name} 409s when the new name matches an existing person
     under a different exact spelling only (an exact-spelling rename already
     merges automatically server-side, since identity is just a shared
     string). Used by person.html's rename modal. */
  window.renameIdentityWithMergePrompt = function (oldName, newName) {
    function attempt(confirmMerge) {
      return fetch('/api/identities/' + encodeURIComponent(oldName), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName, confirm_merge: !!confirmMerge }),
      }).then(function (r) {
        if (r.status === 409) {
          return r.json().then(function (data) {
            const existing = data.existing_identity;
            const wantsMerge = confirm(
              'A person named "' + existing.name + '" already exists (' + existing.count + ' photo(s)).' +
              '\n\nMerge "' + oldName + '" into "' + existing.name + '"?'
            );
            if (!wantsMerge) throw new Error('canceled — a person with that name already exists');
            return attempt(true);
          });
        }
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      });
    }
    return attempt(false);
  };

  /* Generic "add file X to set Y" — used by the search page's bulk-select flow,
     which isn't tied to a single "current" file the way the photo page is. */
  window.assignFileToSet = function (fileIdToAssign, setId) {
    return fetch('/api/files/' + fileIdToAssign + '/sets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ set_id: setId }),
    }).then(function (r) {
      if (!r.ok) throw new Error('Request failed: ' + r.status);
      return r.json();
    });
  };

  /* Shared "is the user typing somewhere" guard for page-global keyboard
     shortcuts (search palette's "/", photo viewer's arrow keys). */
  function isTypingTarget(el) {
    if (!el) return false;
    var tag = el.tagName;
    return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
  }

  /* ------------------------------------------------------------------ */
  /* Search palette — multi-facet chip search, every page.                 */
  /* Type free text; matches across category/tag/face/set/filename are      */
  /* ranked together as suggestions, Enter locks the highlighted one into a  */
  /* removable chip, and chips AND-combine live against a preview grid.      */
  /* ------------------------------------------------------------------ */
  /* Preloaded once, page-wide, the moment this script runs — not gated on the
     palette ever being opened. Every tag/category/identity/set name+count is
     small enough to hold entirely in memory, so suggestion ranking (fuzzy-
     matching what's typed against these) never needs a network round trip;
     only resolving a chip combination to actual matching photos/sets/people
     needs the server (that's real checksum-set work). Reuses the same list
     endpoints/shapes openEntitySearchModal's ENTITY_TYPE_CONFIGS already
     fetches, so there's exactly one source of truth per entity type. */
  var paletteCandidates = null; // [{type, value, label, count}, ...] once loaded
  var paletteCandidatesPromise = (function () {
    var sources = [
      { type: 'category', url: '/api/categories', label: function (c) { return c.name; }, value: function (c) { return c.name; }, count: function (c) { return c.image_count || 0; } },
      { type: 'location', url: '/api/locations', label: function (l) { return l.name; }, value: function (l) { return String(l.id); }, count: function (l) { return l.file_count || 0; } },
      { type: 'city', url: '/api/cities', label: function (c) { return c.name + (c.country ? ', ' + c.country : ''); }, value: function (c) { return String(c.id); }, count: function (c) { return c.file_count || 0; } },
      { type: 'tag', url: '/api/tags', label: function (t) { return t.tag; }, value: function (t) { return t.tag; }, count: function (t) { return t.count || 0; } },
      { type: 'face', url: '/api/identities', label: function (i) { return i.name; }, value: function (i) { return i.name; }, count: function (i) { return i.count || 0; } },
      { type: 'set', url: '/api/sets', label: function (s) {
        var peopleText = (s.people || []).map(function (p) { return '@' + p.name + (p.age != null ? Math.round(p.age) : ''); }).join(' ');
        var aliasText = (s.aliases || []).map(function (a) { return 'aka ' + a; }).join(' ');
        return [s.name, s.studio || '', peopleText, aliasText].filter(Boolean).join(' — ');
      }, value: function (s) { return String(s.id); }, count: function (s) { return s.image_count || 0; } },
    ];
    return Promise.all(sources.map(function (src) {
      return cachedGetJson(src.url).then(function (data) {
        return data.map(function (item) {
          return { type: src.type, value: src.value(item), label: src.label(item), count: src.count(item) };
        });
      });
    })).then(function (lists) {
      paletteCandidates = [].concat.apply([], lists);
    }).catch(function (err) {
      console.error('failed to preload search palette candidates:', err);
      paletteCandidates = [];
    });
  })();

  (function () {
    var overlay = document.getElementById('palette-overlay');
    var input = document.getElementById('palette-input');
    var navInput = document.getElementById('nav-search-input');
    if (!overlay || !input) return;

    var chipsEl = document.getElementById('palette-applied');
    var countEl = document.getElementById('palette-count');
    var suggestionsEl = document.getElementById('palette-suggestions');
    var gridEl = document.getElementById('palette-grid');
    var viewAllEl = document.getElementById('palette-viewall');

    var chips = [];       // [{type, value}] — applied facet filters
    var sortSpec = null;  // {field, dir} parsed from a "field:asc|desc" token, or null
    var suggestions = []; // current in-memory suggestion list (flat, grouped by type in render)
    var highlighted = 0;
    var requestSeq = 0;   // guards against an in-flight grid request resolving out of order

    var FACET_LABELS = { category: 'CAT', location: 'LOC', city: 'CITY', tag: 'TAG', face: 'FACE', set: 'SET', file: 'FILE' };
    // Section headers for the grouped left-pane suggestions.
    var TYPE_HEADINGS = { face: 'PEOPLE', tag: 'TAGS', category: 'CATEGORIES', location: 'LOCATIONS', city: 'CITIES', set: 'SETS', file: 'FILES' };
    var TYPE_ORDER = ['face', 'tag', 'category', 'location', 'city', 'set', 'file'];

    // Text sort syntax: "field:asc|desc" anywhere in the input is parsed out into a
    // sort chip. Aliases map onto the backend's sort keys.
    var SORT_ALIASES = { age: 'age', date: 'added', added: 'added', modified: 'modified',
                         filename: 'filename', name: 'filename', favorites: 'favorites', fav: 'favorites' };
    var SORT_TOKEN_RE = /\b(age|date|added|modified|filename|name|favorites|fav)\s*:\s*(asc|desc)\b/i;

    // Pull a sort token out of `text`; returns {clean, sort} where sort is
    // {field, dir} (backend key) or null and clean is text with the token removed.
    function parseSortToken(text) {
      var m = text.match(SORT_TOKEN_RE);
      if (!m) return { clean: text, sort: null };
      var field = SORT_ALIASES[m[1].toLowerCase()];
      var clean = (text.slice(0, m.index) + text.slice(m.index + m[0].length)).replace(/\s{2,}/g, ' ').trim();
      return { clean: clean, sort: { field: field, dir: m[2].toLowerCase() } };
    }

    /* "cat /outs" — everything up to the last "/" is leftover text the user is
       composing (kept untouched), everything after it is the fragment actively
       matched against suggestions. No "/" at all means the whole box is the
       fragment (the common single-word case), matching plain typing. */
    function fragmentText() {
      var v = input.value;
      var i = v.lastIndexOf('/');
      return i === -1 ? v : v.slice(i + 1);
    }
    function leftoverText() {
      var v = input.value;
      var i = v.lastIndexOf('/');
      return i === -1 ? '' : v.slice(0, i);
    }

    function open() {
      overlay.style.display = 'flex';
      input.value = '';
      input.focus();
      updateSuggestions();
      updateGrid();
    }
    function close() {
      overlay.style.display = 'none';
      chips = [];
      sortSpec = null;
      suggestions = [];
      highlighted = 0;
    }

    // Applied filters (top of the left pane): the facet chips plus, if set, the
    // sort chip (distinctly colored). Each removable.
    function renderChips() {
      chipsEl.innerHTML = '';
      chips.forEach(function (c, i) {
        var chip = document.createElement('span');
        chip.className = 'palette-chip palette-chip-' + c.type;
        var tag = document.createElement('span');
        tag.className = 'palette-chip-tag';
        tag.textContent = FACET_LABELS[c.type] || c.type.toUpperCase();
        var val = document.createElement('span');
        val.textContent = c.value;
        var x = document.createElement('span');
        x.className = 'palette-chip-x';
        x.textContent = '×';
        chip.appendChild(tag);
        chip.appendChild(val);
        chip.appendChild(x);
        chip.title = 'remove';
        chip.addEventListener('click', function () {
          chips.splice(i, 1);
          updateSuggestions();
          updateGrid();
        });
        chipsEl.appendChild(chip);
      });
      if (sortSpec) {
        var schip = document.createElement('span');
        schip.className = 'palette-chip palette-chip-sort';
        var stag = document.createElement('span');
        stag.className = 'palette-chip-tag';
        stag.textContent = 'SORT';
        var sval = document.createElement('span');
        sval.textContent = sortSpec.field + ' ' + (sortSpec.dir === 'asc' ? '▲' : '▼');
        var sx = document.createElement('span');
        sx.className = 'palette-chip-x';
        sx.textContent = '×';
        schip.appendChild(stag); schip.appendChild(sval); schip.appendChild(sx);
        schip.title = 'remove sort';
        schip.addEventListener('click', function () {
          sortSpec = null;
          renderChips();
          updateGrid();
        });
        chipsEl.appendChild(schip);
      }
    }

    // Left-pane "available filters", grouped by type with a header per group;
    // click (or Enter on the highlighted row) applies it as a filter chip.
    function renderSuggestions() {
      suggestionsEl.innerHTML = '';
      if (!suggestions.length) {
        var empty = document.createElement('div');
        empty.className = 'palette-empty';
        empty.textContent = paletteCandidates === null ? '— loading… —' : '— no matching filters —';
        suggestionsEl.appendChild(empty);
        return;
      }
      var lastType = null;
      suggestions.forEach(function (s, i) {
        if (s.type !== lastType) {
          lastType = s.type;
          var h = document.createElement('div');
          h.className = 'palette-suggestion-group';
          h.textContent = TYPE_HEADINGS[s.type] || s.type.toUpperCase();
          suggestionsEl.appendChild(h);
        }
        var row = document.createElement('div');
        row.className = 'palette-suggestion-item' + (i === highlighted ? ' is-highlighted' : '');
        row.style.borderLeftColor = i === highlighted ? 'var(--accent)' : 'transparent';
        var tag = document.createElement('span');
        tag.className = 'palette-suggestion-tag palette-chip-' + s.type;
        tag.textContent = FACET_LABELS[s.type] || s.type.toUpperCase();
        var label = document.createElement('span');
        label.className = 'palette-suggestion-label';
        label.textContent = s.label;
        var count = document.createElement('span');
        count.className = 'palette-suggestion-count';
        count.textContent = s.count;
        row.appendChild(tag);
        row.appendChild(label);
        row.appendChild(count);
        row.addEventListener('click', function () {
          highlighted = i;
          commitHighlighted();
        });
        suggestionsEl.appendChild(row);
      });
    }

    function section(label, count) {
      var wrap = document.createElement('div');
      wrap.className = 'palette-section';
      var heading = document.createElement('div');
      heading.className = 'palette-section-title';
      heading.textContent = '> ' + label + ' (' + count + ')_';
      wrap.appendChild(heading);
      var tiles = document.createElement('div');
      tiles.className = 'palette-tiles';
      wrap.appendChild(tiles);
      return { wrap: wrap, tiles: tiles };
    }

    function makeTile(href, thumbSrc, alt, caption, extraClass, metaEl) {
      var tile = document.createElement('div');
      tile.className = 'card palette-tile' + (extraClass ? ' ' + extraClass : '');
      var link = document.createElement('a');
      link.className = 'card-img-link';
      link.href = href;
      var img = document.createElement('img');
      img.loading = 'lazy';
      img.src = thumbSrc;
      img.alt = alt;
      link.appendChild(img);
      var name = document.createElement('span');
      name.className = 'palette-tile-name';
      name.textContent = caption;
      tile.appendChild(link);
      tile.appendChild(name);
      if (metaEl) tile.appendChild(metaEl);
      return tile;
    }

    // A set's studio + linked-face "@name" (+ age/gender when known) tags, on
    // their own line under the tile's name — mirrors _macros.html's
    // set_meta_line. null when the set has neither.
    function buildSetTileMeta(s) {
      if (!s.studio && !(s.people && s.people.length)) return null;
      var meta = document.createElement('span');
      meta.className = 'set-meta-line';
      if (s.studio) {
        var studioSpan = document.createElement('span');
        studioSpan.className = 'set-studio';
        studioSpan.textContent = s.studio;
        meta.appendChild(studioSpan);
      }
      (s.people || []).forEach(function (p) {
        var tag = document.createElement('span');
        tag.className = 'set-people-tag';
        tag.textContent = '@' + p.name;
        if (p.age != null) {
          var ageSpan = document.createElement('span');
          ageSpan.className = 'set-person-age' + (p.gender === 'male' ? ' gender-male' : p.gender === 'female' ? ' gender-female' : '');
          ageSpan.textContent = Math.round(p.age);
          tag.appendChild(ageSpan);
        }
        meta.appendChild(tag);
      });
      return meta;
    }

    // A photo tile that, when clicked, opens the FULL result set as the photo
    // viewer's watch queue starting at this photo (not just the previewed tiles).
    function photoTile(m) {
      var tile = makeTile('/photo/' + m.id, '/thumb/' + m.id, m.filename, m.title || m.filename);
      tile.dataset.fileId = m.id;
      var link = tile.querySelector('.card-img-link');
      if (link) link.addEventListener('click', function (e) { e.preventDefault(); openResultsAsQueue(m.id); });
      return tile;
    }

    function photoSection(label, list) {
      if (!list || !list.length) return;
      var sec = section(label, list.length);
      list.forEach(function (m) { sec.tiles.appendChild(photoTile(m)); });
      gridEl.appendChild(sec.wrap);
    }

    function renderGrid(data) {
      gridEl.innerHTML = '';
      var hasAny = (data.media || []).length || (data.sets || []).length || (data.people || []).length
        || (data.tag_matches || []).length || (data.file_matches || []).length;
      if (!hasAny) {
        var empty = document.createElement('div');
        empty.className = 'palette-empty';
        empty.textContent = chips.length || fragmentText().trim() ? '— no results for this query —' : 'Type to search, or pick a filter…';
        gridEl.appendChild(empty);
        return;
      }
      if ((data.sets || []).length) {
        var setSec = section('SETS', data.sets.length);
        data.sets.forEach(function (s) {
          setSec.tiles.appendChild(makeTile(
            '/sets/' + s.id,
            s.thumb_file_id != null ? '/thumb/' + s.thumb_file_id : '',
            s.name, s.name, 'palette-tile-set', buildSetTileMeta(s)
          ));
        });
        if (chips.length) {
          var viewSetsLink = document.createElement('a');
          viewSetsLink.className = 'sub palette-section-viewall';
          viewSetsLink.style.marginLeft = '10px';
          viewSetsLink.href = buildSetsFilterHref();
          viewSetsLink.textContent = 'View as list →';
          setSec.wrap.querySelector('.palette-section-title').appendChild(viewSetsLink);
        }
        gridEl.appendChild(setSec.wrap);
      }
      if ((data.people || []).length) {
        var peopleSec = section('PEOPLE', data.people.length);
        data.people.forEach(function (p) {
          peopleSec.tiles.appendChild(makeTile(
            '/person/' + encodeURIComponent(p.name),
            p.face_id != null ? '/face-crop/manual:' + p.face_id : '',
            p.name, p.name, 'palette-tile-person'
          ));
        });
        gridEl.appendChild(peopleSec.wrap);
      }
      // Free-text matches, each in its own section; applied-filter photos last.
      photoSection('TAGS', data.tag_matches);
      photoSection('FILES', data.file_matches);
      photoSection('PHOTOS', data.media);
    }

    // Open the full result set (chips + free text, in the current sort order) as
    // the photo viewer's watch queue, starting at `startId` (or the first result).
    // Replaces the old "view all on /search" — results are browsed as a queue.
    function openResultsAsQueue(startId) {
      var q = fragmentText().trim();
      fetch('/api/search-ids', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          q: q, chips: chips,
          sort: sortSpec ? sortSpec.field : '',
          order: sortSpec ? sortSpec.dir : 'desc',
        }),
      })
        .then(function (r) { if (!r.ok) throw new Error('search-ids ' + r.status); return r.json(); })
        .then(function (data) {
          var ids = data.ids || [];
          if (!ids.length) { showToast('No results to open.'); return; }
          var cursor = startId != null ? ids.indexOf(startId) : 0;
          if (cursor < 0) cursor = 0;
          var label = chips.map(function (c) { return c.type + ':' + c.value; }).join(' + ') || (q || 'search');
          sessionStorage.setItem('photoQueue', JSON.stringify({ ids: ids, cursor: cursor, label: 'Search: ' + label }));
          sessionStorage.setItem('photoQueueNavigating', '1');
          window.location.href = '/photo/' + ids[cursor];
        })
        .catch(function (err) { showToast('Failed to open results: ' + err.message); });
    }

    // Chip-encoded href targeting /sets' own f= filtering (alongside its
    // favorite/studio filters) — used by the palette's "View as list" link under
    // SETS (the only place a chip combination still resolves to a list page).
    function buildSetsFilterHref() {
      var parts = chips.map(function (c) { return 'f=' + encodeURIComponent(c.type + ':' + c.value); });
      return '/sets?' + parts.join('&');
    }

    /* Instant, offline, client-side suggestion ranking — no network call. Uses
       the same subsequence fuzzy matcher as openEntitySearchModal (fuzzyScore,
       above). Empty fragment shows the top candidates by their own global
       count (a quick browse/entry point); a typed fragment fuzzy-filters and
       sorts by match quality. Counts shown are each candidate's own total —
       not intersected with the currently active chips, which would need the
       server; the grid (updateGrid) is the actual live combined result. */
    function computeSuggestions() {
      var fragment = fragmentText().trim();
      var activeKeys = {};
      chips.forEach(function (c) { activeKeys[c.type + ':' + c.value] = true; });
      var pool = (paletteCandidates || []).filter(function (c) {
        return !activeKeys[c.type + ':' + c.value];
      });
      // Group by type, fuzzy-filter within each (or top-by-count when the
      // fragment is empty), take the top few per group, and emit in TYPE_ORDER
      // so renderSuggestions can print a header per contiguous group.
      var byType = {};
      pool.forEach(function (c) {
        var score;
        if (!fragment) {
          score = -c.count;   // ascending sort → highest count first
        } else {
          score = fuzzyScore(fragment, c.label);
          if (score === null) return;
        }
        (byType[c.type] = byType[c.type] || []).push({ item: c, score: score });
      });
      var out = [];
      TYPE_ORDER.forEach(function (t) {
        var arr = byType[t];
        if (!arr) return;
        arr.sort(function (a, b) { return a.score - b.score; });
        arr.slice(0, 6).forEach(function (s) { out.push(s.item); });
      });
      return out;
    }

    function updateSuggestions() {
      suggestions = paletteCandidates === null ? [] : computeSuggestions();
      highlighted = 0;
      renderChips();
      renderSuggestions();
    }

    function updateGrid() {
      var seq = ++requestSeq;
      var q = fragmentText().trim();
      fetch('/api/search-palette', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          q: q, chips: chips,
          sort: sortSpec ? sortSpec.field : '',
          order: sortSpec ? sortSpec.dir : 'desc',
        }),
      })
        .then(function (r) {
          if (!r.ok) {
            return r.text().then(function (body) {
              throw new Error('search-palette ' + r.status + ': ' + body);
            });
          }
          return r.json();
        })
        .then(function (data) {
          if (seq !== requestSeq) return; // a newer request already landed
          renderGrid(data);
          countEl.textContent = data.total_count + ' RESULTS';
          if (data.total_count > 0) {
            viewAllEl.style.display = '';
            viewAllEl.textContent = 'Browse ' + data.total_count + ' result' + (data.total_count === 1 ? '' : 's') + ' →';
          } else {
            viewAllEl.style.display = 'none';
          }
        })
        .catch(function (err) {
          if (seq !== requestSeq) return;
          console.error('search palette grid query failed:', err);
          gridEl.innerHTML = '';
          var errEl = document.createElement('div');
          errEl.className = 'palette-empty';
          errEl.textContent = 'search error — see browser console for details';
          gridEl.appendChild(errEl);
        });
    }

    function commitHighlighted() {
      var s = suggestions[Math.min(highlighted, suggestions.length - 1)];
      if (!s) return;
      chips.push({ type: s.type, value: s.value });
      input.value = leftoverText();
      input.focus();
      var len = input.value.length;
      input.setSelectionRange(len, len);
      updateSuggestions();
      updateGrid();
    }

    var gridDebounceTimer = null;
    input.addEventListener('input', function () {
      // A "field:asc|desc" token anywhere in the box is pulled out into the sort
      // chip and disappears from the input.
      var parsed = parseSortToken(input.value);
      if (parsed.sort) {
        input.value = parsed.clean;
        sortSpec = parsed.sort;
      }
      updateSuggestions(); // instant, pure in-memory work; also re-renders chips (incl. sort)
      clearTimeout(gridDebounceTimer);
      gridDebounceTimer = setTimeout(updateGrid, 150);
    });

    if (viewAllEl) {
      viewAllEl.addEventListener('click', function (e) {
        e.preventDefault();
        openResultsAsQueue(null);
      });
    }

    // "📍 Has location" toggles a `location:any` chip — the sentinel the backend
    // reads as "has any location metadata" (see the location facet in web.py).
    // Same chip-add code path as a suggestion click, just with a fixed value.
    var hasLocBtn = document.getElementById('palette-hasloc-btn');
    if (hasLocBtn) {
      hasLocBtn.addEventListener('click', function () {
        var idx = -1;
        chips.forEach(function (c, i) { if (c.type === 'location' && c.value === 'any') idx = i; });
        if (idx === -1) chips.push({ type: 'location', value: 'any' });
        else chips.splice(idx, 1);
        updateSuggestions();
        updateGrid();
      });
    }

    // paletteCandidates may still be loading the first time the palette opens
    // (a handful of small GETs, normally done well before that) — refresh
    // suggestions once they land so an early keystroke isn't stuck empty.
    paletteCandidatesPromise.then(function () {
      if (overlay.style.display === 'flex') updateSuggestions();
    });

    input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        highlighted = Math.min(highlighted + 1, Math.max(0, suggestions.length - 1));
        renderSuggestions();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        highlighted = Math.max(highlighted - 1, 0);
        renderSuggestions();
      } else if (e.key === 'Enter') {
        e.preventDefault();
        // Enter applies the highlighted filter suggestion when there is one;
        // otherwise it opens the current results as a queue.
        if (suggestions.length) commitHighlighted();
        else openResultsAsQueue(null);
      } else if (e.key === 'Backspace' && input.value === '') {
        // Peel off the sort chip first, then the most-recent filter chip.
        if (sortSpec) {
          e.preventDefault();
          sortSpec = null;
          renderChips();
          updateGrid();
        } else if (chips.length) {
          e.preventDefault();
          chips.pop();
          updateSuggestions();
          updateGrid();
        }
      } else if (e.key === 'Escape') {
        e.preventDefault();
        close();
      }
    });

    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) close();
    });

    if (navInput) {
      navInput.addEventListener('focus', function (e) {
        e.target.blur();
        open();
      });
    }
    document.addEventListener('keydown', function (e) {
      if (photoGridOpen) return;
      if (e.key !== '/') return;
      if (isTypingTarget(document.activeElement)) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      open();
    });
  })();

  // File-info "Taken" cell holds a raw unix timestamp — render it as a locale
  // date/time client-side (no server-side date filter exists).
  document.querySelectorAll('.taken-at-ts').forEach(function (el) {
    var ts = parseFloat(el.dataset.ts);
    if (!isNaN(ts)) el.textContent = new Date(ts * 1000).toLocaleString();
  });

  const fileId = window.MEDIA_FILE_ID;
  if (!fileId) return;

  /* Arrow-key photo navigation via the watch queue. The queue (file ids +
     cursor + a display label) is written to sessionStorage by base.html's
     click-capture script whenever a photo is opened from a grid (gallery,
     set/category detail, search, similar). If the current photo isn't part
     of a live queue — arrived via a non-grid link, a bookmark, or the queue
     is stale — there's nothing to browse: Left/Right do nothing. */
  (function () {
    const STORAGE_KEY = 'photoQueue';
    const currentId = parseInt(fileId, 10);

    function loadQueue() {
      try {
        const raw = JSON.parse(sessionStorage.getItem(STORAGE_KEY));
        if (raw && Array.isArray(raw.ids) && typeof raw.cursor === 'number') return raw;
      } catch (e) { /* ignore corrupt storage */ }
      return null;
    }

    let queue = loadQueue();
    if (queue) {
      const idx = queue.ids.indexOf(currentId);
      if (idx !== -1) {
        queue.cursor = idx;
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(queue));
      } else {
        // This photo isn't part of the last recorded queue — stale/unrelated.
        sessionStorage.removeItem(STORAGE_KEY);
        queue = null;
      }
    }

    window.__photoQueue = queue;

    // Neon progress line under the navbar: fills to the current position in the
    // queue (cursor+1 of N). Hidden when this photo isn't part of a queue.
    const progressEl = document.getElementById('queue-progress');
    if (progressEl) {
      if (queue && queue.ids.length > 0) {
        progressEl.style.width = ((queue.cursor + 1) / queue.ids.length * 100) + '%';
        progressEl.style.display = 'block';
      } else {
        progressEl.style.display = 'none';
      }
    }

    function goTo(id) {
      // Tells the beforeunload guard above this is in-queue browsing, not
      // abandoning a pending custom action — see its comment for why.
      sessionStorage.setItem('photoQueueNavigating', '1');
      window.location.href = '/photo/' + id;
    }

    // Move `delta` (+1 next / -1 previous) within the watch queue — shared by the
    // arrow keys and the touch swipe below.
    function stepQueue(delta) {
      if (!queue) return;
      var next = queue.cursor + delta;
      if (next < 0 || next >= queue.ids.length) return;
      queue.cursor = next;
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(queue));
      goTo(queue.ids[queue.cursor]);
    }

    // Warm the browser cache with the neighbouring full-res images (±2) so a
    // Left/Right/swipe step shows instantly instead of downloading /image on
    // navigation. /image is served with immutable cache headers, so these stay
    // cached. Deferred so the CURRENT photo's image gets bandwidth priority first.
    if (queue && queue.ids.length > 1) {
      setTimeout(function () {
        [1, -1, 2, -2].forEach(function (d) {
          var i = queue.cursor + d;
          if (i >= 0 && i < queue.ids.length) {
            var im = new Image();
            im.src = '/image/' + queue.ids[i];
          }
        });
      }, 350);
    }

    function isTypingTarget(el) {
      if (!el) return false;
      const tag = el.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
    }

    document.addEventListener('keydown', function (e) {
      if (photoGridOpen) return;
      if (isTypingTarget(document.activeElement)) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;

      // Up/Down scroll the displayed image inside #photo-stage (the element
      // made scrollable via overflow:auto in FILL/1:1 modes). Works regardless
      // of queue state; a visual no-op in FIT mode where nothing overflows.
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        const stage = document.getElementById('photo-stage');
        if (!stage) return;
        e.preventDefault(); // stop the browser page from also scrolling
        // Fixed ~120px step: predictable, wheel-like nudges that let you
        // fine-tune framing without overshooting on tall images.
        const step = 120;
        stage.scrollBy({ top: e.key === 'ArrowDown' ? step : -step, behavior: 'smooth' });
        return;
      }

      if (!queue) return;
      if (e.key === 'ArrowLeft') stepQueue(-1);
      else if (e.key === 'ArrowRight') stepQueue(1);
    });

    // Phone-first: swipe left = next photo, swipe right = previous — same queue
    // navigation as the arrow keys. Only fires when the image isn't zoomed/pannable
    // horizontally (FIT mode), so it doesn't fight panning in FILL / 1:1 modes.
    var stage = document.getElementById('photo-stage');
    if (stage && queue) {
      var tsx = 0, tsy = 0, tracking = false;
      stage.addEventListener('touchstart', function (e) {
        if (e.touches.length !== 1) { tracking = false; return; }
        tsx = e.touches[0].clientX; tsy = e.touches[0].clientY; tracking = true;
      }, { passive: true });
      stage.addEventListener('touchend', function (e) {
        if (!tracking) return;
        tracking = false;
        var t = e.changedTouches[0];
        var dx = t.clientX - tsx, dy = t.clientY - tsy;
        if (Math.abs(dx) < 60 || Math.abs(dx) < Math.abs(dy) * 1.3) return;  // not a horizontal flick
        if (stage.scrollWidth > stage.clientWidth + 4) return;  // zoomed/pannable — leave to native pan
        stepQueue(dx < 0 ? 1 : -1);
      }, { passive: true });
    }
  })();

  /* Space — a 3-wide thumbnail grid of every photo in the current watch
     queue (window.__photoQueue, set up by the IIFE just above), so a long
     queue can be jumped into directly instead of stepping through it one
     Left/Right at a time. Built entirely client-side from ids already in
     memory — no fetches, since this exists specifically to be the fast
     alternative to a network-heavy preview. Toggles via `photoGridOpen`
     (declared near the top of this file), which the Left/Right, "/", and
     F/Esc handlers above/below already check to no-op while this is open. */
  (function () {
    let overlay = null;
    let cellEls = [];
    let selectedIndex = 0;

    // Per-photo caption metadata (name + set names). The queue only carries ids,
    // so name/sets are fetched in one batch from /api/files/meta and cached here
    // across grid opens (queue contents don't change within a page). captionEls
    // maps id -> the caption <div> so a returning fetch can fill cells that were
    // already rendered.
    const metaCache = {};        // id -> { name, sets: [{id, name}] }
    let captionEls = {};         // id -> caption element (rebuilt each render)

    function applyCaption(el, meta) {
      if (!meta) { el.textContent = ''; return; }
      el.textContent = '';
      const nameEl = document.createElement('div');
      nameEl.className = 'queue-grid-caption-name';
      nameEl.textContent = meta.name || '';
      nameEl.title = meta.name || '';
      el.appendChild(nameEl);
      if (meta.sets && meta.sets.length) {
        const setsEl = document.createElement('div');
        setsEl.className = 'queue-grid-caption-sets';
        const setNames = meta.sets.map(function (s) { return s.name; });
        setsEl.textContent = setNames.join(', ');
        setsEl.title = setNames.join(', ');
        el.appendChild(setsEl);
      }
    }

    // Fetch name+sets for any queue ids not already cached, in one POST, then
    // fill their (already-rendered) caption elements. Non-blocking and
    // non-fatal: a failure just leaves captions empty rather than breaking the
    // grid (which is usable for navigation without them).
    function fillCaptions(ids) {
      const missing = ids.filter(function (id) { return !(id in metaCache); });
      if (!missing.length) return;
      fetch('/api/files/meta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids: missing }),
      })
        .then(function (r) { return r.ok ? r.json() : []; })
        .then(function (rows) {
          rows.forEach(function (row) {
            metaCache[row.id] = row;
            if (captionEls[row.id]) applyCaption(captionEls[row.id], row);
          });
        })
        .catch(function () { /* non-fatal: captions just stay empty */ });
    }

    // Select mode ("Add queue to set"): the same grid, but Enter/Shift+Arrows/
    // click toggle a multi-selection instead of navigating, and Ctrl+Enter (or
    // the footer button) sends the picked items to a chosen set.
    let selectMode = false;
    let targetSet = null;
    const picked = new Set();     // queue indices chosen for the set
    const alreadyIn = new Set();  // queue indices already in the target set (greyed, unselectable)
    let rangeAnchor = null;       // anchor index for Shift range-select

    function goTo(id) {
      // Tells the beforeunload guard above this is in-queue browsing, not
      // abandoning a pending custom action — see its comment for why.
      sessionStorage.setItem('photoQueueNavigating', '1');
      window.location.href = '/photo/' + id;
    }

    function isTypingTarget(el) {
      if (!el) return false;
      const tag = el.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
    }

    function buildOverlay() {
      const el = document.createElement('div');
      el.className = 'queue-grid-overlay';
      el.addEventListener('click', function (e) {
        if (e.target === el) closeGrid();
      });
      const panel = document.createElement('div');
      panel.className = 'queue-grid-panel';
      el.appendChild(panel);

      const footer = document.createElement('div');
      footer.className = 'queue-grid-footer';
      const hint = document.createElement('span');
      hint.className = 'queue-grid-hint';
      hint.textContent = 'Enter select · Shift+Arrows range · click to toggle · Ctrl+Enter add · Esc cancel';
      const commitBtn = document.createElement('button');
      commitBtn.type = 'button';
      commitBtn.className = 'stage-btn-accent queue-grid-commit-btn';
      commitBtn.addEventListener('click', commitSelection);
      footer.appendChild(hint);
      footer.appendChild(commitBtn);
      el.appendChild(footer);

      document.body.appendChild(el);
      return { el: el, panel: panel, footer: footer, commitBtn: commitBtn };
    }

    function render() {
      const queue = window.__photoQueue;
      overlay.panel.innerHTML = '';
      captionEls = {};   // rebuilt below; stale nodes are discarded with innerHTML
      cellEls = queue.ids.map(function (id, i) {
        const cell = document.createElement('button');
        cell.type = 'button';
        // Keep cells out of the keyboard focus path: a focused <button> fires a
        // native click on Enter, which would race the single keydown handler
        // below and make Enter ambiguous. With tabindex -1 + blur-on-click, Enter
        // is handled ONLY by that handler (toggle-select in select mode, open in
        // preview mode) — never by a phantom button activation.
        cell.tabIndex = -1;
        cell.className = 'queue-grid-cell' + (i === queue.cursor ? ' is-current' : '');
        const img = document.createElement('img');
        img.src = '/thumb/' + id;
        img.loading = 'lazy';
        cell.appendChild(img);
        // Caption (name + set names) sits over the bottom of the thumbnail. It's
        // pointer-events:none (see CSS) so it never intercepts the cell's click.
        // Filled asynchronously by fillCaptions() once the batch meta fetch
        // returns — cells render immediately without blocking on the network.
        const caption = document.createElement('div');
        caption.className = 'queue-grid-caption';
        applyCaption(caption, metaCache[id]);
        cell.appendChild(caption);
        captionEls[id] = caption;
        cell.addEventListener('click', function (e) {
          if (!selectMode) { goTo(id); return; }
          cell.blur();   // don't let the clicked cell keep focus (see tabindex note above)
          setSelected(i);
          if (alreadyIn.has(i)) return;   // greyed out — already in the set
          if (e.shiftKey && rangeAnchor !== null) {
            selectRange(rangeAnchor, i);
          } else {
            togglePick(i);
            rangeAnchor = i;
          }
        });
        overlay.panel.appendChild(cell);
        return cell;
      });
      setSelected(queue.cursor);
      refreshMarks();
      updateFooter();
      fillCaptions(queue.ids);   // async; fills captions when the batch returns
    }

    function setSelected(i) {
      if (cellEls[selectedIndex]) cellEls[selectedIndex].classList.remove('is-selected');
      selectedIndex = i;
      const cell = cellEls[selectedIndex];
      if (cell) {
        cell.classList.add('is-selected');
        cell.scrollIntoView({ block: 'center' });
      }
    }

    function moveSelection(delta) {
      const next = Math.max(0, Math.min(cellEls.length - 1, selectedIndex + delta));
      setSelected(next);
    }

    // The grid is responsive (grid-template-columns: repeat(auto-fill,
    // minmax(170px, 1fr))), so the column count depends on viewport width and
    // is NOT a fixed constant. Measure it live from the rendered DOM: the first
    // row is every leading cell that shares cellEls[0]'s offsetTop; the count of
    // those cells IS the column count. Recompute on each Up/Down press because
    // the window may have been resized since the grid opened. Fall back to a
    // single column so a Down press never becomes a no-op / moves by 0.
    function currentColumns() {
      if (!cellEls.length) return 1;
      const top0 = cellEls[0].offsetTop;
      let cols = cellEls.findIndex(function (c) { return c.offsetTop > top0; });
      if (cols <= 0) cols = cellEls.length;   // -1: all cells fit in one row
      return cols;
    }

    function togglePick(i) {
      if (alreadyIn.has(i)) return;
      if (picked.has(i)) picked.delete(i); else picked.add(i);
      refreshMarks();
      updateFooter();
    }

    function selectRange(a, b) {
      const lo = Math.min(a, b), hi = Math.max(a, b);
      for (let i = lo; i <= hi; i++) {
        if (!alreadyIn.has(i)) picked.add(i);
      }
      refreshMarks();
      updateFooter();
    }

    function refreshMarks() {
      cellEls.forEach(function (cell, i) {
        cell.classList.toggle('is-picked', picked.has(i));
        cell.classList.toggle('is-in-set', alreadyIn.has(i));
      });
    }

    function updateFooter() {
      if (!overlay || !overlay.commitBtn) return;
      const n = picked.size;
      overlay.commitBtn.textContent = 'Add ' + n + ' item' + (n === 1 ? '' : 's') +
        ' to ' + (targetSet ? targetSet.name : '');
      overlay.commitBtn.disabled = n === 0;
    }

    function commitSelection() {
      if (!selectMode || !targetSet || picked.size === 0) return;
      const queue = window.__photoQueue;
      // Map picked indices -> file ids, deduped (a queue can repeat a file id,
      // e.g. multiple face matches in one photo).
      const ids = Array.from(new Set(
        Array.from(picked).sort(function (a, b) { return a - b; })
          .map(function (i) { return queue.ids[i]; })
      ));
      overlay.commitBtn.disabled = true;
      overlay.commitBtn.textContent = 'Adding…';
      // Sequential, not Promise.all — the backend shares one SQLite connection
      // per db with no locking, so parallel writes corrupt cursor state (same
      // reason as the search page's bulk add-to-set).
      let added = 0, firstErr = null;
      ids.reduce(function (chain, id) {
        return chain.then(function () {
          return window.assignFileToSet(id, targetSet.id)
            .then(function () { added++; })
            .catch(function (err) { if (!firstErr) firstErr = err; });
        });
      }, Promise.resolve()).then(function () {
        if (firstErr) {
          overlay.commitBtn.disabled = false;
          updateFooter();
          showToast('Added ' + added + ' of ' + ids.length + ' — failed: ' + firstErr.message);
        } else {
          closeGrid();
          showToast('Added ' + added + ' item' + (added === 1 ? '' : 's') + ' to "' + targetSet.name + '".');
        }
      });
    }

    function openGrid(opts) {
      const queue = window.__photoQueue;
      if (!queue || !queue.ids.length) return;
      opts = opts || {};
      selectMode = !!opts.selectMode;
      targetSet = opts.set || null;
      picked.clear();
      alreadyIn.clear();
      rangeAnchor = null;
      if (!overlay) overlay = buildOverlay();
      overlay.el.classList.toggle('select-mode', selectMode);
      render();
      overlay.el.classList.add('open');
      photoGridOpen = true;

      // In select mode, grey out (but still show) items already in the set.
      if (selectMode && targetSet) {
        fetch('/api/sets/' + targetSet.id + '/file-ids')
          .then(function (r) { return r.ok ? r.json() : { file_ids: [] }; })
          .then(function (data) {
            const member = new Set(data.file_ids || []);
            queue.ids.forEach(function (id, i) {
              if (member.has(id)) { alreadyIn.add(i); picked.delete(i); }
            });
            refreshMarks();
            updateFooter();
          })
          .catch(function () { /* non-fatal: just don't grey anything */ });
      }
    }

    function closeGrid() {
      if (overlay) {
        overlay.el.classList.remove('open');
        overlay.el.classList.remove('select-mode');
      }
      photoGridOpen = false;
      selectMode = false;
    }

    // Opened by the sidebar's "Add queue to set" button after a set is picked.
    window.openQueueSelectGrid = function (set) { openGrid({ selectMode: true, set: set }); };

    function moveAndMaybeRange(delta, shift) {
      if (shift && rangeAnchor === null) rangeAnchor = selectedIndex;
      moveSelection(delta);
      if (shift) selectRange(rangeAnchor, selectedIndex);
      else rangeAnchor = selectedIndex;
    }

    document.addEventListener('keydown', function (e) {
      if (e.key === ' ' || e.key === 'Spacebar') {
        if (isTypingTarget(document.activeElement)) return;
        if (e.altKey || e.ctrlKey || e.metaKey) return;
        if (!photoGridOpen && !window.__photoQueue) return;
        e.preventDefault();
        if (photoGridOpen) closeGrid(); else openGrid();   // Space is always navigate mode
        return;
      }
      if (!photoGridOpen) return;
      if (e.key === 'Escape') { e.preventDefault(); closeGrid(); return; }

      if (selectMode) {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); commitSelection(); return; }
        if (e.key === 'Enter') { e.preventDefault(); togglePick(selectedIndex); rangeAnchor = selectedIndex; return; }
        if (e.key === 'ArrowLeft') { e.preventDefault(); moveAndMaybeRange(-1, e.shiftKey); return; }
        if (e.key === 'ArrowRight') { e.preventDefault(); moveAndMaybeRange(1, e.shiftKey); return; }
        if (e.key === 'ArrowUp') { e.preventDefault(); moveAndMaybeRange(-currentColumns(), e.shiftKey); return; }
        if (e.key === 'ArrowDown') { e.preventDefault(); moveAndMaybeRange(currentColumns(), e.shiftKey); return; }
        return;
      }

      if (e.key === 'Enter') { e.preventDefault(); goTo(window.__photoQueue.ids[selectedIndex]); return; }
      if (e.key === 'ArrowLeft') { e.preventDefault(); moveSelection(-1); return; }
      if (e.key === 'ArrowRight') { e.preventDefault(); moveSelection(1); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); moveSelection(-currentColumns()); return; }
      if (e.key === 'ArrowDown') { e.preventDefault(); moveSelection(currentColumns()); return; }
    });
  })();

  /* "Custom action" banner — renders whatever setPhotoCustomAction() last
     stashed as a persistent button at the very top of the sidebar (see
     #sidebar-custom-action in photo.html). One-shot: cleared from
     sessionStorage as soon as its request succeeds, so it doesn't reappear
     on the next photo/reload. Also offers an explicit Cancel — previously
     the only way to stop seeing an abandoned action was to complete it or
     manually clear storage, so a "never mind" banner just followed you
     through every photo view in the tab indefinitely. */
  (function () {
    const raw = sessionStorage.getItem('photoCustomAction');
    if (!raw) return;
    let action;
    try {
      action = JSON.parse(raw);
    } catch (e) {
      sessionStorage.removeItem('photoCustomAction');
      return;
    }
    if (!action || !action.buttonLabel || !action.request) {
      sessionStorage.removeItem('photoCustomAction');
      return;
    }

    const mount = document.getElementById('sidebar-custom-action');
    if (!mount) return;

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn-similar btn-primary';
    btn.textContent = action.buttonLabel;
    const cancelBtn = document.createElement('button');
    cancelBtn.type = 'button';
    cancelBtn.className = 'btn-similar';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.style.marginLeft = '6px';
    const status = document.createElement('div');
    status.className = 'sub';
    status.style.marginTop = '6px';
    status.style.display = 'none';

    btn.addEventListener('click', function () {
      btn.disabled = true;
      cancelBtn.disabled = true;
      fetch(action.request.url, {
        method: action.request.method || 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: action.request.body ? JSON.stringify(action.request.body) : undefined,
      })
        .then(function (r) { if (!r.ok) throw new Error('Request failed: ' + r.status); return r.json(); })
        .then(function (data) {
          sessionStorage.removeItem('photoCustomAction');
          status.textContent = (action.successMessage || 'Done.').replace(/\{\{(\w+)\}\}/g, function (_, key) {
            return data[key] !== undefined ? data[key] : '';
          });
          status.style.display = 'block';
          btn.style.display = 'none';
          cancelBtn.style.display = 'none';
        })
        .catch(function (err) {
          btn.disabled = false;
          cancelBtn.disabled = false;
          status.textContent = 'Failed: ' + err.message;
          status.style.display = 'block';
        });
    });

    cancelBtn.addEventListener('click', function () {
      const subject = action.count != null && action.targetLabel
        ? action.count + ' photo(s) will NOT be added to "' + action.targetLabel + '"'
        : 'This action will NOT be completed';
      if (!confirm(subject + ' if you cancel.\n\nProceed?')) return;
      sessionStorage.removeItem('photoCustomAction');
      btn.remove();
      cancelBtn.remove();
      status.textContent = 'Canceled.';
      status.style.display = 'block';
    });

    mount.appendChild(btn);
    mount.appendChild(cancelBtn);
    mount.appendChild(status);
  })();

  const tagList = document.getElementById('tag-list');

  function renderTags(tags) {
    tagList.innerHTML = '';
    tags.forEach(function (tag) {
      const span = document.createElement('span');
      const isNegative = tag.polarity === 'negative';
      span.className = isNegative ? 'tag-negative' : 'tag-removable';
      span.dataset.tagId = tag.id;

      const display = document.createElement('span');
      display.className = 'tag-label-display';
      if (isNegative) {
        const labelSpan = document.createElement('span');
        labelSpan.className = 'tag-negative-label';
        labelSpan.textContent = tag.label;
        display.appendChild(labelSpan);
      } else {
        display.appendChild(document.createTextNode(tag.label));
      }
      span.appendChild(display);

      const heartBtn = document.createElement('button');
      heartBtn.type = 'button';
      heartBtn.className = 'heart-btn-inline tag-heart-btn';
      heartBtn.style.fontSize = '0.85em';
      heartBtn.title = 'Favorite this tag (right-click to lower)';
      window.renderHeart(heartBtn, tag.favorite);
      span.appendChild(heartBtn);

      const editBtn = document.createElement('button');
      editBtn.className = 'tag-edit-btn';
      editBtn.type = 'button';
      editBtn.title = 'Edit label';
      editBtn.textContent = '✏️';
      span.appendChild(editBtn);

      const btn = document.createElement('button');
      btn.className = 'rm';
      btn.type = 'button';
      btn.textContent = '×';
      btn.title = 'Remove tag';
      // no per-button listener here — handled by delegation on tagList below,
      // so both server-rendered (page load) and dynamically-added chips work.

      span.appendChild(btn);
      tagList.appendChild(span);
    });
    // Detected-object chips (gray, read-only) live in the same list but aren't
    // part of the add/remove round-trip — re-append them after every rebuild.
    (window.DETECTED_CLASSES || []).forEach(function (cls) {
      const span = document.createElement('span');
      span.className = 'tag-detected';
      span.title = 'Auto-detected';
      span.textContent = cls;
      tagList.appendChild(span);
    });
  }

  // Event delegation: covers the initial server-rendered tag chips too,
  // not just ones created by renderTags() after an add/remove round-trip.
  // Manually-added tags (blue/red) are just deleted on ×. A detected-object
  // chip (gray) has no row to delete — × instead files a negative tag for it,
  // which both hides it and removes the underlying detection server-side.
  function updateTagLabel(tagId, label) {
    return fetch('/api/files/' + fileId + '/tags/' + encodeURIComponent(tagId), {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label: label }),
    })
      .then(function (r) {
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      })
      .then(function (data) {
        renderTags(data.tags);
      });
  }

  /* Tag favorite is a counter too — shared by the click (+1) and contextmenu (-1)
     delegated handlers below. Renders "♥ N"/"♡" via the shared renderHeart. */
  function bumpTagFavorite(heartBtn, chip, delta) {
    fetch('/api/tags/' + chip.dataset.tagId + '/favorite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ delta: delta }),
    })
      .then(function (r) { if (!r.ok) throw new Error('Request failed'); return r.json(); })
      .then(function (data) { window.renderHeart(heartBtn, data.count); })
      .catch(function (err) { showToast('Failed to update favorite: ' + err.message); });
  }

  if (tagList) {
    tagList.addEventListener('contextmenu', function (e) {
      const heartBtn = e.target.closest('.tag-heart-btn');
      if (!heartBtn) return;
      const chip = heartBtn.closest('.tag-removable, .tag-negative');
      if (!chip) return;
      e.preventDefault();
      bumpTagFavorite(heartBtn, chip, -1);
    });
    tagList.addEventListener('click', function (e) {
      const heartBtn = e.target.closest('.tag-heart-btn');
      if (heartBtn) {
        const chip = heartBtn.closest('.tag-removable, .tag-negative');
        if (!chip) return;
        bumpTagFavorite(heartBtn, chip, 1);
        return;
      }

      const editBtn = e.target.closest('.tag-edit-btn');
      if (editBtn) {
        const chip = editBtn.closest('.tag-removable, .tag-negative');
        if (!chip) return;
        const display = chip.querySelector('.tag-label-display');
        const currentLabel = display.textContent.trim();
        const input = document.createElement('input');
        input.type = 'text';
        input.value = currentLabel;
        input.style.width = '80px';
        input.style.fontSize = '0.85em';
        let settled = false;

        function restoreDisplay() {
          if (input.isConnected) input.replaceWith(display);
        }
        function commit() {
          if (settled) return;
          settled = true;
          const newLabel = input.value.trim();
          if (!newLabel || newLabel === currentLabel) {
            restoreDisplay();
            return;
          }
          updateTagLabel(chip.dataset.tagId, newLabel).catch(function (err) {
            showToast('Failed to update tag: ' + err.message);
          });
        }
        input.addEventListener('keydown', function (ev) {
          if (ev.key === 'Enter') { ev.preventDefault(); commit(); }
          else if (ev.key === 'Escape') { settled = true; restoreDisplay(); }
        });
        input.addEventListener('blur', commit);
        display.replaceWith(input);
        input.focus();
        input.select();
        return;
      }

      const btn = e.target.closest('.rm');
      if (!btn) return;
      const manualChip = btn.closest('.tag-removable, .tag-negative');
      if (manualChip) {
        removeTag(manualChip.dataset.tagId);
        return;
      }
      const detectedChip = btn.closest('.tag-detected');
      if (detectedChip) {
        const label = detectedChip.dataset.detectedLabel;
        if (!label) return;
        btn.disabled = true;
        fetch('/api/files/' + fileId + '/tags', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tag: label, polarity: 'negative' }),
        })
          .then(function (r) {
            if (!r.ok) throw new Error('Request failed: ' + r.status);
            return r.json();
          })
          .then(function () {
            // The detection row is gone server-side too — reload so the
            // gray chip disappears and the new red one renders correctly.
            location.reload();
          })
          .catch(function (err) {
            btn.disabled = false;
            showToast('Failed to reject tag: ' + err.message);
          });
      }
    });
  }

  function addTag(tag, polarity) {
    fetch('/api/files/' + fileId + '/tags', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tag: tag, polarity: polarity || 'positive' }),
    })
      .then(function (r) {
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      })
      .then(function (data) {
        renderTags(data.tags);
      })
      .catch(function (err) {
        showToast('Failed to add tag: ' + err.message);
      });
  }

  function removeTag(tagId) {
    fetch('/api/files/' + fileId + '/tags/' + encodeURIComponent(tagId), {
      method: 'DELETE',
    })
      .then(function (r) {
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      })
      .then(function (data) {
        renderTags(data.tags);
      })
      .catch(function (err) {
        showToast('Failed to remove tag: ' + err.message);
      });
  }

  const addTagBtn = document.getElementById('add-tag-btn');
  if (addTagBtn) {
    addTagBtn.addEventListener('click', function () {
      openEntitySearchModal({
        type: 'tag',
        onResolved: function (entity) { addTag(entity.name, 'positive'); },
      });
    });
  }

  const tagNegativeBtn = document.getElementById('tag-negative-btn');
  if (tagNegativeBtn) {
    tagNegativeBtn.addEventListener('click', function () {
      openEntitySearchModal({
        type: 'tag',
        title: 'Mark tag as NOT present',
        onResolved: function (entity) { addTag(entity.name, 'negative'); },
      });
    });
  }

  const reindexBtn = document.getElementById('reindex-tags-btn');
  const reindexStatus = document.getElementById('reindex-tags-status');
  if (reindexBtn) {
    reindexBtn.addEventListener('click', function () {
      reindexBtn.disabled = true;
      reindexStatus.style.display = 'block';
      reindexStatus.textContent = 'Reindexing…';
      fetch('/api/files/' + fileId + '/reindex', { method: 'POST' })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Request failed'); });
          return r.json();
        })
        .then(function () { location.reload(); })
        .catch(function (err) {
          reindexBtn.disabled = false;
          reindexStatus.textContent = 'Failed: ' + err.message;
        });
    });
  }

  /* Generic drag-to-draw-a-box-on-the-photo, shared by "Add face" and "Label region" */
  const photoWrap = document.getElementById('photo-image-wrap');
  const photoImg = document.getElementById('photo-image');

  // Browsers start a native "drag this image out" operation on mousedown+move over
  // an <img>, which fights with our custom box-drawing and can end in a save/open-image
  // action. draggable="false" + CSS -webkit-user-drag stop most of it; this catches the rest.
  if (photoImg) {
    photoImg.addEventListener('dragstart', function (e) { e.preventDefault(); });
  }

  function wireBoxDraw(triggerBtn, activeLabel, idleLabel, onBoxDrawn) {
    if (!triggerBtn || !photoWrap || !photoImg) return null;
    const state = { drawing: false, dragging: false, startX: 0, startY: 0, box: null };

    triggerBtn.addEventListener('click', function () {
      if (state.drawing) {
        deactivate();
        return;
      }
      deactivateOthers(state);
      state.drawing = true;
      photoWrap.classList.add('add-face-active');
      triggerBtn.textContent = activeLabel;
    });

    function deactivate() {
      state.drawing = false;
      photoWrap.classList.remove('add-face-active');
      triggerBtn.textContent = idleLabel;
    }
    state.deactivate = deactivate;

    photoWrap.addEventListener('mousedown', function (e) {
      if (!state.drawing) return;
      e.preventDefault();
      state.dragging = true;
      const rect = photoWrap.getBoundingClientRect();
      state.startX = e.clientX - rect.left;
      state.startY = e.clientY - rect.top;
      state.box = document.createElement('div');
      state.box.className = 'face-draw-box';
      state.box.style.left = state.startX + 'px';
      state.box.style.top = state.startY + 'px';
      state.box.style.width = '0px';
      state.box.style.height = '0px';
      photoWrap.appendChild(state.box);
    });

    photoWrap.addEventListener('mousemove', function (e) {
      if (!state.dragging || !state.box) return;
      const rect = photoWrap.getBoundingClientRect();
      const curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
      const curY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
      const left = Math.min(state.startX, curX);
      const top = Math.min(state.startY, curY);
      state.box.style.left = left + 'px';
      state.box.style.top = top + 'px';
      state.box.style.width = Math.abs(curX - state.startX) + 'px';
      state.box.style.height = Math.abs(curY - state.startY) + 'px';
    });

    window.addEventListener('mouseup', function () {
      if (!state.dragging) return;
      state.dragging = false;
      const wasDrawing = state.drawing;
      deactivate();
      if (!wasDrawing || !state.box) return;

      const cssX1 = parseFloat(state.box.style.left);
      const cssY1 = parseFloat(state.box.style.top);
      const cssW = parseFloat(state.box.style.width);
      const cssH = parseFloat(state.box.style.height);
      state.box.remove();
      state.box = null;

      if (cssW < 10 || cssH < 10) return;

      const scaleX = photoImg.naturalWidth / photoImg.clientWidth;
      const scaleY = photoImg.naturalHeight / photoImg.clientHeight;
      const bbox = [
        cssX1 * scaleX,
        cssY1 * scaleY,
        (cssX1 + cssW) * scaleX,
        (cssY1 + cssH) * scaleY,
      ];
      onBoxDrawn(bbox);
    });

    return state;
  }

  function deactivateOthers(exceptState) {
    boxDrawStates.forEach(function (s) {
      if (s !== exceptState) s.deactivate();
    });
  }

  const boxDrawStates = [];

  /* Hover a "Detected objects" chip (or a labeled-region chip) to see WHERE
     on the photo it was found — same idea as the swipe-review fullview's
     Right-arrow highlight (swipe-core.js), just triggered by hover instead
     of a keypress, and reusing the existing drag-to-draw box's own
     naturalWidth/clientWidth scaling (photoImg) so it lines up correctly in
     whatever fit mode is active, same as .face-draw-box already does for
     drawing. window.DETECTED_BBOXES ({class_name: [x1,y1,x2,y2]} in ORIGINAL
     image pixels, see database.get_detection_bboxes) is only set on
     photo.html; harmless no-op anywhere it's undefined/empty. */
  if (photoWrap && photoImg) {
    let hoverBox = null;

    function hideHoverBbox() {
      if (hoverBox) { hoverBox.remove(); hoverBox = null; }
    }

    function showHoverBbox(bbox, className) {
      hideHoverBbox();
      if (!photoImg.naturalWidth || !photoImg.clientWidth) return; // image not loaded yet
      const scaleX = photoImg.clientWidth / photoImg.naturalWidth;
      const scaleY = photoImg.clientHeight / photoImg.naturalHeight;
      const [x1, y1, x2, y2] = bbox;
      hoverBox = document.createElement('div');
      hoverBox.className = className || 'face-draw-box';
      hoverBox.style.left = (x1 * scaleX) + 'px';
      hoverBox.style.top = (y1 * scaleY) + 'px';
      hoverBox.style.width = ((x2 - x1) * scaleX) + 'px';
      hoverBox.style.height = ((y2 - y1) * scaleY) + 'px';
      photoWrap.appendChild(hoverBox);
    }

    document.querySelectorAll('.tag-detected[data-detected-label]').forEach(function (chip) {
      const bboxes = window.DETECTED_BBOXES || {};
      const bbox = bboxes[chip.dataset.detectedLabel];
      if (!bbox) return; // detected before bboxes were stored, or a stale/renamed label — nothing to show
      chip.addEventListener('mouseenter', function () { showHoverBbox(bbox); });
      chip.addEventListener('mouseleave', hideHoverBbox);
    });

    /* Hover a face (its chip in the Faces menu, or its at-a-glance avatar) to
       highlight WHERE that face is on the photo — same overlay + coordinate
       mapping as the detected-object hover, with a distinct solid box so a face
       highlight doesn't read as a draw-in-progress. Face bbox is [x1,y1,x2,y2]
       in ORIGINAL image pixels (see _combined_faces_for_file). */
    document.querySelectorAll('.face-chip[data-bbox], .face-glance[data-bbox]').forEach(function (el) {
      let bbox;
      try { bbox = JSON.parse(el.getAttribute('data-bbox')); } catch (e) { return; }
      if (!Array.isArray(bbox) || bbox.length !== 4) return;
      el.addEventListener('mouseenter', function () { showHoverBbox(bbox, 'face-hilite-box'); });
      el.addEventListener('mouseleave', hideHoverBbox);
    });
  }

  /* Add face */
  const addFaceBtn = document.getElementById('add-face-btn');
  const addFaceStatus = document.getElementById('add-face-status');

  const addFaceState = wireBoxDraw(addFaceBtn, 'Click and drag on the photo…', '＋ Add face', function (bbox) {
    addFaceStatus.style.display = 'block';
    addFaceStatus.textContent = 'Detecting…';
    fetch('/api/files/' + fileId + '/faces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bbox: bbox }),
    })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Request failed'); });
        return r.json();
      })
      .then(function (data) {
        addFaceStatus.style.display = 'none';
        openFaceNamingModal(data.id);
      })
      .catch(function (err) {
        addFaceStatus.textContent = 'Failed: ' + err.message;
      });
  });
  if (addFaceState) boxDrawStates.push(addFaceState);

  /* Label region (spatial tag) — positive ("this area IS X") and negative
     ("this area is confirmed NOT X", a hard negative for CLIP training, see
     manual_db.add_spatial_tag/clip_tag_classifier.py) share the same
     draw-a-box-then-pick-a-tag flow, differing only in the polarity posted. */
  const labelRegionStatus = document.getElementById('label-region-status');

  function wireLabelRegion(btnId, polarity, modalTitle) {
    const btn = document.getElementById(btnId);
    const idleLabel = btn ? btn.textContent : '';
    const state = wireBoxDraw(btn, 'Click and drag on the photo…', idleLabel, function (bbox) {
      openEntitySearchModal({
        type: 'tag',
        title: modalTitle,
        onResolved: function (entity) {
          if (labelRegionStatus) { labelRegionStatus.style.display = 'block'; labelRegionStatus.textContent = 'Saving…'; }
          fetch('/api/files/' + fileId + '/tags/region', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ label: entity.name, bbox: bbox, polarity: polarity }),
          })
            .then(function (r) {
              if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Request failed'); });
              return r.json();
            })
            .then(function () {
              location.reload();
            })
            .catch(function (err) {
              if (labelRegionStatus) { labelRegionStatus.style.display = 'block'; labelRegionStatus.textContent = 'Failed: ' + err.message; }
            });
        },
      });
    });
    if (state) boxDrawStates.push(state);
  }

  wireLabelRegion('label-region-btn', 'positive', 'Label this region');
  wireLabelRegion('label-region-negative-btn', 'negative', 'Label this region as NOT…');

  /* Label person — drag a box around a person's body; it becomes a searchable
     find-by-body crop (mirrors "Add face" but CLIP-embeds the body crop via
     /api/files/{id}/bodies). Deep-linked from the find-by-body page's "Manually
     label person" button via the #label-person hash. */
  const labelPersonStatus = document.getElementById('label-person-status');
  const labelPersonBtn = document.getElementById('label-person-btn');
  const labelPersonState = wireBoxDraw(labelPersonBtn, 'Drag a box around the person…', '🧍 Label person', function (bbox) {
    if (labelPersonStatus) { labelPersonStatus.style.display = 'block'; labelPersonStatus.textContent = 'Embedding…'; }
    fetch('/api/files/' + fileId + '/bodies', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bbox: bbox }),
    })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Request failed'); });
        return r.json();
      })
      .then(function () {
        if (labelPersonStatus) {
          labelPersonStatus.textContent = 'Saved. ';
          var a = document.createElement('a');
          a.href = '/body-similar/' + fileId;
          a.textContent = 'Find similar people →';
          labelPersonStatus.appendChild(a);
        }
      })
      .catch(function (err) {
        if (labelPersonStatus) { labelPersonStatus.style.display = 'block'; labelPersonStatus.textContent = 'Failed: ' + err.message; }
      });
  });
  if (labelPersonState) boxDrawStates.push(labelPersonState);

  /* Find pill — 🔲 region segment: click to enter draw mode (segment shows ✏️ +
     the photo gets a crosshair), then drag a box around a thing (e.g. a wall);
     the crop is CLIP-embedded and ranked (tile index if built, else whole-image),
     opening the matches in the browsable watch-queue. */
  const searchRegionBtn = document.querySelector('.find-by-region-btn');
  const searchRegionState = wireBoxDraw(searchRegionBtn, '✏️', '🔲', function (bbox) {
    openMatchesAsQueue({
      el: searchRegionBtn,
      url: '/api/files/' + fileId + '/region-search',
      fetchInit: { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ bbox: bbox }) },
      label: 'Region matches',
      extractIds: function (data) { return (data.results || []).map(function (r) { return r.file_id; }); },
      onEmpty: function () { if (window.showToast) showToast('No matches found for that region.'); },
    });
  });
  if (searchRegionState) boxDrawStates.push(searchRegionState);

  // Deep-link: /photo/{id}#label-person (from find-by-body) enters draw mode on load.
  if (window.location.hash === '#label-person' && labelPersonBtn) {
    labelPersonBtn.click();
  }

  /* "Open ranked matches as a browsable watch-queue" — the UX the face 🔎
     pioneered (Left/Right browse + Space grid) instead of a separate results
     page. Shared verbatim by find-by-face, find-by-body, and find-similar-images:
     fetch the ranked matches, seed the queue with their file_ids, jump to the
     first. Swaps the clicked control to ⏳ while the request is in flight. */
  function openMatchesAsQueue(opts) {
    var el = opts.el;
    if (!el || el.dataset.busy) return;
    el.dataset.busy = '1';
    var glyph = el.textContent;
    el.textContent = '⏳';
    fetch(opts.url, opts.fetchInit)
      .then(function (r) { if (!r.ok) throw new Error('status ' + r.status); return r.json(); })
      .then(function (data) {
        var seen = {}, ids = [];
        (opts.extractIds(data) || []).forEach(function (id) {
          if (id != null && !seen[id]) { seen[id] = 1; ids.push(id); }
        });
        if (!ids.length) {
          el.textContent = glyph; delete el.dataset.busy;
          if (opts.onEmpty) opts.onEmpty(data);
          else if (window.showToast) showToast('No matches found.');
          return;
        }
        setPhotoQueue(ids, opts.label, 0);
        sessionStorage.setItem('photoQueueNavigating', '1');
        window.location.href = '/photo/' + ids[0];
      })
      .catch(function (err) {
        el.textContent = glyph; delete el.dataset.busy;
        if (window.showToast) showToast('Failed: ' + err.message);
      });
  }

  /* Face 🔎 — similar faces. Looser 0.3 threshold: same-person crops often land
     ~0.35-0.45, and this is a browse-and-skip queue (restores the reach of the
     old "expand similar" slider we removed). */
  document.querySelectorAll('.find-similar-faces-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var ref = btn.getAttribute('data-face-ref');
      if (!ref) return;
      openMatchesAsQueue({
        el: btn,
        url: '/api/faces/' + encodeURIComponent(ref) + '/similar?threshold=0.3',
        label: 'Similar faces',
        extractIds: function (data) { return (data.results || []).map(function (f) { return f.file_id; }); },
        onEmpty: function () { if (window.showToast) showToast('No similar faces found.'); },
      });
    });
  });

  /* Find pill — 📏 distance segment: whole-image CLIP similarity into the queue. */
  var findDistanceBtn = document.querySelector('.find-by-distance-btn');
  if (findDistanceBtn) {
    findDistanceBtn.addEventListener('click', function () {
      openMatchesAsQueue({
        el: findDistanceBtn,
        url: '/api/files/' + findDistanceBtn.dataset.fileId + '/similar',
        label: 'Similar images',
        extractIds: function (data) { return (data.results || []).map(function (r) { return r.file_id; }); },
        onEmpty: function (data) { if (window.showToast) showToast((data && data.message) || 'No similar images found.'); },
      });
    });
  }

  /* Find pill — 🧍 body segment: find-by-body into the queue. First open of a
     not-yet-indexed photo runs a person-only YOLO/CLIP pass server-side (hence
     the ⏳). If the photo has multiple people and none is pre-selected the API
     returns no ranked results — fall back to the /body-similar page so the user
     can pick which person. If NO person was detected at all, pop a recovery
     modal offering a forced person-only reindex or manual labeling (the same
     two escape hatches the dedicated /body-similar page has). */
  var findBodyBtn = document.querySelector('.find-by-body-btn');
  if (findBodyBtn) {
    var bodyFileId = findBodyBtn.dataset.fileId;

    function runFindByBody() {
      openMatchesAsQueue({
        el: findBodyBtn,
        url: '/api/files/' + bodyFileId + '/body-similar',
        label: 'Similar people',
        extractIds: function (data) { return (data.results || []).map(function (r) { return r.id; }); },
        onEmpty: function (data) {
          if (data && data.no_people) { openBodyRecoveryModal(); return; }
          if (data && data.crops && data.crops.length > 1) {
            window.location.href = '/body-similar/' + bodyFileId;
            return;
          }
          if (window.showToast) showToast((data && data.message) || 'No similar people found.');
        },
      });
    }

    function openBodyRecoveryModal() {
      if (!window.openModal) {   // no modal on this page — degrade to the full page
        window.location.href = '/body-similar/' + bodyFileId;
        return;
      }
      openModal('No people detected', function (box) {
        var msg = document.createElement('div');
        msg.className = 'sub';
        msg.style.marginBottom = '12px';
        msg.textContent = 'The general index found no person here — it may have been '
          + 'labeled as child/crowd/etc. Force a person-only reindex, or draw the box yourself.';
        box.appendChild(msg);

        var actions = document.createElement('div');
        actions.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;';

        var reBtn = document.createElement('button');
        reBtn.type = 'button';
        reBtn.className = 'btn-similar';
        reBtn.style.width = 'auto';
        reBtn.textContent = '🔄 Reindex now';
        reBtn.addEventListener('click', function () {
          reBtn.disabled = true;
          reBtn.textContent = 'Reindexing…';
          fetch('/api/files/' + bodyFileId + '/body-reindex', { method: 'POST' })
            .then(function (r) {
              if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || ('status ' + r.status)); });
              return r.json();
            })
            .then(function () { closeModal(); runFindByBody(); })  // retry with fresh crops
            .catch(function (err) {
              reBtn.disabled = false;
              reBtn.textContent = '🔄 Reindex now';
              if (window.showToast) showToast('Reindex failed: ' + err.message);
            });
        });

        var labelBtn = document.createElement('button');
        labelBtn.type = 'button';
        labelBtn.className = 'btn-similar';
        labelBtn.style.width = 'auto';
        labelBtn.textContent = '🧍 Manually label person';
        labelBtn.addEventListener('click', function () {
          closeModal();
          var lp = document.getElementById('label-person-btn');
          if (lp) lp.click();   // enter the draw-a-box-around-the-person flow, right here
          else window.location.href = '/photo/' + bodyFileId + '#label-person';
        });

        actions.appendChild(reBtn);
        actions.appendChild(labelBtn);
        box.appendChild(actions);
      });
    }

    findBodyBtn.addEventListener('click', runFindByBody);
  }

  /* Face naming modal — click any face chip (named or "Unknown") to name/rename
     it. Goes through the shared entity picker: fuzzy-search known people, pick
     one, or type a new name (or none at all — "Save without a name" confirms a
     distinct, still-unnamed person; the server auto-generates a placeholder
     like "Unnamed N", renamable later exactly like any other name). The
     embedding-similarity match, if any, is wired in as an extra one-click
     suggestion alongside the regular search results. */
  function openFaceNamingModal(faceRef) {
    function saveName(name) {
      fetch('/api/faces/' + faceRef + '/identity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name || null }),
      })
        .then(function (r) {
          if (!r.ok) throw new Error('Request failed: ' + r.status);
          return r.json();
        })
        .then(function () { location.reload(); })
        .catch(function (err) { showToast('Failed to save name: ' + err.message); });
    }

    openEntitySearchModal({
      type: 'identity',
      title: 'Name this face',
      previewImage: '/face-crop/' + faceRef,
      allowEmpty: true,
      extraSuggestion: function (resolve, box) {
        fetch('/api/faces/' + faceRef + '/suggestions')
          .then(function (r) { return r.json(); })
          .then(function (data) {
            const top = data.suggestions && data.suggestions[0];
            if (!top) return;
            const suggestBtn = document.createElement('button');
            suggestBtn.type = 'button';
            suggestBtn.className = 'btn-similar';
            suggestBtn.style.fontSize = '0.85em';
            suggestBtn.style.marginTop = '8px';
            suggestBtn.textContent = 'Looks like ' + top.name + '? (' + top.score.toFixed(2) + ')';
            suggestBtn.addEventListener('click', function () { resolve({ name: top.name }); });
            box.appendChild(suggestBtn);
          })
          .catch(function () {});
      },
      onResolved: function (entity) { saveName(entity.name); },
    });
  }

  document.querySelectorAll('.face-name-btn').forEach(function (el) {
    el.addEventListener('click', function (e) {
      e.preventDefault();
      openFaceNamingModal(el.dataset.faceRef);
    });
  });

  document.querySelectorAll('.face-reject-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const ref = btn.dataset.faceRef;
      btn.disabled = true;
      fetch('/api/faces/' + ref + '/reject', { method: 'POST' })
        .then(function (r) {
          if (!r.ok) throw new Error('Request failed: ' + r.status);
          return r.json();
        })
        .then(function () {
          const chip = document.querySelector('.face-chip[data-face-ref="' + ref + '"]');
          if (chip) chip.remove();
        })
        .catch(function (err) {
          btn.disabled = false;
          showToast('Failed to reject face: ' + err.message);
        });
    });
  });

  /* Whole-photo identity assignment — "this person is in this photo" with no
     face crop at all (missed detection, face turned away, back of the head,
     etc.). Goes through the same shared entity picker as face naming, minus
     the face-crop preview (there is none) and minus allowEmpty (an empty name
     has no meaning here — face naming's "Save without a name" placeholder
     exists to name a distinct-but-unidentified *face*; there's no face object
     here to attach a placeholder to). Since this path exists specifically for
     the no-face case, every successful assignment is exactly the case where
     face-based similarity search can't find this person elsewhere — so the
     confirmation also surfaces a direct link into body/outfit similarity
     search (/body-similar/{file_id}) as the next step, right where the
     assignment happened. */
  const assignIdentityBtn = document.getElementById('assign-identity-btn');
  const assignIdentityStatus = document.getElementById('assign-identity-status');
  const identityChipsWrap = document.getElementById('identity-photo-chips-wrap');
  const identityChipsContainer = document.getElementById('identity-photo-chips');

  function wireIdentityPhotoChip(chip) {
    const removeBtn = chip.querySelector('.identity-photo-remove-btn');
    if (!removeBtn) return;
    removeBtn.addEventListener('click', function () {
      const name = chip.dataset.identityName;
      removeBtn.disabled = true;
      fetch('/api/files/' + fileId + '/identity-assignments/' + encodeURIComponent(name), { method: 'DELETE' })
        .then(function (r) {
          if (!r.ok) throw new Error('Request failed: ' + r.status);
          return r.json();
        })
        .then(function () {
          chip.remove();
          if (identityChipsContainer && !identityChipsContainer.children.length && identityChipsWrap) {
            identityChipsWrap.style.display = 'none';
          }
        })
        .catch(function (err) {
          removeBtn.disabled = false;
          showToast('Failed to remove assignment: ' + err.message);
        });
    });
  }

  function appendIdentityPhotoChip(name) {
    if (!identityChipsContainer) return;
    if (identityChipsContainer.querySelector('[data-identity-name="' + name + '"]')) return;
    const chip = document.createElement('div');
    chip.className = 'face-chip identity-photo-chip';
    chip.dataset.identityName = name;

    const nameSpan = document.createElement('span');
    nameSpan.className = 'face-name';

    const link = document.createElement('a');
    link.href = '/person/' + encodeURIComponent(name);
    link.textContent = name;
    nameSpan.appendChild(link);

    const bodyLink = document.createElement('a');
    bodyLink.className = 'heart-btn-inline';
    bodyLink.href = '/body-similar/' + fileId;
    bodyLink.title = 'No face for ' + name + ' — search other photos by body instead';
    bodyLink.style.fontSize = '0.85em';
    bodyLink.textContent = '⛶ by body';
    nameSpan.appendChild(bodyLink);

    chip.appendChild(nameSpan);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'rm identity-photo-remove-btn';
    removeBtn.type = 'button';
    removeBtn.title = 'Remove this assignment';
    removeBtn.textContent = '×';
    chip.appendChild(removeBtn);

    identityChipsContainer.appendChild(chip);
    wireIdentityPhotoChip(chip);
    if (identityChipsWrap) identityChipsWrap.style.display = 'block';
  }

  if (identityChipsContainer) {
    identityChipsContainer.querySelectorAll('.identity-photo-chip').forEach(wireIdentityPhotoChip);
  }

  if (assignIdentityBtn) {
    assignIdentityBtn.addEventListener('click', function () {
      openEntitySearchModal({
        type: 'identity',
        title: 'This is… (no face crop)',
        onResolved: function (entity) {
          const name = entity.name;
          if (!name) return;
          assignIdentityStatus.style.display = 'block';
          assignIdentityStatus.textContent = 'Saving…';
          fetch('/api/files/' + fileId + '/identity-assignments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name }),
          })
            .then(function (r) {
              if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Request failed'); });
              return r.json();
            })
            .then(function () {
              appendIdentityPhotoChip(name);
              assignIdentityStatus.textContent =
                'Saved. No face found for ' + name + ' — search other photos by body instead?';
              const cta = document.createElement('a');
              cta.href = '/body-similar/' + fileId;
              cta.className = 'btn-similar';
              cta.style.marginLeft = '6px';
              cta.style.fontSize = '0.9em';
              cta.textContent = '⛶ Search by body';
              assignIdentityStatus.appendChild(document.createElement('br'));
              assignIdentityStatus.appendChild(cta);
            })
            .catch(function (err) {
              assignIdentityStatus.textContent = 'Failed: ' + err.message;
            });
        },
      });
    });
  }

  /* "Find similar faces" (🔎 next to each face chip) opens the matches as a
     browsable photo watch-queue — see the .find-similar-faces-btn handler in
     the photo-page init block above (it hits /api/faces/{ref}/similar, seeds
     setPhotoQueue with the result file_ids, and navigates to the first). */

  const detectFacesBtn = document.getElementById('detect-faces-btn');
  const detectFacesStatus = document.getElementById('detect-faces-status');
  if (detectFacesBtn) {
    detectFacesBtn.addEventListener('click', function () {
      detectFacesBtn.disabled = true;
      detectFacesStatus.style.display = 'block';
      detectFacesStatus.textContent = 'Detecting faces…';
      fetch('/api/files/' + fileId + '/detect-faces', { method: 'POST' })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Request failed'); });
          return r.json();
        })
        .then(function () { location.reload(); })
        .catch(function (err) {
          detectFacesBtn.disabled = false;
          detectFacesStatus.textContent = 'Failed: ' + err.message;
        });
    });
  }

  /* Scan all frames — manual, single-image action for animated GIF/WEBP. The scan
     runs in a background thread server-side (POST just kicks it off), so we poll
     the progress endpoint to show live status rather than blocking on one long
     request — a many-frame file can take a long time. */
  const scanAllFramesBtn = document.getElementById('scan-all-frames-btn');
  const scanAllFramesStatus = document.getElementById('scan-all-frames-status');
  if (scanAllFramesBtn) {
    function pollFrameScanProgress() {
      fetch('/api/files/' + fileId + '/scan-all-frames/progress')
        .then(function (r) { return r.json(); })
        .then(function (job) {
          if (!job.done) {
            scanAllFramesStatus.textContent =
              'Scanning frame ' + job.frames_processed + '/' + job.frame_count + '… ' +
              job.faces_found + ' faces, ' + job.objects_found + ' objects found so far.';
            setTimeout(pollFrameScanProgress, 1000);
            return;
          }
          if (job.error) {
            scanAllFramesBtn.disabled = false;
            scanAllFramesStatus.textContent = 'Failed: ' + job.error;
            return;
          }
          scanAllFramesStatus.textContent =
            'Done: ' + job.frames_processed + '/' + job.frame_count + ' frames, ' +
            job.faces_found + ' faces, ' + job.objects_found + ' objects found.';
          setTimeout(function () { location.reload(); }, 1000);
        })
        .catch(function (err) {
          scanAllFramesBtn.disabled = false;
          scanAllFramesStatus.textContent = 'Failed: ' + err.message;
        });
    }

    scanAllFramesBtn.addEventListener('click', function () {
      scanAllFramesBtn.disabled = true;
      scanAllFramesStatus.style.display = 'block';
      scanAllFramesStatus.textContent = 'Starting scan…';
      fetch('/api/files/' + fileId + '/scan-all-frames', { method: 'POST' })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Request failed'); });
          return r.json();
        })
        .then(function () { pollFrameScanProgress(); })
        .catch(function (err) {
          scanAllFramesBtn.disabled = false;
          scanAllFramesStatus.textContent = 'Failed: ' + err.message;
        });
    });
  }

  /* Category assignment — a photo can belong to any number of categories,
     mirroring how sets already work (see wireSetChip/appendSetChip below). */
  const categoryCurrent = document.getElementById('category-current');
  const categoryPickerBtn = document.getElementById('category-picker-btn');

  function wireCategoryChip(span) {
    const removeBtn = span.querySelector('.category-remove-btn');
    if (removeBtn) {
      removeBtn.addEventListener('click', function () {
        removeFileCategory(span.dataset.categoryId);
      });
    }
  }

  function buildCategoryChip(category) {
    const span = document.createElement('span');
    span.className = 'tag-removable';
    span.style.marginBottom = '4px';
    span.style.display = 'inline-flex';
    span.dataset.categoryId = category.id;

    const link = document.createElement('a');
    link.href = '/search?category=' + encodeURIComponent(category.name);
    link.style.color = '#fff';
    link.textContent = category.name;
    span.appendChild(link);

    if (category.source === 'auto') {
      const autoSpan = document.createElement('span');
      autoSpan.style.opacity = '.7';
      autoSpan.textContent = ' (auto)';
      span.appendChild(autoSpan);
    }

    const removeBtn = document.createElement('button');
    removeBtn.className = 'rm category-remove-btn';
    removeBtn.type = 'button';
    removeBtn.title = 'Remove this category';
    removeBtn.textContent = '×';
    span.appendChild(removeBtn);

    wireCategoryChip(span);
    return span;
  }

  function renderCategories(categories) {
    if (!categoryCurrent) return;
    categoryCurrent.innerHTML = '';
    if (!categories || !categories.length) {
      const span = document.createElement('span');
      span.className = 'sub';
      span.textContent = 'Uncategorized.';
      categoryCurrent.appendChild(span);
      return;
    }
    categories.forEach(function (category) {
      categoryCurrent.appendChild(buildCategoryChip(category));
    });
  }

  function appendCategoryChip(category) {
    // Drop the "Uncategorized." placeholder if it's the only thing there.
    if (categoryCurrent.children.length === 1 && categoryCurrent.firstElementChild.tagName === 'SPAN'
        && !categoryCurrent.firstElementChild.dataset.categoryId) {
      categoryCurrent.innerHTML = '';
    }
    if (categoryCurrent.querySelector('[data-category-id="' + category.id + '"]')) return;
    categoryCurrent.appendChild(buildCategoryChip(category));
  }

  function addFileCategory(categoryId) {
    return fetch('/api/files/' + fileId + '/categories', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category_id: categoryId }),
    }).then(function (r) {
      if (!r.ok) throw new Error('Request failed: ' + r.status);
      return r.json();
    });
  }

  function removeFileCategory(categoryId) {
    fetch('/api/files/' + fileId + '/categories/' + categoryId, { method: 'DELETE' })
      .then(function (r) {
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      })
      .then(function () {
        const chip = categoryCurrent.querySelector('[data-category-id="' + categoryId + '"]');
        if (chip) chip.remove();
        if (!categoryCurrent.children.length) renderCategories([]);
      })
      .catch(function (err) {
        showToast('Failed to remove category: ' + err.message);
      });
  }

  function openCategoryPickerModal() {
    openEntitySearchModal({
      type: 'category',
      onResolved: function (entity) {
        addFileCategory(entity.id)
          .then(function () { appendCategoryChip({ id: entity.id, name: entity.name }); })
          .catch(function (err) { showToast('Failed to add category: ' + err.message); });
      },
    });
  }

  if (categoryPickerBtn) categoryPickerBtn.addEventListener('click', openCategoryPickerModal);
  if (categoryCurrent) categoryCurrent.querySelectorAll('[data-category-id]').forEach(wireCategoryChip);

  /* Location assignment — a photo can be assigned to any number of named
     places, mirroring the category flow above. When the photo carries EXIF GPS
     (window.MEDIA_FILE_GPS) and a just-assigned location has no coordinates of
     its own, offer to seed it with this photo's GPS. */
  const locationCurrent = document.getElementById('location-current');
  const locationPickerBtn = document.getElementById('location-picker-btn');

  function wireLocationChip(span) {
    const removeBtn = span.querySelector('.location-remove-btn');
    if (removeBtn) {
      removeBtn.addEventListener('click', function () {
        removeFileLocation(span.dataset.locationId);
      });
    }
  }

  function buildLocationChip(location) {
    const span = document.createElement('span');
    span.className = 'tag-removable';
    span.style.marginBottom = '4px';
    span.style.display = 'inline-flex';
    span.dataset.locationId = location.id;

    const link = document.createElement('a');
    link.href = '/locations/' + location.id;
    link.style.color = '#fff';
    link.textContent = location.name;
    span.appendChild(link);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'rm location-remove-btn';
    removeBtn.type = 'button';
    removeBtn.title = 'Remove this location';
    removeBtn.textContent = '×';
    span.appendChild(removeBtn);

    wireLocationChip(span);
    return span;
  }

  function renderLocations(locations) {
    if (!locationCurrent) return;
    locationCurrent.innerHTML = '';
    if (!locations || !locations.length) {
      const span = document.createElement('span');
      span.className = 'sub';
      span.textContent = 'No location.';
      locationCurrent.appendChild(span);
      return;
    }
    locations.forEach(function (location) {
      locationCurrent.appendChild(buildLocationChip(location));
    });
  }

  function appendLocationChip(location) {
    // Drop the "No location." placeholder if it's the only thing there.
    if (locationCurrent.children.length === 1 && locationCurrent.firstElementChild.tagName === 'SPAN'
        && !locationCurrent.firstElementChild.dataset.locationId) {
      locationCurrent.innerHTML = '';
    }
    if (locationCurrent.querySelector('[data-location-id="' + location.id + '"]')) return;
    locationCurrent.appendChild(buildLocationChip(location));
  }

  function addFileLocation(locationId) {
    return fetch('/api/files/' + fileId + '/locations', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ location_id: locationId }),
    }).then(function (r) {
      if (!r.ok) throw new Error('Request failed: ' + r.status);
      return r.json();
    });
  }

  function removeFileLocation(locationId) {
    fetch('/api/files/' + fileId + '/locations/' + locationId, { method: 'DELETE' })
      .then(function (r) {
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      })
      .then(function () {
        const chip = locationCurrent.querySelector('[data-location-id="' + locationId + '"]');
        if (chip) chip.remove();
        if (!locationCurrent.children.length) renderLocations([]);
      })
      .catch(function (err) {
        showToast('Failed to remove location: ' + err.message);
      });
  }

  // If this photo has EXIF GPS and the just-assigned location has none, offer to
  // seed the location's coordinates from the photo (PUT /api/locations/{id}).
  function maybeOfferPhotoGps(location) {
    const gps = window.MEDIA_FILE_GPS;
    if (!gps || location.gps_lat != null) return;
    if (!confirm('Use this photo\'s GPS (' + gps.lat.toFixed(5) + ', ' + gps.lon.toFixed(5) +
                 ') as the coordinates for "' + location.name + '"?')) return;
    fetch('/api/locations/' + location.id, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gps_lat: gps.lat, gps_lon: gps.lon }),
    })
      .then(function (r) { if (!r.ok) throw new Error('Request failed: ' + r.status); return r.json(); })
      .then(function () { showToast('Saved GPS for "' + location.name + '".'); })
      .catch(function (err) { showToast('Failed to save GPS: ' + err.message); });
  }

  function openLocationPickerModal() {
    openEntitySearchModal({
      type: 'location',
      onResolved: function (entity) {
        addFileLocation(entity.id)
          .then(function () {
            appendLocationChip({ id: entity.id, name: entity.name });
            maybeOfferPhotoGps(entity);
          })
          .catch(function (err) { showToast('Failed to add location: ' + err.message); });
      },
    });
  }

  if (locationPickerBtn) locationPickerBtn.addEventListener('click', openLocationPickerModal);
  if (locationCurrent) locationCurrent.querySelectorAll('[data-location-id]').forEach(wireLocationChip);
  window.openLocationPickerModal = openLocationPickerModal;

  /* Set assignment — a photo can belong to any number of sets */
  const setCurrent = document.getElementById('set-current');
  const setPickerBtn = document.getElementById('set-picker-btn');

  // A set's studio + linked-face "@name" tags (each an {name, age, gender}
  // object — age/gender only when that identity has an estimate), always
  // rendered together on their own line directly under the set's name
  // (mirrors _macros.html's set_meta_line). Returns null when the set has
  // neither, so callers can skip appending an empty line.
  function buildSetMetaLine(set) {
    if (!set.studio && !(set.people && set.people.length)) return null;
    const meta = document.createElement('span');
    meta.className = 'set-meta-line';
    if (set.studio) {
      const studioSpan = document.createElement('span');
      studioSpan.className = 'set-studio';
      studioSpan.textContent = set.studio;
      meta.appendChild(studioSpan);
    }
    (set.people || []).forEach(function (p) {
      const tag = document.createElement('span');
      tag.className = 'set-people-tag';
      tag.textContent = '@' + p.name;
      if (p.age != null) {
        const ageSpan = document.createElement('span');
        ageSpan.className = 'set-person-age' + (p.gender === 'male' ? ' gender-male' : p.gender === 'female' ? ' gender-female' : '');
        ageSpan.textContent = Math.round(p.age);
        tag.appendChild(ageSpan);
      }
      meta.appendChild(tag);
    });
    return meta;
  }

  function wireSetChip(span) {
    const heartBtn = span.querySelector('.set-heart-btn');
    if (heartBtn) wireHeartButton(heartBtn, '/api/sets/' + span.dataset.setId + '/favorite');
    const removeBtn = span.querySelector('.set-remove-btn');
    if (removeBtn) {
      removeBtn.addEventListener('click', function () {
        removeSet(span.dataset.setId);
      });
    }
  }

  function renderSets(sets) {
    setCurrent.innerHTML = '';
    if (!sets || !sets.length) {
      const span = document.createElement('span');
      span.className = 'sub';
      span.textContent = 'Not in any set.';
      setCurrent.appendChild(span);
      return;
    }
    sets.forEach(function (set) {
      const span = document.createElement('span');
      span.className = 'tag-removable';
      span.style.marginBottom = '4px';
      span.style.display = 'inline-flex';
      span.style.flexDirection = 'column';
      span.style.alignItems = 'flex-start';
      span.style.gap = '2px';
      span.dataset.setId = set.id;

      const row = document.createElement('span');
      row.style.display = 'flex';
      row.style.alignItems = 'center';
      row.style.gap = '5px';

      const link = document.createElement('a');
      link.href = '/sets/' + set.id;
      link.style.color = '#fff';
      link.textContent = set.name;
      row.appendChild(link);

      const heartBtn = document.createElement('button');
      heartBtn.type = 'button';
      heartBtn.className = 'heart-btn-inline set-heart-btn';
      heartBtn.style.fontSize = '0.85em';
      heartBtn.title = 'Favorite this set (right-click to lower)';
      window.renderHeart(heartBtn, set.favorite);
      row.appendChild(heartBtn);

      const removeBtn = document.createElement('button');
      removeBtn.className = 'rm set-remove-btn';
      removeBtn.type = 'button';
      removeBtn.title = 'Remove from this set';
      removeBtn.textContent = '×';
      row.appendChild(removeBtn);

      span.appendChild(row);
      const metaLine = buildSetMetaLine(set);
      if (metaLine) span.appendChild(metaLine);

      setCurrent.appendChild(span);
      wireSetChip(span);
    });
  }

  function assignSetById(setId) {
    return window.assignFileToSet(fileId, setId);
  }

  function appendSetChip(set) {
    // Drop the "Not in any set." placeholder if it's the only thing there.
    if (setCurrent.children.length === 1 && setCurrent.firstElementChild.tagName === 'SPAN'
        && !setCurrent.firstElementChild.dataset.setId) {
      setCurrent.innerHTML = '';
    }
    if (setCurrent.querySelector('[data-set-id="' + set.id + '"]')) return;
    const span = document.createElement('span');
    span.className = 'tag-removable';
    span.style.marginBottom = '4px';
    span.style.display = 'inline-flex';
    span.style.flexDirection = 'column';
    span.style.alignItems = 'flex-start';
    span.style.gap = '2px';
    span.dataset.setId = set.id;

    const row = document.createElement('span');
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.gap = '5px';

    const link = document.createElement('a');
    link.href = '/sets/' + set.id;
    link.style.color = '#fff';
    link.textContent = set.name;
    row.appendChild(link);

    const heartBtn = document.createElement('button');
    heartBtn.type = 'button';
    heartBtn.className = 'heart-btn-inline set-heart-btn';
    heartBtn.style.fontSize = '0.85em';
    heartBtn.title = 'Favorite this set (right-click to lower)';
    window.renderHeart(heartBtn, 0);
    row.appendChild(heartBtn);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'rm set-remove-btn';
    removeBtn.type = 'button';
    removeBtn.title = 'Remove from this set';
    removeBtn.textContent = '×';
    row.appendChild(removeBtn);

    span.appendChild(row);
    const metaLine = buildSetMetaLine(set);
    if (metaLine) span.appendChild(metaLine);

    setCurrent.appendChild(span);
    wireSetChip(span);
  }

  function removeSet(setId) {
    fetch('/api/files/' + fileId + '/sets/' + setId, { method: 'DELETE' })
      .then(function (r) {
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      })
      .then(function () {
        const chip = setCurrent.querySelector('[data-set-id="' + setId + '"]');
        if (chip) chip.remove();
        if (!setCurrent.children.length) renderSets([]);
      })
      .catch(function (err) {
        showToast('Failed to remove set: ' + err.message);
      });
  }

  function openSetPickerModal() {
    // Opened from the photo page → surface CLIP-ranked suggested sets inside the
    // picker (suggestForFileId). See openEntitySearchModal's set-suggestion block.
    openSetSearchModal(function (set) {
      assignSetById(set.id)
        .then(function (data) { appendSetChip(data); })
        .catch(function (err) { showToast('Failed to add set: ' + err.message); });
    }, undefined, { suggestForFileId: fileId });
  }

  if (setPickerBtn) {
    setPickerBtn.addEventListener('click', openSetPickerModal);
  }

  // When more than one photo is in the watch queue, split "Add to set" into
  // "Add this to set" (this photo) + "Add queue to set" (pick a set, then
  // curate which queue items to add in the space-grid overlay).
  const setPickerQueueBtn = document.getElementById('set-picker-queue-btn');
  const photoQueue = window.__photoQueue;
  function addQueueToSet() {
    openSetSearchModal(function (set) { window.openQueueSelectGrid(set); });
  }
  if (setPickerQueueBtn && photoQueue && photoQueue.ids.length > 1) {
    if (setPickerBtn) setPickerBtn.textContent = '＋ Add this to set';
    setPickerQueueBtn.style.display = '';
    setPickerQueueBtn.addEventListener('click', addQueueToSet);
  }

  // The Set pod trigger. When there are no sets yet it's a direct "＋ Add set"
  // action (data-pod-action, handled by the pod controller). If a queue is
  // present too, upgrade it here (queue is only known at runtime) into a split
  // pill: left half adds THIS photo, right half adds the WHOLE queue.
  const setPodTrigger = document.getElementById('set-pod-trigger');
  if (setPodTrigger && setPodTrigger.dataset.podAction === 'add-set'
      && photoQueue && photoQueue.ids && photoQueue.ids.length) {
    setPodTrigger.dataset.podAction = '';   // segments handle their own clicks now
    setPodTrigger.classList.add('stage-pod-split');
    setPodTrigger.innerHTML = '';
    const photoSeg = document.createElement('span');
    photoSeg.className = 'split-seg';
    photoSeg.textContent = '＋ photo';
    photoSeg.title = 'Add this photo to a set';
    const divider = document.createElement('span');
    divider.className = 'split-div';
    const queueSeg = document.createElement('span');
    queueSeg.className = 'split-seg';
    queueSeg.textContent = '＋ queue';
    queueSeg.title = 'Add the whole queue to a set';
    setPodTrigger.appendChild(photoSeg);
    setPodTrigger.appendChild(divider);
    setPodTrigger.appendChild(queueSeg);
    // stopPropagation so the click never bubbles to the pod controller (which
    // would otherwise toggle the menu / pin the pod open).
    photoSeg.addEventListener('click', function (e) { e.stopPropagation(); openSetPickerModal(); });
    queueSeg.addEventListener('click', function (e) { e.stopPropagation(); addQueueToSet(); });
  }

  if (setCurrent) {
    setCurrent.querySelectorAll('[data-set-id]').forEach(wireSetChip);
  }

  // Suggested sets are now surfaced INSIDE the set picker modal (see
  // openEntitySearchModal's suggestForFileId block) instead of a separate
  // "✨ Suggest sets" button + inline list on the page.

  /* Photo (file) favorite heart */
  const fileHeartBtn = document.getElementById('file-heart-btn');
  if (fileHeartBtn) wireHeartButton(fileHeartBtn, '/api/files/' + fileId + '/favorite');

  /* Photo title — click the pencil to edit in place, matching the tag-label-edit
     interaction (Enter/blur saves, Escape cancels). An empty title clears it back
     to showing the filename. */
  const titleEditBtn = document.getElementById('photo-title-edit-btn');
  const titleDisplay = document.getElementById('photo-title-display');
  if (titleEditBtn && titleDisplay) {
    titleEditBtn.addEventListener('click', function () {
      const currentText = titleDisplay.textContent.trim();
      const input = document.createElement('input');
      input.type = 'text';
      input.value = currentText;
      input.placeholder = 'Title…';
      input.style.fontSize = '1rem';
      input.style.width = '320px';
      input.style.maxWidth = '60vw';
      let settled = false;

      function restoreDisplay() {
        if (input.isConnected) input.replaceWith(titleDisplay);
      }
      function commit() {
        if (settled) return;
        settled = true;
        const newTitle = input.value.trim();
        fetch('/api/files/' + fileId + '/title', {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: newTitle }),
        })
          .then(function (r) {
            if (!r.ok) throw new Error('Request failed: ' + r.status);
            return r.json();
          })
          .then(function () {
            location.reload();
          })
          .catch(function (err) {
            restoreDisplay();
            showToast('Failed to save title: ' + err.message);
          });
      }
      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); commit(); }
        else if (e.key === 'Escape') { settled = true; restoreDisplay(); }
      });
      input.addEventListener('blur', commit);
      titleDisplay.replaceWith(input);
      input.focus();
      input.select();
    });
  }

  /* Named-face favorite hearts */
  document.querySelectorAll('.face-heart-btn').forEach(function (btn) {
    wireHeartButton(btn, '/api/faces/' + btn.dataset.faceRef + '/favorite');
  });

  /* Age & gender estimation (experimental, MiVOLO via isolated venv — see
     age_estimator.py). Self-contained block: delete this along with the button in
     photo.html and the one web.py endpoint to remove the feature entirely. Results
     are shown inline next to each face's name (".face-age-gender"), not as a
     separate list. */
  const estimateAgeBtn = document.getElementById('estimate-age-btn');
  const estimateAgeStatus = document.getElementById('estimate-age-status');
  if (estimateAgeBtn) {
    estimateAgeBtn.addEventListener('click', function () {
      estimateAgeBtn.disabled = true;
      estimateAgeStatus.style.display = 'block';
      estimateAgeStatus.textContent = 'Estimating… (first run downloads models, can take a while)';
      fetch('/api/files/' + fileId + '/estimate-age', { method: 'POST' })
        .then(function (r) {
          if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Request failed'); });
          return r.json();
        })
        .then(function (data) {
          estimateAgeBtn.disabled = false;
          if (data.message) {
            estimateAgeStatus.textContent = data.message;
            return;
          }
          estimateAgeStatus.style.display = 'none';
          data.results.forEach(function (est) {
            const el = document.querySelector('.face-age-gender[data-face-ref="' + est.face_ref + '"]');
            if (!el) return;
            el.textContent = est.age !== null && est.age !== undefined ? Math.round(est.age) : '';
            el.classList.remove('gender-male', 'gender-female');
            if (est.gender === 'male') el.classList.add('gender-male');
            else if (est.gender === 'female') el.classList.add('gender-female');
          });
        })
        .catch(function (err) {
          estimateAgeBtn.disabled = false;
          estimateAgeStatus.style.display = 'block';
          estimateAgeStatus.textContent = 'Failed: ' + err.message;
        });
    });
  }

  // ---- Find by body: body-index build banner (body_similar.html) ----
  const bodyIndexBanner = document.getElementById('body-index-banner');
  if (bodyIndexBanner) {
    const bodyIndexText = document.getElementById('body-index-text');
    const bodyIndexBtn = document.getElementById('body-index-build-btn');
    let bodyIndexWasRunning = false;

    function refreshBodyIndexBanner() {
      fetch('/api/body-index/status')
        .then(function (r) { return r.json(); })
        .then(function (s) {
          if (s.running) {
            bodyIndexWasRunning = true;
            bodyIndexBanner.style.display = '';
            bodyIndexBtn.style.display = 'none';
            bodyIndexText.textContent = 'Building body index\u2026 ' + s.done + ' / ' + s.total;
            setTimeout(refreshBodyIndexBanner, 1500);
          } else if (bodyIndexWasRunning) {
            // build finished while we were watching — reload to pick up new results
            location.reload();
          } else if (s.error) {
            bodyIndexBanner.style.display = '';
            bodyIndexText.textContent = 'Body index build failed: ' + s.error;
          } else if (s.pending > 0) {
            bodyIndexBanner.style.display = '';
            bodyIndexBtn.style.display = '';
            bodyIndexText.textContent = s.pending + ' photo(s) are not in the body index yet.';
          } else {
            bodyIndexBanner.style.display = 'none';
          }
        });
    }

    bodyIndexBtn.addEventListener('click', function () {
      bodyIndexBtn.disabled = true;
      fetch('/api/body-index/start', { method: 'POST' })
        .then(function () {
          bodyIndexWasRunning = true;
          bodyIndexBtn.disabled = false;
          refreshBodyIndexBanner();
        });
    });

    refreshBodyIndexBanner();
  }

  // ---- Photo stage: fit-mode cycling + overlay sidebar (photo.html) ----
  const photoStage = document.getElementById('photo-stage');
  if (photoStage) {
    const FIT_MODES = [
      { key: 'fit',     cls: 'fit-contain', label: 'FIT' },
      { key: 'fill',    cls: 'fit-cover',   label: 'FILL' },
      { key: 'stretch', cls: 'fit-stretch', label: 'STRETCH' },
      { key: 'pixel',   cls: 'fit-pixel',   label: '1:1' },
    ];
    const fitBtn = document.getElementById('fit-mode-btn');
    const fitLabel = document.getElementById('fit-mode-label');

    /* Mouse-wheel zoom toward the cursor + click-drag pan (images only; the
       video element is never touched — `photoImg` is null for videos). State is a
       single scale + translate applied as `translate(tx,ty) scale(s)` on
       #photo-image, whose transform-origin is 0 0 (see style.css). With that
       origin, a local point l maps to screen  L0 + t + s*l, so the point under
       the cursor stays fixed across a zoom step when we solve for the new t:
         t' = t + (cursor - rect.left) * (1 - s'/s)
       (rect.left is the image's current on-screen left, i.e. L0 + t = s*l + L0,
       so cursor-rect.left = s*l and the identity falls out). Reused across the
       fit-mode handlers below so zoom and fit modes never fight. */
    let zScale = 1, zTx = 0, zTy = 0;
    function applyZoom() {
      if (!photoImg) return;
      photoImg.style.transform = 'translate(' + zTx + 'px,' + zTy + 'px) scale(' + zScale + ')';
      photoImg.style.cursor = zScale > 1 ? 'grab' : '';
    }
    function resetZoom() {
      zScale = 1; zTx = 0; zTy = 0;
      if (photoImg) { photoImg.style.transform = ''; photoImg.style.cursor = ''; }
    }
    // Best-effort pan clamp: when the image is larger than the stage on an axis,
    // don't let a gap open at its edges; when smaller, keep it inside the stage.
    // Works off live bounding rects, so it's fit-mode- and scroll-agnostic.
    function clampPan() {
      if (!photoImg || zScale <= 1) return;
      const s = photoStage.getBoundingClientRect();
      const i = photoImg.getBoundingClientRect();
      let cx = 0, cy = 0;
      if (i.width >= s.width) {
        if (i.left > s.left) cx = s.left - i.left;
        else if (i.right < s.right) cx = s.right - i.right;
      } else {
        if (i.left < s.left) cx = s.left - i.left;
        else if (i.right > s.right) cx = s.right - i.right;
      }
      if (i.height >= s.height) {
        if (i.top > s.top) cy = s.top - i.top;
        else if (i.bottom < s.bottom) cy = s.bottom - i.bottom;
      } else {
        if (i.top < s.top) cy = s.top - i.top;
        else if (i.bottom > s.bottom) cy = s.bottom - i.bottom;
      }
      if (cx || cy) { zTx += cx; zTy += cy; applyZoom(); }
    }
    if (photoImg) {
      const ZOOM_STEP = 1.15, ZOOM_MIN = 1, ZOOM_MAX = 8;
      photoStage.addEventListener('wheel', function (e) {
        // Always swallow the wheel so the page/stage never scrolls under us.
        e.preventDefault();
        const factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
        const newScale = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zScale * factor));
        if (newScale === zScale) return;
        const r = photoImg.getBoundingClientRect();
        const k = 1 - newScale / zScale;
        zTx += (e.clientX - r.left) * k;
        zTy += (e.clientY - r.top) * k;
        zScale = newScale;
        if (zScale <= 1) { zScale = 1; zTx = 0; zTy = 0; }
        applyZoom();
        clampPan();
      }, { passive: false });

      // Drag-to-pan, active only while zoomed in. mousemove/up live on window so
      // the drag survives the cursor leaving the image.
      let dragging = false, lastX = 0, lastY = 0;
      photoImg.addEventListener('mousedown', function (e) {
        if (zScale <= 1) return;
        dragging = true;
        lastX = e.clientX; lastY = e.clientY;
        photoImg.style.cursor = 'grabbing';
        e.preventDefault();
      });
      window.addEventListener('mousemove', function (e) {
        if (!dragging) return;
        zTx += e.clientX - lastX;
        zTy += e.clientY - lastY;
        lastX = e.clientX; lastY = e.clientY;
        applyZoom();
        clampPan();
      });
      function endDrag() {
        if (!dragging) return;
        dragging = false;
        if (photoImg) photoImg.style.cursor = zScale > 1 ? 'grab' : '';
      }
      window.addEventListener('mouseup', endDrag);
      photoStage.addEventListener('mouseleave', endDrag);
    }

    function setFit(key) {
      resetZoom(); // zoom and fit modes must not fight — every mode change clears zoom
      const mode = FIT_MODES.find(function (m) { return m.key === key; }) || FIT_MODES[0];
      FIT_MODES.forEach(function (m) { photoStage.classList.remove(m.cls); });
      photoStage.classList.add(mode.cls);
      if (fitLabel) fitLabel.textContent = mode.label;
      try { localStorage.setItem('mm_fit', mode.key); } catch (e) {}
    }
    function cycleFit() {
      const cur = FIT_MODES.findIndex(function (m) { return photoStage.classList.contains(m.cls); });
      setFit(FIT_MODES[(cur + 1) % FIT_MODES.length].key);
    }
    try { setFit(localStorage.getItem('mm_fit') || 'fit'); } catch (e) { setFit('fit'); }
    if (fitBtn) fitBtn.addEventListener('click', cycleFit);

    // Drag-to-draw (add face / label region) maps clicks through the img element
    // box, which only equals the visible image outside FILL mode — so entering a
    // draw tool snaps back to FIT.
    ['add-face-btn', 'label-region-btn', 'label-region-negative-btn', 'label-person-btn'].forEach(function (id) {
      const btn = document.getElementById(id);
      if (btn) btn.addEventListener('click', function () {
        if (photoStage.classList.contains('fit-cover')) setFit('fit');
      });
    });

    // ---- Corner pods: hover opens (with a small close delay to cross the
    // trigger→menu gap), click pins; outside-click / Esc closes. One pin max. ----
    const pods = Array.prototype.slice.call(photoStage.querySelectorAll('[data-pod]'));
    function closeAllPods(exceptEl) {
      pods.forEach(function (p) {
        if (p !== exceptEl) { p.classList.remove('is-open'); p.__pinned = false; }
      });
    }
    pods.forEach(function (pod) {
      const trigger = pod.querySelector('.stage-pod-btn');
      // Categories, and an EMPTY Sets pod, are click-to-open only — their
      // collapsed trigger doubles as a label/direct-add button, so a hover
      // popover would fight the pointer. A POPULATED Sets pod (.has-sets) is the
      // exception: it opens on HOVER, swapping its trigger name for the in-place
      // set link (same screen spot), so hovering replaces the button with the
      // clickable set. Faces and the rest (info/frames/tags) keep hover-open.
      const isSetWithSets = pod.dataset.podKind === 'set' && pod.classList.contains('has-sets');
      const clickOnly = (pod.dataset.podKind === 'set' || pod.dataset.podKind === 'category'
                         || pod.dataset.podKind === 'location')
                        && !isSetWithSets;
      let hideTimer = null;
      function openPod() { if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; } pod.classList.add('is-open'); }
      function scheduleClose() {
        if (pod.__pinned) return;
        if (hideTimer) clearTimeout(hideTimer);
        hideTimer = setTimeout(function () { pod.classList.remove('is-open'); }, 160);
      }
      if (!clickOnly) {
        pod.addEventListener('mouseenter', openPod);
        pod.addEventListener('mouseleave', scheduleClose);
      }
      if (trigger) {
        trigger.addEventListener('click', function (e) {
          e.stopPropagation();
          // An empty Sets/Categories trigger is a direct "add" shortcut: it opens
          // the picker modal straight away instead of toggling the (near-empty) menu.
          const action = trigger.dataset.podAction;
          if (action === 'add-category') { closeAllPods(null); openCategoryPickerModal(); return; }
          if (action === 'add-location') { closeAllPods(null); openLocationPickerModal(); return; }
          if (action === 'add-set') { closeAllPods(null); openSetPickerModal(); return; }
          pod.__pinned = !pod.__pinned;
          if (pod.__pinned) { closeAllPods(pod); openPod(); }
          else pod.classList.remove('is-open');
        });
      }
      // Any interaction inside the menu (add/remove a tag, pick a category, etc.)
      // pins the pod open so it doesn't vanish when the mouse wanders off or a
      // picker modal opens mid-action.
      pod.addEventListener('click', function (e) {
        if (trigger && trigger.contains(e.target)) return; // the trigger toggles (above)
        pod.__pinned = true;
        openPod();
      });
    });
    document.addEventListener('click', function (e) {
      // Only a click on the page itself closes pods — not a click inside a pod, nor
      // inside a modal/palette a pod opened (e.g. the add-tag / add-to-set picker).
      if (e.target.closest('[data-pod], .modal-overlay, .palette-overlay')) return;
      closeAllPods(null);
    });

    // Quick "＋ Add to set" in the top File-info pod reuses the set picker.
    const fileinfoAddSet = document.getElementById('fileinfo-add-set-btn');
    if (fileinfoAddSet) fileinfoAddSet.addEventListener('click', openSetPickerModal);

    // Frame stepper for animated images (GIF/WEBP). The browser can't read the
    // currently-shown animation frame, so we scrub server-rendered frames: each step
    // swaps #photo-image's src to /api/files/{id}/frame/{n} (a static, non-playing
    // frame). `selectedFrameIndex` stays null while the GIF is still autoplaying
    // (nothing picked yet); once the user steps, it holds the shown frame index so
    // "📸 Capture" can grab that exact frame by index.
    const frameCount = Number(window.MEDIA_FRAME_COUNT) || 1;
    let selectedFrameIndex = null;
    (function wireFrameStepper() {
      const stepper = document.getElementById('frame-stepper');
      const img = document.getElementById('photo-image');
      if (!stepper || !img || frameCount <= 1) return;
      const slider = document.getElementById('frame-slider');
      const counter = document.getElementById('frame-counter');
      const prevBtn = document.getElementById('frame-prev-btn');
      const nextBtn = document.getElementById('frame-next-btn');

      function showFrame(n) {
        const idx = Math.max(0, Math.min(n, frameCount - 1));
        selectedFrameIndex = idx;
        // Swapping src to a single-frame JPEG stops the animation and pins this frame.
        img.src = '/api/files/' + fileId + '/frame/' + idx;
        if (slider) slider.value = String(idx);
        if (counter) counter.textContent = (idx + 1) + ' / ' + frameCount;
      }

      if (prevBtn) prevBtn.addEventListener('click', function () {
        showFrame((selectedFrameIndex === null ? 0 : selectedFrameIndex) - 1);
      });
      if (nextBtn) nextBtn.addEventListener('click', function () {
        showFrame((selectedFrameIndex === null ? -1 : selectedFrameIndex) + 1);
      });
      if (slider) slider.addEventListener('input', function () {
        showFrame(parseInt(slider.value, 10) || 0);
      });

      // Keyboard: [ / , = prev, ] / . = next (ignored while typing / grid open).
      window.addEventListener('keydown', function (e) {
        if (photoGridOpen) return;
        const tag = (e.target && e.target.tagName) || '';
        if (tag === 'INPUT' || tag === 'TEXTAREA') return;
        if (e.key === '[' || e.key === ',') { e.preventDefault(); prevBtn && prevBtn.click(); }
        else if (e.key === ']' || e.key === '.') { e.preventDefault(); nextBtn && nextBtn.click(); }
      });
    })();

    // Capture the current frame → a hidden still image, then open it.
    //  • <video>: draw the currently-shown frame (currentTime) to a canvas + upload.
    //  • animated <img>: no canvas — ask the server to extract the frame the stepper
    //    is showing (selectedFrameIndex, or 0 if still autoplaying) by index.
    const captureBtn = document.getElementById('capture-frame-btn');
    const captureSrc = document.getElementById('photo-video') || document.getElementById('photo-image');
    if (captureBtn && captureSrc) {
      const isAnimatedImage = captureSrc.tagName === 'IMG' && frameCount > 1;
      captureBtn.addEventListener('click', function () {
        const status = document.getElementById('capture-frame-status');
        function say(t) { if (status) { status.style.display = ''; status.textContent = t; } }
        function onCaptured(r) {
          if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || ('Request failed: ' + r.status)); });
          return r.json();
        }
        function onError(err) { captureBtn.disabled = false; say(''); showToast('Capture failed: ' + err.message); }

        if (isAnimatedImage) {
          const idx = selectedFrameIndex === null ? 0 : selectedFrameIndex;
          captureBtn.disabled = true;
          say('Capturing frame ' + (idx + 1) + '…');
          fetch('/api/files/' + fileId + '/capture-frame-index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_index: idx }),
          })
            .then(onCaptured)
            .then(function (data) { window.location.href = '/photo/' + data.file_id; })
            .catch(onError);
          return;
        }

        const isVideo = captureSrc.tagName === 'VIDEO';
        const w = isVideo ? captureSrc.videoWidth : captureSrc.naturalWidth;
        const h = isVideo ? captureSrc.videoHeight : captureSrc.naturalHeight;
        if (!w || !h) { say(isVideo ? 'Video not ready yet — press play/seek first.' : 'Image not ready yet.'); return; }
        const canvas = document.createElement('canvas');
        canvas.width = w; canvas.height = h;
        try {
          canvas.getContext('2d').drawImage(captureSrc, 0, 0, w, h);
        } catch (e) { say('Could not read this frame.'); return; }
        captureBtn.disabled = true;
        say('Capturing…');
        canvas.toBlob(function (blob) {
          if (!blob) { captureBtn.disabled = false; say('Capture failed.'); return; }
          const fd = new FormData();
          fd.append('frame', blob, 'frame.jpg');
          fd.append('time_ms', String(isVideo ? Math.round((captureSrc.currentTime || 0) * 1000) : 0));
          fetch('/api/files/' + fileId + '/capture-frame', { method: 'POST', body: fd })
            .then(onCaptured)
            .then(function (data) { window.location.href = '/photo/' + data.file_id; })
            .catch(onError);
        }, 'image/jpeg', 0.92);
      });
    }

    window.addEventListener('keydown', function (e) {
      if (photoGridOpen) return;
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'f' || e.key === 'F') cycleFit();
      if (e.key === 'Escape') closeAllPods(null);
    });
  }

  /* Manual set ordering (set_detail.html, sort=manual): per-card ▲/▼ buttons and
     a position number input reorder the grid in place, then persist the full new
     order (each card's data-file-id, in DOM order) to /api/sets/{id}/reorder. The
     server writes dense positions, so the whole order — not a two-row swap — is
     sent on every change; that keeps up/down and the number input on one path. */
  (function wireManualReorder() {
    var grid = document.getElementById('set-grid');
    if (!grid || !grid.dataset.manual) return;
    var setId = grid.dataset.setId;

    function cards() {
      return Array.prototype.slice.call(grid.querySelectorAll('.card'));
    }

    function renumber() {
      cards().forEach(function (card, i) {
        var input = card.querySelector('.pos-input');
        if (input) input.value = i + 1;
      });
    }

    var saveTimer = null;
    function persist() {
      if (saveTimer) clearTimeout(saveTimer);
      saveTimer = setTimeout(function () {
        var ids = cards().map(function (card) {
          return parseInt(card.dataset.fileId, 10);
        }).filter(function (n) { return !isNaN(n); });
        fetch('/api/sets/' + setId + '/reorder', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_ids: ids }),
        }).catch(function () { /* best-effort; order is re-derivable from the DOM */ });
      }, 300);
    }

    function moveTo(card, index) {
      var list = cards();
      var clamped = Math.max(0, Math.min(index, list.length - 1));
      var ref = list[clamped];
      if (ref === card) return;
      // Insert before ref when moving up, after ref when moving down.
      if (list.indexOf(card) < clamped) {
        grid.insertBefore(card, ref.nextSibling);
      } else {
        grid.insertBefore(card, ref);
      }
      renumber();
      persist();
    }

    grid.addEventListener('click', function (e) {
      var btn = e.target.closest && e.target.closest('.reorder-up, .reorder-down');
      if (!btn) return;
      var card = btn.closest('.card');
      var idx = cards().indexOf(card);
      moveTo(card, btn.classList.contains('reorder-up') ? idx - 1 : idx + 1);
    });

    grid.addEventListener('change', function (e) {
      if (!e.target.classList.contains('pos-input')) return;
      var card = e.target.closest('.card');
      var pos = parseInt(e.target.value, 10);
      if (isNaN(pos)) { renumber(); return; }
      moveTo(card, pos - 1); // 1-based input -> 0-based index
    });

    renumber();
  })();

})();
