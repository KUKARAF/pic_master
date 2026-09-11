// A/B identity review for /find_all_faces: one unknown face on the left, the
// currently-selected candidate person on the right, arrow keys to cycle the
// ranked identities, Enter to confirm.
//
// Deliberately NOT built on swipe-core.js: that engine is a binary yes/no card
// stack (one question per card, up/down = the whole decision space), and this
// screen's core interaction is "pick one of N identities for this face" —
// left/right already mean "previous/next candidate" here, which is exactly the
// axis swipe-core spends on skip/full-view. Bending it would have cost both
// features their meaning on every other page that uses it, so this is its own
// self-contained module with no dependency on (and no effect on) swipe-core.
//
// The face's top-K candidate list comes precomputed from face_candidates (the
// single scoring worker's materialized output), so cycling identities is free —
// no server round-trip per arrow key. Only R (rotate) has to re-rank, because a
// rotated face is a genuinely different embedding.

(function () {
  // Kept module-private rather than exported: the page owns exactly one review
  // instance, and a second one sharing this overlay could only ever fight over it.
  var _fullviewOverlay = null;
  var _fullviewOpen = false;

  // Reuses swipe-core's .swipe-fullview-* styling (already in style.css) rather
  // than a second near-identical overlay skin — the visual is the same "show me
  // the whole photo" affordance, only the trigger key differs.
  function showFullview(src) {
    if (!_fullviewOverlay) {
      var overlay = document.createElement('div');
      overlay.className = 'swipe-fullview-overlay';
      var frame = document.createElement('div');
      frame.className = 'swipe-fullview-frame';
      frame.appendChild(document.createElement('img'));
      overlay.appendChild(frame);
      overlay.addEventListener('click', hideFullview);
      document.body.appendChild(overlay);
      _fullviewOverlay = overlay;
    }
    _fullviewOverlay.querySelector('img').src = src;
    _fullviewOverlay.classList.add('open');
    _fullviewOpen = true;
  }

  function hideFullview() {
    if (_fullviewOverlay) _fullviewOverlay.classList.remove('open');
    _fullviewOpen = false;
  }

  // Same guard swipe-core.js uses, plus the modal: the identity picker (N) is a
  // combobox that owns every arrow key while it's open, and base.html's single
  // #modal-overlay is the one and only thing that can be on top of this page.
  function isBlockedTarget(target) {
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return true;
    var modal = document.getElementById('modal-overlay');
    return !!modal && modal.style.display !== 'none' && modal.style.display !== '';
  }

  window.initFaceReview = function (config) {
    var root = document.getElementById(config.rootId);
    if (!root) return;

    // Buffer target: advancing must be instant, and every refill is a
    // /api/face-review/next round-trip that also builds each card's candidate
    // list — so keep a few ahead and top up in the background, never on demand.
    var BUFFER_SIZE = 5;
    var REFILL_AT = 2;

    var reviewEl = root.querySelector('.review-ab');
    var emptyEl = root.querySelector('.review-empty-slot');
    var controlsEl = root.querySelector('.review-controls');
    var statusEl = root.querySelector('.review-status');
    var hintEl = root.querySelector('.review-hint');
    var unknownImg = root.querySelector('.review-unknown-img');
    var unknownMeta = root.querySelector('.review-unknown-meta');
    var unknownLink = root.querySelector('.review-unknown-link');
    var candImg = root.querySelector('.review-candidate-img');
    var candName = root.querySelector('.review-candidate-name');
    var candScore = root.querySelector('.review-candidate-score');
    var candList = root.querySelector('.review-candidate-list');

    var queue = [];            // cards not yet decided, [0] is on screen
    var seen = {};             // every ref ever buffered — the exclude list, see below
    var history = [];          // [{card, action, rotate}] — most recent last
    var fetching = false;
    var remaining = null;      // server's count of unmatched faces still in the pool
    var sel = 0;               // index into the current card's candidate list
    var rotate = 0;            // absolute quarter-turns CW for the current card

    function card() { return queue[0] || null; }

    function candidates() {
      var c = card();
      return (c && c.candidates) || [];
    }

    // Sent in full on every refill (like swipe-core's `known`): the pool is
    // re-ranked server-side from scratch each time, so an already-seen face left
    // out of the exclude list just comes straight back at the top and the buffer
    // looks permanently stuck. Skipped faces are in here too — a skip means
    // "not now", and it must not resurface for the rest of the session.
    function excludeList() { return Object.keys(seen); }

    function setStatus(text) { statusEl.textContent = text; }

    function renderStatus() {
      if (remaining == null) { setStatus(''); return; }
      var n = Math.max(remaining, 0);
      setStatus(n.toLocaleString() + (n === 1 ? ' face' : ' faces') + ' still to review');
    }

    function rotationDelta() {
      // /face-crop already renders the face upright per its stored angle and the
      // card's own `rotation`, so the CSS turn is the delta from what's served —
      // same reasoning as openFaceNamingModal's preview in app.js.
      var c = card();
      var base = c ? (c.rotation || 0) : 0;
      return ((rotate - base) % 4 + 4) % 4;
    }

    function renderCandidates() {
      var list = candidates();
      candList.innerHTML = '';
      if (!list.length) {
        candName.textContent = 'No candidate identities';
        candScore.textContent = 'nobody named scores above the match floor for this face';
        candImg.removeAttribute('src');
        candImg.style.visibility = 'hidden';
        var none = document.createElement('div');
        none.className = 'sub';
        none.textContent = 'Press N to save this face as a new person, or ↓ to say it is nobody known.';
        candList.appendChild(none);
        return;
      }
      if (sel >= list.length) sel = 0;
      var pick = list[sel];
      candImg.style.visibility = '';
      candImg.src = '/face-crop/' + encodeURIComponent(pick.ref);
      candName.textContent = pick.name;
      candScore.textContent = 'match score ' + pick.score.toFixed(3) +
        (list.length > 1 ? '  ·  ' + (sel + 1) + ' of ' + list.length + ' (← →)' : '');

      // The whole ranked list stays visible with its scores: the point of this
      // screen is seeing WHY the top pick won and how close the runners-up are,
      // not just being handed an answer.
      list.forEach(function (cand, i) {
        var row = document.createElement('button');
        row.type = 'button';
        row.className = 'review-cand' + (i === sel ? ' is-selected' : '');
        row.title = 'Select ' + cand.name + ' (Enter confirms)';
        var img = document.createElement('img');
        img.className = 'review-cand-img';
        img.src = '/face-crop/' + encodeURIComponent(cand.ref);
        img.width = 36;
        img.height = 36;
        img.loading = 'lazy';
        var name = document.createElement('span');
        name.className = 'review-cand-name';
        name.textContent = cand.name;
        var score = document.createElement('span');
        score.className = 'review-cand-score';
        score.textContent = cand.score.toFixed(3);
        row.appendChild(img);
        row.appendChild(name);
        row.appendChild(score);
        row.addEventListener('click', function () { sel = i; renderCandidates(); });
        candList.appendChild(row);
      });
    }

    function render() {
      var c = card();
      if (!c) {
        reviewEl.style.display = 'none';
        hintEl.style.display = 'none';
        controlsEl.style.display = 'none';
        setStatus('');
        emptyEl.style.display = '';
        if (fetching) emptyEl.innerHTML = '<div class="swipe-empty swipe-searching">SEARCHING<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>';
        else renderEmptyState();
        return;
      }
      emptyEl.style.display = 'none';
      reviewEl.style.display = '';
      hintEl.style.display = '';
      controlsEl.style.display = '';
      unknownImg.src = '/face-crop/' + encodeURIComponent(c.ref);
      unknownImg.style.transform = 'rotate(' + (rotationDelta() * 90) + 'deg)';
      unknownLink.href = '/photo/' + c.file_id;
      unknownMeta.innerHTML = window.swipeCardMeta(c) +
        (rotate ? ' · <strong>rotated ' + (rotate * 90) + '°</strong>' : '');
      renderCandidates();
      renderStatus();
    }

    function advance(persist) {
      var c = card();
      if (!c) return;
      queue.shift();
      history.push({ card: c, action: persist, rotate: rotate });
      // Only a persisted decision takes a face out of the server-side pool, so
      // only those move the remaining count — a skip leaves it there for later.
      if (persist !== 'skip' && remaining != null) remaining--;
      sel = 0;
      rotate = queue.length ? (queue[0].rotation || 0) : 0;
      render();
      fetchMore();
    }

    function post(url, body) {
      return fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
      }).then(function (r) {
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      });
    }

    function saveIdentity(name) {
      var c = card();
      if (!c) return;
      // The rotation the user landed on IS part of the decision: it's what the
      // candidate list was re-ranked against, so it has to be persisted with the
      // name or the stored embedding stays the scrambled one.
      post('/api/faces/' + encodeURIComponent(c.ref) + '/identity', { name: name || null, rotate: rotate })
        .catch(function (err) { showToast('Failed to save: ' + err.message); });
      advance('confirm');
    }

    function confirmSelected() {
      var list = candidates();
      if (!list.length) return;
      saveIdentity(list[sel].name);
    }

    function rejectAll() {
      var c = card();
      if (!c) return;
      post('/api/faces/' + encodeURIComponent(c.ref) + '/reject')
        .catch(function (err) { showToast('Failed to reject: ' + err.message); });
      advance('reject');
    }

    function undo() {
      if (!history.length) return;
      var last = history.pop();
      if (last.action !== 'skip') {
        fetch('/api/faces/' + encodeURIComponent(last.card.ref) + '/decision', { method: 'DELETE' })
          .catch(function () {});
        if (remaining != null) remaining++;
      }
      queue.unshift(last.card);
      sel = 0;
      rotate = last.rotate;
      render();
    }

    function saveAsNew() {
      var c = card();
      if (!c) return;
      // Same picker /photo uses (its ＋ Create row and its "Save without a name"
      // path both land on onResolved), so a face can be filed under a brand-new
      // person — or under no name at all — without leaving the review.
      window.openEntitySearchModal({
        type: 'identity',
        title: 'Save as a new person',
        previewImage: '/face-crop/' + encodeURIComponent(c.ref),
        allowEmpty: true,
        onResolved: function (entity) { saveIdentity(entity.name); },
      });
    }

    function rotateFace() {
      var c = card();
      if (!c) return;
      rotate = (rotate + 1) % 4;
      unknownImg.style.transform = 'rotate(' + (rotationDelta() * 90) + 'deg)';
      unknownMeta.innerHTML = window.swipeCardMeta(c) +
        (rotate ? ' · <strong>rotated ' + (rotate * 90) + '°</strong>' : '');
      // Rotating changes the embedding, so the stored top-K no longer applies —
      // re-rank live against the named matrix, otherwise the user would be
      // confirming a name that was picked for the mis-aligned vector.
      candScore.textContent = 're-ranking rotated face…';
      var ref = c.ref;
      fetch('/api/faces/' + encodeURIComponent(ref) + '/candidates?rotate=' + rotate + '&limit=8')
        .then(function (r) { return r.json(); })
        .then(function (data) {
          // Drop a stale response: the user has either moved past this face or
          // pressed R again, in which case this ranking describes an orientation
          // the face is no longer at (the endpoint echoes the orientation its
          // ranking belongs to in `rotation`).
          var cur = card();
          if (!cur || cur.ref !== ref) return;
          if (data.rotation != null && data.rotation !== rotate) return;
          cur.candidates = data.candidates || [];
          sel = 0;
          renderCandidates();
        })
        .catch(function (err) { showToast('Re-rank failed: ' + err.message); renderCandidates(); });
    }

    function fetchMore() {
      if (fetching || queue.length > REFILL_AT) return;
      fetching = true;
      var avoid = document.getElementById(config.avoidToggleId);
      var params = new URLSearchParams({
        count: String(BUFFER_SIZE - queue.length),
        avoid_existing: String(!avoid || avoid.checked),
        candidates: '8',
      });
      var wasEmpty = queue.length === 0;
      if (wasEmpty) render();   // paint the searching state while this is in flight
      fetch('/api/face-review/next?' + params.toString(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exclude: excludeList() }),
      })
        .then(function (r) {
          if (!r.ok) throw new Error('Request failed: ' + r.status);
          return r.json();
        })
        .then(function (data) {
          fetching = false;
          remaining = data.remaining;
          (data.cards || []).forEach(function (c) {
            if (seen[c.ref]) return;
            seen[c.ref] = true;
            queue.push(c);
          });
          if (wasEmpty) {
            rotate = queue.length ? (queue[0].rotation || 0) : 0;
            sel = 0;
          }
          if (wasEmpty || queue.length <= 1) render();
          else renderStatus();
        })
        .catch(function (err) {
          fetching = false;
          if (wasEmpty) {
            emptyEl.style.display = '';
            emptyEl.innerHTML = '<div class="swipe-empty">Could not load faces: ' + escapeHtml(err.message) + '</div>';
          }
        });
    }

    /* ---------------- empty states ----------------
       The old page had one message for every reason the queue could be empty,
       which was actively misleading: "no more suggestions" reads as "you're
       done" when the real cause is that nobody is named yet, or that the
       scoring worker has never run (in which case face_candidates is empty and
       NOTHING will ever show up until it does). Each of those needs a different
       action, so each gets its own copy. */
    function renderEmptyState() {
      emptyEl.innerHTML = '<div class="swipe-empty">Checking why the queue is empty…</div>';
      Promise.all([
        fetch('/api/identities').then(function (r) { return r.json(); }).catch(function () { return []; }),
        fetch('/api/face-scoring/status').then(function (r) { return r.json(); }).catch(function () { return {}; }),
      ]).then(function (res) {
        var identities = res[0] || [];
        var status = res[1] || {};
        if (status.running) { renderScoringProgress(status); return; }
        if (!identities.length) {
          emptyEl.innerHTML =
            '<div class="alert"><p style="margin:0 0 8px;"><strong>Nobody is named yet.</strong></p>' +
            '<p class="sub" style="margin:0;">This screen matches unknown faces against people you have already named — ' +
            'with an empty name list there is nothing to match against. Name a handful of faces on a photo page first ' +
            '(<a href="/faces">Faces</a>), then come back.</p></div>';
          return;
        }
        // `pending` is live (faces still needing a score at the current scoring
        // generation), independent of whether a job ever ran in this process —
        // unlike done/total/scored, which describe only the last/current run and
        // are 0 in a fresh process. So it is the one signal that separates "never
        // scored / stale after a rename" from "genuinely all reviewed".
        var pending = status.pending || 0;
        emptyEl.innerHTML =
          '<div class="alert">' +
            (status.error
              ? '<p style="margin:0 0 8px;"><strong>Face scoring failed:</strong> ' + escapeHtml(String(status.error)) + '</p>'
              : pending > 0
                ? '<p style="margin:0 0 8px;"><strong>' + pending.toLocaleString() + ' face(s) have never been scored.</strong> ' +
                  'Until scoring runs they cannot appear here — this queue only shows faces that already have a ranked match.</p>'
                : '<p style="margin:0 0 8px;"><strong>Nothing left to review.</strong> Every scored face is either named, rejected, or below the match floor.</p>') +
            '<p class="sub" style="margin:0 0 10px;">Scoring is what fills this queue: it ranks every unknown face against ' +
            'everyone you have named. Run it after naming new people — each confirm adds reference faces but does not ' +
            're-rank the rest of the library on its own.</p>' +
            '<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">' +
              '<button type="button" class="btn-similar review-score-btn">↻ Run face scoring</button>' +
              '<span class="sub review-score-status"></span>' +
            '</div>' +
          '</div>';
        emptyEl.querySelector('.review-score-btn').addEventListener('click', function () {
          startScoring();
        });
      });
    }

    function renderScoringProgress(status) {
      emptyEl.innerHTML =
        '<div class="alert"><p style="margin:0 0 6px;"><strong>Scoring faces…</strong></p>' +
        '<p class="sub review-score-status" style="margin:0;"></p></div>';
      paintScoring(status);
      pollScoring();
    }

    function paintScoring(status) {
      var el = emptyEl.querySelector('.review-score-status');
      if (!el) return;
      if (status.error) { el.textContent = 'failed: ' + status.error; return; }
      el.textContent = (status.done || 0).toLocaleString() + ' / ' + (status.total || 0).toLocaleString() +
        ((status.scored != null) ? ' · ' + status.scored.toLocaleString() + ' scored' : '');
    }

    function pollScoring() {
      fetch('/api/face-scoring/status')
        .then(function (r) { return r.json(); })
        .then(function (status) {
          paintScoring(status);
          if (status.running) { setTimeout(pollScoring, 1000); return; }
          // Scoring finished: the pool it just wrote is exactly what this page
          // reads, so retry the buffer instead of making the user reload.
          fetchMore();
        })
        .catch(function (err) { paintScoring({ error: err.message }); });
    }

    function startScoring() {
      var btn = emptyEl.querySelector('.review-score-btn');
      if (btn) { btn.disabled = true; btn.textContent = 'Scoring…'; }
      post('/api/face-scoring/start?full=0')
        .then(function () { renderScoringProgress({ done: 0, total: 0 }); })
        .catch(function (err) {
          if (btn) { btn.disabled = false; btn.textContent = '↻ Run face scoring'; }
          showToast('Could not start scoring: ' + err.message);
        });
    }

    document.addEventListener('keydown', function (e) {
      // Registered once, and it closes the full view before anything else can
      // act on that keypress — pressing Escape/F/Enter to dismiss the photo must
      // not also confirm or skip the face underneath.
      if (_fullviewOpen) {
        e.preventDefault();
        hideFullview();
        return;
      }
      if (isBlockedTarget(e.target)) return;

      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        undo();
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      var c = card();
      if (!c) return;
      var key = e.key;
      if (key === 'ArrowRight' || key === 'ArrowLeft') {
        var list = candidates();
        if (!list.length) return;
        e.preventDefault();
        sel = (sel + (key === 'ArrowRight' ? 1 : list.length - 1)) % list.length;
        renderCandidates();
      } else if (key === 'Enter') {
        e.preventDefault();
        confirmSelected();
      } else if (key === 'ArrowDown' || key.toLowerCase() === 'x') {
        e.preventDefault();
        rejectAll();
      } else if (key === 'ArrowUp') {
        e.preventDefault();
        advance('skip');
      } else if (key.toLowerCase() === 'n') {
        e.preventDefault();
        saveAsNew();
      } else if (key.toLowerCase() === 'r') {
        e.preventDefault();
        rotateFace();
      } else if (key.toLowerCase() === 'f') {
        e.preventDefault();
        showFullview('/image/' + c.file_id);
      }
    });

    // Every key also has a button: the keyboard is the fast path, not the only one.
    root.querySelectorAll('[data-review-action]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var action = btn.dataset.reviewAction;
        var c = card();
        if (action === 'confirm') confirmSelected();
        else if (action === 'reject') rejectAll();
        else if (action === 'skip') { if (c) advance('skip'); }
        else if (action === 'new') saveAsNew();
        else if (action === 'rotate') rotateFace();
        else if (action === 'full') { if (c) showFullview('/image/' + c.file_id); }
        else if (action === 'undo') undo();
      });
    });

    var avoidToggle = document.getElementById(config.avoidToggleId);
    if (avoidToggle) {
      // Changing the ordering preference only affects faces not yet fetched —
      // refetching would throw away a buffer the user can still act on.
      avoidToggle.addEventListener('change', function () { fetchMore(); });
    }

    fetchMore();
  };
})();
