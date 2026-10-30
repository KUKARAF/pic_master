// File manager (/files) — client side of the colour-coded folder view.
// Only loaded when the backend passes `fm`. Everything built here uses
// textContent / DOM APIs (never innerHTML with filenames).
(function () {
  'use strict';

  var ONLY_UNSCANNED_KEY = 'mm_fm_only_unscanned';

  function relTime(ts) {
    var secs = Math.max(0, Math.round(Date.now() / 1000 - ts));
    if (secs < 45) return 'just now';
    var mins = Math.round(secs / 60);
    if (mins < 60) return mins + ' min ago';
    var hours = Math.round(mins / 60);
    if (hours < 24) return hours + ' h ago';
    var days = Math.round(hours / 24);
    if (days < 30) return days + ' day' + (days === 1 ? '' : 's') + ' ago';
    return new Date(ts * 1000).toLocaleDateString();
  }

  // "Folder totals as of <relative time>" — absolute time in the tooltip.
  document.querySelectorAll('[data-fm-ts]').forEach(function (el) {
    var ts = parseInt(el.getAttribute('data-fm-ts'), 10);
    if (!ts) return;
    el.textContent = relTime(ts);
    el.title = new Date(ts * 1000).toLocaleString();
  });

  // ---- Show non-media files: server-side (?show_other=1), other params preserved.
  var showOther = document.getElementById('fm-show-other');
  if (showOther) {
    var params = new URLSearchParams(window.location.search);
    showOther.checked = params.get('show_other') === '1';
    showOther.addEventListener('change', function () {
      var p = new URLSearchParams(window.location.search);
      if (showOther.checked) { p.set('show_other', '1'); } else { p.delete('show_other'); }
      var qs = p.toString();
      window.location.search = qs ? '?' + qs : '';
    });
  }

  // ---- Only not fully scanned: client-side hide, remembered in localStorage.
  var onlyUnscanned = document.getElementById('fm-only-unscanned');
  function applyOnlyUnscanned() {
    var on = onlyUnscanned.checked;
    var visible = 0, total = 0;
    document.querySelectorAll('.fm-tile').forEach(function (tile) {
      var st = tile.getAttribute('data-state');
      var hide = on && (st === 'scanned' || st === 'ignored' || st === 'trashed');
      tile.hidden = hide;
      total++;
      if (!hide) visible++;
    });
    document.querySelectorAll('.fm-folder-row').forEach(function (row) {
      var hide = on && row.getAttribute('data-done') === '1';
      row.hidden = hide;
      total++;
      if (!hide) visible++;
    });
    var allDone = document.getElementById('fm-all-done');
    if (allDone) allDone.hidden = !(on && total > 0 && visible === 0);
  }
  if (onlyUnscanned) {
    try { onlyUnscanned.checked = localStorage.getItem(ONLY_UNSCANNED_KEY) === '1'; } catch (e) {}
    onlyUnscanned.addEventListener('change', function () {
      try { localStorage.setItem(ONLY_UNSCANNED_KEY, onlyUnscanned.checked ? '1' : '0'); } catch (e) {}
      applyOnlyUnscanned();
    });
    applyOnlyUnscanned();
  }

  // ---- ↻ Refresh inventory: POST start, poll status every 2 s, reload when done.
  var box = document.getElementById('fm-inventory');
  var btn = document.getElementById('fm-inv-refresh');
  var text = document.getElementById('fm-inv-text');
  if (!box || !btn || !text) return;

  function setText(msg) { text.textContent = msg; }

  function poll() {
    fetch('/api/fm/inventory/status')
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (s) {
        if (s.running) {
          setText('Inventory running… ' + (s.total ? (s.done || 0) + '/' + s.total : (s.done || 0)) + ' files');
          setTimeout(poll, 2000);
          return;
        }
        btn.disabled = false;
        if (s.error) { setText('Inventory failed: ' + s.error); return; }
        setText('Inventory done — reloading…');
        window.location.reload();
      })
      .catch(function (err) {
        btn.disabled = false;
        setText('Lost track of the inventory: ' + err.message);
      });
  }

  btn.addEventListener('click', function () {
    btn.disabled = true;
    setText('Starting inventory…');
    fetch('/api/fm/inventory/start', { method: 'POST' })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json().catch(function () { return {}; }); })
      .then(function () { setTimeout(poll, 500); })
      .catch(function (err) {
        btn.disabled = false;
        setText('Could not start inventory: ' + err.message);
      });
  });

  // Page loaded while an inventory is already running — follow it.
  if (box.getAttribute('data-running') === '1') { btn.disabled = true; poll(); }
})();

// ---- 📥 Track this folder / 🧠 Run missing ML here: background jobs with a
// progress panel each (#fm-jobs). Both jobs are server-wide singletons, so on page
// load we attach to whichever is running — even one started from another folder.
(function () {
  'use strict';

  var POLL_MS = 1500;
  var SUMMARY_KEY = 'mm_fm_job_summary';   // survives the post-job reload
  var jobsBox = document.getElementById('fm-jobs');
  var trackBtn = document.getElementById('fm-track-btn');
  var mlBtn = document.getElementById('fm-ml-btn');
  if (!jobsBox || !trackBtn || !mlBtn) return;

  // Summary of a job that finished right before the reload.
  try {
    var pending = sessionStorage.getItem(SUMMARY_KEY);
    if (pending) {
      sessionStorage.removeItem(SUMMARY_KEY);
      var saved = JSON.parse(pending);
      showToast(saved.msg, saved.type);
    }
  } catch (e) {}

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = String(text);
    return node;
  }
  function plural(n, word) { return n + ' ' + word + (n === 1 ? '' : 's'); }
  function folderLabel(path) { return '/' + (path || ''); }
  function pct(done, total) { return total ? Math.min(100, Math.round(100 * done / total)) : 0; }

  function bar(done, total) {
    var outer = el('div', 'fm-job-bar');
    var inner = el('span');
    inner.style.width = pct(done, total) + '%';
    outer.appendChild(inner);
    outer.setAttribute('role', 'progressbar');
    outer.setAttribute('aria-valuemin', '0');
    outer.setAttribute('aria-valuemax', String(total || 0));
    outer.setAttribute('aria-valuenow', String(done || 0));
    return outer;
  }

  // One panel per job kind: {root, title, body, actions}.
  var panels = {};
  function panel(kind) {
    if (panels[kind]) return panels[kind];
    var root = el('div', 'fm-job fm-job-' + kind);
    var title = el('div', 'fm-job-title');
    var body = el('div', 'fm-job-body');
    var actions = el('div', 'fm-job-actions');
    root.appendChild(title);
    root.appendChild(body);
    root.appendChild(actions);
    jobsBox.appendChild(root);
    panels[kind] = { root: root, title: title, body: body, actions: actions };
    return panels[kind];
  }

  function finishAndReload(msg, type) {
    showToast(msg, type);
    try { sessionStorage.setItem(SUMMARY_KEY, JSON.stringify({ msg: msg, type: type })); } catch (e) {}
    setTimeout(function () { window.location.reload(); }, 2500);
  }

  function readJson(r) { return r.json().catch(function () { return {}; }); }

  // ================= Track =================
  var TRACK_PHASES = {
    listing: 'Listing files', hashing: 'Hashing', saving: 'Saving to the library',
    markers: 'Applying folder markers', done: 'Done',
  };

  function renderTrack(s) {
    var p = panel('track');
    p.title.textContent = '📥 Tracking ' + folderLabel(s.path) + (s.running ? '' : ' — finished');
    p.body.replaceChildren();
    var phase = TRACK_PHASES[s.phase] || (s.running ? 'Starting' : '');
    var line = phase;
    if (s.phase === 'listing') line += '… ' + plural(s.done || 0, 'file') + ' seen';
    else if (s.total) line += ' ' + (s.done || 0) + '/' + s.total;
    p.body.appendChild(el('div', 'fm-job-line', line));
    if (s.phase === 'hashing' || s.phase === 'saving') p.body.appendChild(bar(s.done, s.total));
    var stats = 'added ' + (s.added || 0) + ' · already tracked ' + (s.already || 0) +
                ' · duplicates ' + (s.duplicates || 0) + ' · ignored ' + (s.ignored || 0);
    p.body.appendChild(el('div', 'fm-job-line sub', stats));
    if (s.error) p.body.appendChild(el('div', 'fm-job-error', 'Error: ' + s.error));
  }

  function trackSummary(s) {
    var msg = 'Tracked ' + folderLabel(s.path) + ': ' + plural(s.added || 0, 'new file') + ' added';
    if (s.already) msg += ', ' + s.already + ' already tracked';
    if (s.duplicates) msg += ', ' + plural(s.duplicates, 'duplicate') + ' moved to Trash (nothing deleted)';
    if (s.ignored) msg += ', ' + s.ignored + ' ignored';
    return msg + '.';
  }

  function pollTrack() {
    fetch('/api/fm/track/status')
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (s) {
        renderTrack(s);
        if (s.running) {
          trackBtn.disabled = true;
          setTimeout(pollTrack, POLL_MS);
          return;
        }
        trackBtn.disabled = false;
        if (s.error) { showToast('Tracking failed: ' + s.error, 'error'); return; }
        finishAndReload(trackSummary(s), 'success');
      })
      .catch(function (err) {
        trackBtn.disabled = false;
        showToast('Lost track of the track job: ' + err.message, 'error');
      });
  }

  trackBtn.addEventListener('click', function () {
    var folder = trackBtn.dataset.folder || '';
    var n = parseInt(trackBtn.dataset.untracked, 10) || 0;
    var msg = n
      ? 'Track ' + plural(n, 'untracked file') + ' under ' + folderLabel(folder) + ' (subfolders included)?'
      : 'No untracked files are counted under ' + folderLabel(folder) + '. Scan it anyway?';
    msg += '\n\nSame as `media add`: new photos/videos are added to the library; copies of files ' +
           'already in it are not tracked but put in the Trash (nothing is deleted); .mediaignore is respected.';
    if (trackBtn.dataset.inventory !== '1') msg += '\n(Subfolder counts are incomplete until the inventory has run.)';
    if (!confirm(msg)) return;
    trackBtn.disabled = true;
    fetch('/api/fm/track/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: folder }),
    })
      .then(function (r) {
        return readJson(r).then(function (d) {
          if (r.status === 409) { showToast((d.detail || 'A track job is already running') + ' — following it.'); }
          else if (!r.ok) { throw new Error(d.detail || 'HTTP ' + r.status); }
          pollTrack();
        });
      })
      .catch(function (err) {
        trackBtn.disabled = false;
        showToast('Could not start tracking: ' + err.message, 'error');
      });
  });

  // ================= ML =================
  var ML_STATUS = {
    pending: 'pending', running: 'running…', done: 'done', skipped: 'skipped',
    error: 'error', cancelled: 'cancelled',
  };

  function renderMl(s) {
    var p = panel('ml');
    var head = '🧠 ML on ' + folderLabel(s.path);
    if (!s.running) head += s.cancelled ? ' — cancelled' : ' — finished';
    p.title.textContent = head;
    p.body.replaceChildren();
    (s.stages || []).forEach(function (st) {
      var row = el('div', 'fm-ml-row fm-ml-' + (st.status || 'pending'));
      row.appendChild(el('span', 'fm-ml-label', st.label || st.key));
      var stText = ML_STATUS[st.status] || st.status || '';
      var counts = st.total ? (st.done || 0) + '/' + st.total : '';
      if (st.failed) counts += (counts ? ' · ' : '') + st.failed + ' failed';
      row.appendChild(el('span', 'fm-ml-status', stText + (counts ? ' ' + counts : '')));
      row.appendChild(bar(st.done || 0, st.total || 0));
      if (st.message) row.appendChild(el('span', 'fm-ml-msg sub', st.message));
      p.body.appendChild(row);
    });
    if (s.error) p.body.appendChild(el('div', 'fm-job-error', 'Error: ' + s.error));
    p.actions.replaceChildren();
    if (s.running) {
      var cancel = el('button', 'btn-similar fm-job-cancel', 'Cancel');
      cancel.type = 'button';
      cancel.addEventListener('click', function () {
        cancel.disabled = true;
        cancel.textContent = 'Cancelling…';
        fetch('/api/fm/ml/cancel', { method: 'POST' })
          .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); })
          .catch(function (err) {
            cancel.disabled = false;
            cancel.textContent = 'Cancel';
            showToast('Could not cancel: ' + err.message, 'error');
          });
      });
      p.actions.appendChild(cancel);
    }
  }

  function mlSummary(s) {
    var parts = (s.stages || []).filter(function (st) { return st.status !== 'skipped'; })
      .map(function (st) {
        var t = (st.label || st.key) + ' ' + (st.done || 0) + '/' + (st.total || 0);
        return st.failed ? t + ' (' + st.failed + ' failed)' : t;
      });
    var msg = 'ML on ' + folderLabel(s.path) + (s.cancelled ? ' cancelled' : ' finished');
    return msg + (parts.length ? ': ' + parts.join(', ') : '') + '.';
  }

  function pollMl() {
    fetch('/api/fm/ml/status')
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (s) {
        renderMl(s);
        if (s.running) {
          mlBtn.disabled = true;
          setTimeout(pollMl, POLL_MS);
          return;
        }
        mlBtn.disabled = false;
        if (s.error) { showToast('ML run failed: ' + s.error, 'error'); return; }
        finishAndReload(mlSummary(s), s.cancelled ? 'info' : 'success');
      })
      .catch(function (err) {
        mlBtn.disabled = false;
        showToast('Lost track of the ML job: ' + err.message, 'error');
      });
  }

  function startMl(folder, keys) {
    mlBtn.disabled = true;
    return fetch('/api/fm/ml/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: folder, stages: keys }),
    })
      .then(function (r) {
        return readJson(r).then(function (d) {
          if (r.status === 409) { showToast((d.detail || 'An ML job is already running') + ' — following it.'); }
          else if (!r.ok) { throw new Error(d.detail || 'HTTP ' + r.status); }
          pollMl();
        });
      })
      .catch(function (err) {
        mlBtn.disabled = false;
        showToast('Could not start ML: ' + err.message, 'error');
      });
  }

  function openMlDialog(folder, preview) {
    openModal('Run missing ML', function (box) {
      box.appendChild(el('p', 'sub fm-ml-intro',
        plural(preview.files || 0, 'tracked file') + ' under ' + folderLabel(preview.path != null ? preview.path : folder) +
        '. Pick the stages to run — only files missing a stage are processed.'));
      var list = el('div', 'fm-ml-pick');
      var boxes = [];
      (preview.stages || []).forEach(function (st) {
        var label = el('label', 'fm-ml-opt');
        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = st.key;
        cb.checked = !!st.required && (st.missing || 0) > 0;
        cb.disabled = !(st.missing > 0);
        label.appendChild(cb);
        label.appendChild(el('span', 'fm-ml-opt-name', st.label || st.key));
        label.appendChild(el('span', 'sub', (st.missing || 0) + ' missing' + (st.required ? '' : ' · optional')));
        list.appendChild(label);
        boxes.push(cb);
      });
      box.appendChild(list);
      var actions = el('div', 'fm-job-actions');
      var go = el('button', 'stage-btn-accent', 'Start');
      go.type = 'button';
      var cancel = el('button', 'btn-similar', 'Cancel');
      cancel.type = 'button';
      cancel.addEventListener('click', closeModal);
      function syncGo() { go.disabled = !boxes.some(function (b) { return b.checked; }); }
      boxes.forEach(function (b) { b.addEventListener('change', syncGo); });
      syncGo();
      go.addEventListener('click', function () {
        var keys = boxes.filter(function (b) { return b.checked; }).map(function (b) { return b.value; });
        closeModal();
        startMl(folder, keys);
      });
      actions.appendChild(go);
      actions.appendChild(cancel);
      box.appendChild(actions);
    });
  }

  mlBtn.addEventListener('click', function () {
    var folder = mlBtn.dataset.folder || '';
    mlBtn.disabled = true;
    fetch('/api/fm/ml/preview?path=' + encodeURIComponent(folder))
      .then(function (r) {
        if (r.status === 404 || r.status === 405) throw new Error('ML runner not available');
        return readJson(r).then(function (d) {
          if (!r.ok) throw new Error(d.detail || 'HTTP ' + r.status);
          return d;
        });
      })
      .then(function (d) {
        mlBtn.disabled = false;
        var todo = (d.stages || []).some(function (st) { return st.missing > 0; });
        if (!todo) { showToast('Nothing missing — every tracked file under ' + folderLabel(folder) + ' is scanned.', 'success'); return; }
        openMlDialog(folder, d);
      })
      .catch(function (err) {
        mlBtn.disabled = false;
        showToast(err.message === 'ML runner not available' ? err.message : 'ML preview failed: ' + err.message,
                  'error');
      });
  });

  // ================= attach on load =================
  function attachIfRunning(url, poll) {
    fetch(url)
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (s) { if (s && s.running) poll(); })
      .catch(function () {});
  }
  attachIfRunning('/api/fm/track/status', pollTrack);
  attachIfRunning('/api/fm/ml/status', pollMl);
})();
