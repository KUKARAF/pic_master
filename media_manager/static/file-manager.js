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
      var hide = on && (st === 'scanned' || st === 'ignored');
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
