/* Duplicate review groups — shared by /duplicates (exact, our own checksums) and
   /near-duplicates (similar, czkawka). One table per group: a row per copy with
   resolution / length / format / date / size / path. The server pre-ticks the copy
   to keep (biggest resolution, then length, then size); a human changes the ticks
   and presses the group's own button — there is no "resolve all". Unticked copies
   go to the Trash (never deleted); everything ticked = keep all.

   DupReview.renderGroup(group, opts) -> element
     opts.kind      'exact' | 'similar'
     opts.resolve   function(group, trashKeys, carryLabels) -> Promise
*/
(function () {
  function esc(s) { var d = document.createElement('div'); d.textContent = s == null ? '' : String(s); return d.innerHTML; }

  function fmtSize(b) {
    if (b == null) return '—';
    var u = ['B', 'KB', 'MB', 'GB', 'TB'], i = 0;
    while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
    return (i ? b.toFixed(b < 10 ? 1 : 0) : b) + ' ' + u[i];
  }
  function fmtDur(s) {
    if (s == null) return '—';
    s = Math.round(s);
    var h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), sec = s % 60;
    var mm = (h ? String(m).padStart(2, '0') : m), ss = String(sec).padStart(2, '0');
    return (h ? h + ':' : '') + mm + ':' + ss;
  }
  function fmtDate(ts) {
    if (!ts) return '—';
    var d = new Date(ts * 1000);
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0') +
      ' ' + String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  }
  function fmtRes(r) { return (r.width && r.height) ? r.width + '×' + r.height : '—'; }

  // Rows: exact → one per tracked path of the same file; similar → one per member file.
  function rowsOf(group, kind) {
    if (kind === 'exact') {
      return group.copies.map(function (c) {
        return { key: c.path, id: group.id, path: c.path, width: group.width, height: group.height,
                 duration: group.duration, format: group.format, is_video: group.is_video, size: group.size,
                 date: c.modified_time, dateKind: 'file date', missing: !c.on_disk };
      });
    }
    return group.members.map(function (m) {
      return { key: m.id, id: m.id, path: m.path, width: m.width, height: m.height, duration: m.duration,
               format: m.format, is_video: m.is_video, size: m.size,
               date: m.taken_at || m.modified_time, dateKind: m.taken_at ? 'taken' : 'file date',
               broken: m.broken, favorite: m.favorite, tags: m.tag_count, people: m.people_count,
               pathCount: m.path_count };
    });
  }

  // Mark the best value per column (similar groups only — exact copies are identical).
  function bestOf(rows, f) {
    var vals = rows.map(f).filter(function (v) { return v; });
    if (vals.length < 2) return null;
    var max = Math.max.apply(null, vals), min = Math.min.apply(null, vals);
    return max > min ? max : null;
  }

  function renderGroup(group, opts) {
    var kind = opts.kind;
    var rows = rowsOf(group, kind);
    var keep = {};
    (group.keep || []).forEach(function (k) { keep[k] = true; });

    var bestPx = kind === 'similar' ? bestOf(rows, function (r) { return (r.width || 0) * (r.height || 0); }) : null;
    var bestDur = kind === 'similar' ? bestOf(rows, function (r) { return r.duration || 0; }) : null;
    var bestSize = kind === 'similar' ? bestOf(rows, function (r) { return r.size || 0; }) : null;

    var el = document.createElement('div');
    el.className = 'dup-group';
    var title = kind === 'exact'
      ? rows.length + ' copies of the same file'
      : rows.length + ' similar ' + (rows.every(function (r) { return r.is_video; }) ? 'videos' : 'files');
    var hint = kind === 'similar' && group.label && group.label !== 'review'
      ? '<span class="dup-hint" title="' + esc(group.reason) + '">' + esc(group.label.replace(/_/g, ' ')) + '</span>' : '';
    var labels = kind === 'exact' && (group.tag_count || group.people_count || group.favorite)
      ? '<span class="sub">labels are shared by every copy (same content)</span>' : '';

    var html = '<div class="dup-group-head"><strong>' + esc(title) + '</strong>' + hint + labels + '</div>' +
      '<table class="dup-table"><thead><tr>' +
        '<th title="Ticked = keep. Unticked copies go to the Trash.">Keep</th><th></th>' +
        '<th>Resolution</th><th>Length</th><th>Format</th><th>Date</th><th>Size</th><th>Path</th>' +
      '</tr></thead><tbody>';
    rows.forEach(function (r, i) {
      var thumb = (kind === 'similar' || i === 0)
        ? '<a href="/photo/' + r.id + '" target="_blank" rel="noopener"><img src="/thumb/' + r.id + '" alt="" loading="lazy"></a>'
        : '<span class="sub">same</span>';
      var px = (r.width || 0) * (r.height || 0);
      var badges = '';
      if (r.missing) badges += '<span class="dup-badge dup-bad">missing on disk</span>';
      if (r.broken) badges += '<span class="dup-badge dup-bad">damaged</span>';
      if (r.favorite) badges += '<span class="dup-badge">♥ ' + r.favorite + '</span>';
      if (r.people) badges += '<span class="dup-badge">👤 ' + r.people + '</span>';
      if (r.tags) badges += '<span class="dup-badge">🏷 ' + r.tags + '</span>';
      if (r.pathCount > 1) badges += '<span class="dup-badge" title="This content is tracked at ' + r.pathCount + ' paths; trashing it trashes the content">' + r.pathCount + ' paths</span>';
      html += '<tr data-key="' + esc(r.key) + '"' + (keep[r.key] ? ' class="is-keep"' : '') + '>' +
        '<td><input type="checkbox" class="dup-keep"' + (keep[r.key] ? ' checked' : '') + '></td>' +
        '<td class="dup-thumb">' + thumb + '</td>' +
        '<td' + (bestPx && px === bestPx ? ' class="dup-best"' : '') + '>' + fmtRes(r) + '</td>' +
        '<td' + (bestDur && r.duration === bestDur ? ' class="dup-best"' : '') + '>' + (r.is_video ? fmtDur(r.duration) : '—') + '</td>' +
        '<td>' + esc(r.format) + '</td>' +
        '<td title="' + esc(r.dateKind) + '">' + fmtDate(r.date) + '</td>' +
        '<td' + (bestSize && r.size === bestSize ? ' class="dup-best"' : '') + '>' + fmtSize(r.size) + '</td>' +
        '<td class="dup-path"><a href="/photo/' + r.id + '" target="_blank" rel="noopener" title="' + esc(r.path) + '">' + esc(r.path) + '</a>' +
          (badges ? '<div>' + badges + '</div>' : '') + '</td>' +
      '</tr>';
    });
    html += '</tbody></table><div class="dup-actions">' +
      (kind === 'similar' ? '<label class="sub"><input type="checkbox" class="dup-carry" checked> carry tags / people / sets to the kept copy</label>' : '') +
      '<button type="button" class="dup-go"></button><span class="sub dup-msg"></span></div>';
    el.innerHTML = html;

    var btn = el.querySelector('.dup-go');
    var msg = el.querySelector('.dup-msg');
    function trashKeys() {
      return Array.prototype.filter.call(el.querySelectorAll('tbody tr'), function (tr) {
        return !tr.querySelector('.dup-keep').checked;
      }).map(function (tr) { return tr.getAttribute('data-key'); });
    }
    function refresh() {
      el.querySelectorAll('tbody tr').forEach(function (tr) {
        tr.classList.toggle('is-keep', tr.querySelector('.dup-keep').checked);
      });
      var n = trashKeys().length;
      btn.disabled = n === rows.length;
      btn.className = 'dup-go' + (n ? ' dup-go-trash' : '');
      btn.textContent = n === rows.length ? 'Tick at least one copy to keep'
        : n ? '🗑 Trash ' + n + ' unticked'
        : (kind === 'exact' ? 'Keep all copies' : 'Not duplicates — keep all');
    }
    el.addEventListener('change', function (e) { if (e.target.classList.contains('dup-keep')) refresh(); });
    btn.addEventListener('click', function () {
      var keys = trashKeys();
      if (kind === 'similar') keys = keys.map(Number);
      var carry = el.querySelector('.dup-carry');
      btn.disabled = true;
      msg.textContent = '';
      opts.resolve(group, keys, carry ? carry.checked : false)
        .then(function () {
          el.classList.add('is-done');
          el.innerHTML = '<div class="sub">' + (keys.length ? 'Trashed ' + keys.length + ' — restore any time from <a href="/trash">Trash</a>.' : 'Kept all.') + '</div>';
          if (opts.onDone) opts.onDone(group, keys);
        })
        .catch(function (err) { btn.disabled = false; msg.textContent = 'Failed: ' + err.message; });
    });
    refresh();
    return el;
  }

  function postJson(url, body) {
    return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
      .then(function (r) {
        return r.json().catch(function () { return {}; }).then(function (d) {
          if (!r.ok) throw new Error(d.detail || ('HTTP ' + r.status));
          return d;
        });
      });
  }

  window.DupReview = { renderGroup: renderGroup, postJson: postJson };
})();
