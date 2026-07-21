/* app.js — nav widgets (all pages) + photo detail page tag/set/face management */

(function () {
  'use strict';

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
    if (card.people && card.people.length) parts.push('with ' + card.people.map(escapeHtml).join(', '));
    if (card.category && card.category.name) parts.push('category: ' + escapeHtml(card.category.name));
    if (card.sets && card.sets.length) parts.push('in ' + card.sets.map(function (s) { return escapeHtml(s.name); }).join(', '));
    if (card.tags && card.tags.length) parts.push('tags: ' + card.tags.map(escapeHtml).join(', '));
    return parts.join(' · ');
  };

  /* Shared favorite-heart toggle — POSTs {favorite: bool} to `endpoint` and flips the
     button's glyph/class on success. Used by every heart button across the app so
     the toggle behaves identically everywhere. */
  window.wireHeartButton = function (btn, endpoint) {
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      e.stopPropagation();
      const goingTo = !btn.classList.contains('is-favorite');
      btn.disabled = true;
      fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ favorite: goingTo }),
      })
        .then(function (r) {
          if (!r.ok) throw new Error('Request failed: ' + r.status);
          return r.json();
        })
        .then(function () {
          btn.classList.toggle('is-favorite', goingTo);
          btn.textContent = goingTo ? '♥' : '♡';
          btn.disabled = false;
        })
        .catch(function (err) {
          btn.disabled = false;
          alert('Failed to update favorite: ' + err.message);
        });
    });
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

  function openStudioPromptModal(setName, onResolved) {
    openModal('Studio for "' + setName + '"', function (box) {
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'Studio (optional)…';
      input.setAttribute('list', 'studio-prompt-datalist');
      input.autocomplete = 'off';
      input.style.width = '100%';
      input.style.fontSize = '1.2rem';
      input.style.padding = '10px 12px';
      input.style.marginBottom = '8px';
      box.appendChild(input);

      const datalist = document.createElement('datalist');
      datalist.id = 'studio-prompt-datalist';
      box.appendChild(datalist);

      const status = document.createElement('div');
      status.className = 'modal-empty';
      status.textContent = 'Loading studios…';
      box.appendChild(status);

      let submitted = false;
      function submit() {
        if (submitted) return;
        submitted = true;
        input.disabled = true;
        createSet(setName, input.value.trim() || null)
          .then(function (data) { closeModal(); onResolved(data); })
          .catch(function (err) {
            submitted = false;
            input.disabled = false;
            alert('Failed to create set: ' + err.message);
          });
      }

      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); submit(); }
        else if (e.key === 'Escape') { e.preventDefault(); input.value = ''; submit(); }
      });

      fetch('/api/studios')
        .then(function (r) { return r.json(); })
        .then(function (studios) {
          datalist.innerHTML = '';
          studios.forEach(function (s) {
            const opt = document.createElement('option');
            opt.value = s.name;
            datalist.appendChild(opt);
          });
          status.textContent = studios.length
            ? 'Type to search ' + studios.length + ' existing studio(s), or a new one — Enter to confirm, Escape to skip.'
            : 'No studios yet — type one, or press Escape to skip.';
        })
        .catch(function () {
          status.textContent = 'Failed to load studios — Enter to confirm, Escape to skip.';
        });

      setTimeout(function () { input.focus(); }, 0);
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

  // Per-type behavior for openEntitySearchModal — the one place that knows how
  // to list/label/create each kind of entity. `image` is only set for
  // 'identity' (a face-crop thumbnail); every other type renders text-only.
  const ENTITY_TYPE_CONFIGS = {
    set: {
      title: 'Choose a set',
      placeholder: 'Search or create a set…',
      fetchAll: function () { return fetch('/api/sets').then(function (r) { return r.json(); }); },
      label: function (s) { return s.name; },
      secondary: function (s) { return s.studio || ''; },
      image: null,
      // Matches today's exact chained behavior: a new set's studio is its own
      // keyboard-first step, not folded into this one.
      createFn: function (typedName) {
        return new Promise(function (resolve) { openStudioPromptModal(typedName, resolve); });
      },
    },
    category: {
      title: 'Set category',
      placeholder: 'Search or create a category…',
      fetchAll: function () { return fetch('/api/categories').then(function (r) { return r.json(); }); },
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
    tag: {
      title: 'Search or create a tag',
      placeholder: 'Search or type a tag…',
      // Vocab is already the merged "every known label" list (YOLO-World's
      // search terms + manual.db's confirmed tags) — exactly the right
      // candidate pool for this picker, hint-only entries included.
      fetchAll: function () {
        return fetch('/api/vocab').then(function (r) { return r.json(); })
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
      fetchAll: function () { return fetch('/api/identities').then(function (r) { return r.json(); }); },
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

      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = config.placeholder;
      input.autocomplete = 'off';
      input.style.marginBottom = '8px';
      box.appendChild(input);

      const status = document.createElement('div');
      status.className = 'modal-empty';
      status.textContent = 'Loading…';
      box.appendChild(status);

      const list = document.createElement('div');
      list.className = 'modal-list';
      box.appendChild(list);

      const extraBox = document.createElement('div');
      box.appendChild(extraBox);

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
              alert('Failed to create: ' + err.message);
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
              thumb.className = 'modal-list-item-thumb';
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
        let matched;
        if (!query) {
          matched = allItems.slice(0, 50);
        } else {
          matched = allItems
            .map(function (item) { return { item: item, score: fuzzyScore(query, config.label(item)) }; })
            .filter(function (x) { return x.score !== null; })
            .sort(function (a, b) { return a.score - b.score; })
            .slice(0, 50)
            .map(function (x) { return x.item; });
        }
        visible = matched.map(function (item) { return { kind: 'existing', item: item }; });
        if (query) visible.push({ kind: 'create', text: query });
        highlighted = visible.length ? 0 : -1;
        renderList();
      }

      input.addEventListener('input', applyFilter);

      input.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          if (highlighted < visible.length - 1) { highlighted++; updateHighlight(); }
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          if (highlighted > 0) { highlighted--; updateHighlight(); }
        } else if (e.key === 'Enter') {
          e.preventDefault();
          if (highlighted >= 0 && visible[highlighted]) resolveEntry(visible[highlighted]);
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

      config.fetchAll()
        .then(function (items) {
          if (excludeIds.length) {
            items = items.filter(function (item) { return excludeIds.indexOf(item.id) === -1; });
          }
          allItems = items;
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

  function openSetSearchModal(onResolved, excludeSetId) {
    openEntitySearchModal({ type: 'set', excludeIds: excludeSetId, onResolved: onResolved });
  }

  window.openSetSearchModal = openSetSearchModal;

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
  (function () {
    var overlay = document.getElementById('palette-overlay');
    var input = document.getElementById('palette-input');
    var navInput = document.getElementById('nav-search-input');
    if (!overlay || !input) return;

    var chipsEl = document.getElementById('palette-chips');
    var countEl = document.getElementById('palette-count');
    var suggestionsEl = document.getElementById('palette-suggestions');
    var gridEl = document.getElementById('palette-grid');
    var viewAllEl = document.getElementById('palette-viewall');

    var chips = [];       // [{type, value}]
    var suggestions = []; // last response's suggestion list
    var highlighted = 0;
    var requestSeq = 0;   // guards against an in-flight request resolving out of order

    var FACET_LABELS = { category: 'CAT', tag: 'TAG', face: 'FACE', set: 'SET', file: 'FILE' };

    function open() {
      overlay.style.display = 'flex';
      input.value = '';
      input.focus();
      query();
    }
    function close() {
      overlay.style.display = 'none';
      chips = [];
      suggestions = [];
      highlighted = 0;
    }

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
          query();
        });
        chipsEl.appendChild(chip);
      });
    }

    function renderSuggestions() {
      suggestionsEl.innerHTML = '';
      if (!suggestions.length) {
        var empty = document.createElement('div');
        empty.className = 'palette-empty';
        empty.textContent = '— no matches —';
        suggestionsEl.appendChild(empty);
        return;
      }
      suggestions.forEach(function (s, i) {
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

    function renderGrid(media) {
      gridEl.innerHTML = '';
      var summary = chips.map(function (c) { return c.type + ':' + c.value; }).join(' + ') || (input.value.trim() || 'all');
      gridEl.dataset.queueSource = 'Search: ' + summary;
      if (!media.length) {
        var empty = document.createElement('div');
        empty.className = 'palette-empty';
        empty.textContent = chips.length || input.value.trim() ? '— no media matches this query —' : 'Type to search, or pick a suggestion…';
        gridEl.appendChild(empty);
        return;
      }
      media.forEach(function (m) {
        var tile = document.createElement('div');
        tile.className = 'card palette-tile';
        tile.dataset.fileId = m.id;
        var link = document.createElement('a');
        link.className = 'card-img-link';
        link.href = '/photo/' + m.id;
        var img = document.createElement('img');
        img.loading = 'lazy';
        img.src = '/thumb/' + m.id;
        img.alt = m.filename;
        link.appendChild(img);
        var name = document.createElement('span');
        name.className = 'palette-tile-name';
        name.textContent = m.title || m.filename;
        tile.appendChild(link);
        tile.appendChild(name);
        gridEl.appendChild(tile);
      });
    }

    function buildFilterHref() {
      var parts = chips.map(function (c) { return 'f=' + encodeURIComponent(c.type + ':' + c.value); });
      return '/search?' + parts.join('&');
    }

    function commitHighlighted() {
      var s = suggestions[Math.min(highlighted, suggestions.length - 1)];
      if (!s) return;
      chips.push({ type: s.type, value: s.value });
      input.value = '';
      highlighted = 0;
      query();
    }

    function query() {
      var seq = ++requestSeq;
      var q = input.value.trim();
      fetch('/api/search-palette', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: q, chips: chips }),
      })
        .then(function (r) { return r.json(); })
        .then(function (data) {
          if (seq !== requestSeq) return; // a newer keystroke's request already landed
          suggestions = data.suggestions;
          highlighted = 0;
          renderChips();
          renderSuggestions();
          renderGrid(data.media);
          countEl.textContent = data.total_count + ' RESULTS';
          if (data.total_count > data.media.length) {
            viewAllEl.style.display = '';
            viewAllEl.href = buildFilterHref();
            viewAllEl.textContent = 'View all ' + data.total_count + ' results →';
          } else {
            viewAllEl.style.display = 'none';
          }
        })
        .catch(function () { /* transient network hiccup — next keystroke retries */ });
    }

    var debounceTimer = null;
    input.addEventListener('input', function () {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(query, 150);
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
        if (suggestions.length) commitHighlighted();
      } else if (e.key === 'Backspace' && input.value === '') {
        if (chips.length) {
          e.preventDefault();
          chips.pop();
          query();
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
      if (e.key !== '/') return;
      if (isTypingTarget(document.activeElement)) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      e.preventDefault();
      open();
    });
  })();

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

    const hintEl = document.getElementById('stage-hint');
    if (hintEl) {
      hintEl.textContent = queue
        ? queue.label + ' · ' + (queue.cursor + 1) + '/' + queue.ids.length + ' · ← → BROWSE · F FIT · ESC PANEL'
        : 'F FIT · ESC PANEL';
    }

    function goTo(id) {
      window.location.href = '/photo/' + id;
    }

    function isTypingTarget(el) {
      if (!el) return false;
      const tag = el.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
    }

    document.addEventListener('keydown', function (e) {
      if (!queue) return;
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      if (isTypingTarget(document.activeElement)) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;

      if (e.key === 'ArrowLeft') {
        if (queue.cursor <= 0) return;
        queue.cursor -= 1;
      } else {
        if (queue.cursor >= queue.ids.length - 1) return;
        queue.cursor += 1;
      }
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(queue));
      goTo(queue.ids[queue.cursor]);
    });
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
      heartBtn.className = 'heart-btn-inline tag-heart-btn' + (tag.favorite ? ' is-favorite' : '');
      heartBtn.style.fontSize = '0.85em';
      heartBtn.title = 'Favorite this tag';
      heartBtn.textContent = tag.favorite ? '♥' : '♡';
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

  if (tagList) {
    tagList.addEventListener('click', function (e) {
      const heartBtn = e.target.closest('.tag-heart-btn');
      if (heartBtn) {
        const chip = heartBtn.closest('.tag-removable, .tag-negative');
        if (!chip) return;
        const goingTo = !heartBtn.classList.contains('is-favorite');
        fetch('/api/tags/' + chip.dataset.tagId + '/favorite', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ favorite: goingTo }),
        })
          .then(function (r) { if (!r.ok) throw new Error('Request failed'); return r.json(); })
          .then(function () {
            heartBtn.classList.toggle('is-favorite', goingTo);
            heartBtn.textContent = goingTo ? '♥' : '♡';
          })
          .catch(function (err) { alert('Failed to update favorite: ' + err.message); });
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
            alert('Failed to update tag: ' + err.message);
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
            alert('Failed to reject tag: ' + err.message);
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
        alert('Failed to add tag: ' + err.message);
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
        alert('Failed to remove tag: ' + err.message);
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

  /* Label region (spatial tag) */
  const labelRegionBtn = document.getElementById('label-region-btn');
  const labelRegionStatus = document.getElementById('label-region-status');

  const labelRegionState = wireBoxDraw(labelRegionBtn, 'Click and drag on the photo…', '＋ Label region', function (bbox) {
    openEntitySearchModal({
      type: 'tag',
      title: 'Label this region',
      onResolved: function (entity) {
        if (labelRegionStatus) { labelRegionStatus.style.display = 'block'; labelRegionStatus.textContent = 'Saving…'; }
        fetch('/api/files/' + fileId + '/tags/region', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ label: entity.name, bbox: bbox }),
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
  if (labelRegionState) boxDrawStates.push(labelRegionState);

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
        .catch(function (err) { alert('Failed to save name: ' + err.message); });
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
          alert('Failed to reject face: ' + err.message);
        });
    });
  });

  /* "Find similar faces" (🔎 next to each face chip) is now a plain navigation
     link to /search?face_ref=... — see search.html for the results page, which
     reuses the same similar-faces grid/slider UI as searching by person name. */

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

  /* Category assignment — single-valued, unlike sets. Manual assign/clear
     always wins over whatever the ML auto-match set (see category_resolver.py). */
  const categoryCurrent = document.getElementById('category-current');
  const categoryPickerBtn = document.getElementById('category-picker-btn');

  function renderCategory(category) {
    if (!categoryCurrent) return;
    categoryCurrent.innerHTML = '';
    if (!category || !category.name) {
      const span = document.createElement('span');
      span.className = 'sub';
      span.textContent = 'Uncategorized.';
      categoryCurrent.appendChild(span);
      return;
    }
    const span = document.createElement('span');
    span.className = 'tag-removable';
    span.style.marginBottom = '4px';
    span.style.display = 'inline-flex';

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
    removeBtn.className = 'rm';
    removeBtn.type = 'button';
    removeBtn.id = 'category-remove-btn';
    removeBtn.title = 'Clear category';
    removeBtn.textContent = '×';
    removeBtn.addEventListener('click', clearFileCategory);
    span.appendChild(removeBtn);

    categoryCurrent.appendChild(span);
  }

  function assignFileCategory(categoryId) {
    return fetch('/api/files/' + fileId + '/category', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category_id: categoryId }),
    }).then(function (r) {
      if (!r.ok) throw new Error('Request failed: ' + r.status);
      return r.json();
    }).then(function (data) { renderCategory(data.category); });
  }

  function clearFileCategory() {
    fetch('/api/files/' + fileId + '/category', { method: 'DELETE' })
      .then(function (r) {
        if (!r.ok) throw new Error('Request failed: ' + r.status);
        return r.json();
      })
      .then(function () { renderCategory(null); })
      .catch(function (err) { alert('Failed to clear category: ' + err.message); });
  }

  function openCategoryPickerModal() {
    openEntitySearchModal({
      type: 'category',
      onResolved: function (entity) {
        assignFileCategory(entity.id)
          .catch(function (err) { alert('Failed to set category: ' + err.message); });
      },
    });
  }

  const categoryRemoveBtn = document.getElementById('category-remove-btn');
  if (categoryRemoveBtn) categoryRemoveBtn.addEventListener('click', clearFileCategory);
  if (categoryPickerBtn) categoryPickerBtn.addEventListener('click', openCategoryPickerModal);

  /* Set assignment — a photo can belong to any number of sets */
  const setCurrent = document.getElementById('set-current');
  const setPickerBtn = document.getElementById('set-picker-btn');

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
      span.dataset.setId = set.id;

      const link = document.createElement('a');
      link.href = '/sets/' + set.id;
      link.style.color = '#fff';
      link.textContent = set.name;
      span.appendChild(link);
      if (set.studio) {
        const studioSpan = document.createElement('span');
        studioSpan.style.opacity = '.8';
        studioSpan.textContent = ' (' + set.studio + ')';
        span.appendChild(studioSpan);
      }

      const heartBtn = document.createElement('button');
      heartBtn.type = 'button';
      heartBtn.className = 'heart-btn-inline set-heart-btn' + (set.favorite ? ' is-favorite' : '');
      heartBtn.style.fontSize = '0.85em';
      heartBtn.title = 'Favorite this set';
      heartBtn.textContent = set.favorite ? '♥' : '♡';
      span.appendChild(heartBtn);

      const removeBtn = document.createElement('button');
      removeBtn.className = 'rm set-remove-btn';
      removeBtn.type = 'button';
      removeBtn.title = 'Remove from this set';
      removeBtn.textContent = '×';
      span.appendChild(removeBtn);

      setCurrent.appendChild(span);
      wireSetChip(span);
    });
  }

  function assignSetById(setId) {
    return fetch('/api/files/' + fileId + '/sets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ set_id: setId }),
    }).then(function (r) {
      if (!r.ok) throw new Error('Request failed: ' + r.status);
      return r.json();
    });
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
    span.dataset.setId = set.id;
    const link = document.createElement('a');
    link.href = '/sets/' + set.id;
    link.style.color = '#fff';
    link.textContent = set.name;
    span.appendChild(link);
    if (set.studio) {
      const studioSpan = document.createElement('span');
      studioSpan.style.opacity = '.8';
      studioSpan.textContent = ' (' + set.studio + ')';
      span.appendChild(studioSpan);
    }
    const heartBtn = document.createElement('button');
    heartBtn.type = 'button';
    heartBtn.className = 'heart-btn-inline set-heart-btn';
    heartBtn.style.fontSize = '0.85em';
    heartBtn.title = 'Favorite this set';
    heartBtn.textContent = '♡';
    span.appendChild(heartBtn);
    const removeBtn = document.createElement('button');
    removeBtn.className = 'rm set-remove-btn';
    removeBtn.type = 'button';
    removeBtn.title = 'Remove from this set';
    removeBtn.textContent = '×';
    span.appendChild(removeBtn);
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
        alert('Failed to remove set: ' + err.message);
      });
  }

  function openSetPickerModal() {
    openSetSearchModal(function (set) {
      assignSetById(set.id)
        .then(function (data) { appendSetChip(data); })
        .catch(function (err) { alert('Failed to add set: ' + err.message); });
    });
  }

  if (setPickerBtn) {
    setPickerBtn.addEventListener('click', openSetPickerModal);
  }
  if (setCurrent) {
    setCurrent.querySelectorAll('[data-set-id]').forEach(wireSetChip);
  }

  /* Suggested sets — CLIP centroid match, fetched lazily so the page render
     itself isn't blocked on scanning every set. Renders nothing when there's
     nothing worth suggesting (no embedding yet, no sets, already in every
     set, or nothing clears the match threshold) — this mirrors auto-detected
     tags/faces elsewhere in the app, which stay silent rather than nagging. */
  const setSuggestions = document.getElementById('set-suggestions');
  if (setSuggestions && fileId) {
    fetch('/api/files/' + fileId + '/suggested-sets')
      .then(function (r) { return r.ok ? r.json() : { results: [] }; })
      .then(function (data) {
        (data.results || []).forEach(function (set) {
          const chip = document.createElement('a');
          chip.className = 'chip-set';
          chip.href = '#';
          chip.style.marginRight = '6px';
          chip.style.marginBottom = '4px';
          chip.style.display = 'inline-flex';
          chip.style.alignItems = 'center';
          chip.style.gap = '4px';
          chip.title = 'Suggested match — click to add';
          chip.dataset.setId = set.id;

          const nameSpan = document.createElement('span');
          nameSpan.textContent = set.name + (set.studio ? ' (' + set.studio + ')' : '');
          chip.appendChild(nameSpan);

          const scoreSpan = document.createElement('span');
          scoreSpan.className = 'score-badge';
          scoreSpan.style.position = 'static';
          scoreSpan.textContent = set.score.toFixed(2);
          chip.appendChild(scoreSpan);

          chip.addEventListener('click', function (e) {
            e.preventDefault();
            chip.style.pointerEvents = 'none';
            assignSetById(set.id)
              .then(function (data) {
                appendSetChip(data);
                chip.remove();
              })
              .catch(function (err) {
                chip.style.pointerEvents = '';
                alert('Failed to add set: ' + err.message);
              });
          });

          setSuggestions.appendChild(chip);
        });
      })
      .catch(function () {
        // Non-critical: leave the block empty rather than surfacing an error
        // for what is, at worst, a missed convenience suggestion.
      });
  }

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
            alert('Failed to save title: ' + err.message);
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

    function setFit(key) {
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
    ['add-face-btn', 'label-region-btn'].forEach(function (id) {
      const btn = document.getElementById(id);
      if (btn) btn.addEventListener('click', function () {
        if (photoStage.classList.contains('fit-cover')) setFit('fit');
      });
    });

    const sidebarOpenBtn = document.getElementById('sidebar-open-btn');
    const sidebarCloseBtn = document.getElementById('sidebar-close-btn');
    function setSidebar(open) {
      photoStage.classList.toggle('sidebar-open', open);
      if (sidebarOpenBtn) sidebarOpenBtn.style.display = open ? 'none' : '';
      try { localStorage.setItem('mm_sidebar', open ? '1' : '0'); } catch (e) {}
    }
    let sidebarPref = '1';
    try { sidebarPref = localStorage.getItem('mm_sidebar') || '1'; } catch (e) {}
    setSidebar(sidebarPref === '1');
    if (sidebarOpenBtn) sidebarOpenBtn.addEventListener('click', function () { setSidebar(true); });
    if (sidebarCloseBtn) sidebarCloseBtn.addEventListener('click', function () { setSidebar(false); });

    window.addEventListener('keydown', function (e) {
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      if (e.key === 'f' || e.key === 'F') cycleFit();
      if (e.key === 'Escape') setSidebar(!photoStage.classList.contains('sidebar-open'));
    });
  }

})();
