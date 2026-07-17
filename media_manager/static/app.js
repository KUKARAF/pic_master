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
  wireDropdown('tags-dropdown-btn', 'tags-dropdown-menu');
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

  function openSetSearchModal(onResolved) {
    openModal('Choose a set', function (box) {
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'Search or create a set…';
      input.setAttribute('list', 'set-picker-datalist');
      input.autocomplete = 'off';
      input.style.width = '100%';
      input.style.fontSize = '1.2rem';
      input.style.padding = '10px 12px';
      input.style.marginBottom = '8px';
      box.appendChild(input);

      const datalist = document.createElement('datalist');
      datalist.id = 'set-picker-datalist';
      box.appendChild(datalist);

      const status = document.createElement('div');
      status.className = 'modal-empty';
      status.textContent = 'Loading sets…';
      box.appendChild(status);

      let setsByLabel = {};

      function submit() {
        const value = input.value.trim();
        if (!value || input.disabled) return;
        input.disabled = true;
        const match = setsByLabel[value.toLowerCase()];
        if (match) {
          closeModal();
          onResolved(match);
          return;
        }
        // No existing set matches — this is a brand-new set, so ask for its
        // (optional) studio before creating it, rather than assuming none.
        openStudioPromptModal(value, onResolved);
      }

      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); submit(); }
      });

      fetch('/api/sets')
        .then(function (r) { return r.json(); })
        .then(function (sets) {
          setsByLabel = {};
          datalist.innerHTML = '';
          sets.forEach(function (s) {
            const label = s.name + (s.studio ? ' (' + s.studio + ')' : '');
            setsByLabel[label.toLowerCase()] = s;
            const opt = document.createElement('option');
            opt.value = label;
            datalist.appendChild(opt);
          });
          status.textContent = sets.length
            ? 'Type to search ' + sets.length + ' existing set(s) — Enter to add, or type a new name and press Enter to create it.'
            : 'No sets yet — type a name and press Enter to create one.';
        })
        .catch(function () {
          status.textContent = 'Failed to load existing sets — you can still type a new name and press Enter.';
        });

      setTimeout(function () { input.focus(); }, 0);
    });
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

  const fileId = window.MEDIA_FILE_ID;
  if (!fileId) return;

  /* Tag/region-label autocomplete — every label YOLO-World already knows about
     (search_terms.txt + manual.db's confirmed tags), shared by both inputs. */
  const vocabDatalist = document.getElementById('vocab-suggestions');
  if (vocabDatalist) {
    fetch('/api/vocab')
      .then(function (r) { return r.json(); })
      .then(function (data) {
        (data.vocab || []).forEach(function (term) {
          const opt = document.createElement('option');
          opt.value = term;
          vocabDatalist.appendChild(opt);
        });
      })
      .catch(function () { /* autocomplete is a nice-to-have, fail silently */ });
  }

  /* Arrow-key photo navigation. History (the sequence of photos actually visited,
     not just id order) is kept in localStorage so Left/Right behave like browser
     back/forward across full page loads, not just simple id+1/id-1 stepping. */
  (function () {
    const STORAGE_KEY = 'mediaPhotoHistory';
    const MAX_HISTORY = 500;
    const currentId = parseInt(fileId, 10);

    function loadHistory() {
      try {
        const raw = JSON.parse(localStorage.getItem(STORAGE_KEY));
        if (raw && Array.isArray(raw.ids) && typeof raw.cursor === 'number') return raw;
      } catch (e) { /* ignore corrupt storage */ }
      return { ids: [], cursor: -1 };
    }

    function saveHistory(hist) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(hist));
    }

    let hist = loadHistory();
    const existingIndex = hist.ids.indexOf(currentId);
    if (existingIndex !== -1) {
      hist.cursor = existingIndex;
    } else {
      // Arrived here via a link/search, not via arrow nav — drop any stale
      // "forward" entries past the current position, then record this visit.
      hist.ids = hist.ids.slice(0, hist.cursor + 1);
      hist.ids.push(currentId);
      hist.cursor = hist.ids.length - 1;
      if (hist.ids.length > MAX_HISTORY) {
        const drop = hist.ids.length - MAX_HISTORY;
        hist.ids = hist.ids.slice(drop);
        hist.cursor -= drop;
      }
    }
    saveHistory(hist);

    function goTo(id) {
      window.location.href = '/photo/' + id;
    }

    function isTypingTarget(el) {
      if (!el) return false;
      const tag = el.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
    }

    document.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      if (isTypingTarget(document.activeElement)) return;
      if (e.altKey || e.ctrlKey || e.metaKey) return;

      if (e.key === 'ArrowLeft') {
        if (hist.cursor > 0) {
          hist.cursor -= 1;
          saveHistory(hist);
          goTo(hist.ids[hist.cursor]);
        } else {
          fetch('/api/files/' + currentId + '/neighbors')
            .then(function (r) { return r.json(); })
            .then(function (data) {
              if (data.prev == null) return;
              hist.ids.unshift(data.prev);
              hist.cursor = 0;
              saveHistory(hist);
              goTo(data.prev);
            });
        }
      } else {
        if (hist.cursor < hist.ids.length - 1) {
          hist.cursor += 1;
          saveHistory(hist);
          goTo(hist.ids[hist.cursor]);
        } else {
          fetch('/api/files/' + currentId + '/neighbors')
            .then(function (r) { return r.json(); })
            .then(function (data) {
              if (data.next == null) return;
              hist.ids.push(data.next);
              hist.cursor = hist.ids.length - 1;
              saveHistory(hist);
              goTo(data.next);
            });
        }
      }
    });
  })();

  const tagList = document.getElementById('tag-list');
  const tagInput = document.getElementById('tag-input');
  const tagForm = document.getElementById('tag-form');

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
        if (tagInput) tagInput.value = '';
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

  if (tagForm) {
    tagForm.addEventListener('submit', function (e) {
      e.preventDefault();
      const tag = (tagInput ? tagInput.value : '').trim();
      if (tag) addTag(tag, 'positive');
    });
  }

  const tagNegativeBtn = document.getElementById('tag-negative-btn');
  if (tagNegativeBtn) {
    tagNegativeBtn.addEventListener('click', function () {
      const tag = (tagInput ? tagInput.value : '').trim();
      if (tag) addTag(tag, 'negative');
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
  const regionLabelForm = document.getElementById('region-label-form');
  const regionLabelInput = document.getElementById('region-label-input');
  let pendingRegionBbox = null;

  const labelRegionState = wireBoxDraw(labelRegionBtn, 'Click and drag on the photo…', '＋ Label region', function (bbox) {
    pendingRegionBbox = bbox;
    if (regionLabelForm) {
      regionLabelForm.style.display = 'flex';
      if (regionLabelInput) regionLabelInput.focus();
    }
  });
  if (labelRegionState) boxDrawStates.push(labelRegionState);

  if (regionLabelForm) {
    regionLabelForm.addEventListener('submit', function (e) {
      e.preventDefault();
      const label = (regionLabelInput ? regionLabelInput.value : '').trim();
      if (!label || !pendingRegionBbox) return;
      if (labelRegionStatus) { labelRegionStatus.style.display = 'block'; labelRegionStatus.textContent = 'Saving…'; }
      fetch('/api/files/' + fileId + '/tags/region', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: label, bbox: pendingRegionBbox }),
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
    });
  }

  /* Face naming modal — click any face chip (named or "Unknown") to name/rename it.
     Keyboard-first, same shape as the set picker: one big autofocused input backed
     by a native <datalist> of every known person; Enter saves whatever's typed,
     whether it matches an existing person (renames onto them) or is brand new
     (naming doesn't need a separate create step — identity is just a string). An
     embedding-similarity suggestion, if any, shows as a one-click shortcut below. */
  function openFaceNamingModal(faceRef) {
    openModal('Name this face', function (box) {
      const thumb = document.createElement('img');
      thumb.src = '/face-crop/' + faceRef;
      thumb.width = 80;
      thumb.height = 80;
      thumb.style.borderRadius = '6px';
      thumb.style.display = 'block';
      thumb.style.marginBottom = '10px';
      box.appendChild(thumb);

      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'Search or type a name…';
      input.setAttribute('list', 'face-naming-datalist');
      input.autocomplete = 'off';
      input.style.width = '100%';
      input.style.fontSize = '1.2rem';
      input.style.padding = '10px 12px';
      input.style.marginBottom = '8px';
      box.appendChild(input);

      const datalist = document.createElement('datalist');
      datalist.id = 'face-naming-datalist';
      box.appendChild(datalist);

      const status = document.createElement('div');
      status.className = 'modal-empty';
      status.textContent = 'Loading known people…';
      box.appendChild(status);

      const suggestBox = document.createElement('div');
      suggestBox.style.marginTop = '8px';
      box.appendChild(suggestBox);

      function saveName(name) {
        if (!name || input.disabled) return;
        input.disabled = true;
        fetch('/api/faces/' + faceRef + '/identity', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name }),
        })
          .then(function (r) {
            if (!r.ok) throw new Error('Request failed: ' + r.status);
            return r.json();
          })
          .then(function () {
            closeModal();
            location.reload();
          })
          .catch(function (err) {
            input.disabled = false;
            alert('Failed to save name: ' + err.message);
          });
      }

      input.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') { e.preventDefault(); saveName(input.value.trim()); }
      });

      fetch('/api/identities')
        .then(function (r) { return r.json(); })
        .then(function (identities) {
          datalist.innerHTML = '';
          identities.forEach(function (i) {
            const opt = document.createElement('option');
            opt.value = i.name;
            datalist.appendChild(opt);
          });
          status.textContent = identities.length
            ? 'Type to search ' + identities.length + ' known people — Enter to name/rename this face.'
            : 'No known people yet — type a name and press Enter.';
        })
        .catch(function () {
          status.textContent = 'Failed to load known people — you can still type a name and press Enter.';
        });

      fetch('/api/faces/' + faceRef + '/suggestions')
        .then(function (r) { return r.json(); })
        .then(function (data) {
          const top = data.suggestions && data.suggestions[0];
          if (!top) return;
          suggestBox.innerHTML = '';
          const suggestBtn = document.createElement('button');
          suggestBtn.type = 'button';
          suggestBtn.className = 'btn-similar';
          suggestBtn.style.fontSize = '0.85em';
          suggestBtn.textContent = 'Looks like ' + top.name + '? (' + top.score.toFixed(2) + ')';
          suggestBtn.addEventListener('click', function () { saveName(top.name); });
          suggestBox.appendChild(suggestBtn);
        })
        .catch(function () {});

      setTimeout(function () { input.focus(); }, 0);
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
            let text = s.pending + ' photo(s) are not in the body index yet.';
            if (s.no_detections > 0) {
              text += ' (' + s.no_detections + ' more need object indexing first \u2014 run media index.)';
            }
            bodyIndexText.textContent = text;
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
