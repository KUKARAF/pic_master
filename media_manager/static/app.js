/* app.js — photo detail page tag management */

(function () {
  'use strict';

  const fileId = window.MEDIA_FILE_ID;
  if (!fileId) return;

  const tagList = document.getElementById('tag-list');
  const tagInput = document.getElementById('tag-input');
  const tagForm = document.getElementById('tag-form');

  function renderTags(tags) {
    tagList.innerHTML = '';
    tags.forEach(function (tag) {
      const span = document.createElement('span');
      span.className = 'tag-removable';
      span.dataset.tag = tag;

      const label = document.createTextNode(tag);
      span.appendChild(label);

      const btn = document.createElement('button');
      btn.className = 'rm';
      btn.type = 'button';
      btn.textContent = '×';
      btn.title = 'Remove tag';
      btn.addEventListener('click', function () {
        removeTag(tag);
      });

      span.appendChild(btn);
      tagList.appendChild(span);
    });
  }

  function addTag(tag) {
    fetch('/api/files/' + fileId + '/tags', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tag: tag }),
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

  function removeTag(tag) {
    fetch('/api/files/' + fileId + '/tags/' + encodeURIComponent(tag), {
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
      if (tag) addTag(tag);
    });
  }

  /* Tags dropdown toggle in nav */
  var tagsBtn = document.getElementById('tags-dropdown-btn');
  var tagsMenu = document.getElementById('tags-dropdown-menu');
  if (tagsBtn && tagsMenu) {
    tagsBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      tagsMenu.classList.toggle('open');
    });
    document.addEventListener('click', function () {
      if (tagsMenu) tagsMenu.classList.remove('open');
    });
  }
})();
