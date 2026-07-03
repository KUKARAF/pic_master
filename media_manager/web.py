"""
FastAPI web server for media_manager — gallery UI with CLIP search, tags, and similar-image discovery.
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

THUMB_SIZE = (400, 400)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

_HERE = Path(__file__).parent


def _gray_placeholder() -> bytes:
    """Return a gray 400×400 JPEG as bytes (used when thumbnail generation fails)."""
    from PIL import Image as PILImage
    img = PILImage.new('RGB', (400, 400), color=(180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)
    return buf.getvalue()


def _make_thumbnail(src_path: str, dst_path: str) -> bool:
    """Resize src to 400 px wide, save as JPEG at dst_path. Returns True on success."""
    from PIL import Image as PILImage
    try:
        with PILImage.open(src_path) as img:
            img = img.convert('RGB')
            w, h = img.size
            if w == 0:
                return False
            new_w = 400
            new_h = max(1, int(h * new_w / w))
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            img.save(dst_path, format='JPEG', quality=80)
            return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(data_root: str) -> FastAPI:
    """Create and return the FastAPI application rooted at data_root."""
    data_root = os.path.abspath(data_root)
    media_dir = os.path.join(data_root, '.media')

    if not os.path.isdir(media_dir):
        raise RuntimeError(
            f"No media repository found at '{data_root}'. "
            "Run 'media init' inside that directory first."
        )

    db_path = os.path.join(media_dir, 'media.db')
    thumbs_dir = os.path.join(media_dir, 'thumbs')
    os.makedirs(thumbs_dir, exist_ok=True)

    # Import here so the module can be imported without heavy deps loaded
    from media_manager.database import Database

    db = Database(db_path)

    app = FastAPI(title='media gallery')

    # Static files and templates
    static_dir = _HERE / 'static'
    templates_dir = _HERE / 'templates'

    app.mount('/static', StaticFiles(directory=str(static_dir)), name='static')
    templates = Jinja2Templates(directory=str(templates_dir))

    # ------------------------------------------------------------------
    # Internal helpers bound to this app instance
    # ------------------------------------------------------------------

    def _file_or_404(file_id: int):
        row = db.get_file_by_id(file_id)
        if row is None:
            raise HTTPException(status_code=404, detail='File not found')
        return row

    def _row_to_dict(row, tags=None):
        return {
            'id': row['id'],
            'path': row['path'],
            'size': row['size'],
            'modified_time': row['modified_time'],
            'checksum': row['checksum'],
            'last_hashed': row['last_hashed'],
            'tags': tags if tags is not None else [],
        }

    def _enrich_rows(rows):
        """Add tags to a list of (id, path, has_embedding) rows."""
        result = []
        for row in rows:
            file_id = row[0]
            path = row[1]
            has_embedding = row[2] if len(row) > 2 else 0
            tags = db.get_tags(file_id)
            result.append({
                'id': file_id,
                'path': path,
                'filename': os.path.basename(path),
                'has_embedding': bool(has_embedding),
                'tags': tags,
            })
        return result

    # ------------------------------------------------------------------
    # Thumbnail / image serving
    # ------------------------------------------------------------------

    @app.get('/thumb/{file_id}')
    def serve_thumb(file_id: int):
        row = db.get_file_by_id(file_id)
        if row is None:
            return Response(content=_gray_placeholder(), media_type='image/jpeg', status_code=404)

        rel_path = row['path']
        ext = os.path.splitext(rel_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            return Response(content=_gray_placeholder(), media_type='image/jpeg')

        abs_path = os.path.join(data_root, rel_path)
        thumb_path = os.path.join(thumbs_dir, f'{file_id}.jpg')

        if not os.path.isfile(thumb_path):
            if not os.path.isfile(abs_path) or not _make_thumbnail(abs_path, thumb_path):
                return Response(content=_gray_placeholder(), media_type='image/jpeg')

        return FileResponse(thumb_path, media_type='image/jpeg')

    @app.get('/image/{file_id}')
    def serve_image(file_id: int):
        row = _file_or_404(file_id)
        abs_path = os.path.join(data_root, row['path'])
        if not os.path.isfile(abs_path):
            raise HTTPException(status_code=404, detail='Image file not found on disk')
        return FileResponse(abs_path)

    # ------------------------------------------------------------------
    # HTML pages
    # ------------------------------------------------------------------

    @app.get('/', response_class=HTMLResponse)
    def gallery_page(request: Request, page: int = 1):
        limit = 60
        offset = (page - 1) * limit
        rows = db.list_files_with_embedding_flag(limit=limit, offset=offset)
        files = _enrich_rows(rows)
        total = db.count_files()
        all_tags = db.list_all_tags()
        return templates.TemplateResponse('gallery.html', {
            'request': request,
            'files': files,
            'page': page,
            'total': total,
            'limit': limit,
            'all_tags': all_tags,
        })

    @app.get('/photo/{file_id}', response_class=HTMLResponse)
    def photo_page(request: Request, file_id: int):
        row = _file_or_404(file_id)
        tags = db.get_tags(file_id)
        all_tags = db.list_all_tags()
        detected_classes = db.get_detected_classes(file_id)
        file_info = dict(row)
        file_info['filename'] = os.path.basename(file_info['path'])
        file_info['tags'] = tags
        ext = os.path.splitext(file_info['path'])[1].lower()
        file_info['is_image'] = ext in IMAGE_EXTENSIONS
        return templates.TemplateResponse('photo.html', {
            'request': request,
            'file': file_info,
            'detected_classes': detected_classes,
            'all_tags': all_tags,
        })

    @app.get('/search', response_class=HTMLResponse)
    def search_page(request: Request, q: str = '', tag: str = '', face_id: str = ''):
        files = []
        message = ''
        all_tags = db.list_all_tags()

        if face_id:
            message = 'Face search is coming soon — run <code>media faces</code> to detect faces first.'

        elif tag:
            rows = db.get_files_by_tag(tag, limit=200)
            # rows are (file_id, path) — no embedding flag
            for file_id, path in rows:
                tags = db.get_tags(file_id)
                files.append({
                    'id': file_id,
                    'path': path,
                    'filename': os.path.basename(path),
                    'has_embedding': False,
                    'tags': tags,
                })

        elif q:
            # YOLO-World object detection search
            try:
                from media_manager.media_manager import _parse_query
                tokens = _parse_query(q)
                if not tokens:
                    message = 'No searchable terms found in query.'
                else:
                    rows = db.search_by_classes(tokens, limit=50)
                    if not rows:
                        message = 'No matches found — run <code>media index</code> to detect objects first.'
                    else:
                        for file_id, path, score in rows:
                            tags = db.get_tags(file_id)
                            files.append({
                                'id': file_id,
                                'path': path,
                                'filename': os.path.basename(path),
                                'has_embedding': True,
                                'tags': tags,
                                'score': round(score, 4),
                            })
            except Exception as exc:
                message = f'Search error: {exc}'

        return templates.TemplateResponse('search.html', {
            'request': request,
            'files': files,
            'q': q,
            'tag': tag,
            'face_id': face_id,
            'message': message,
            'all_tags': all_tags,
        })

    @app.get('/similar/{file_id}', response_class=HTMLResponse)
    def similar_page(request: Request, file_id: int):
        row = _file_or_404(file_id)
        filename = os.path.basename(row['path'])
        files = []
        message = ''
        all_tags = db.list_all_tags()

        emb_bytes = db.get_embedding(file_id)
        if emb_bytes is None:
            message = (
                f'"{filename}" has no embedding yet — '
                'run <code>media index</code> to build CLIP embeddings.'
            )
        else:
            try:
                import numpy as np
                query_emb = np.frombuffer(emb_bytes, dtype=np.float32)

                all_embs = db.get_all_embeddings()
                matrix = np.stack([
                    np.frombuffer(r[2], dtype=np.float32) for r in all_embs
                ])
                scores = matrix.dot(query_emb).tolist()
                ranked = sorted(
                    zip([r[0] for r in all_embs], [r[1] for r in all_embs], scores),
                    key=lambda x: x[2],
                    reverse=True,
                )
                # skip self
                ranked = [(fid, path, score) for fid, path, score in ranked if fid != file_id][:20]

                for fid, path, score in ranked:
                    tags = db.get_tags(fid)
                    files.append({
                        'id': fid,
                        'path': path,
                        'filename': os.path.basename(path),
                        'has_embedding': True,
                        'tags': tags,
                        'score': round(score, 4),
                    })
            except Exception as exc:
                message = f'Similarity search error: {exc}'

        return templates.TemplateResponse('similar.html', {
            'request': request,
            'source_file': dict(row),
            'source_filename': filename,
            'source_id': file_id,
            'files': files,
            'message': message,
            'all_tags': all_tags,
        })

    # ------------------------------------------------------------------
    # JSON API
    # ------------------------------------------------------------------

    @app.get('/api/files')
    def api_list_files(limit: int = 100, offset: int = 0):
        rows = db.list_files_with_embedding_flag(limit=limit, offset=offset)
        result = []
        for row in rows:
            file_id, path, has_emb = row[0], row[1], row[2]
            tags = db.get_tags(file_id)
            result.append({
                'id': file_id,
                'path': path,
                'has_embedding': bool(has_emb),
                'tags': tags,
            })
        return result

    @app.get('/api/files/{file_id}')
    def api_get_file(file_id: int):
        row = _file_or_404(file_id)
        tags = db.get_tags(file_id)
        info = _row_to_dict(row, tags)
        detected = db.get_detected_classes(file_id)
        info['detected_classes'] = detected
        return info

    class TagBody(BaseModel):
        tag: str

    @app.post('/api/files/{file_id}/tags')
    def api_add_tag(file_id: int, body: TagBody):
        _file_or_404(file_id)
        tag = body.tag.strip()
        if not tag:
            raise HTTPException(status_code=400, detail='Tag must not be empty')
        db.add_tag(file_id, tag)
        return {'tags': db.get_tags(file_id)}

    @app.delete('/api/files/{file_id}/tags/{tag}')
    def api_remove_tag(file_id: int, tag: str):
        _file_or_404(file_id)
        db.remove_tag(file_id, tag)
        return {'tags': db.get_tags(file_id)}

    @app.get('/api/tags')
    def api_list_tags():
        rows = db.list_all_tags()
        return [{'tag': r[0], 'count': r[1]} for r in rows]

    @app.get('/api/faces')
    def api_list_faces():
        return {
            'faces': [],
            'message': 'Face indexing is not implemented yet. Run `media faces` when available.',
        }

    return app
