#!/usr/bin/env bash
# Bulk-upload a folder of photos/videos to a running `media web` server.
#
# Every file is POSTed on its own to /api/files/upload, which stores it under
# uploads/<bucket>/ in the library and REJECTS content the library already has
# (same full-file hash). With --delete-uploaded, a local file is removed only after
# the server confirmed it saved it — never on a failure or a rejection.
#
# Usage: ./upload_folder.sh [options] FOLDER
#   -u, --url URL            server base URL (default: $MEDIA_URL or https://127.0.0.1:8000)
#   -d, --delete-uploaded    delete each local file once the server has saved it
#       --delete-duplicates  with -d: also delete files the library already has
#   -n, --no-recursive       only FOLDER itself, not its subfolders
#       --strict-tls         verify the server certificate (default: accept the
#                            self-signed cert `media web` generates)
#       --dry-run            list what would be uploaded, upload nothing
#   -h, --help               show this help
set -euo pipefail

URL="${MEDIA_URL:-https://127.0.0.1:8000}"
DELETE=0
DELETE_DUPS=0
RECURSIVE=1
INSECURE=1
DRY_RUN=0
FOLDER=""

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -u|--url) URL="${2:?--url needs a value}"; shift 2 ;;
        -d|--delete-uploaded) DELETE=1; shift ;;
        --delete-duplicates) DELETE_DUPS=1; shift ;;
        -n|--no-recursive) RECURSIVE=0; shift ;;
        --strict-tls) INSECURE=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; FOLDER="${1:-}"; break ;;
        -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
        *) [ -n "$FOLDER" ] && { echo "Only one FOLDER, please." >&2; exit 2; }; FOLDER="$1"; shift ;;
    esac
done

[ -n "$FOLDER" ] || { usage >&2; exit 2; }
[ -d "$FOLDER" ] || { echo "Not a folder: $FOLDER" >&2; exit 2; }
if [ "$DELETE_DUPS" -eq 1 ] && [ "$DELETE" -eq 0 ]; then
    echo "--delete-duplicates only makes sense together with --delete-uploaded" >&2; exit 2
fi
command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }

ENDPOINT="${URL%/}/api/files/upload"
CURL_OPTS=(-sS --max-time 3600)
[ "$INSECURE" -eq 1 ] && CURL_OPTS+=(-k)

# Same media extensions the library tracks (media_manager/formats.py).
FIND_ARGS=("$FOLDER")
[ "$RECURSIVE" -eq 0 ] && FIND_ARGS+=(-maxdepth 1)
FIND_ARGS+=(-type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.gif'
    -o -iname '*.bmp' -o -iname '*.webp' -o -iname '*.tif' -o -iname '*.tiff'
    -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.m4v' \)
    -not -path '*/.media/*' -print0)

RESP="$(mktemp)"
trap 'rm -f "$RESP"' EXIT

saved=0; dups=0; failed=0; deleted=0; total=0
while IFS= read -r -d '' f; do
    total=$((total + 1))
    if [ "$DRY_RUN" -eq 1 ]; then echo "would upload: $f"; continue; fi
    # curl's -F treats ; and , in a bare filename as separators; quoting the
    # path (with " and \ escaped) makes any filename safe.
    q="${f//\\/\\\\}"; q="${q//\"/\\\"}"
    # On a connection failure curl still prints 000 for %{http_code} (and exits non-zero).
    code="$(curl "${CURL_OPTS[@]}" -o "$RESP" -w '%{http_code}' -F "file=@\"$q\"" "$ENDPOINT")" || true
    case "$code" in
        200)
            saved=$((saved + 1))
            echo "saved      $f"
            if [ "$DELETE" -eq 1 ]; then rm -f -- "$f" && deleted=$((deleted + 1)); fi
            ;;
        409)
            dups=$((dups + 1))
            echo "duplicate  $f"
            if [ "$DELETE_DUPS" -eq 1 ]; then rm -f -- "$f" && deleted=$((deleted + 1)); fi
            ;;
        000)
            failed=$((failed + 1))
            echo "FAILED     $f (could not reach $ENDPOINT)" >&2
            ;;
        *)
            failed=$((failed + 1))
            echo "FAILED     $f (HTTP $code: $(head -c 300 "$RESP"))" >&2
            ;;
    esac
done < <(find "${FIND_ARGS[@]}")

if [ "$DRY_RUN" -eq 1 ]; then
    echo "$total file(s) would be uploaded to $ENDPOINT"
    exit 0
fi
echo "---"
echo "$total file(s): $saved saved, $dups already in library, $failed failed, $deleted deleted locally"
[ "$failed" -eq 0 ]
