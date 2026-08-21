"""Offline city reference data from GeoNames (cities15000, CC-BY 4.0).

Downloaded once via `media geo fetch-cities` into a `cities` table in media.db, so a
photo can be labeled with the NAME of its nearest known city instead of raw
coordinates — with no runtime network/CDN calls (same offline, no-data-leak rule as
the /locations map). Attribution: this data is from GeoNames
(https://www.geonames.org/), licensed CC-BY 4.0.

This module only downloads + parses the dump; the `cities` table, nearest-city lookup
and name search live on Database (database.py). haversine() is shared here so both the
reverse-geocode (nearest_city) and the search's distance sort can use one implementation.
"""
import io
import math
import urllib.request
import zipfile

# "cities with a population > 15000" (~26k rows) — the big-cities set. Swap the tail
# for cities5000 / cities1000 to trade coverage for size (see tag_retraining-style notes).
CITIES15000_URL = 'https://download.geonames.org/export/dump/cities15000.zip'


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in kilometres."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def parse_dump(text):
    """Yield (name, lat, lon, country, admin1, population) tuples from a GeoNames
    cities dump (tab-separated, no header). Relevant 0-indexed columns:
    1=name, 4=latitude, 5=longitude, 8=country code, 10=admin1 code, 14=population.
    Rows with an unparseable lat/lon are skipped (loudly counted by the caller)."""
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) < 15:
            continue
        try:
            lat = float(fields[4])
            lon = float(fields[5])
        except ValueError:
            continue
        try:
            population = int(fields[14]) if fields[14] else 0
        except ValueError:
            population = 0
        yield (fields[1], lat, lon, fields[8], fields[10], population)


def fetch_cities(db, url=CITIES15000_URL, log=print):
    """Download the GeoNames cities dump and (re)populate the media.db `cities` table
    via db.replace_cities(). Wipes and rebuilds, so a re-run always reflects the current
    dump. Returns the number of cities loaded. Raises on any network/zip/parse failure —
    loud, never a silent half-built table."""
    log(f'[geo] downloading {url} ...')
    with urllib.request.urlopen(url, timeout=180) as resp:  # noqa: S310 (fixed GeoNames host)
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        txt_name = next(n for n in zf.namelist() if n.endswith('.txt'))
        text = zf.read(txt_name).decode('utf-8')
    rows = list(parse_dump(text))
    if not rows:
        raise RuntimeError(f'GeoNames dump at {url} parsed to zero cities — refusing to wipe the table')
    db.replace_cities(rows)
    log(f'[geo] loaded {len(rows)} cities into media.db')
    return len(rows)
