"""Resolve the media-worker connection config.

The worker config lives at ``<data_root>/.media/worker.json`` (alongside
media.db / error.db). It records the RNS destination *address* (hex hash) of the
worker to offload heavy ML to, and whether offloading is *enabled*.

Resolution order is env-overrides-file so a deployment can point at a different
worker without editing the repo:
    MEDIA_WORKER_ADDR    overrides the address
    MEDIA_WORKER_ENABLED overrides the enabled flag

Per this project's "no silent failures" rule, a present-but-corrupt worker.json
is surfaced as a ValueError (with its path), never silently ignored.
"""
import json
import os

WORKER_JSON_NAME = "worker.json"


def config_path(data_root: str) -> str:
    """Absolute path to the worker config file for the given data root."""
    return os.path.join(data_root, '.media', WORKER_JSON_NAME)


def load(data_root: str) -> dict:
    """Load worker config, applying env overrides over the on-disk file.

    Returns a dict with keys ``address`` (str or None) and ``enabled`` (bool).
    A missing file yields defaults; a present-but-unparseable file raises
    ValueError naming the path (never silently swallowed).
    """
    path = config_path(data_root)
    data = {}
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Failed to read worker config at {path}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(
                f"Worker config at {path} must be a JSON object, got {type(data).__name__}")

    address = data.get('address')
    enabled = data.get('enabled')

    env_addr = os.environ.get('MEDIA_WORKER_ADDR')
    if env_addr:
        address = env_addr

    env_enabled = os.environ.get('MEDIA_WORKER_ENABLED')
    if env_enabled is not None:
        enabled = env_enabled.lower() not in ('0', 'false', 'no', '')

    if enabled is None:
        enabled = bool(address)

    return {'address': address, 'enabled': bool(enabled)}


def save(data_root: str, address: str, enabled: bool = True) -> str:
    """Write the worker config file atomically; return its path.

    Creates the parent .media directory if it does not already exist.
    """
    path = config_path(data_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {'address': address, 'enabled': bool(enabled)}
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def address_hash_bytes(address: str) -> bytes:
    """Convert a hex RNS destination-hash string to raw bytes.

    RNS destination hashes are 16 bytes (32 hex chars). Raises ValueError with a
    helpful message if the string is not valid hex or not the expected length.
    """
    try:
        raw = bytes.fromhex(address)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Worker address {address!r} is not a valid hex string: {e}") from e
    if len(raw) != 16:
        raise ValueError(
            f"Worker address {address!r} must be 16 bytes (32 hex chars), "
            f"got {len(raw)} bytes")
    return raw
