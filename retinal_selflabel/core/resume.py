import hashlib
import json
import os

def resume_dir_for(out_dir):
    d = os.path.join(out_dir, "_resume")
    os.makedirs(d, exist_ok=True)
    return d

def cache_key(*parts):
    raw = "__".join(str(p) for p in parts)
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in raw)
    if len(safe) > 150:
        safe = safe[:120] + "_" + hashlib.md5(raw.encode()).hexdigest()[:8]
    return safe

def cached_json(resume_dir, key, compute_fn):
    path = os.path.join(resume_dir, key + ".json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f), True
        except Exception:
            pass  # recompute
    result = compute_fn()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp, path)  # atomic on the same filesystem
    return result, False

def is_done(resume_dir, key):
    return os.path.exists(os.path.join(resume_dir, key + ".json"))
