"""
The ten-second look.

This runs every ten minutes and does almost nothing: it asks the shop
database whether any frame is waiting for a cut-out, and writes down that
the robot is alive. Only if something IS waiting does the real cutter get
woken, with its half-gigabyte of libraries and its model.

Deliberately written with nothing but what Python already has, so this
step needs no installing at all. An idle day therefore costs seconds, not
minutes -- which is what makes running every ten minutes sensible.
"""

import datetime
import json
import os
import urllib.error
import urllib.request

SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_KEY = (os.environ.get("SUPABASE_KEY") or "").strip()


def key_role():
    try:
        import base64
        part = SUPABASE_KEY.split(".")[1]
        part += "=" * (-len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part)).get("role", "?")
    except Exception:
        return "not a legacy JWT key"
MAX_TRIES = int(os.environ.get("MAX_TRIES", "3"))


def call(path, method="GET", body=None, prefer=None):
    req = urllib.request.Request(SUPABASE_URL + path, method=method)
    req.add_header("apikey", SUPABASE_KEY)
    req.add_header("Authorization", "Bearer " + SUPABASE_KEY)
    req.add_header("Content-Type", "application/json")
    if prefer:
        req.add_header("Prefer", prefer)
    data = json.dumps(body).encode() if body is not None else None
    try:
        with urllib.request.urlopen(req, data, timeout=45) as r:
            text = r.read().decode() or "[]"
    except urllib.error.HTTPError as e:
        raise RuntimeError("database answered %d: %s" % (e.code, e.read().decode()[:200]))
    try:
        return json.loads(text)
    except Exception:
        return []


def waiting(p):
    if p.get("tryOnLocked"):
        return False
    if p.get("tryOnOk") is False:
        return False          # try-on switched off by hand in admin: leave it
    if (p.get("tryOnFails") or 0) >= MAX_TRIES:
        return False
    if p.get("tryOnReady") and p.get("tryOnUrl"):
        return False
    ang = p.get("anglePhotos") if isinstance(p.get("anglePhotos"), dict) else {}
    return bool(p.get("rawFrontUrl") or p.get("originalUrl") or ang.get("front") or p.get("url"))


def main():
    now = datetime.datetime.utcnow().isoformat() + "Z"
    state = {"lastLook": now, "ok": False}
    failed = False
    recut = (os.environ.get("RECUT") or "").strip()
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("::error::STOP: the SUPABASE_KEY secret is missing or empty.")
        failed = True
    else:
        print("::notice::Database %s, key role: %s" % (SUPABASE_URL, key_role()))
    count = 0
    try:
        rows = call("/rest/v1/products?select=id,data&limit=3000")
        frames = [r.get("data") or {} for r in rows if not str(r.get("id") or "").startswith("__")]
        if recut.lower() == "reset":
            # forgive every rejection: the fault was ours (a wrong secret),
            # not the photos. The cutter then tries them all again.
            cleared = []
            for r in rows:
                d = r.get("data") or {}
                if str(r.get("id") or "").startswith("__"):
                    continue
                if (d.get("tryOnFails") or 0) > 0:
                    d["tryOnFails"] = 0
                    d.pop("tryOnFail", None)
                    cleared.append({"id": str(r.get("id")), "data": d, "updated_at": now})
            if cleared:
                call("/rest/v1/products?on_conflict=id", "POST", cleared,
                     "resolution=merge-duplicates,return=minimal")
            print("Reset: %d frame(s) forgiven and back in the queue." % len(cleared))
            recut = ""
        count = sum(1 for p in frames if waiting(p))
        rejected = sum(1 for p in frames if (p.get("tryOnFails") or 0) >= MAX_TRIES)
        # say WHY, so a rejection can be read from the run page
        reasons = {}
        for p in frames:
            if (p.get("tryOnFails") or 0) >= 1 and p.get("tryOnFail"):
                reasons.setdefault(str(p.get("tryOnFail"))[:160], []).append(str(p.get("name") or p.get("id"))[:24])
        for why, names in sorted(reasons.items(), key=lambda kv: -len(kv[1]))[:6]:
            print("  %d frame(s): %s  [%s]" % (len(names), why, ", ".join(names[:4])))
        state = {"lastLook": now, "ok": True, "waiting": count,
                 "rejected": rejected, "frames": len(frames)}
        if recut:
            count = max(count, 1)     # a trial cut was asked for: wake the cutter
        print("::notice::Frames: %d. Waiting: %d. Rejected: %d.%s" % (len(frames), count, rejected,
              ("  Trial re-cut asked for: " + recut) if recut else ""))
    except Exception as e:
        state["error"] = str(e)[:300]
        print("::error::STOP: could not read the database: %s" % str(e)[:200].replace("\n", " "))
        failed = True

    # The health line in admin reads this row.
    try:
        old = call("/rest/v1/products?select=id,data&id=eq.__robot")
        keep = (old[0].get("data") if old else None) or {}
        keep.update(state)
        call("/rest/v1/products?on_conflict=id", "POST",
             [{"id": "__robot", "data": keep, "updated_at": now}],
             "resolution=merge-duplicates,return=minimal")
    except Exception as e:
        print("Could not write the health note: %s" % str(e)[:150])

    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write("waiting=%d\n" % count)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
