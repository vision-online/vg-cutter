"""
Vision Gallery — the cut-out robot.

Runs by itself on GitHub Actions. Every hour it asks the shop's own
database which frames are still waiting for a try-on picture, cuts each
one, puts the cut-out on Cloudinary, and writes the address back onto the
frame. Nobody opens Colab, nobody renames a file, nobody presses anything.

The cutting method is the one the shop settled on, unchanged:

  1. no erosion or trimming rule anywhere -- the rim is NEVER thinned
  2. a first mask from the full BiRefNet
  3. a trimap: a band of "unknown" either side of the rim edge
  4. alpha matting in that band, so the edge is soft and a crystal or
     tinted rim keeps its see-through look instead of being cut to a line
  5. the inner lens cleared to full transparency, the outer rim untouched
  6. for the front try-on only: the arms removed end piece to end piece,
     by dropping outer columns that are too short to be a rim

What it needs to be told (GitHub -> Settings -> Secrets):
  SUPABASE_URL   the shop database address
  SUPABASE_KEY   the service_role key (a robot cannot sign in like a person)
Cloudinary is read from the shop's own settings row; it can also be given
as CLOUDINARY_NAME and CLOUDINARY_PRESET if it is ever missing there.
"""

import io
import json
import os
import sys
import time

import numpy as np
import requests
from PIL import Image

START = time.time()
MAX_MINUTES = float(os.environ.get("MAX_MINUTES", "50"))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "40"))
MAX_TRIES = int(os.environ.get("MAX_TRIES", "3"))

SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_KEY = (os.environ.get("SUPABASE_KEY") or "").strip()


def key_role():
    """Which kind of key this is, read from inside it -- without ever
    printing the key. 'service_role' is right; 'anon' cannot write."""
    try:
        import base64
        part = SUPABASE_KEY.split(".")[1]
        part += "=" * (-len(part) % 4)
        return json.loads(base64.urlsafe_b64decode(part)).get("role", "?")
    except Exception:
        return "not a legacy JWT key (sb_... keys are not accepted here)"


def log(*a):
    print(*a, flush=True)


def out_of_time():
    return (time.time() - START) / 60.0 > MAX_MINUTES


# ---------------------------------------------------------------- database

def sb_headers(extra=None):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": "Bearer " + SUPABASE_KEY,
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def fetch_products():
    """Every real frame. Settings rows have ids starting __ and are
    skipped here in Python: in the database's LIKE language an underscore
    is a wildcard, so a 'not like __*' filter would hide every frame."""
    url = (SUPABASE_URL + "/rest/v1/products?select=id,data&order=id.asc&limit=3000")
    r = requests.get(url, headers=sb_headers(), timeout=60)
    if not r.ok:
        raise RuntimeError("database answered %d: %s" % (r.status_code, r.text[:200]))
    rows = r.json() or []
    out = []
    for row in rows:
        rid = str(row.get("id") or "")
        if rid.startswith("__"):
            continue
        data = row.get("data") if isinstance(row.get("data"), dict) else {}
        data["id"] = rid
        out.append(data)
    return out


def fetch_setting(key):
    url = (SUPABASE_URL + "/rest/v1/products?select=id,data&id=eq.__" + key)
    try:
        r = requests.get(url, headers=sb_headers(), timeout=30)
        if not r.ok:
            return None
        rows = r.json() or []
        return rows[0]["data"] if rows and isinstance(rows[0].get("data"), dict) else None
    except Exception:
        return None


def save_product(product):
    """Write the frame back, whole, exactly as the admin does."""
    import datetime
    body = [{
        "id": str(product["id"]),
        "data": product,
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }]
    r = requests.post(
        SUPABASE_URL + "/rest/v1/products?on_conflict=id",
        headers=sb_headers({"Prefer": "resolution=merge-duplicates,return=minimal"}),
        data=json.dumps(body), timeout=60)
    r.raise_for_status()


# ---------------------------------------------------------------- cloudinary

def cloudinary_config():
    name = (os.environ.get("CLOUDINARY_NAME") or "").strip()
    preset = (os.environ.get("CLOUDINARY_PRESET") or "").strip()
    if name and preset:
        return name, preset
    for key in ("settings", "cloudinary", "shop"):
        s = fetch_setting(key) or {}
        c = s.get("cloudinary") if isinstance(s.get("cloudinary"), dict) else s
        if isinstance(c, dict) and c.get("name") and c.get("preset"):
            return c["name"], c["preset"]
    return name, preset


def cloudinary_upload(png_bytes, public_hint, name, preset):
    r = requests.post(
        "https://api.cloudinary.com/v1_1/%s/image/upload" % name,
        data={"upload_preset": preset},
        files={"file": (public_hint + ".png", png_bytes, "image/png")},
        timeout=180)
    if not r.ok:
        raise RuntimeError("Cloudinary refused: %s %s" % (r.status_code, r.text[:200]))
    j = r.json()
    url = j.get("secure_url") or j.get("url")
    if not url:
        raise RuntimeError("Cloudinary gave no address back")
    return url


# ---------------------------------------------------------------- the model

_model = None


def model():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModelForImageSegmentation
        wanted = os.environ.get("BIREFNET_MODEL", "ZhengPeng7/BiRefNet")
        try:
            m = AutoModelForImageSegmentation.from_pretrained(wanted, trust_remote_code=True)
            log("Model loaded:", wanted)
        except Exception as e:
            log("Full model would not load (%s). Using the lite model instead." % str(e)[:120])
            m = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet_lite", trust_remote_code=True)
        m.eval()
        torch.set_grad_enabled(False)
        torch.set_num_threads(int(os.environ.get("THREADS", "4")))
        _model = m
    return _model


def coarse_mask(img):
    """BiRefNet's own opinion of the frame, as a 0..1 mask at photo size."""
    import cv2
    import torch
    from torchvision import transforms
    prep = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = prep(img).unsqueeze(0)
    with torch.no_grad():
        pred = model()(x)[-1].sigmoid().cpu()[0, 0].numpy()
    pred = cv2.resize(pred, img.size, interpolation=cv2.INTER_LINEAR)
    return pred.astype(np.float32)


# ---------------------------------------------------------------- the method

def make_trimap(mask01, band_px):
    """
    0 = surely background, 1 = surely frame, 0.5 = unknown.
    The unknown band straddles the rim edge, so the matting -- not a hard
    threshold -- decides every pixel near the rim. The rim itself is never
    eroded; erosion only marks what is safely inside.
    """
    import cv2
    hard = (mask01 > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1))
    sure_fg = cv2.erode(hard, k)
    sure_bg = 1 - cv2.dilate(hard, k)
    tri = np.full(hard.shape, 0.5, np.float32)
    tri[sure_fg == 1] = 1.0
    tri[sure_bg == 1] = 0.0
    return tri


def matte(img, trimap):
    """Continuous alpha in the unknown band, at a reduced size for speed."""
    import cv2
    from pymatting import estimate_alpha_lkm, estimate_alpha_cf
    rgb = np.asarray(img.convert("RGB")).astype(np.float64) / 255.0
    H, W = trimap.shape
    scale = min(1.0, 900.0 / max(H, W))
    if scale < 1.0:
        sw, sh = int(W * scale), int(H * scale)
        rgb_s = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        tri_s = cv2.resize(trimap, (sw, sh), interpolation=cv2.INTER_NEAREST)
    else:
        rgb_s, tri_s = rgb, trimap
    try:
        a = estimate_alpha_lkm(rgb_s, tri_s.astype(np.float64))
    except Exception:
        a = estimate_alpha_cf(rgb_s, tri_s.astype(np.float64))
    a = np.clip(a, 0, 1)
    if scale < 1.0:
        a = cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    a = np.where(trimap == 1.0, 1.0, np.where(trimap == 0.0, 0.0, a))
    return a.astype(np.float32)


def _border(mask):
    b = np.zeros_like(mask, dtype=bool)
    b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
    return b & mask


def largest_component(alpha):
    """Keep the frame; drop stray specks the model may have kept."""
    from scipy import ndimage
    lab, n = ndimage.label(alpha > 0.5)
    if n <= 1:
        return alpha
    sizes = ndimage.sum(alpha > 0.5, lab, range(1, n + 1))
    keep = int(np.argmax(sizes)) + 1
    out = alpha.copy()
    out[(lab != keep) & (lab != 0)] = 0.0
    return out


def lens_holes(alpha):
    """The enclosed openings inside the rim -- i.e. the two lenses."""
    from scipy import ndimage
    frame = alpha > 0.5
    not_frame = ~frame
    outside = ndimage.binary_propagation(_border(not_frame), mask=not_frame)
    holes = not_frame & ~outside
    lab, n = ndimage.label(holes)
    if n == 0:
        return np.zeros_like(holes), outside
    sizes = ndimage.sum(holes, lab, range(1, n + 1))
    big = [i + 1 for i, s in enumerate(sizes) if s > holes.size * 0.004]
    return np.isin(lab, big), outside


def clear_inner_lens(alpha):
    """
    Make the glass fully transparent -- including a tinted or mirrored
    lens the model kept as solid, and a hinge tongue showing through the
    corner. The outer rim is not touched: nothing here operates on it.
    """
    import cv2
    lens, outside = lens_holes(alpha)
    if not lens.any():
        return alpha
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(alpha.shape[1] * 0.024) | 1,) * 2)
    lens_closed = cv2.morphologyEx(lens.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    lens_closed &= ~outside
    out = alpha.copy()
    out[lens_closed] = 0.0
    return out


def cut_end_to_end(alpha):
    """
    The front try-on picture: arms removed, end pieces kept.

    Walking in from each side, a column belonging to an arm is far
    shorter than a column belonging to a rim. Columns shorter than 12%
    of the frame's depth are dropped; the walk stops at the first proper
    column, which is the end piece. Nothing is thinned: whole columns of
    arm are dropped, never a pixel of what is kept.
    """
    A = alpha > 0.2
    heights = A.sum(axis=0)
    if not heights.any():
        return alpha
    depth = float(heights.max())
    thin = heights < depth * 0.12
    W = alpha.shape[1]
    left = 0
    while left < W and (heights[left] == 0 or thin[left]):
        left += 1
    right = W - 1
    while right > left and (heights[right] == 0 or thin[right]):
        right -= 1
    out = alpha.copy()
    out[:, :left] = 0.0
    out[:, right + 1:] = 0.0
    return out


def measurements(alpha):
    """Lens centres, widths, bridge and front width, in pixels and as
    fractions of the front width -- the numbers the try-on engine uses."""
    from scipy import ndimage
    lens, _ = lens_holes(alpha)
    ys, xs = np.where(alpha > 0.2)
    if not xs.size:
        return {}
    front_w = int(xs.max() - xs.min() + 1)
    front_h = int(ys.max() - ys.min() + 1)
    out = {"frontWidthPx": front_w, "frontHeightPx": front_h}
    lab, n = ndimage.label(lens)
    if n < 1:
        return out
    boxes = []
    for i in range(1, n + 1):
        yy, xx = np.where(lab == i)
        boxes.append((xx.min(), xx.max(), yy.min(), yy.max(), xx.size))
    boxes.sort(key=lambda b: -b[4])
    boxes = sorted(boxes[:2], key=lambda b: b[0])
    eyes = []
    for (x0, x1, y0, y1, _s) in boxes:
        eyes.append({
            "centreX": float((x0 + x1) / 2.0),
            "centreY": float((y0 + y1) / 2.0),
            "widthPx": int(x1 - x0 + 1),
            "heightPx": int(y1 - y0 + 1),
            "centreXFrac": float(((x0 + x1) / 2.0 - xs.min()) / front_w),
        })
    out["lenses"] = eyes
    if len(boxes) == 2:
        out["bridgePx"] = int(boxes[1][0] - boxes[0][1])
        out["pupilGapPx"] = float(eyes[1]["centreX"] - eyes[0]["centreX"])
    return out


def cut(photo_bytes, front=True):
    """One photo in, one transparent PNG (and its numbers) out."""
    import cv2
    img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    if max(img.size) > 1600:
        s = 1600.0 / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)

    m = coarse_mask(img)                                  # step 2
    m = largest_component(m)
    band_px = max(6, min(24, int(round(12 * max(img.size) / 1440.0))))
    tri = make_trimap(m, band_px)                         # step 3
    a = matte(img, tri)                                   # step 4
    a = clear_inner_lens(a)                               # step 5
    if front:
        a = cut_end_to_end(a)                             # step 6

    nums = measurements(a)
    rgba = np.dstack([np.asarray(img), (a * 255).round().astype(np.uint8)])
    ys, xs = np.where(a > 0.02)
    if ys.size:
        pad = 4
        y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad + 1)
        rgba = rgba[y0:y1, x0:x1]
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue(), nums


# ---------------------------------------------------------------- the round

def front_photo(p):
    ang = p.get("anglePhotos") if isinstance(p.get("anglePhotos"), dict) else {}
    # the photo AS TAKEN first (d16.62 admin uploads it plain); the
    # sharpened copies only if that is all an older frame has
    for v in (p.get("rawFrontUrl"), p.get("originalUrl"), ang.get("front"), p.get("url")):
        if isinstance(v, str) and v.startswith("http"):
            return v
    return None


def angle_photo(p):
    ang = p.get("anglePhotos") if isinstance(p.get("anglePhotos"), dict) else {}
    for v in (p.get("rawAngleUrl"), ang.get("angle")):
        if isinstance(v, str) and v.startswith("http"):
            return v
    for g in (p.get("gallery") or []):
        if isinstance(g, dict) and g.get("view") == "angle" and str(g.get("url", "")).startswith("http"):
            return g["url"]
    return None


def side_photo(p):
    ang = p.get("anglePhotos") if isinstance(p.get("anglePhotos"), dict) else {}
    for v in (p.get("rawSideUrl"), p.get("sidePhoto"), ang.get("side")):
        if isinstance(v, str) and v.startswith("http"):
            return v
    return None


def waiting(p):
    """A frame is waiting if it has a photo but no ready cut-out."""
    if p.get("tryOnLocked"):
        return False
    if p.get("tryOnOk") is False:
        return False          # try-on switched off by hand in admin: leave it          # a cut-out the shop supplied by hand: never touch
    if (p.get("tryOnFails") or 0) >= MAX_TRIES:
        return False          # already rejected: waiting for a better photo
    if p.get("tryOnReady") and p.get("tryOnUrl"):
        return False
    return bool(front_photo(p))


def plain_reason(err):
    """Turn a technical failure into something a photographer can act on."""
    t = str(err).lower()
    if "no frame" in t or "nothing" in t or "empty" in t:
        return "no frame found in the photo"
    if "cloudinary" in t:
        return "the photo store refused the upload"
    if "timed out" in t or "timeout" in t or "connection" in t:
        return "the photo could not be downloaded"
    if "lens" in t:
        return "the lenses could not be made clear"
    return "the photo could not be cut (%s)" % str(err)[:80]


def alert_numbers():
    """Numbers to message, from the shop settings: number:key, number:key."""
    raw = os.environ.get("ALERT_NUMBERS") or ""
    if not raw:
        for key in ("settings", "shop"):
            s = fetch_setting(key) or {}
            raw = s.get("alertWhatsApp") or s.get("alertNumbers") or ""
            if raw:
                break
    out = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        num, _, key = part.partition(":")
        num = "".join(ch for ch in num if ch.isdigit())
        if num and key.strip():
            out.append((num, key.strip()))
    return out


def whatsapp(text):
    """Message the shop. Free service, so a failure here is never fatal."""
    for num, key in alert_numbers():
        try:
            requests.get("https://api.callmebot.com/whatsapp.php",
                         params={"phone": num, "text": text, "apikey": key},
                         timeout=45)
            log("    alert sent to %s" % num[-4:])
        except Exception as e:
            log("    alert to %s failed: %s" % (num[-4:], str(e)[:80]))


def health(patch):
    """The line the admin health check reads."""
    try:
        import datetime
        now = datetime.datetime.utcnow().isoformat() + "Z"
        rows = requests.get(SUPABASE_URL + "/rest/v1/products?select=id,data&id=eq.__robot",
                            headers=sb_headers(), timeout=30).json() or []
        keep = (rows[0].get("data") if rows else None) or {}
        keep.update(patch)
        keep["lastRun"] = now
        requests.post(SUPABASE_URL + "/rest/v1/products?on_conflict=id",
                      headers=sb_headers({"Prefer": "resolution=merge-duplicates,return=minimal"}),
                      data=json.dumps([{"id": "__robot", "data": keep, "updated_at": now}]),
                      timeout=30)
    except Exception as e:
        log("Could not write the health note: %s" % str(e)[:120])


def download(url):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        log("STOP: SUPABASE_URL or SUPABASE_KEY is missing from the repository secrets.")
        return 1

    log("Database: %s  key: %s" % (SUPABASE_URL, key_role()))
    try:
        products = fetch_products()
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "JWT" in msg:
            log("::error::STOP: the database rejected the key. Replace the SUPABASE_KEY secret "
                "with the service_role key (Supabase -> Settings -> API Keys -> Legacy).")
        else:
            log("::error::STOP: could not read the database: %s" % msg[:300].replace("\n", " "))
        health({"ok": False, "error": msg[:300]})
        return 1
    todo = [p for p in products if waiting(p)]
    recut = (os.environ.get("RECUT") or "").strip()
    if recut:
        # a trial: cut one frame again even though it already has a cut-out
        if recut.lower() == "first":
            pick = next((p for p in products if front_photo(p) and not p.get("tryOnLocked")), None)
        else:
            pick = next((p for p in products if str(p.get("id")) == recut or str(p.get("sku")) == recut), None)
        if pick is None:
            log("Trial re-cut: no frame matched '%s'." % recut)
        elif pick not in todo:
            pick["tryOnFails"] = 0
            todo.insert(0, pick)
            log("Trial re-cut of %s (%s)." % (pick.get("id"), (pick.get("name") or "")[:40]))
    log("Catalogue: %d frames. Waiting for a cut-out: %d." % (len(products), len(todo)))
    if not todo:
        log("Nothing to do.")
        return 0

    cname, cpreset = cloudinary_config()
    if not cname or not cpreset:
        log("::error::STOP: no Cloudinary account name and unsigned preset found "
            "(not in the shop settings, and not in the repository secrets).")
        return 1
    log("Cloudinary: %s / %s" % (cname, cpreset))

    done = failed = 0
    rejected = []
    for p in todo[:MAX_FRAMES]:
        if out_of_time():
            log("Time is up for this run; the rest go in the next one.")
            break
        pid = p.get("id")
        try:
            log("--- %s : %s" % (pid, (p.get("name") or "")[:50]))
            png, nums = cut(download(front_photo(p)), front=True)
            url = cloudinary_upload(png, "tryon-" + str(pid), cname, cpreset)
            p["tryOnUrl"] = url
            p["tryOnReady"] = True
            p["tryOnNumbers"] = nums
            p["tryOnCutBy"] = "robot"

            for src, field, label in ((angle_photo(p), "tryOnAngleUrl", "45 degree"),
                                      (side_photo(p), "tryOnSideUrl", "90 degree side")):
                if src and not p.get(field) and not out_of_time():
                    try:
                        png2, _ = cut(download(src), front=False)
                        p[field] = cloudinary_upload(
                            png2, "tryon-" + field + "-" + str(pid), cname, cpreset)
                        log("    %s cut too" % label)
                    except Exception as e:
                        log("    %s photo skipped: %s" % (label, str(e)[:110]))

            p.pop("tryOnFail", None)
            p["tryOnFails"] = 0
            save_product(p)
            done += 1
            log("    done -> %s" % url)
        except Exception as e:
            failed += 1
            why = plain_reason(e)
            tries = (p.get("tryOnFails") or 0) + 1
            p["tryOnFails"] = tries
            p["tryOnFail"] = why
            p["tryOnReady"] = False
            try:
                save_product(p)
            except Exception:
                pass
            log("::warning::%s FAILED (%d of %d): %s" % (pid, tries, MAX_TRIES, why))
            if tries >= MAX_TRIES:
                rejected.append("%s: %s" % (p.get("name") or pid, why))

    if rejected:
        whatsapp("Vision Gallery - frames rejected\n\n" + "\n".join(rejected[:10]) +
                 "\n\nThese frames are NOT on the website. Please take a proper "
                 "photo again and save it in admin.")

    health({"ok": True, "cut": done, "failed": failed,
            "rejectedNow": len(rejected),
            "waiting": max(0, len(todo) - done)})
    log("::notice::Finished. Cut %d, failed %d, rejected %d, still waiting %d.%s"
        % (done, failed, len(rejected), max(0, len(todo) - done),
           ("  Rejected: " + "; ".join(rejected[:5])) if rejected else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
