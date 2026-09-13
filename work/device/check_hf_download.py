#!/usr/bin/env python3
"""Exercise the exact HTTP path `ModelStore.kt` uses, without needing the phone.

The app fetches ~7.9 GB of context binaries from a PRIVATE Hugging Face repo. That
makes the download less trivial than it looks, and the failure modes surface on device
hours later as an unhelpful "Create From Binary failure". The Kotlin follows redirects
BY HAND on three assumptions; this checks all three against the live repo:

  1. an unauthenticated request is refused        -> the token is genuinely required
  2. `huggingface.co/.../resolve/...` answers 302 with a PRE-SIGNED CDN url, and that
     hop must NOT carry the `Authorization` header (the signature is in the query
     string, and an object store generally rejects a request carrying both)
  3. `Range` survives the redirect, so a dropped 1.5 GB transfer resumes rather than
     restarting and appending a second copy onto the partial file

It also downloads a small slice from the middle of one file and compares it against the
local artefact, which proves the bytes on the CDN are the bytes we built.

  usage: py -3.10 work/device/check_hf_download.py [name]
"""
import sys
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from huggingface_hub import get_token

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://huggingface.co/AbrahamPJ/neodragon-npu-s25u/resolve/main"
AUTH_HOST = "huggingface.co"


def raw_get(url, token=None, rng=None, redirect=False):
    """One hop, no automatic redirects -- mirrors instanceFollowRedirects=false."""
    req = Request(url, method="GET")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    if rng:
        req.add_header("Range", rng)

    class NoRedirect(__import__("urllib.request", fromlist=["x"]).HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    import urllib.request as u
    opener = u.build_opener() if redirect else u.build_opener(NoRedirect)
    try:
        r = opener.open(req, timeout=60)
        return r.status, dict(r.headers), r
    except HTTPError as e:
        return e.code, dict(e.headers), e


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "vaedecsn"
    url = f"{BASE}/{name}_v79.bin"
    token = get_token()
    local = ROOT / "work" / "device" / f"{name}_v79.bin"
    print(f"target: {url}\nlocal : {local} ({local.stat().st_size:,} B)\n")

    ok = True

    # --- 1. no token ---------------------------------------------------------
    code, _, _ = raw_get(url)
    good = code in (401, 403, 404)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] unauthenticated -> {code} "
          f"(expect 401/403/404: the repo really is private)")

    # --- 2. token -> redirect to a pre-signed CDN url -------------------------
    code, hdr, _ = raw_get(url, token=token)
    loc = hdr.get("Location", "")
    good = code in (301, 302, 303, 307, 308) and bool(loc)
    ok &= good
    host = loc.split("/")[2] if "//" in loc else "?"
    print(f"[{'PASS' if good else 'FAIL'}] authenticated   -> {code} "
          f"-> {host}")
    print(f"         signed url: {'yes' if ('Signature' in loc or 'X-Amz' in loc) else 'no'}"
          f"   host is {AUTH_HOST}: {host == AUTH_HOST}")
    if not good:
        print("cannot continue without a redirect target")
        return 1

    # --- 3. follow WITHOUT auth, with a Range --------------------------------
    size = local.stat().st_size
    start, length = size // 2, 65536
    code, hdr, body = raw_get(loc, rng=f"bytes={start}-{start+length-1}")
    good = code == 206
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] CDN + Range, no auth -> {code} "
          f"(expect 206 Partial Content)")

    if good:
        got = body.read()
        with open(local, "rb") as f:
            f.seek(start)
            want = f.read(length)
        same = got == want
        ok &= same
        print(f"[{'PASS' if same else 'FAIL'}] {len(got):,} bytes from offset "
              f"{start:,} match the local artefact")

    # --- 4. the design decision: does auth to the CDN actually break? --------
    code2, _, _ = raw_get(loc, token=token, rng=f"bytes={start}-{start+255}")
    print(f"[info] CDN WITH the Authorization header -> {code2}"
          + ("  (would have worked, but forwarding it still leaks the token"
             " to a third-party host)" if code2 == 206 else
             "  (confirms the header must be dropped on the redirect)"))

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOMETHING FAILED -- see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
