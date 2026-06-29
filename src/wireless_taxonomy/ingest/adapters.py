

def detect_source_hint(source_url: str, page_text: str = "") -> str:
    haystack = f"{source_url}\n{page_text[:1000]}".lower()
    if "sigcomm" in haystack:
        return "sigcomm"
    if "dl.acm.org" in haystack or "acm" in haystack:
        return "acm"
    if "ieeexplore.ieee.org" in haystack or "ieee" in haystack:
        return "ieee"
    if "usenix.org" in haystack:
        return "usenix"
    if "openreview.net" in haystack:
        return "openreview"
    return "generic"
