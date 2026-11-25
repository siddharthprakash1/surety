"""Which website a WebJev host belongs to. Hosts that share a registrable domain's name are one website when a plan
splits or spreads sites: en. and de.wikipedia.org, apple.com and apps.apple.com, amazon.com and amazon.de. Errs toward
merging (weather.com and weather.gov are one), which only makes a split stricter."""
from functools import lru_cache


@lru_cache(maxsize=1)
def _psl():
    from publicsuffixlist import PublicSuffixList   # imported here: scoring containers load webjev.py without it
    return PublicSuffixList(only_icann=False)


def psl_version() -> str:
    from importlib.metadata import version
    return version("publicsuffixlist")


@lru_cache(maxsize=None)
def website(host: str) -> str:
    """The registrable domain without its public suffix: apps.apple.com -> apple, amazon.co.jp -> amazon,
    juliaserano.blogspot.com -> juliaserano; a host that is itself a public suffix (gov.uk) stays as it is."""
    h = host.lower().rstrip(".")
    reg = _psl().privatesuffix(h) or h
    suffix = _psl().publicsuffix(reg) or ""
    return reg[: -len(suffix) - 1] if suffix and reg.endswith("." + suffix) else reg
