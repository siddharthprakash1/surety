import pytest

from surety.eval.sites import website


@pytest.mark.parametrize("host, site", [
    ("en.wikipedia.org", "wikipedia"), ("de.wikipedia.org", "wikipedia"),
    ("apps.apple.com", "apple"), ("apple.com.au", "apple"), ("amazon.co.jp", "amazon"), ("amazon.com", "amazon"),
    ("gettested.cdc.gov", "cdc"), ("bbc.co.uk", "bbc"), ("gov.uk", "gov.uk"),
    ("juliaserano.blogspot.com", "juliaserano"), ("WWW.Example.COM.", "example"),
])
def test_website_groups_hosts_of_one_site(host, site):
    assert website(host) == site
