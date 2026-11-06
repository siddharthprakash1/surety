import functools
import http.server
import os
import threading

import pytest

if os.environ.get("SURETY_LIVE") != "1":
    pytest.skip("set SURETY_LIVE=1 to run browser tests", allow_module_level=True)
pytest.importorskip("playwright")

from surety.env import Action, PlaywrightEnv  # noqa: E402

pytestmark = pytest.mark.live

HTML = """<html><head><title>Fixture</title></head><body>
<p>Hello</p><button>Go</button><button disabled>Off</button><input placeholder="Query">
<input type="password" name="pw" value="secret"><div style="height:3000px"></div><button>Far</button>
</body></html>"""


@pytest.fixture
def url(tmp_path):
    (tmp_path / "index.html").write_text(HTML)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    httpd.shutdown()


def test_observe_finds_button_and_text_input(url):
    env = PlaywrightEnv(url)
    try:
        page = env.observe()
        by_name = {e.name: e for e in page.elements}
        assert by_name["Go"].role == "button" and by_name["Query"].role == "input"
        assert by_name["pw"].value == "" and page.title == "Fixture" and "Hello" in page.text
        assert "Off" not in by_name and "Far" not in by_name
        env.act(Action("type", by_name["Query"].id, "shoes"))
        assert {e.name: e.value for e in env.observe().elements}["Query"] == "shoes"
    finally:
        env.close()
