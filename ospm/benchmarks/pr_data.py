"""Hand-labelled pull requests for the review and security agents.

Each PR lists files as lines prefixed with '+', '-' or ' ' (context); `make_diff` builds a valid unified diff.
Labels follow the conventions in examples/inkwell/project.toml:

review:   debug_print (print in src/ except src/inkwell/cli.py), bare_except, todo_without_issue,
          missing_tests (>=3 added lines of src/*.py and no file under tests/ changed), breakpoint_left,
          mutable_default_arg, wildcard_import, missing_type_hints (new public function in src/ lacking
          parameter or return annotations)
security: hardcoded_secret, sql_injection, command_injection, unsafe_deserialization, insecure_tls,
          weak_crypto, path_traversal, code_eval
"""

from __future__ import annotations


def make_diff(files: dict[str, list[str]]) -> str:
    out: list[str] = []
    for path, lines in files.items():
        old = sum(1 for l in lines if l[:1] in (" ", "-"))
        new = sum(1 for l in lines if l[:1] in (" ", "+"))
        out.append(f"diff --git a/{path} b/{path}")
        out.append(f"--- {'/dev/null' if old == 0 else 'a/' + path}")
        out.append(f"+++ b/{path}")
        out.append(f"@@ -{1 if old else 0},{old} +1,{new} @@")
        out.extend(lines)
    return "\n".join(out) + "\n"


def _pr(pid, title, body, files, review, security):
    for path, lines in files.items():
        for l in lines:
            if l[:1] not in ("+", "-", " "):
                raise ValueError(f"{pid} {path}: bad diff line {l!r}")
    return {"id": pid, "number": int(pid.split("-")[1]), "title": title, "body": body,
            "diff": make_diff(files), "review": sorted(review), "security": sorted(security)}


T = "tests/test_{}.py"

PRS = [
    _pr("pr-1", "fix(feeds): emit RFC 822 pubDate in UTC", "Fixes #113.", {
        "src/inkwell/feeds/rss.py": [
            " from datetime import datetime",
            "+from datetime import timezone",
            "+",
            "+",
            "+def format_pubdate(value: datetime) -> str:",
            "+    \"\"\"Format a datetime as an RFC 822 date in UTC.\"\"\"",
            "+    return value.astimezone(timezone.utc).strftime(\"%a, %d %b %Y %H:%M:%S +0000\")",
        ],
        T.format("feeds"): [
            "+def test_pubdate_is_utc():",
            "+    assert format_pubdate(datetime(2024, 1, 1, 12, tzinfo=CET)).endswith(\"+0000\")",
        ],
    }, [], []),
    _pr("pr-2", "feat: expose reading time to templates", "Adds page.reading_time.", {
        "src/inkwell/markdown/stats.py": [
            "+WORDS_PER_MINUTE = 220",
            "+",
            "+def reading_time(text):",
            "+    words = len(text.split())",
            "+    print(f\"counted {words} words\")",
            "+    return max(1, round(words / WORDS_PER_MINUTE))",
        ],
        T.format("stats"): [
            "+def test_reading_time():",
            "+    assert reading_time(\"word \" * 440) == 2",
        ],
    }, ["debug_print", "missing_type_hints"], []),
    _pr("pr-3", "plugins: keep going when a plugin fails to import", "", {
        "src/inkwell/plugins/loader.py": [
            " def load_all(names: list[str]) -> list[Plugin]:",
            "     plugins = []",
            "     for name in names:",
            "-        plugins.append(import_plugin(name))",
            "+        try:",
            "+            plugins.append(import_plugin(name))",
            "+        except:",
            "+            logger.warning(\"could not load plugin %s\", name)",
            "     return plugins",
        ],
        T.format("loader"): [
            "+def test_broken_plugin_is_skipped(tmp_path):",
            "+    assert load_all([\"does_not_exist\"]) == []",
        ],
    }, ["bare_except"], []),
    _pr("pr-4", "WIP: image cache", "Still experimenting.", {
        "src/inkwell/assets/cache.py": [
            "+class ImageCache:",
            "+    def __init__(self, root: Path) -> None:",
            "+        self.root = root",
            "+",
            "+    def get(self, key: str) -> bytes | None:",
            "+        # TODO: handle eviction when the cache grows too large",
            "+        breakpoint()",
            "+        path = self.root / key",
            "+        return path.read_bytes() if path.exists() else None",
        ],
    }, ["breakpoint_left", "missing_tests", "todo_without_issue"], []),
    _pr("pr-5", "feat: taxonomy registry", "Part of #118.", {
        "src/inkwell/feeds/taxonomy.py": [
            "+REGISTRY: dict[str, list] = {}",
            "+",
            "+def register(name: str, pages: list = []) -> None:",
            "+    pages.extend(REGISTRY.get(name, []))",
            "+    REGISTRY[name] = pages",
        ],
        T.format("taxonomy"): [
            "+def test_register():",
            "+    register(\"cuisine\", [page])",
            "+    assert REGISTRY[\"cuisine\"] == [page]",
        ],
    }, ["mutable_default_arg"], []),
    _pr("pr-6", "refactor: split markdown extensions into a package", "", {
        "src/inkwell/markdown/__init__.py": [
            "-from inkwell.markdown.tables import TableExtension",
            "-from inkwell.markdown.footnotes import FootnoteExtension",
            "+from inkwell.markdown.ext import *",
            "+",
            "+DEFAULT_EXTENSIONS = [TableExtension(), FootnoteExtension()]",
        ],
        T.format("markdown"): [
            "+def test_default_extensions_loaded():",
            "+    assert len(DEFAULT_EXTENSIONS) == 2",
        ],
    }, ["wildcard_import"], []),
    _pr("pr-7", "cli: print a build summary", "Shows pages built and elapsed time.", {
        "src/inkwell/cli.py": [
            "     site = build_site(config)",
            "+    elapsed = time.perf_counter() - started",
            "+    print(f\"Built {len(site.pages)} pages in {elapsed:.2f}s\")",
            "+    print(f\"Output written to {config.output_dir}\")",
        ],
        T.format("cli"): [
            "+def test_build_prints_summary(runner):",
            "+    result = runner.invoke(cli, [\"build\"])",
            "+    assert \"Built\" in result.output",
        ],
    }, [], []),
    _pr("pr-8", "feat: deploy helper for GitHub Pages", "Adds `inkwell deploy`.", {
        "src/inkwell/deploy.py": [
            "+import requests",
            "+",
            "+GITHUB_TOKEN = \"ghp_9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b2a1f0e\"",
            "+",
            "+def trigger_pages_build(repo: str) -> int:",
            "+    resp = requests.post(",
            "+        f\"https://api.github.com/repos/{repo}/pages/builds\",",
            "+        headers={\"Authorization\": f\"token {GITHUB_TOKEN}\"},",
            "+        verify=False,",
            "+    )",
            "+    return resp.status_code",
        ],
        T.format("deploy"): [
            "+def test_trigger(monkeypatch):",
            "+    monkeypatch.setattr(requests, \"post\", fake_post)",
            "+    assert trigger_pages_build(\"o/r\") == 201",
        ],
    }, [], ["hardcoded_secret", "insecure_tls"]),
    _pr("pr-9", "feat: search pages from the dev server", "", {
        "src/inkwell/search.py": [
            "+import sqlite3",
            "+",
            "+def search(conn: sqlite3.Connection, query: str) -> list[tuple]:",
            "+    cur = conn.cursor()",
            "+    cur.execute(f\"SELECT url, title FROM pages WHERE title LIKE '%{query}%'\")",
            "+    return cur.fetchall()",
        ],
        T.format("search"): [
            "+def test_search(db):",
            "+    assert search(db, \"intro\") == [(\"/intro/\", \"Intro\")]",
        ],
    }, [], ["sql_injection"]),
    _pr("pr-10", "feat: user-defined build hooks", "Runs `hooks.post_page` commands from inkwell.toml.", {
        "src/inkwell/hooks.py": [
            "+import subprocess",
            "+",
            "+def run_post_page_hook(command: str, page_path: str) -> None:",
            "+    subprocess.run(f\"{command} {page_path}\", shell=True, check=True)",
        ],
        T.format("hooks"): [
            "+def test_hook_runs(tmp_path):",
            "+    run_post_page_hook(\"touch\", str(tmp_path / \"x\"))",
        ],
    }, [], ["command_injection"]),
    _pr("pr-11", "perf: share the parsed-page cache between CI runs", "Downloads the cache from the artifact URL.", {
        "src/inkwell/cache.py": [
            "+import pickle",
            "+import urllib.request",
            "+",
            "+def fetch_remote_cache(url: str) -> dict:",
            "+    with urllib.request.urlopen(url) as resp:",
            "+        return pickle.loads(resp.read())",
        ],
        T.format("cache"): [
            "+def test_fetch_remote_cache(http_server):",
            "+    assert fetch_remote_cache(http_server.url) == {}",
        ],
    }, [], ["unsafe_deserialization"]),
    _pr("pr-12", "config: accept inkwell.yaml", "", {
        "src/inkwell/config.py": [
            "+import yaml",
            "+",
            "+def load_yaml_config(path: Path) -> dict:",
            "+    with open(path) as fh:",
            "+        return yaml.load(fh, Loader=yaml.Loader)",
        ],
        T.format("config"): [
            "+def test_yaml_config(tmp_path):",
            "+    (tmp_path / \"inkwell.yaml\").write_text(\"title: x\")",
            "+    assert load_yaml_config(tmp_path / \"inkwell.yaml\")[\"title\"] == \"x\"",
        ],
    }, [], ["unsafe_deserialization"]),
    _pr("pr-13", "feat: password-protected previews", "", {
        "src/inkwell/preview_auth.py": [
            "+import hashlib",
            "+",
            "+def hash_password(password: str) -> str:",
            "+    return hashlib.md5(password.encode()).hexdigest()",
            "+",
            "+def check_password(password: str, stored: str) -> bool:",
            "+    return hash_password(password) == stored",
        ],
        T.format("preview_auth"): [
            "+def test_roundtrip():",
            "+    assert check_password(\"pw\", hash_password(\"pw\"))",
        ],
    }, [], ["weak_crypto"]),
    _pr("pr-14", "perf: content-addressed render cache", "MD5 is only used as a cache key here.", {
        "src/inkwell/render_cache.py": [
            "+import hashlib",
            "+",
            "+def cache_key(content: bytes) -> str:",
            "+    # not security sensitive: only used to detect unchanged inputs",
            "+    return hashlib.md5(content, usedforsecurity=False).hexdigest()",
        ],
        T.format("render_cache"): [
            "+def test_cache_key_stable():",
            "+    assert cache_key(b\"a\") == cache_key(b\"a\")",
        ],
    }, [], []),
    _pr("pr-15", "feat: preview share links", "", {
        "src/inkwell/share.py": [
            "+import random",
            "+import string",
            "+",
            "+def new_share_token(length: int = 24) -> str:",
            "+    alphabet = string.ascii_letters + string.digits",
            "+    return \"\".join(random.choice(alphabet) for _ in range(length))",
        ],
        T.format("share"): [
            "+def test_token_length():",
            "+    assert len(new_share_token()) == 24",
        ],
    }, [], ["weak_crypto"]),
    _pr("pr-16", "serve: serve files straight from the output dir", "", {
        "src/inkwell/serve.py": [
            "+import os",
            "+",
            "+def resolve_request(root: str, request_path: str) -> bytes:",
            "+    path = os.path.join(root, request_path.lstrip(\"/\"))",
            "+    with open(path, \"rb\") as fh:",
            "+        return fh.read()",
        ],
        T.format("serve"): [
            "+def test_resolve(tmp_path):",
            "+    (tmp_path / \"index.html\").write_text(\"hi\")",
            "+    assert resolve_request(str(tmp_path), \"/index.html\") == b\"hi\"",
        ],
    }, [], ["path_traversal"]),
    _pr("pr-17", "fix(serve): refuse paths outside the output directory", "Fixes #122.", {
        "src/inkwell/serve.py": [
            "+from pathlib import Path",
            "+",
            "+def resolve_request(root: Path, request_path: str) -> Path | None:",
            "+    candidate = (root / request_path.lstrip(\"/\")).resolve()",
            "+    if not candidate.is_relative_to(root.resolve()):",
            "+        return None",
            "+    return candidate",
        ],
        T.format("serve"): [
            "+def test_traversal_blocked(tmp_path):",
            "+    assert resolve_request(tmp_path, \"/../../etc/passwd\") is None",
        ],
    }, [], []),
    _pr("pr-18", "feat: inline expressions in templates", "Allows `{= 1 + 2 =}` in Markdown.", {
        "src/inkwell/markdown/expressions.py": [
            "+import re",
            "+",
            "+EXPR = re.compile(r\"\\{=(.+?)=\\}\")",
            "+",
            "+def render_expressions(markdown_text: str) -> str:",
            "+    return EXPR.sub(lambda m: str(eval(m.group(1))), markdown_text)",
        ],
        T.format("expressions"): [
            "+def test_expression():",
            "+    assert render_expressions(\"{= 1 + 2 =}\") == \"3\"",
        ],
    }, [], ["code_eval"]),
    _pr("pr-19", "feat: use git commit date for lastmod", "Fixes #189.", {
        "src/inkwell/feeds/sitemap.py": [
            "+import subprocess",
            "+",
            "+def last_commit_date(path: str) -> str:",
            "+    out = subprocess.run(",
            "+        [\"git\", \"log\", \"-1\", \"--format=%cI\", \"--\", path],",
            "+        capture_output=True, text=True, check=True,",
            "+    )",
            "+    return out.stdout.strip()",
        ],
        T.format("sitemap"): [
            "+def test_last_commit_date(repo):",
            "+    assert last_commit_date(\"README.md\").startswith(\"20\")",
        ],
    }, [], []),
    _pr("pr-20", "feat: analytics-free webmention support", "Reads the API key from the environment.", {
        "src/inkwell/webmention.py": [
            "+import os",
            "+",
            "+def api_key() -> str:",
            "+    return os.environ[\"INKWELL_WEBMENTION_KEY\"]",
            "+",
            "+def password_from_env() -> str:",
            "+    return os.getenv(\"INKWELL_PREVIEW_PASSWORD\", \"\")",
        ],
        T.format("webmention"): [
            "+def test_api_key(monkeypatch):",
            "+    monkeypatch.setenv(\"INKWELL_WEBMENTION_KEY\", \"k\")",
            "+    assert api_key() == \"k\"",
        ],
    }, [], []),
    _pr("pr-21", "fix(search): use a parameterised query", "", {
        "src/inkwell/search.py": [
            " def search(conn: sqlite3.Connection, query: str) -> list[tuple]:",
            "     cur = conn.cursor()",
            "-    cur.execute(f\"SELECT url, title FROM pages WHERE title LIKE '%{query}%'\")",
            "+    cur.execute(\"SELECT url, title FROM pages WHERE title LIKE ?\", (f\"%{query}%\",))",
            "     return cur.fetchall()",
        ],
        T.format("search"): [
            "+def test_search_quotes(db):",
            "+    assert search(db, \"it's\") == []",
        ],
    }, [], []),
    _pr("pr-22", "docs: fix typos in CLI reference", "", {
        "docs/cli.md": [
            "-The server will recieve requests on port 8000.",
            "+The server will receive requests on port 8000.",
        ],
    }, [], []),
    _pr("pr-23", "feat(i18n): load translation catalogs", "", {
        "src/inkwell/i18n/catalog.py": [
            "+import json",
            "+from pathlib import Path",
            "+",
            "+def load_catalog(lang, root=Path(\"i18n\")):",
            "+    path = root / f\"{lang}.json\"",
            "+    return json.loads(path.read_text(encoding=\"utf-8\"))",
        ],
    }, ["missing_tests", "missing_type_hints"], []),
    _pr("pr-24", "fix: footnote ordering", "Fixes #110.", {
        "src/inkwell/markdown/footnotes.py": [
            "+import pdb",
            "+",
            "+def number_footnotes(refs: list[str]) -> dict[str, int]:",
            "+    order: dict[str, int] = {}",
            "+    pdb.set_trace()",
            "+    for ref in refs:",
            "+        order.setdefault(ref, len(order) + 1)",
            "+    print(order)",
            "+    return order",
        ],
        T.format("footnotes"): [
            "+def test_number_footnotes():",
            "+    assert number_footnotes([\"a\", \"b\", \"a\"]) == {\"a\": 1, \"b\": 2}",
        ],
    }, ["breakpoint_left", "debug_print"], []),
    _pr("pr-25", "feat: webhook notifications on build", "", {
        "src/inkwell/notify.py": [
            "+import requests",
            "+",
            "+SIGNING_SECRET = \"s3cr3t-signing-key-2024-prod\"",
            "+",
            "+def notify(url: str, event: str, payload: dict = {}) -> None:",
            "+    payload[\"event\"] = event",
            "+    payload[\"signature\"] = sign(payload, SIGNING_SECRET)",
            "+    requests.post(url, json=payload, timeout=10, verify=False)",
        ],
        T.format("notify"): [
            "+def test_notify(httpserver):",
            "+    notify(httpserver.url_for(\"/\"), \"built\")",
        ],
    }, ["mutable_default_arg"], ["hardcoded_secret", "insecure_tls"]),
    _pr("pr-26", "feat(assets): convert images to webp via ImageMagick", "", {
        "src/inkwell/assets/webp.py": [
            "+import os",
            "+",
            "+def to_webp(filename: str) -> None:",
            "+    os.system(\"convert \" + filename + \" \" + filename.rsplit(\".\", 1)[0] + \".webp\")",
        ],
        T.format("webp"): [
            "+def test_to_webp(tmp_path):",
            "+    to_webp(str(tmp_path / \"a.png\"))",
        ],
    }, [], ["command_injection"]),
    _pr("pr-27", "feat(feeds): atom feed groundwork", "", {
        "src/inkwell/feeds/atom.py": [
            "+# FIXME(#212): entries are not sorted by updated date yet",
            "+# TODO: support per-tag feeds",
            "+def build_atom(entries: list[Entry]) -> str:",
            "+    return render_template(\"atom.xml\", entries=entries)",
        ],
        T.format("atom"): [
            "+def test_build_atom():",
            "+    assert build_atom([]).startswith(\"<?xml\")",
        ],
    }, ["todo_without_issue"], []),
    _pr("pr-28", "serve: debug console endpoint", "Handy for theme development.", {
        "src/inkwell/serve_debug.py": [
            "+def handle_debug(request: Request) -> Response:",
            "+    code = request.args.get(\"code\", \"\")",
            "+    namespace: dict[str, object] = {}",
            "+    exec(code, namespace)",
            "+    return Response(str(namespace.get(\"result\")))",
        ],
        T.format("serve_debug"): [
            "+def test_debug_endpoint(client):",
            "+    assert client.get(\"/_debug?code=result=1\").text == \"1\"",
        ],
    }, [], ["code_eval"]),
    _pr("pr-29", "themes: read theme.yaml safely", "", {
        "src/inkwell/themes/meta.py": [
            "+import yaml",
            "+",
            "+def load_theme_meta(path: Path) -> dict:",
            "+    with open(path, encoding=\"utf-8\") as fh:",
            "+        return yaml.safe_load(fh) or {}",
        ],
        T.format("theme_meta"): [
            "+def test_load_theme_meta(tmp_path):",
            "+    (tmp_path / \"theme.yaml\").write_text(\"name: x\")",
            "+    assert load_theme_meta(tmp_path / \"theme.yaml\") == {\"name\": \"x\"}",
        ],
    }, [], []),
    _pr("pr-30", "tests: shared fixtures for site builds", "", {
        "tests/conftest.py": [
            "+import pytest",
            "+",
            "+@pytest.fixture",
            "+def site(tmp_path):",
            "+    print(\"building fixture site in\", tmp_path)",
            "+    return make_site(tmp_path)",
        ],
    }, [], []),
    _pr("pr-31", "plugins: log the traceback of failing hooks", "", {
        "src/inkwell/plugins/runner.py": [
            " def run_hook(hook: Hook, page: Page) -> None:",
            "-    hook(page)",
            "+    try:",
            "+        hook(page)",
            "+    except Exception:",
            "+        logger.exception(\"hook %s failed on %s\", hook.__name__, page.path)",
        ],
        T.format("runner"): [
            "+def test_failing_hook_logged(caplog):",
            "+    run_hook(boom, page)",
            "+    assert \"failed\" in caplog.text",
        ],
    }, [], []),
    _pr("pr-32", "assets: download remote images at build time", "", {
        "src/inkwell/assets/remote.py": [
            "+import ssl",
            "+import urllib.request",
            "+",
            "+def download(url: str) -> bytes:",
            "+    ctx = ssl._create_unverified_context()",
            "+    with urllib.request.urlopen(url, context=ctx) as resp:",
            "+        return resp.read()",
        ],
        T.format("remote"): [
            "+def test_download(https_server):",
            "+    assert download(https_server.url) == b\"img\"",
        ],
    }, [], ["insecure_tls"]),
    _pr("pr-33", "chore: helper to reset a search table", "", {
        "src/inkwell/search_admin.py": [
            "+def reset_table(conn: sqlite3.Connection, table_name: str) -> None:",
            "+    conn.execute(\"DROP TABLE IF EXISTS %s\" % table_name)",
            "+    conn.execute(\"CREATE TABLE %s (url TEXT, title TEXT)\" % table_name)",
        ],
        T.format("search_admin"): [
            "+def test_reset(db):",
            "+    reset_table(db, \"pages\")",
        ],
    }, [], ["sql_injection"]),
    _pr("pr-34", "feat(plugins): typed registration API", "Part of #174.", {
        "src/inkwell/plugins/api.py": [
            "+from typing import Callable",
            "+",
            "+HookFn = Callable[[\"Page\"], None]",
            "+",
            "+def register_hook(name: str, fn: HookFn, *, priority: int = 100) -> None:",
            "+    _HOOKS.setdefault(name, []).append((priority, fn))",
            "+    _HOOKS[name].sort(key=lambda item: item[0])",
        ],
        T.format("api"): [
            "+def test_register_hook_priority():",
            "+    register_hook(\"x\", f, priority=1)",
        ],
    }, [], []),
    _pr("pr-35", "themes: install themes from a zip archive", "`inkwell theme install theme.zip`", {
        "src/inkwell/themes/install.py": [
            "+import os",
            "+import zipfile",
            "+",
            "+def install_theme(archive: str, dest: str) -> None:",
            "+    with zipfile.ZipFile(archive) as zf:",
            "+        for member in zf.namelist():",
            "+            target = os.path.join(dest, member)",
            "+            os.makedirs(os.path.dirname(target), exist_ok=True)",
            "+            with open(target, \"wb\") as fh:",
            "+                fh.write(zf.read(member))",
        ],
        T.format("install"): [
            "+def test_install_theme(tmp_path, theme_zip):",
            "+    install_theme(theme_zip, str(tmp_path))",
        ],
    }, [], ["path_traversal"]),
    _pr("pr-36", "feat: share links use the secrets module", "Replaces #15's approach.", {
        "src/inkwell/share.py": [
            "-import random",
            "+import secrets",
            "+",
            "+def new_share_token(nbytes: int = 18) -> str:",
            "+    return secrets.token_urlsafe(nbytes)",
        ],
        T.format("share"): [
            "+def test_token_unique():",
            "+    assert new_share_token() != new_share_token()",
        ],
    }, [], []),
    _pr("pr-37", "feat: render math with an external katex binary", "", {
        "src/inkwell/markdown/math.py": [
            "+import subprocess",
            "+",
            "+def render_math(tex: str) -> str:",
            "+    out = subprocess.check_output(\"katex --display-mode '\" + tex + \"'\", shell=True)",
            "+    return out.decode()",
        ],
        T.format("math"): [
            "+def test_render_math():",
            "+    assert \"katex\" in render_math(\"x^2\")",
        ],
    }, [], ["command_injection"]),
    _pr("pr-38", "feat: legacy config import", "", {
        "src/inkwell/legacy.py": [
            "+import marshal",
            "+from pathlib import Path",
            "+",
            "+def import_legacy_config(path: Path) -> dict:",
            "+    return marshal.loads(path.read_bytes())",
        ],
    }, ["missing_tests"], ["unsafe_deserialization"]),
    _pr("pr-39", "feat: S3 upload for build artifacts", "", {
        "scripts/upload.py": [
            "+AWS_SECRET_ACCESS_KEY = \"q7Zr9vT2LmX4pK8sW1nB6yH3cF0dJ5gE2uR7tA9z\"",
            "+",
            "+def upload(path):",
            "+    client = boto3.client(\"s3\", aws_secret_access_key=AWS_SECRET_ACCESS_KEY)",
            "+    client.upload_file(path, \"inkwell-artifacts\", path)",
        ],
    }, [], ["hardcoded_secret"]),
    _pr("pr-40", "feat(cli): `inkwell open` opens the built site", "", {
        "src/inkwell/cli.py": [
            "+@cli.command()",
            "+def open_site(config):",
            "+    import webbrowser",
            "+    webbrowser.open(config.output_dir.as_uri())",
        ],
        T.format("cli_open"): [
            "+def test_open_site(monkeypatch, runner):",
            "+    monkeypatch.setattr(webbrowser, \"open\", lambda url: None)",
        ],
    }, ["missing_type_hints"], []),
]
