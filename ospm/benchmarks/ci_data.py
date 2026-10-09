"""Hand-labelled CI failure logs across ecosystems (pytest, go, jest, cargo, gradle, docker, scanners...).

category: test_failure | lint | type_check | build | dependency | timeout | infra_flaky | security_scan
"""

from __future__ import annotations


def _c(cid, name, category, log, conclusion="failure"):
    return {"id": cid, "name": name, "conclusion": conclusion, "category": category, "log": log.strip("\n")}


CI_LOGS = [
    # ------------------------------------------------------------------ test_failure
    _c("ci-01", "tests (3.12)", "test_failure", """
2026-03-02T10:01:12Z ============================= test session starts ==============================
2026-03-02T10:01:12Z collected 412 items
2026-03-02T10:01:40Z tests/test_feeds.py ........F...
2026-03-02T10:01:41Z ___________________________ test_rss_pubdate_utc ___________________________
2026-03-02T10:01:41Z     def test_rss_pubdate_utc():
2026-03-02T10:01:41Z >       assert item.pub_date.endswith("+0000")
2026-03-02T10:01:41Z E       AssertionError: assert False
2026-03-02T10:01:41Z E        +  where False = <built-in method endswith of str object>('+0000')
2026-03-02T10:01:45Z =========================== short test summary info ============================
2026-03-02T10:01:45Z FAILED tests/test_feeds.py::test_rss_pubdate_utc - AssertionError: assert False
2026-03-02T10:01:45Z ======================== 1 failed, 411 passed in 33.10s =========================
2026-03-02T10:01:45Z ##[error]Process completed with exit code 1.
"""),
    _c("ci-02", "unit-tests", "test_failure", """
--- FAIL: TestSlugify (0.00s)
    --- FAIL: TestSlugify/unicode_title (0.00s)
        slug_test.go:41: Slugify("Café au lait") = "caf-au-lait", want "cafe-au-lait"
FAIL
FAIL	github.com/inkwell/inkwell-go/slug	0.012s
ok  	github.com/inkwell/inkwell-go/render	0.231s
FAIL
Error: Process completed with exit code 1.
"""),
    _c("ci-03", "test (node 20)", "test_failure", """
> inkwell-theme@2.1.0 test
> jest --ci

 FAIL  src/__tests__/menu.test.ts
  ● Menu › highlights the active page

    expect(received).toContain(expected) // indexOf

    Expected substring: "active"
    Received string:    "menu-item"

      18 |     render(<Menu current="/about" />);
    > 19 |     expect(screen.getByText("About").className).toContain("active");

Tests:       1 failed, 57 passed, 58 total
npm ERR! Test failed.  See above for more details.
"""),
    _c("ci-04", "pytest", "test_failure", """
tests/test_cli.py::test_new_into_git_repo FAILED                          [ 61%]
=================================== FAILURES ===================================
___________________________ test_new_into_git_repo ____________________________
tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-0/test_new_into_git_repo0')
    def test_new_into_git_repo(tmp_path):
        (tmp_path / ".git").mkdir()
>       result = runner.invoke(cli, ["new", str(tmp_path)])
E       inkwell.errors.ProjectError: directory is not empty: /tmp/pytest-of-runner/pytest-0/test_new_into_git_repo0
src/inkwell/scaffold.py:33: ProjectError
FAILED tests/test_cli.py::test_new_into_git_repo - inkwell.errors.ProjectError: directory is not empty
1 failed, 230 passed, 2 skipped in 19.88s
"""),
    _c("ci-05", "cargo test", "test_failure", """
running 24 tests
test parser::tests::parses_front_matter ... ok
test parser::tests::table_alignment ... FAILED

failures:

---- parser::tests::table_alignment stdout ----
thread 'parser::tests::table_alignment' panicked at src/parser.rs:212:9:
assertion `left == right` failed
  left: None
 right: Some(Center)

failures:
    parser::tests::table_alignment

test result: FAILED. 23 passed; 1 failed; 0 ignored; 0 measured
error: test failed, to rerun pass `--lib`
"""),
    _c("ci-06", "integration", "test_failure", """
> Task :core:test

com.inkwell.core.PermalinkTest > midnightPostKeepsLocalDate() FAILED
    org.opentest4j.AssertionFailedError: expected: <2024/05/01> but was: <2024/05/02>
        at com.inkwell.core.PermalinkTest.midnightPostKeepsLocalDate(PermalinkTest.java:58)

38 tests completed, 1 failed

FAILURE: Build failed with an exception.
* What went wrong:
Execution failed for task ':core:test'.
> There were failing tests. See the report at: file:///home/runner/work/core/build/reports/tests/test/index.html
"""),
    # ------------------------------------------------------------------ lint
    _c("ci-07", "lint", "lint", """
Run ruff check .
src/inkwell/feeds/rss.py:12:1: F401 [*] `datetime.timezone` imported but unused
src/inkwell/cli.py:88:89: E501 Line too long (104 > 88)
Found 2 errors.
[*] 1 fixable with the `--fix` option.
Error: Process completed with exit code 1.
"""),
    _c("ci-08", "format", "lint", """
Run black --check src tests
would reformat src/inkwell/markdown/tables.py
would reformat tests/test_tables.py

Oh no! 💥 💔 💥
2 files would be reformatted, 87 files would be left unchanged.
Error: Process completed with exit code 1.
"""),
    _c("ci-09", "eslint", "lint", """
> eslint . --max-warnings 0

/home/runner/work/theme/src/menu.tsx
  12:7  error  'unusedVar' is assigned a value but never used  no-unused-vars
  30:5  error  Expected '===' and instead saw '=='              eqeqeq

✖ 2 problems (2 errors, 0 warnings)
Error: Process completed with exit code 1.
"""),
    _c("ci-10", "pre-commit", "lint", """
trim trailing whitespace.................................................Failed
- hook id: trailing-whitespace
- exit code: 1
- files were modified by this hook

Fixing docs/plugins.md

check yaml...............................................................Passed
end of files.............................................................Passed
Error: Process completed with exit code 1.
"""),
    _c("ci-11", "golangci-lint", "lint", """
run golangci-lint
slug/slug.go:27:2: ineffectual assignment to err (ineffassign)
	err = normalize(s)
	^
render/html.go:104:15: Error return value of `w.Write` is not checked (errcheck)
	w.Write(buf)
	        ^
level=error msg="issues found"
Error: issues found
"""),
    _c("ci-12", "flake8", "lint", """
flake8 src/ tests/
src/inkwell/plugins/loader.py:44:5: E722 do not use bare 'except'
src/inkwell/plugins/loader.py:51:80: W291 trailing whitespace
tests/test_loader.py:3:1: F811 redefinition of unused 'pytest' from line 1
Error: Process completed with exit code 1.
"""),
    # ------------------------------------------------------------------ type_check
    _c("ci-13", "typecheck", "type_check", """
Run mypy src
src/inkwell/config.py:58: error: Incompatible types in assignment (expression has type "str | None", variable has type "str")  [assignment]
src/inkwell/feeds/atom.py:22: error: Argument 1 to "format_date" has incompatible type "date"; expected "datetime"  [arg-type]
Found 2 errors in 2 files (checked 64 source files)
Error: Process completed with exit code 1.
"""),
    _c("ci-14", "tsc", "type_check", """
> tsc --noEmit

src/menu.tsx(19,24): error TS2339: Property 'current' does not exist on type 'MenuProps'.
src/utils/url.ts(7,3): error TS2322: Type 'string | undefined' is not assignable to type 'string'.

Found 2 errors in 2 files.
Error: Process completed with exit code 2.
"""),
    _c("ci-15", "pyright", "type_check", """
pyright 1.1.380
/home/runner/work/inkwell/src/inkwell/i18n/locale.py
  /home/runner/work/inkwell/src/inkwell/i18n/locale.py:31:16 - error: "lang" is possibly unbound (reportPossiblyUnbound)
  /home/runner/work/inkwell/src/inkwell/i18n/locale.py:40:12 - error: Return type "list[str] | None" is not assignable to declared return type "list[str]"
2 errors, 0 warnings, 0 informations
"""),
    _c("ci-16", "mypy (strict)", "type_check", """
src/inkwell/plugins/api.py:14: error: Function is missing a return type annotation  [no-untyped-def]
src/inkwell/plugins/api.py:14: note: Use "-> None" if function does not return a value
src/inkwell/plugins/api.py:27: error: Call to untyped function "register" in typed context  [no-untyped-call]
Found 2 errors in 1 file (checked 64 source files)
"""),
    _c("ci-17", "type check", "type_check", """
Run python -m mypy --config-file pyproject.toml
src/inkwell/assets/images.py:73: error: Item "None" of "Image | None" has no attribute "resize"  [union-attr]
Found 1 error in 1 file (checked 64 source files)
##[error]Process completed with exit code 1.
"""),
    _c("ci-18", "typecheck-theme", "type_check", """
> vue-tsc --noEmit
src/components/LangSwitcher.vue:12:7 - error TS2345: Argument of type 'number' is not assignable to parameter of type 'string'.
12       setLang(index)
         ~~~~~~~~~~~~~~
Found 1 error.
"""),
    # ------------------------------------------------------------------ build
    _c("ci-19", "build wheel", "build", """
Run python -m build
* Building wheel...
Traceback (most recent call last):
  File "/tmp/build-env/lib/python3.12/site-packages/setuptools/build_meta.py", line 325, in _get_build_requires
  File "setup.py", line 14
    version=__version__
           ^
SyntaxError: invalid syntax. Perhaps you forgot a comma?
ERROR Backend subprocess exited when trying to invoke build_wheel
"""),
    _c("ci-20", "go build", "build", """
go build ./...
# github.com/inkwell/inkwell-go/render
render/html.go:88:2: undefined: escapeAttr
render/html.go:91:9: cannot use n (variable of type int) as string value in return statement
Error: Process completed with exit code 1.
"""),
    _c("ci-21", "cargo build", "build", """
   Compiling inkwell-core v0.4.0 (/home/runner/work/inkwell-core)
error[E0308]: mismatched types
  --> src/render.rs:41:20
   |
41 |     let n: usize = count_words(&page);
   |            -----   ^^^^^^^^^^^^^^^^^^ expected `usize`, found `u32`
error: could not compile `inkwell-core` (lib) due to 1 previous error
"""),
    _c("ci-22", "docker build", "build", """
#9 [builder 4/6] COPY src/ ./src
#9 ERROR: failed to calculate checksum of ref 7a1c: "/src": not found
------
 > [builder 4/6] COPY src/ ./src:
------
Dockerfile:12
--------------------
  12 | >>> COPY src/ ./src
--------------------
ERROR: failed to solve: failed to compute cache key: "/src": not found
"""),
    _c("ci-23", "webpack", "build", """
> theme@2.1.0 build
> webpack --mode production

ERROR in ./src/index.ts 3:0-35
Module not found: Error: Can't resolve './components/Footer' in '/home/runner/work/theme/src'
webpack 5.91.0 compiled with 1 error in 4211 ms
npm ERR! code ELIFECYCLE
"""),
    _c("ci-24", "docs build", "build", """
Running Sphinx v7.3.7
building [html]: targets for 48 source files that are out of date
/home/runner/work/inkwell/docs/plugins.rst:88: WARNING: undefined label: 'hook-reference'
Warning, treated as error:
/home/runner/work/inkwell/docs/plugins.rst:88:undefined label: 'hook-reference'
make: *** [Makefile:20: html] Error 2
"""),
    # ------------------------------------------------------------------ dependency
    _c("ci-25", "install", "dependency", """
Run pip install -e ".[dev]"
ERROR: Cannot install inkwell and inkwell[dev]==0.11.0 because these package versions have conflicting dependencies.

The conflict is caused by:
    inkwell 0.11.0 depends on markdown-it-py>=3.1
    mdit-py-plugins 0.4.0 depends on markdown-it-py<3.0

ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/
"""),
    _c("ci-26", "npm ci", "dependency", """
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! Found: react@19.0.0
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^18.0.0" from @headlessui/react@1.7.18
npm ERR! Fix the upstream dependency conflict, or retry with --force or --legacy-peer-deps
"""),
    _c("ci-27", "deps", "dependency", """
Collecting pygments==2.99.0
ERROR: Could not find a version that satisfies the requirement pygments==2.99.0 (from versions: 2.17.2, 2.18.0)
ERROR: No matching distribution found for pygments==2.99.0
Error: Process completed with exit code 1.
"""),
    _c("ci-28", "go mod", "dependency", """
go: downloading github.com/yuin/goldmark v1.9.0
go: github.com/inkwell/inkwell-go imports
	github.com/yuin/goldmark/extension/cjk: module github.com/yuin/goldmark@v1.9.0 found, but does not contain package github.com/yuin/goldmark/extension/cjk
Error: Process completed with exit code 1.
"""),
    _c("ci-29", "poetry install", "dependency", """
Installing dependencies from lock file

pyproject.toml changed significantly since poetry.lock was last generated. Run `poetry lock [--no-update]` to fix the lock file.
Error: Process completed with exit code 1.
"""),
    _c("ci-30", "cargo fetch", "dependency", """
    Updating crates.io index
error: failed to select a version for the requirement `pulldown-cmark = "^0.15"`
candidate versions found which didn't match: 0.12.2, 0.12.1, 0.12.0, ...
location searched: crates.io index
required by package `inkwell-core v0.4.0 (/home/runner/work/inkwell-core)`
"""),
    # ------------------------------------------------------------------ timeout
    _c("ci-31", "tests (3.11)", "timeout", """
tests/test_serve.py::test_live_reload_websocket
##[error]The job running on runner GitHub Actions 12 has exceeded the maximum execution time of 30 minutes.
##[error]The operation was canceled.
""", conclusion="cancelled"),
    _c("ci-32", "e2e", "timeout", """
Running 14 tests using 2 workers
  ✓  [chromium] › home.spec.ts:3:1 › renders home page (1.2s)
  ✘  [chromium] › search.spec.ts:8:1 › search returns results (30.0s)

  1) [chromium] › search.spec.ts:8:1 › search returns results
    Test timeout of 30000ms exceeded.
    Error: page.waitForSelector: Target closed
"""),
    _c("ci-33", "pytest -x", "timeout", """
tests/test_watch.py::test_rebuild_on_partial_change
+++++++++++++++++++++++++++++++++++ Timeout +++++++++++++++++++++++++++++++++++
~~~~~~~~~~~~~~~~~~~~~~~~~ Stack of Thread-2 (140230) ~~~~~~~~~~~~~~~~~~~~~~~~~~
  File "src/inkwell/watch.py", line 77, in _loop
    event = self._queue.get()
+++++++++++++++++++++++++++++++++++ Timeout +++++++++++++++++++++++++++++++++++
Failed: Timeout >60.0s
"""),
    _c("ci-34", "go test -race", "timeout", """
panic: test timed out after 10m0s
	running tests:
		TestWatcherDebounce (9m58s)

goroutine 34 [chan receive, 9 minutes]:
github.com/inkwell/inkwell-go/watch.(*Watcher).wait(...)
FAIL	github.com/inkwell/inkwell-go/watch	600.021s
"""),
    _c("ci-35", "benchmarks", "timeout", """
Running benchmark suite (4000 pages)...
[bench] build_full: 182.4s
##[error]The job running on runner ubuntu-latest has exceeded the maximum execution time of 15 minutes.
""", conclusion="timed_out"),
    _c("ci-36", "windows tests", "timeout", """
tests\\test_serve.py::test_shutdown_releases_port
Error: The operation was canceled.
##[error]The job has exceeded the maximum execution time of 45m0s
""", conclusion="cancelled"),
    # ------------------------------------------------------------------ infra_flaky
    _c("ci-37", "setup python", "infra_flaky", """
Run actions/setup-python@v5
Error: Failed to download https://github.com/actions/python-versions/releases/download/3.12.3/python-3.12.3-linux-22.04-x64.tar.gz
Error: read ECONNRESET
"""),
    _c("ci-38", "install", "infra_flaky", """
Collecting jinja2>=3.1
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out. (read timeout=15)")': /packages/jinja2-3.1.4.tar.gz
ERROR: Could not install packages due to an OSError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Max retries exceeded (Caused by ReadTimeoutError)
"""),
    _c("ci-39", "npm ci", "infra_flaky", """
npm ERR! code E503
npm ERR! 503 Service Unavailable - GET https://registry.npmjs.org/postcss - Service Unavailable
npm ERR! A complete log of this run can be found in: /home/runner/.npm/_logs/debug-0.log
"""),
    _c("ci-40", "docker publish", "infra_flaky", """
#12 pushing layers
#12 ERROR: failed to push ghcr.io/inkwell/inkwell:pr-412: unexpected status from PUT request: 502 Bad Gateway
------
ERROR: failed to solve: failed to push ghcr.io/inkwell/inkwell:pr-412: 502 Bad Gateway
"""),
    _c("ci-41", "macos tests", "infra_flaky", """
The hosted runner: GitHub Actions 34 lost communication with the server. Anything in your workflow that
terminates the runner process, starves it for CPU/Memory, or blocks its network access can cause this error.
"""),
    _c("ci-42", "checkout", "infra_flaky", """
Run actions/checkout@v4
Fetching the repository
  /usr/bin/git -c protocol.version=2 fetch --no-tags --prune --depth=1 origin +refs/pull/412/merge
  Error: fatal: unable to access 'https://github.com/inkwell/inkwell/': The requested URL returned error: 500
The process '/usr/bin/git' failed with exit code 128
Waiting 13 seconds before trying again
  Error: fatal: unable to access 'https://github.com/inkwell/inkwell/': The requested URL returned error: 500
"""),
    # ------------------------------------------------------------------ security_scan
    _c("ci-43", "pip-audit", "security_scan", """
Run pip-audit -r requirements.txt
Found 1 known vulnerability in 1 package
Name   Version ID                  Fix Versions
------ ------- ------------------- ------------
jinja2 3.1.3   GHSA-h75v-3vvj-5mfj 3.1.4
Error: Process completed with exit code 1.
"""),
    _c("ci-44", "npm audit", "security_scan", """
# npm audit report

postcss  <8.4.31
Severity: moderate
PostCSS line return parsing error - https://github.com/advisories/GHSA-7fh5-64p2-3v2j
fix available via `npm audit fix`
1 moderate severity vulnerability
"""),
    _c("ci-45", "gitleaks", "security_scan", """
    ○
    │╲
    │ ○
    ○ ░
    ░    gitleaks

Finding:     AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
Secret:      wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
RuleID:      aws-secret-access-key
File:        scripts/deploy.sh
Line:        7
leaks found: 1
"""),
    _c("ci-46", "trivy", "security_scan", """
ghcr.io/inkwell/inkwell:pr-412 (debian 12.5)
Total: 2 (HIGH: 1, CRITICAL: 1)
│ openssl  │ CVE-2024-5535 │ CRITICAL │ 3.0.11-1  │ 3.0.13-1 │ openssl: SSL_select_next_proto buffer overread
│ libexpat │ CVE-2024-45491│ HIGH     │ 2.5.0-1   │ 2.5.0-1+deb12u1 │ libexpat: integer overflow
Error: Process completed with exit code 1.
"""),
    _c("ci-47", "bandit", "security_scan", """
Run bandit -r src -ll
>> Issue: [B602:subprocess_popen_with_shell_equals_true] subprocess call with shell=True identified, security issue.
   Severity: High   Confidence: High
   Location: src/inkwell/plugins/exec.py:31:8
--------------------------------------------------
Code scanned: Total lines of code: 6123
Total issues (by severity): High: 1
"""),
    _c("ci-48", "codeql", "security_scan", """
Analyzing python...
Interpreting results.
CodeQL found 1 new alert:
  py/path-injection (error): This path depends on a user-provided value. src/inkwell/serve.py:142
Error: Code Scanning could not process the submitted SARIF file: 1 high severity alert blocks merge per branch protection.
"""),
]
