"""Hand-labelled issue-triage benchmark for the fictional project "inkwell" (see examples/inkwell).

Labelling guide
* kind: bug | feature | docs | question  (None for spam: kind is not meaningful there)
* decision: accept | needs-info | out-of-scope | duplicate | spam
* priority: P0 critical (data loss, security, crash/broken install for most users)
            P1 major (core functionality broken without workaround, significant regression, high-demand roadmap item)
            P2 normal, P3 minor/cosmetic/questions; every non-accepted issue is P3.
"""

from __future__ import annotations


def _i(number, title, body, kind, decision, priority, reactions=0, duplicate_of=None):
    return {
        "number": number,
        "title": title,
        "body": body.strip("\n"),
        "reactions": reactions,
        "labels": {"kind": kind, "decision": decision, "priority": priority},
        "duplicate_of": duplicate_of,
    }


ISSUES = [
    _i(101, "Live reload stops working after editing a theme partial", """
**inkwell version:** 0.9.2, Python 3.12, macOS 14

Steps to reproduce:
1. `inkwell serve`
2. edit `themes/minimal/partials/footer.html` and save

Expected: the browser refreshes with the new footer.
Actual: nothing happens; the terminal shows no rebuild. Editing Markdown content still triggers a reload,
only changes to partials are ignored. Restarting `serve` picks the change up.
""", "bug", "accept", "P1", reactions=6),
    _i(102, "Support for Mermaid diagrams in code fences", """
It would be great if fenced code blocks with the language `mermaid` were rendered as diagrams
(e.g. by emitting a `<div class="mermaid">` and letting themes include mermaid.js). Many docs sites
need sequence and flow diagrams. Happy to help implement it as a Markdown extension.
""", "feature", "accept", "P2", reactions=4),
    _i(103, "Typo in quickstart: 'inkwel build'", """
The quickstart page says to run `inkwel build` (missing an L). Copy-pasting it fails, which is confusing for
first-time users.
""", "docs", "accept", "P3"),
    _i(104, "How do I set a custom permalink format for posts?", """
I want my posts at /blog/2024/my-post/ instead of /posts/my-post/. I looked through inkwell.toml options but
couldn't find anything about URL patterns. Is there a setting for this?
""", "question", "accept", "P3"),
    _i(105, "Best SEO backlinks cheap!!!", """
Boost your ranking today!!! https://seo-boost.example.com https://cheap-links.example.net
https://rank-fast.example.org  Contact us for packages.
""", None, "spam", "P3"),
    _i(106, "it doesnt work", """
build fails on my machine pls fix
""", "bug", "needs-info", "P3"),
    _i(107, "Add a WYSIWYG editor for writing posts in the browser", """
Writing Markdown by hand is hard for non-technical members of our team. Could inkwell ship a WYSIWYG editor
that runs in the browser and saves posts directly?
""", "feature", "out-of-scope", "P3", reactions=2),
    _i(108, "inkwell build deletes files outside the output directory", """
Version 0.10.0. With `output_dir = "../site"` in inkwell.toml, running `inkwell build --clean` removed every file
in the PARENT directory of my project, not just ../site. I lost uncommitted work in sibling folders.

Steps: create project, set output_dir to a relative path with `..`, run `inkwell build --clean`.
Expected: only the output directory is cleaned. Actual: the parent directory is wiped.
""", "bug", "accept", "P0", reactions=12),
    _i(109, "Incremental builds: only re-render changed pages", """
On my 4,000-page site every `inkwell build` takes ~3 minutes even if I change a single post. It would be a big
improvement if inkwell tracked dependencies (content file -> template -> output) and only re-rendered the pages
whose inputs changed.
""", "feature", "accept", "P1", reactions=25),
    _i(110, "Footnotes render in wrong order when a footnote is referenced twice", """
inkwell 0.9.4. Markdown:

```
First[^a], second[^b], again[^a].
[^a]: Alpha
[^b]: Beta
```

Expected footnote list: 1 Alpha, 2 Beta. Actual: Beta is numbered 1 and Alpha 2, and the back-links point to
the wrong references.
""", "bug", "accept", "P2"),
    _i(111, "Live reload broken when I change a partial template", """
When running the dev server, edits to files in my theme's partials folder don't trigger a reload. Content edits
do. I have to restart serve every time. Using 0.9.3.
""", "bug", "duplicate", "P3", duplicate_of=101),
    _i(112, "Is it possible to use inkwell with Tailwind CSS?", """
I'd like to build my theme with Tailwind. Do I need a separate build step, or is there a plugin that runs the
Tailwind CLI as part of `inkwell build`?
""", "question", "accept", "P3"),
    _i(113, "RSS feed uses local time instead of UTC for pubDate", """
inkwell 0.9.4 on a machine in UTC+2. The generated feed.xml has `<pubDate>` values that are two hours off and
have no timezone suffix, so feed readers show the wrong time. RFC 822 dates in RSS should include the offset.
""", "bug", "accept", "P2"),
    _i(114, "Plugin API documentation is missing the list of available hooks", """
The plugin guide shows how to register a hook but never lists which hooks exist or what arguments they receive.
I had to read the source to find `on_page_render`. A reference table would help plugin authors a lot.
""", "docs", "accept", "P2", reactions=3),
    _i(115, "Built-in hosting so I can publish with one command", """
Could inkwell offer hosting so that `inkwell publish` uploads my site to inkwell's servers and gives me a URL?
Then I wouldn't have to set up Netlify or a server.
""", "feature", "out-of-scope", "P3"),
    _i(116, "Crypto trading signals group - join now", """
Earn 300% monthly with our crypto signals!! Join telegram https://t.me/example-signals
https://signals.example.com https://pump.example.io
""", None, "spam", "P3"),
    _i(117, "Images broken", """
After updating, images on my site are broken. Help.
""", "bug", "needs-info", "P3"),
    _i(118, "Allow multiple taxonomies besides tags and categories", """
I run a recipe site and would like to define custom taxonomies like `cuisine` and `diet` in inkwell.toml, each
with its own listing pages and feeds, the same way tags and categories work today.
""", "feature", "accept", "P2", reactions=5),
    _i(119, "`inkwell serve` crashes on Windows with UnicodeDecodeError", """
inkwell 0.10.0, Windows 11, Python 3.12.

```
Traceback (most recent call last):
  File "...\\inkwell\\cli.py", line 212, in serve
  File "...\\inkwell\\watch.py", line 48, in _read
UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d in position 1203
```

Happens with any page that contains non-ASCII characters (e.g. curly quotes). `inkwell build` works fine;
only `serve` crashes. Looks like files are opened without encoding="utf-8".
""", "bug", "accept", "P1", reactions=9),
    _i(120, "What's the recommended way to deploy to GitHub Pages?", """
Is there an official GitHub Actions workflow for building an inkwell site and publishing it to GitHub Pages?
""", "question", "accept", "P3"),
    _i(121, "Add user login so readers can have accounts", """
It would be cool if readers could sign up and log in on my inkwell site to save their favourite posts.
""", "feature", "out-of-scope", "P3"),
    _i(122, "Security: path traversal in dev server allows reading arbitrary files", """
`inkwell serve` serves files for URLs like `/../../etc/passwd` (URL-encoded as %2e%2e%2f). Tested on 0.10.0:

```
curl http://localhost:8000/%2e%2e/%2e%2e/%2e%2e/etc/passwd
```

returns the file. The dev server binds to 0.0.0.0 by default, so anyone on the same network can read files
from the developer's machine.
""", "bug", "accept", "P0", reactions=7),
    _i(123, "Broken link in README to the themes gallery", """
The "Browse themes" link in the README points to https://inkwell.dev/themes which returns 404. The gallery
seems to have moved to /gallery.
""", "docs", "accept", "P3"),
    _i(124, "Add `--drafts` flag to build command to include draft posts", """
Posts with `draft: true` in front matter are always skipped. For previewing on a staging server I'd like
`inkwell build --drafts` to include them.
""", "feature", "accept", "P2", reactions=3),
    _i(125, "Please make builds incremental, full rebuilds take minutes", """
Every build re-renders the whole site. With a few thousand pages this takes several minutes. Can inkwell only
rebuild pages that changed?
""", "feature", "duplicate", "P3", duplicate_of=109),
    _i(126, "Table alignment syntax ignored in Markdown tables", """
Using inkwell 0.9.4:

```
| Left | Center | Right |
|:-----|:------:|------:|
| a    |   b    |     c |
```

The generated `<td>` elements have no `style="text-align: ..."`/align attribute, so every column is left
aligned. GitHub renders the same table correctly.
""", "bug", "accept", "P2"),
    _i(127, "Error when building", """
I get an error when I run build.
""", "bug", "needs-info", "P3"),
    _i(128, "Comment system for blog posts", """
Please add a built-in comment system so visitors can leave comments under posts, with moderation.
""", "feature", "out-of-scope", "P3", reactions=1),
    _i(129, "Get 50% discount code on essay writing service", """
Students! Get your essays written by professionals. Use discount code ESSAY50 at https://essays.example.com
https://essays.example.com/order https://essays.example.com/prices
""", None, "spam", "P3"),
    _i(130, "Regression in 0.10.0: front matter dates parsed as strings", """
After upgrading from 0.9.4 to 0.10.0 all my posts are sorted alphabetically instead of by date. In templates
`page.date` is now a string ("2024-03-01") instead of a date, so `page.date.year` raises an UndefinedError.

Expected: dates in YAML front matter are parsed into date objects like in 0.9.x.
""", "bug", "accept", "P1", reactions=8),
    _i(131, "How can I add a custom Jinja2 filter?", """
I want a `| reading_time` filter in my templates. Where do I register custom filters - in a plugin or in the
theme?
""", "question", "accept", "P3"),
    _i(132, "Responsive images: generate WebP variants", """
The image pipeline already creates resized JPEG/PNG variants for `srcset`. Generating WebP versions as well
(with a `<picture>` element) would cut page weight significantly.
""", "feature", "accept", "P2", reactions=6),
    _i(133, "Minor: CLI help text for `serve` has inconsistent capitalization", """
`inkwell serve --help` shows "--port  Port to listen on" but "--host  the host to bind". Purely cosmetic.
""", "bug", "accept", "P3"),
    _i(134, "Document how to write a theme from scratch", """
The docs explain how to customise the default theme, but there is no guide for creating a new theme: required
templates, available variables, how assets are copied. A step-by-step tutorial would be very helpful.
""", "docs", "accept", "P2", reactions=4),
    _i(135, "Add a shopping cart plugin with Stripe checkout", """
I'd like to sell t-shirts from my inkwell site. Can you add a shopping cart with Stripe payments?
""", "feature", "out-of-scope", "P3"),
    _i(136, "Sitemap includes draft pages", """
inkwell 0.10.0: pages with `draft: true` are not rendered (correct) but they still appear in sitemap.xml,
which makes search engines hit 404s. Reproduced with a fresh `inkwell new` site and one draft post.
""", "bug", "accept", "P2"),
    _i(137, "serve not working", """
localhost doesn't load anymore
""", "bug", "needs-info", "P3"),
    _i(138, "Multilingual sites: per-language content folders", """
For our docs we need English, German and Japanese versions. Proposal: `content/en/`, `content/de/`, ... with
language-specific URLs (/de/...), translated strings in the theme, and hreflang links between translations.
""", "feature", "accept", "P1", reactions=18),
    _i(139, "UnicodeDecodeError on Windows when running serve", """
On Windows the dev server dies with `UnicodeDecodeError: 'charmap' codec can't decode byte` as soon as a page
contains an em dash. Build works. Windows 10, inkwell 0.10.0.
""", "bug", "duplicate", "P3", duplicate_of=119),
    _i(140, "pip install inkwell 0.10.1 fails: templates not included in the wheel", """
`pip install inkwell==0.10.1` succeeds, but every command then fails:

```
jinja2.exceptions.TemplateNotFound: base.html
```

The 0.10.1 wheel on PyPI contains no `inkwell/themes/default/*.html` files - looks like package data was
dropped from the build config. 0.10.0 is fine. This breaks every new installation.
""", "bug", "accept", "P0", reactions=15),
    _i(141, "Can I use inkwell to generate documentation for a Python library?", """
Does inkwell have something like autodoc to pull docstrings from my package, or is it only for blogs?
""", "question", "accept", "P3"),
    _i(142, "Visit my casino site for free spins", """
Free spins every day!!! https://casino.example.com https://casino.example.com/bonus https://slots.example.net
""", None, "spam", "P3"),
    _i(143, "Integrate Google Analytics tracking by default", """
Every site needs analytics. inkwell should inject the Google Analytics tracking snippet into every page by
default and provide a dashboard of visitor statistics.
""", "feature", "out-of-scope", "P3"),
    _i(144, "Syntax highlighting fails for code blocks with language 'tsx'", """
inkwell 0.10.0. A fenced block tagged `tsx` is rendered as plain text and the build log shows
`WARNING: no lexer for alias 'tsx'`. `typescript` and `jsx` work. Pygments supports tsx since 2.12, so the alias
map in inkwell may be outdated.
""", "bug", "accept", "P2"),
    _i(145, "Clarify difference between `base_url` and `site_url` in config docs", """
The configuration reference lists both `base_url` and `site_url` with nearly identical descriptions. It's not
clear which one is used for feeds vs. links.
""", "docs", "accept", "P3"),
    _i(146, "Hook to modify page context before rendering", """
Plugins can transform Markdown but can't add variables to the template context. A `before_render(page, context)`
hook would let plugins inject computed data (e.g. related posts) without monkeypatching.
""", "feature", "accept", "P2", reactions=4),
    _i(147, "RSS problem", """
my feed reader shows weird stuff
""", "bug", "needs-info", "P3"),
    _i(148, "Admin panel to manage posts from the browser", """
Please add an admin panel where I can create, edit and delete posts and upload images through a web UI.
""", "feature", "out-of-scope", "P3", reactions=1),
    _i(149, "Build takes 10x longer since 0.10 with many images", """
Site with ~800 images. `inkwell build` on 0.9.4: 25s. On 0.10.0: 4m10s. Profiling shows every image is
re-encoded on every build - the image cache in `.inkwell-cache/` is never hit because the cache key now includes
the file's mtime in nanoseconds, which changes on checkout in CI.
""", "bug", "accept", "P1", reactions=5),
    _i(150, "Does inkwell support math equations with KaTeX?", """
I write a lot of maths. Is there built-in support for $...$ and $$...$$ with KaTeX or MathJax?
""", "question", "accept", "P3"),
    _i(151, "Nice to have: colored output in CLI", """
It would be nice to have colored log output in the terminal (warnings in yellow, errors in red). Low priority.
""", "feature", "accept", "P3"),
    _i(152, "Templates missing after installing 0.10.1 from PyPI", """
Fresh virtualenv, `pip install inkwell` gives 0.10.1 and `inkwell build` immediately fails with
TemplateNotFound base.html. Downgrading to 0.10.0 fixes it.
""", "bug", "duplicate", "P3", duplicate_of=140),
    _i(153, "Heading anchors not unique when two headings share the same text", """
A page with two `## Example` headings gets two elements with `id="example"`, so the table of contents and
direct links always jump to the first one. Expected `example` and `example-1` like most Markdown renderers.
inkwell 0.10.0.
""", "bug", "accept", "P2", reactions=2),
    _i(154, "Buy now: cheap followers for Instagram", """
Buy now 10k followers for $5!! https://followers.example.com https://followers.example.com/buy
https://growth.example.biz
""", None, "spam", "P3"),
    _i(155, "Add migration guide from Jekyll", """
Many people come from Jekyll. A guide that maps Jekyll concepts (_posts, _layouts, Liquid tags, _config.yml) to
inkwell equivalents would make migration much easier.
""", "docs", "accept", "P2", reactions=6),
    _i(156, "Dev server exposes .env file contents", """
Running `inkwell serve` in a project with a `.env` file: `curl http://localhost:8000/.env` returns my API keys.
The dev server serves dotfiles from the project root instead of only the output directory. inkwell 0.10.0.
""", "bug", "accept", "P0", reactions=4),
    _i(157, "Server-side rendering so pages are generated per request", """
Instead of building everything up front, could inkwell run as a server that renders each Markdown page when
it's requested? That way I don't need to rebuild after edits.
""", "feature", "out-of-scope", "P3"),
    _i(158, "Plugin crash", """
My plugin crashes inkwell.
""", "bug", "needs-info", "P3"),
    _i(159, "Atom feed per tag", """
Currently there is a single site-wide feed. Generating `/tags/<tag>/atom.xml` would let readers subscribe to
only the topics they care about.
""", "feature", "accept", "P2", reactions=3),
    _i(160, "`inkwell new` fails if target directory contains a .git folder", """
```
$ git init blog && inkwell new blog
Error: directory 'blog' is not empty
```

A directory containing only `.git` (and maybe `.gitignore`) should be treated as empty so you can scaffold
into a freshly cloned repository. inkwell 0.10.0, Linux.
""", "bug", "accept", "P2"),
    _i(161, "Where should static files like favicon.ico go?", """
Should I put favicon.ico and robots.txt in the theme's static folder or in the content folder? They are not
copied to the output right now.
""", "question", "accept", "P3"),
    _i(162, "Support TOML front matter in addition to YAML", """
Hugo users are used to `+++` TOML front matter. Supporting it (detected by the delimiter) would ease migration
and avoid YAML's surprising type coercions.
""", "feature", "accept", "P2", reactions=2),
    _i(163, "pubDate in RSS feed is not in UTC", """
The RSS pubDate is generated in my local timezone with no offset, so readers display the wrong publish time.
""", "bug", "duplicate", "P3", duplicate_of=113),
    _i(164, "Add a database backend to store posts in PostgreSQL", """
Markdown files don't scale for our newsroom. Could inkwell store posts in PostgreSQL and query them at build
or request time?
""", "feature", "out-of-scope", "P3"),
    _i(165, "Plugin hooks run twice during `serve` rebuilds", """
With a plugin that registers `on_page_render`, the hook is called twice per page on every rebuild triggered by
`serve` (but once during `build`). My plugin appends a footer, so pages end up with two footers. inkwell 0.10.0.
Minimal plugin attached below:

```python
def on_page_render(page):
    page.html += "<footer>x</footer>"
```
""", "bug", "accept", "P1", reactions=3),
    _i(166, "Spelling mistake in CLI reference ('recieve')", """
The CLI reference page says "the server will recieve requests". Should be "receive".
""", "docs", "accept", "P3"),
    _i(167, "Business proposal", """
Hello dear, I am a prince with a large inheritance and need your help to transfer funds. Please send your bank
details to claim your share. https://prince.example.com https://funds.example.com https://bank.example.org
""", None, "spam", "P3"),
    _i(168, "Relative links between Markdown files are not rewritten to .html", """
In `content/guide/a.md` I link `[next](b.md)`. The output keeps `href="b.md"`, which 404s. Expected inkwell to
rewrite links to other content files into their output URLs. inkwell 0.10.0.
""", "bug", "accept", "P2", reactions=4),
    _i(169, "Expose word count and reading time to templates", """
Themes often show "5 min read". Exposing `page.word_count` and `page.reading_time` in the template context
would avoid every theme reimplementing it.
""", "feature", "accept", "P2"),
    _i(170, "Weird bug with themes", """
Something is off with my theme since the update.
""", "bug", "needs-info", "P3"),
    _i(171, "Is there a way to exclude a folder from the build?", """
I keep notes in `content/_private/` and don't want them published. Is there an ignore setting?
""", "question", "accept", "P3"),
    _i(172, "Crash on startup for all users on Python 3.13: ImportError", """
Every command fails immediately on Python 3.13:

```
ImportError: cannot import name 'cgi' from ...
```

inkwell 0.10.1 imports the `cgi` module, which was removed in Python 3.13. Nobody on 3.13 can use inkwell at all.
""", "bug", "accept", "P0", reactions=11),
    _i(173, "Visual drag-and-drop page builder", """
Could inkwell include a drag-and-drop page builder so designers can lay out landing pages visually?
""", "feature", "out-of-scope", "P3"),
    _i(174, "Stable plugin API with versioning for 1.0", """
Plugin authors need guarantees. Proposal: freeze the hook signatures for 1.0, add an `api_version` to plugin
metadata, and emit deprecation warnings for one minor release before removing anything.
""", "feature", "accept", "P1", reactions=30),
    _i(175, "Duplicate heading ids generated for repeated headings", """
If a page has the same heading text twice, both headings get the same id attribute and anchor links break.
""", "bug", "duplicate", "P3", duplicate_of=153),
    _i(176, "Make the live reload port configurable", """
The live reload websocket always uses port 35729, which conflicts with other tools I run. Please add a
`--livereload-port` option (and config key).
""", "feature", "accept", "P2"),
    _i(177, "Explain plugin load order in docs", """
When two plugins register the same hook, which runs first? The docs don't say, and it matters for plugins that
transform the same HTML.
""", "docs", "accept", "P2"),
    _i(178, "Free crypto airdrop claim now", """
Claim your free crypto airdrop before it ends!! https://airdrop.example.com https://claim.example.io
https://wallet.example.net
""", None, "spam", "P3"),
    _i(179, "Date-based permalinks off by one day for posts near midnight", """
A post with `date: 2024-05-01T23:30:00+02:00` gets the permalink /2024/05/02/... because the date is converted
to UTC before the permalink is formatted. Expected the permalink to use the date as written. inkwell 0.10.0.
""", "bug", "accept", "P2"),
    _i(180, "How do I enable footnotes?", """
The Markdown guide mentions footnotes but `[^1]` shows up literally in my pages. Do I need to enable an extension?
""", "question", "accept", "P3"),
    _i(181, "Can't build", """
build broke, error
""", "bug", "needs-info", "P3"),
    _i(182, "Email newsletter sending service built into inkwell", """
Please add the ability to collect subscriber emails and send newsletters when I publish a new post.
""", "feature", "out-of-scope", "P3"),
    _i(183, "Draft preview mode in `serve`", """
When running `inkwell serve` I'd like drafts to be rendered (with a visible "DRAFT" banner) so I can preview
them, while `build` keeps excluding them.
""", "feature", "accept", "P2", reactions=2),
    _i(184, "Typo in error message: 'Coudn't find config'", """
Running inkwell outside a project prints "Coudn't find config inkwell.toml". Should be "Couldn't".
""", "bug", "accept", "P3"),
    _i(185, "WebP image variants for responsive images", """
Feature request: when generating responsive image sizes, also produce WebP files and use a picture element.
""", "feature", "duplicate", "P3", duplicate_of=132),
    _i(186, "Incremental build cache not invalidated when theme changes", """
With the experimental `incremental = true`, changing `themes/x/base.html` does not re-render any pages; the old
layout stays until I delete `.inkwell-cache`. Templates should be part of each page's dependency set.
inkwell 0.11.0.dev3.
""", "bug", "accept", "P1", reactions=3),
    _i(187, "README badges point to old CI service", """
The build badge in the README still points to Travis CI, which hasn't been used for a year. It should use the
GitHub Actions badge.
""", "docs", "accept", "P3"),
    _i(188, "Let readers create accounts and bookmark posts", """
Feature idea: readers register on the site and can bookmark posts and get recommendations.
""", "feature", "out-of-scope", "P3"),
    _i(189, "Sitemap lastmod uses build time instead of page modification time", """
Every URL in sitemap.xml gets `<lastmod>` equal to the build time, so search engines think every page changed on
every deploy. It should use the page's `updated` front matter or the file's last commit date. inkwell 0.10.0.
""", "bug", "accept", "P2"),
    _i(190, "Top SEO services for your website", """
We offer top SEO services, guaranteed first page ranking. Contact us https://seo.example.com
https://seo.example.com/pricing https://seo.example.com/contact
""", None, "spam", "P3"),
    _i(191, "How to use inkwell with Netlify?", """
What build command and publish directory should I configure on Netlify for an inkwell site?
""", "question", "accept", "P3"),
    _i(192, "Build silently drops pages whose filenames contain spaces", """
inkwell 0.10.1: `content/my first post.md` is skipped without any warning, so it's missing from production.
We only noticed weeks later. 0.10.0 handled it (output `my-first-post/`). Probably the new glob pattern in
`discover_pages()`.
""", "bug", "accept", "P0", reactions=6),
    _i(193, "Language switcher helper for multilingual themes", """
Once multilingual sites land, themes will need a helper to list the current page's translations, e.g.
`{% for t in page.translations %}` with language name and URL.
""", "feature", "accept", "P2", reactions=3),
    _i(194, "Feeds are wrong", """
my feed is wrong please fix
""", "bug", "needs-info", "P3"),
    _i(195, "Tag pages not generated when tag contains a slash", """
A post tagged `ci/cd` produces no tag page and the tag link 404s. Expected the slash to be slugified (ci-cd).
inkwell 0.10.0.
""", "bug", "accept", "P2"),
    _i(196, "Can I write pages in reStructuredText?", """
I have existing docs in .rst. Can inkwell render reStructuredText, or only Markdown?
""", "question", "accept", "P3"),
]
