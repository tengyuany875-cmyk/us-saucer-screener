"""Shared category navigation; standard library only for weekly compatibility."""
NAV_CSS = """
.category-nav { display:grid; grid-template-columns:1.25fr 1fr; gap:4px; padding:4px;
  margin-bottom:22px; border:1px solid var(--line); border-radius:12px; background:var(--surface); }
.category-nav a { display:flex; align-items:center; justify-content:center; min-height:46px;
  padding:6px 4px; border-radius:8px; color:var(--muted); text-decoration:none; font-size:13px; font-weight:600; }
.category-nav a[aria-current="page"] { color:var(--accent); background:var(--accent-soft); }
@media(max-width:350px) { .category-nav a { font-size:12px; } }
"""


def nav(active: str) -> str:
    return '<nav class="category-nav" aria-label="スクリーナーのカテゴリー">' + ''.join(
        f'<a href="{path}"' + (' aria-current="page"' if key == active else '') + f'>{label}</a>'
        for key,path,label in [('weekly','index.html','週足ソーサーボトム'),('daily','daily.html','日足押し目')]) + '</nav>'


def add_navigation(source: str) -> str:
    if '<main>' not in source or '</style>' not in source:
        raise ValueError('Weekly HTML structure is unavailable')
    if '<!-- categories:start -->' in source:
        return source
    source = source.replace('</style>',NAV_CSS+'</style>',1)
    return source.replace('<main>','<main><!-- categories:start -->'+nav('weekly')+'<!-- categories:end -->',1)
