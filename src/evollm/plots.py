"""Visualisations of a run: lineage, occupancy, block usage, self-sufficiency.

Emits one self-contained HTML file — no CDN, no build step — so it can be
opened off the filesystem on a login node or published as-is.

Every panel is drawn against **its own room's step axis**. Rooms advance
independently by design (§4.5: "generations should not be in lockstep, only
tokens"), so a shared x-axis would misrepresent the run; instead each room's
line simply ends where that room got to.

Colours are the reference categorical palette, used in its documented order and
within its documented caps: four slots for the line charts (adjacent pairlist)
and the first three for the lineage panel (all-pairs).
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

# Reference categorical palette, light / dark. Order is the CVD-safety
# mechanism, not cosmetic — never cycle or reorder it.
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500"]
ORIGIN_ORDER = ["birth", "refill", "seed"]


def _read(run_dir: Path) -> dict[str, list[dict]]:
    rooms: dict[str, list[dict]] = {}
    for path in sorted((run_dir / "events").glob("*.jsonl")):
        events = []
        for line in path.open():
            line = line.strip()
            if line:
                events.append(json.loads(line))
        rooms[path.stem] = events
    return rooms


# ── data reduction ────────────────────────────────────────────────────────
def _series(events: list[dict], t0: float) -> dict:
    """Time series on the shared clock, in hours since the run began.

    Room step counters are independent and are meant to be (\u00a74.5), so
    plotting rooms against "step" put four different clocks on one axis. Every
    event carries a wall-clock stamp, so hours-since-start is the one axis on
    which the rooms are actually comparable.
    """
    pop, usage, backlog = [], [], []
    for e in events:
        if e["type"] != "occupancy":
            continue
        h = (e["t"] - t0) / 3600.0
        cap = e.get("capacity_blocks") or 1
        pop.append((h, e["agents"]))
        usage.append((h, 100.0 * (cap - e.get("free_blocks", 0)) / cap))
        backlog.append((h, e.get("mean_backlog", 0)))

    marks = []
    for e in events:
        if e["type"] == "birth" and e.get("origin") == "birth":
            marks.append(((e["t"] - t0) / 3600.0, 1))
        elif e["type"] == "refill":
            marks.append(((e["t"] - t0) / 3600.0, 0))
    marks.sort()
    selfsuf, window = [], 400
    for i in range(len(marks)):
        chunk = marks[max(0, i - window):i + 1]
        if len(chunk) >= 40:
            selfsuf.append((marks[i][0],
                            100.0 * sum(c[1] for c in chunk) / len(chunk)))

    # Departures per hour: how much of the population is in motion, and when.
    outs = sorted((e["t"] - t0) / 3600.0 for e in events if e["type"] == "move")
    rate, span = [], 0.25
    if outs:
        edge = outs[0]
        while edge <= outs[-1]:
            n = sum(1 for o in outs if edge <= o < edge + span)
            rate.append((edge + span / 2, n / span))
            edge += span
    return {"population": pop, "usage": usage, "backlog": backlog,
            "selfsuf": _thin(selfsuf, 700), "moves": rate}


def _thin(points: list, limit: int) -> list:
    if len(points) <= limit:
        return points
    stride = len(points) / limit
    return [points[int(i * stride)] for i in range(limit)]


def _population(rooms: dict[str, list[dict]]) -> dict:
    """One record per agent across the whole world, on a single clock.

    Rooms keep independent step counters, but every event also carries a
    wall-clock stamp, and a move lands in its destination a median 2.7s later
    — so `t` orders events across rooms exactly, with nothing to estimate.
    An agent's room at any moment is simply the log its events are landing in,
    which makes migration fall out of the data rather than needing its own
    event type.
    """
    per: dict[str, list[tuple[float, str, dict]]] = defaultdict(list)
    for room, events in rooms.items():
        for e in events:
            a = e.get("agent")
            if a:
                per[a].append((e["t"], room, e))

    agents: dict[str, dict] = {}
    for a, rows in per.items():
        rows.sort(key=lambda r: r[0])
        segments = []
        for t, room, _ in rows:
            if segments and segments[-1][2] == room:
                segments[-1][1] = t
            else:
                segments.append([t, t, room])
        rec = {"t0": rows[0][0], "t1": rows[-1][0], "segments": segments,
               "parents": None, "generation": 0, "origin": "seed",
               "moves": len(segments) - 1}
        for _, _, e in rows:
            if e["type"] == "birth":
                rec["parents"] = e.get("parents")
                rec["generation"] = e.get("generation", 0)
                rec["origin"] = e.get("origin", "seed")
                break
        agents[a] = rec
    return agents


def _kinship_groups(agents: dict) -> list[dict]:
    """Split the population into kinship groups — connected components of the
    parent graph, following *both* parents.

    This population is not one tree. Grouping by first parent alone left 44% of
    parent links pointing outside the group; the connected components are the
    real families, and there are thousands of them, most of size one.
    """
    parent = {a: a for a in agents}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, rec in agents.items():
        for p in (rec.get("parents") or []):
            if p in agents:
                ra, rb = find(a), find(p)
                if ra != rb:
                    parent[ra] = rb

    buckets: dict[str, list[str]] = defaultdict(list)
    for a in agents:
        buckets[find(a)].append(a)
    groups = []
    for members in buckets.values():
        if len(members) < 2:
            continue                      # a lone agent is not a family tree
        members.sort(key=lambda a: agents[a]["t0"])
        groups.append({
            "members": members,
            "t0": min(agents[m]["t0"] for m in members),
            "t1": max(agents[m]["t1"] for m in members),
            "maxgen": max(agents[m]["generation"] for m in members),
        })
    groups.sort(key=lambda g: -len(g["members"]))
    return groups


def _tidy(agents: dict, members: list[str]) -> dict[str, float]:
    """Classic tidy-tree x positions: leaves left to right, a parent centred
    over its children. Structure follows the first parent; the second parent's
    link is drawn but does not shape the layout."""
    inside = set(members)
    kids: dict[str, list[str]] = defaultdict(list)
    roots = []
    for a in members:
        ps = [p for p in (agents[a].get("parents") or []) if p in inside]
        if ps:
            kids[ps[0]].append(a)
        else:
            roots.append(a)
    for v in kids.values():
        v.sort(key=lambda a: agents[a]["t0"])

    x: dict[str, float] = {}
    leaf = 0.0
    for root in roots:
        stack = [(root, False)]
        while stack:
            node, done = stack.pop()
            if done:
                cs = kids.get(node, [])
                x[node] = sum(x[c] for c in cs) / len(cs) if cs else leaf
                continue
            cs = kids.get(node, [])
            if not cs:
                x[node] = leaf
                leaf += 1
                continue
            stack.append((node, True))
            for c in reversed(cs):
                stack.append((c, False))
    return x


# ── rendering ─────────────────────────────────────────────────────────────
_CSS = """
:root{color-scheme:light;
--surface:#fbfcfb;--panel:#ffffff;--ink:#0d1210;--ink-2:#4c5450;
--ink-3:#7d857f;--grid:#e6eae7;--rule:#dfe3e0;--tint:#f2f6f3;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;
--good:#0ca30c;--warning:#fab219;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
color-scheme:dark;--surface:#141715;--panel:#1c201e;--ink:#f2f5f3;
--ink-2:#a8b1ab;--ink-3:#78817b;--grid:#2b302d;--rule:#333935;--tint:#1f2421;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--good:#0ca30c;--warning:#fab219;}}
:root[data-theme="dark"]{color-scheme:dark;--surface:#141715;--panel:#1c201e;
--ink:#f2f5f3;--ink-2:#a8b1ab;--ink-3:#78817b;--grid:#2b302d;--rule:#333935;
--tint:#1f2421;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--good:#0ca30c;--warning:#fab219;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface);color:var(--ink);
font-family:Archivo,ui-sans-serif,system-ui,-apple-system,sans-serif;
font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased;}
.wrap{max-width:1180px;margin:0 auto;padding:44px 22px 72px;
display:flex;flex-direction:column;gap:22px}
.masthead{display:flex;flex-direction:column;gap:4px;
padding-bottom:18px;border-bottom:1px solid var(--rule)}
h1{font-family:"Instrument Serif",Georgia,serif;font-weight:400;
font-size:40px;line-height:1.08;margin:0;letter-spacing:-.012em;
text-wrap:balance}
.run{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;
color:var(--ink-3);letter-spacing:.02em}
.sub{color:var(--ink-2);font-size:13.5px;margin:0;max-width:66ch;
text-wrap:pretty}
h2{font-size:15px;font-weight:600;margin:0;letter-spacing:-.004em;color:var(--ink)}
.eyebrow{font-size:10.5px;font-weight:600;letter-spacing:.1em;
text-transform:uppercase;color:var(--ink-3)}
.note{color:var(--ink-3);font-size:11.5px;margin:8px 0 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));
gap:10px}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:10px;
padding:13px 15px 14px;display:flex;flex-direction:column;gap:2px}
.tile .k{color:var(--ink-2);font-size:10.5px;font-weight:600;
text-transform:uppercase;letter-spacing:.08em}
.tile .v{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:25px;
font-weight:500;letter-spacing:-.028em;font-variant-numeric:tabular-nums}
.tile .m{color:var(--ink-3);font-size:11.5px}
.pill{align-self:flex-start;margin-top:5px;display:inline-flex;
align-items:center;gap:5px;padding:2px 8px;border-radius:999px;
background:var(--tint);border:1px solid var(--rule);
font-size:10.5px;font-weight:600;letter-spacing:.03em;color:var(--ink-2)}
.pill i{width:6px;height:6px;border-radius:999px;flex:none}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px;align-items:flex-start}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:9px;padding:8px 8px 6px;flex:0 1 auto}
.card .cap{font-family:"JetBrains Mono",ui-monospace,monospace;font-size:10.5px;color:var(--ink-3);margin-top:5px;letter-spacing:-.01em}
.card .cap b{color:var(--ink-2);font-weight:600}
.scroll{overflow:auto;max-height:78vh;border:1px solid var(--rule);border-radius:8px;margin-top:12px;background:var(--surface)}
.scroll svg{max-width:none}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:12px;
padding:18px 20px 12px;margin:0;display:flex;flex-direction:column;gap:0;
overflow-x:auto}
.legend{display:flex;flex-wrap:wrap;gap:15px;margin:12px 0 4px}
.legend span{display:inline-flex;align-items:center;gap:6px;
color:var(--ink-2);font-size:11.5px;
font-family:"JetBrains Mono",ui-monospace,monospace}
.sw{width:10px;height:10px;border-radius:3px;flex:none;display:inline-block}
svg{display:block;max-width:100%;height:auto}
svg text{font-family:"JetBrains Mono",ui-monospace,monospace}
.tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .09s;
background:var(--panel);border:1px solid var(--rule);border-radius:9px;
padding:8px 11px;font-size:11.5px;box-shadow:0 8px 26px rgba(0,0,0,.16);
z-index:9;color:var(--ink);
font-family:"JetBrains Mono",ui-monospace,monospace;
font-variant-numeric:tabular-nums}
.tip b{font-weight:600}
details{margin:2px 0 0}
summary{cursor:pointer;color:var(--ink-2);font-size:12px}
summary:focus-visible{outline:2px solid var(--s1);outline-offset:3px}
table{border-collapse:collapse;margin-top:10px;font-size:12px;
font-family:"JetBrains Mono",ui-monospace,monospace;
font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--rule);padding:5px 10px;text-align:right}
th{color:var(--ink-2);font-weight:600;text-align:left;background:var(--tint)}
td:first-child,th:first-child{text-align:left}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

_JS = """
const tip=document.createElement('div');tip.className='tip';
document.body.appendChild(tip);
document.querySelectorAll('[data-chart]').forEach(function(fig){
 const d=JSON.parse(fig.querySelector('script[type="application/json"]').textContent);
 const svg=fig.querySelector('svg'),cur=fig.querySelector('.cursor');
 svg.addEventListener('mousemove',function(ev){
  const b=svg.getBoundingClientRect(),vb=svg.viewBox.baseVal;
  const px=(ev.clientX-b.left)/b.width*vb.width;
  if(px<d.p.l||px>vb.width-d.p.r){tip.style.opacity=0;cur.setAttribute('opacity',0);return;}
  const t=(px-d.p.l)/(vb.width-d.p.l-d.p.r)*(d.x1-d.x0)+d.x0;
  let rows='';
  d.series.forEach(function(s){
   if(!s.pts.length)return;
   let best=s.pts[0],bd=1e18;
   for(const p of s.pts){const dd=Math.abs(p[0]-t);if(dd<bd){bd=dd;best=p;}}
   if(best[0]<d.x0-1||bd>(d.x1-d.x0)*0.06)return;
   rows+='<div><span class="sw" style="display:inline-block;background:'+s.color+
     '"></span> '+s.name+' <b>'+d.fmt.replace('{}',
     (Math.round(best[1]*10)/10).toLocaleString())+'</b></div>';
  });
  if(!rows){tip.style.opacity=0;cur.setAttribute('opacity',0);return;}
  cur.setAttribute('x1',px);cur.setAttribute('x2',px);cur.setAttribute('opacity',1);
  tip.innerHTML='<div style="color:var(--ink-2);margin-bottom:3px">'+
    (Math.round(t*10)/10)+' h</div>'+rows;
  tip.style.opacity=1;
  tip.style.left=Math.min(ev.clientX+14,innerWidth-190)+'px';
  tip.style.top=(ev.clientY-12)+'px';
 });
 svg.addEventListener('mouseleave',function(){tip.style.opacity=0;
  cur.setAttribute('opacity',0);});
});
"""


def _fmt(n: float) -> str:
    return f"{n:,.0f}" if abs(n) >= 100 else f"{n:,.1f}".rstrip("0").rstrip(".")


def _line_chart(cid: str, title: str, subtitle: str,
                series: list[tuple[str, list, str]], unit: str,
                w: int = 1100, h: int = 250) -> str:
    pad = {"l": 62, "r": 96, "t": 14, "b": 30}
    pts_all = [p for _, pts, _ in series for p in pts]
    if not pts_all:
        return ""
    x0, x1 = 0, max(p[0] for p in pts_all) or 1
    y1 = max(p[1] for p in pts_all) or 1
    y1 *= 1.12
    iw, ih = w - pad["l"] - pad["r"], h - pad["t"] - pad["b"]

    def sx(v):
        return pad["l"] + (v - x0) / (x1 - x0) * iw

    def sy(v):
        return pad["t"] + ih - (v / y1) * ih

    out = [f'<figure class="panel" data-chart id="{cid}" style="margin:0 0 20px">',
           f"<h2>{html.escape(title)}</h2><p class='sub' style='margin:0 0 10px'>"
           f"{html.escape(subtitle)}</p>"]
    out.append('<div class="legend">' + "".join(
        f'<span><i class="sw" style="background:{c}"></i>{html.escape(n)}</span>'
        for n, _, c in series) + "</div>")
    out.append(f'<svg viewBox="0 0 {w} {h}" role="img">')
    # recessive grid + y ticks
    for i in range(5):
        v = y1 * i / 4
        y = sy(v)
        out.append(f'<line x1="{pad["l"]}" x2="{w - pad["r"]}" y1="{y:.1f}" '
                   f'y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>')
        out.append(f'<text x="{pad["l"] - 9}" y="{y + 4:.1f}" text-anchor="end" '
                   f'font-size="11" fill="var(--ink-3)">{_fmt(v)}</text>')
    for i in range(5):
        v = x0 + (x1 - x0) * i / 4
        out.append(f'<text x="{sx(v):.1f}" y="{h - 9}" text-anchor="middle" '
                   f'font-size="11" fill="var(--ink-3)">{_fmt(v)}</text>')
    for name, pts, colour in series:
        if not pts:
            continue
        d = " ".join(("M" if i == 0 else "L") + f"{sx(x):.1f},{sy(y):.1f}"
                     for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{colour}" '
                   f'stroke-width="2" stroke-linejoin="round"/>')
        # direct label at the line end: identity is never colour alone
        lx, ly = pts[-1]
        out.append(f'<text x="{sx(lx) + 7:.1f}" y="{sy(ly) + 4:.1f}" '
                   f'font-size="11" fill="var(--ink-2)">{html.escape(name)}</text>')
    out.append(f'<text x="{w - pad["r"]}" y="{h - 9}" text-anchor="end" '
               f'font-size="10" fill="var(--ink-3)">hours</text>')
    out.append('<line class="cursor" y1="%d" y2="%d" stroke="var(--ink-3)" '
               'stroke-width="1" stroke-dasharray="3 3" opacity="0"/>'
               % (pad["t"], pad["t"] + ih))
    out.append("</svg>")
    payload = {"p": pad, "x0": x0, "x1": x1, "fmt": unit,
               "series": [{"name": n, "color": c, "pts": _thin(p, 900)}
                          for n, p, c in series]}
    out.append('<script type="application/json">' + json.dumps(payload) + "</script>")
    out.append("</figure>")
    return "".join(out)


def _family_card(agents: dict, group: dict, room_colour: dict,
                 limit: int = 150) -> str:
    """One kinship group as a tidy tree: time down, family across.

    Each card has its own vertical scale — a family that lived four minutes and
    one that lived two hours are both worth reading — so the caption carries
    the real duration.
    """
    members = group["members"][:limit]
    truncated = len(group["members"]) - len(members)
    x = _tidy(agents, members)
    if not x:
        return ""
    cols = max(x.values()) + 1
    w = int(min(1100, max(210, cols * 3.4)))
    h = 232
    pad = {"l": 8, "r": 8, "t": 10, "b": 8}
    t0, t1 = group["t0"], group["t1"]
    span = max(t1 - t0, 1e-9)
    iw = w - pad["l"] - pad["r"]
    ih = h - pad["t"] - pad["b"]

    def sx(c):
        return pad["l"] + (c / max(cols - 1, 1)) * iw

    def sy(t):
        return pad["t"] + (t - t0) / span * ih

    out = [f'<div class="card"><svg viewBox="0 0 {w} {h}" role="img">']
    for a in members:                       # links first, under the lifelines
        for pnt in (agents[a].get("parents") or []):
            if pnt in x:
                y = sy(agents[a]["t0"])
                out.append(f'<path d="M{sx(x[pnt]):.1f},{y:.1f} '
                           f'L{sx(x[a]):.1f},{y:.1f}" fill="none" '
                           f'stroke="var(--ink-3)" stroke-width=".9" opacity=".55"/>')
    for a in members:
        rec = agents[a]
        cx = sx(x[a])
        # Agents migrate constantly, and most stays are sub-pixel at this
        # scale. Drawing them anyway triples the file for marks nobody can
        # see, so a stay shorter than a pixel contributes its migration dot
        # and is otherwise folded into the run around it.
        drawn = []
        for s0, s1, room in rec["segments"]:
            if drawn and sy(s1) - sy(drawn[-1][0]) < 1.0:
                drawn[-1] = (drawn[-1][0], s1, drawn[-1][2], drawn[-1][3] + 1)
            else:
                drawn.append((s0, s1, room, 0))
        for j, (s0, s1, room, folded) in enumerate(drawn):
            y0, y1 = sy(s0), sy(s1)
            hops = f" \u00b7 +{folded} brief stays" if folded else ""
            out.append(
                f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y0:.1f}" '
                f'y2="{max(y1, y0 + 1.4):.1f}" '
                f'stroke="{room_colour.get(room, "var(--s1)")}" '
                f'stroke-width="2" stroke-linecap="round">'
                f'<title>{a} \u00b7 gen {rec["generation"]} \u00b7 {rec["origin"]}'
                f' \u00b7 {room} \u00b7 {(s1 - s0) / 60:.1f} min here'
                f'{hops}</title></line>')
            if j:
                out.append(f'<circle cx="{cx:.1f}" cy="{y0:.1f}" r="1.7" '
                           f'fill="var(--ink)" opacity=".7"/>')
    mins = span / 60
    extra = f" · {truncated:,} more not drawn" if truncated else ""
    out.append("</svg>"
               f'<div class="cap"><b>{len(group["members"]):,} agents</b> · '
               f'to generation {group["maxgen"]} · {mins:,.0f} min{extra}</div>'
               "</div>")
    return "".join(out)


def _families_panel(agents: dict, groups: list[dict], room_names: list[str],
                    colours: list[str], limit: int) -> str:
    if not groups:
        return ""
    room_colour = {r: colours[i % len(colours)] for i, r in enumerate(room_names)}
    shown = groups[:limit]
    covered = sum(len(g["members"]) for g in shown)
    singles = len(agents) - sum(len(g["members"]) for g in groups)
    cards = "".join(_family_card(agents, g, room_colour) for g in shown)
    return ("<figure class='panel' style='margin:0'>"
            "<div class='eyebrow'>kinship groups</div>"
            "<h2>Family trees</h2>"
            f"<p class='sub' style='margin:6px 0 0'>The {len(shown)} largest "
            f"families of the {len(groups):,} with more than one member, "
            f"covering {covered:,} agents. A further {singles:,} agents "
            "("
            f"{singles / max(len(agents), 1):.0%}) never reproduced and have no "
            "relatives, so they form no tree. Time runs downward within each "
            "family on its own scale; colour is the room, so a lifeline that "
            "changes colour has migrated.</p>"
            '<div class="legend">' + "".join(
                f'<span><i class="sw" style="background:{room_colour[r]}"></i>{r}</span>'
                for r in room_names) +
            '<span><i class="sw" style="background:var(--ink-3);height:2px;'
            'border-radius:0"></i>parent link</span></div>'
            f'<div class="cards">{cards}</div></figure>')


def _tiles(stats: list[tuple]) -> str:
    """Summary before detail. The self-sufficiency tile also carries a pill:
    the number alone does not say whether a population is standing up."""
    cells = []
    for row in stats:
        k, v, m = row[0], row[1], row[2]
        pill = ""
        if len(row) > 3 and row[3]:
            label, colour = row[3]
            pill = (f'<span class="pill"><i style="background:{colour}"></i>'
                    f'{html.escape(label)}</span>')
        cells.append(f'<div class="tile"><div class="k">{html.escape(k)}</div>'
                     f'<div class="v">{html.escape(v)}</div>'
                     f'<div class="m">{html.escape(m)}</div>{pill}</div>')
    return '<div class="tiles">' + "".join(cells) + "</div>"


def _table(rows: list[list[str]], head: list[str], caption: str) -> str:
    return ("<details><summary>" + html.escape(caption) + "</summary><table><tr>"
            + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r)
                      + "</tr>" for r in rows) + "</table></details>")


def build_html(run_dir: str | Path, out_path: str | Path,
               max_families: int = 28) -> Path:
    run_dir, out_path = Path(run_dir), Path(out_path)
    rooms = _read(run_dir)
    if not rooms:
        raise ValueError(f"no event logs under {run_dir}/events")
    names = sorted(rooms)
    stamps = [e["t"] for evs in rooms.values() for e in evs if "t" in e]
    t0, t1 = min(stamps), max(stamps)
    series = {r: _series(rooms[r], t0) for r in names}

    agents = _population(rooms)
    groups = _kinship_groups(agents)

    births = sum(1 for a in agents.values() if a["origin"] == "birth")
    refills = sum(1 for a in agents.values() if a["origin"] == "refill")
    deaths = sum(1 for evs in rooms.values() for e in evs if e["type"] == "death")
    moves = sum(1 for evs in rooms.values() for e in evs if e["type"] == "move")
    maxgen = max((a["generation"] for a in agents.values()), default=0)
    ss = births / (births + refills) if (births + refills) else 0.0
    travelled = sum(1 for a in agents.values() if a["moves"])

    verdict = ("self-sustaining", "var(--good)") if ss >= 0.5 else (
        ("at parity", "var(--warning)") if ss >= 0.4
        else ("carried by immigration", "var(--ink-3)"))
    body = ["<header class='masthead'>"
            "<div class='eyebrow'>evoLLM run telemetry</div>"
            "<h1>Lineage &amp; scarcity</h1>"
            f"<div class='run'>{html.escape(run_dir.name)}</div>"
            "<p class='sub'>Rooms keep their own step counters and are meant to "
            "drift apart (\u00a74.5). Every event also carries a wall-clock "
            "stamp, so the panels below share one axis \u2014 hours since the "
            "run began \u2014 on which the four rooms are actually "
            "comparable.</p></header>",
            _tiles([
                ("Descendant births", f"{births:,}", "children of two parents"),
                ("Immigrants", f"{refills:,}", "refills to hold the floor"),
                ("Self-sufficiency", f"{ss:.3f}", "births / (births + refills)",
                 verdict),
                ("Deepest lineage", f"gen {maxgen}", "generations from a founder"),
                ("Deaths", f"{deaths:,}", "all scarcity events"),
                ("Migrations", f"{moves:,}",
                 f"{travelled:,} of {len(agents):,} agents moved"),
            ])]

    def collect(key):
        return [(r, series[r][key], (SERIES_LIGHT * 3)[i])
                for i, r in enumerate(names)]

    body.append(_line_chart(
        "pop", "Population", "Living agents per room. Refill holds a floor "
        "under each room, so a falling line means deaths are outrunning both "
        "births and immigration.", collect("population"), "{} agents"))
    body.append(_line_chart(
        "mem", "Block usage per GPU", "Share of each room's authoritative pool "
        "held by live agents \u2014 KV cache plus adapters. A room at 100% is "
        "one where the next token kills somebody.",
        collect("usage"), "{}% of pool"))
    body.append(_line_chart(
        "ss", "Self-sufficiency", "Share of new agents arriving by descent "
        "rather than immigration, over a sliding window of 400 arrivals. "
        "Rising toward 100% is takeoff; flat is a population being carried.",
        collect("selfsuf"), "{}% by descent"))
    body.append(_line_chart(
        "mv", "Migration rate", "Departures per hour from each room. Movement "
        "is what couples the rooms together \u2014 an agent carries its "
        "context and its lineage across.", collect("moves"), "{}/hour"))
    body.append(_line_chart(
        "bl", "Observation backlog", "Mean unread tokens per agent. An agent "
        "absorbs its queue at a bounded rate, so this is also how far behind "
        "the present it is acting.", collect("backlog"), "{} tokens"))

    body.append(_families_panel(agents, groups, names, SERIES_LIGHT,
                                limit=max_families))

    body.append(_table(
        [[r, f"{(max(e['t'] for e in rooms[r]) - t0) / 3600:.1f}h",
          f"{sum(1 for e in rooms[r] if e['type'] == 'birth' and e.get('origin') == 'birth'):,}",
          f"{sum(1 for e in rooms[r] if e['type'] == 'refill'):,}",
          f"{sum(1 for e in rooms[r] if e['type'] == 'death'):,}",
          f"{sum(1 for e in rooms[r] if e['type'] == 'move'):,}",
          f"{series[r]['population'][-1][1]:.0f}" if series[r]["population"] else "0"]
         for r in names],
        ["room", "reached", "births", "refills", "deaths", "departures",
         "final population"],
        "Table view \u2014 per-room totals"))

    doc = ("<title>Lineage &amp; Scarcity</title>"
           '<link rel="preconnect" href="https://fonts.googleapis.com">'
           '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
           'family=Archivo:wght@400;500;600&family=Instrument+Serif&'
           'family=JetBrains+Mono:wght@400;500;600&display=swap">'
           f"<style>{_CSS}</style>"
           f"<div class='wrap'>{''.join(body)}</div><script>{_JS}</script>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return out_path
