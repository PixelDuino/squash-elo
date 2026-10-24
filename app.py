"""Squash Elo - web version (Streamlit).
Anyone can VIEW. Only people with the password can add players / enter, edit or delete results.
Data lives in squash_data.json inside a separate private GitHub repo, so it survives restarts.
"""
import base64
import html
import json
import uuid
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

MODE = 3  # Extra tabs when logged in:  0 = none  |  1 = Awards + Fun  |  2 = "Stats" (everyone's numbers), no Fun  |  3 = "Stats" with switchable views

START_RATING, K_FACTOR = 1000, 32
MIN_GAMES = 3  # games a player needs before average-based awards (Most Dominant, Brick Wall) count
# Player Efficiency Rating: how much each stat counts (weights add up to 1.0)
# Every player keeps one colour on the rating graphs: colours are handed out in the order players were added.
PLAYER_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22",
                 "#393b79", "#e6550d", "#31a354", "#756bb1", "#843c39", "#d6a300", "#00a6a6", "#c51b8a", "#637939",
                 "#5254a3", "#7f7f7f"]
PER_WEIGHTS = {"win": 0.30,   # win percentage
               "diff": 0.25,  # average point difference per game (points won minus points lost)
               "off": 0.15,   # average points won per game
               "dfn": 0.15,   # average points lost per game (fewer is better)
               "sos": 0.15}   # strength of schedule: average rating of opponents played
MAX_PENDING = 30  # most results that can be waiting for approval at once
SHUTOUT_BONUS = 5  # extra rating points for winning 11-0 (the loser only gets the normal drop)
CLOSE_MARGIN = 3  # a game decided by this many points or fewer counts as a close game (Nail Biter)
URL = f"https://api.github.com/repos/{st.secrets['DATA_REPO']}/contents/squash_data.json"
HEAD = {"Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"}


# ---------- storage ----------
@st.cache_data(ttl=30, show_spinner=False)  # keep a copy for 30 seconds so tapping around is instant
def load():
    r = requests.get(URL, headers=HEAD, timeout=15)
    if r.status_code == 404:  # file doesn't exist yet
        return {"players": {}, "matches": []}, None
    r.raise_for_status()
    j = r.json()
    return json.loads(base64.b64decode(j["content"])), j["sha"]


def save(data, sha, msg):
    body = {"message": msg, "content": base64.b64encode(json.dumps(data, indent=2).encode()).decode()}
    if sha:
        body["sha"] = sha
    r = requests.put(URL, headers=HEAD, json=body, timeout=15)
    if r.status_code in (409, 422):  # the saved file changed since this page loaded it: don't overwrite it
        load.clear()
        st.session_state.flash_err = ("Someone else saved a change a moment ago, so yours was not applied. "
                                      "The latest data has been loaded. Please check it and try again.")
        st.rerun()
    r.raise_for_status()
    load.clear()  # the next page load fetches the fresh version


# ---------- elo ----------
def elo(r1, r2, s1):
    e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    return r1 + K_FACTOR * (s1 - e1), r2 + K_FACTOR * ((1 - s1) - (1 - e1))


SCORE_RULES = ("Games go to 11, win by 2 (11-9, 12-10, 13-11, 14-12). "
               "At 14-14 the next point wins, so the highest score is 15-14.")


def valid_game(a, b):
    hi, lo = max(a, b), min(a, b)
    return (hi == 11 and lo <= 9) or (12 <= hi <= 14 and hi - lo == 2) or (hi == 15 and lo == 14)


def shutout_bonus(a, b):
    """Winning 11-0 earns the winner a bonus on top of the normal rating change."""
    return SHUTOUT_BONUS if sorted((a, b)) == [0, 11] else 0


def play(r1, r2, a, b, bonus=0):
    """New ratings after one game. The bonus goes to the winner only; the loser just gets the normal Elo drop."""
    n1, n2 = elo(r1, r2, 1 if a > b else 0)
    return (n1 + bonus, n2) if a > b else (n1, n2 + bonus)


def recompute(players, matches):
    """Replay every match in order from the starting rating (11-0 bonuses included).
    Used after editing or deleting a match."""
    for p in players:
        players[p] = START_RATING
    for m in matches:
        a, b = m["score"]
        r1, r2 = players[m["p1"]], players[m["p2"]]
        n1, n2 = play(r1, r2, a, b, m.get("bonus", 0))  # matches saved before the bonus existed have no bonus
        m.update(p1_before=r1, p1_after=n1, p2_before=r2, p2_after=n2, winner=m["p1"] if a > b else m["p2"])
        players[m["p1"]], players[m["p2"]] = n1, n2


def rating_timeline(matches):
    """READ-ONLY: each player's rating after their 1st game, 2nd game, 3rd game... (Game 0 = starting rating).
    Everyone's nth game lines up at the same point, so a player with more games just goes further along.
    Never changes saved data."""
    played, rows = {}, []
    for m in matches:
        for who, key in ((m["p1"], "p1"), (m["p2"], "p2")):
            if who not in played:
                played[who] = 0
                rows.append({"Game": 0, "Player": who, "Rating": m[key + "_before"]})
            played[who] += 1
            rows.append({"Game": played[who], "Player": who, "Rating": m[key + "_after"]})
    return pd.DataFrame(rows, columns=["Game", "Player", "Rating"])


def player_color(name):
    order = list(players)
    return PLAYER_COLORS[order.index(name) % len(PLAYER_COLORS)] if name in order else "#7f7f7f"


def rename_player(players, matches, old, new):
    """Give a player a new name everywhere (ratings and every match they appear in). Keeps their place in the
    player order, so their graph colour doesn't change."""
    items = list(players.items())
    players.clear()
    for k, v in items:
        players[new if k == old else k] = v
    for m in matches:
        for k in ("p1", "p2", "winner"):
            if m.get(k) == old:
                m[k] = new


def forget_names():
    """Clear remembered dropdown choices that may still point at an old or removed name."""
    for k in ("chart_sel", "hist_player", "manage_who", "card_player"):
        st.session_state.pop(k, None)
    for k in [k for k in st.session_state if str(k).startswith("hist_opp_")]:
        st.session_state.pop(k, None)


def player_stats(matches):
    """READ-ONLY: games, wins, points scored and points conceded for every player."""
    out = {}
    for m in matches:
        a, b = m["score"]
        for who, got, gave in ((m["p1"], a, b), (m["p2"], b, a)):
            d = out.setdefault(who, {"g": 0, "w": 0, "f": 0, "a": 0})
            d["g"] += 1
            d["w"] += 1 if got > gave else 0
            d["f"] += got
            d["a"] += gave
    return out


def build_awards(matches, everyone=False):
    """READ-ONLY: returns [(award name, ranked rows, text if nobody qualifies), ...].
    Each ranked row is (player, stat text, place). Tied players share a place."""
    ps = player_stats(matches)
    plural = lambda n, word: f"{n} {word}{'' if n == 1 else 's'}"

    def ranked(scores, fmt):  # scores: player -> number, higher is better
        items = sorted(scores.items(), key=lambda kv: -kv[1])
        return [(p, fmt(v), 1 + sum(1 for _, o in items if o > v + 1e-9), v) for p, v in items]

    # Giant Killer: biggest rating gap overcome in a win (rating before the match)
    upset = {}
    for m in matches:
        win, lose = ("p1", "p2") if m["score"][0] > m["score"][1] else ("p2", "p1")
        gap = m[lose + "_before"] - m[win + "_before"]
        if gap > 0:
            upset[m[win]] = max(upset.get(m[win], 0), gap)

    # Hot Streak: longest run of consecutive wins
    run, longest = {}, {}
    for m in matches:
        a, b = m["score"]
        for who, won in ((m["p1"], a > b), (m["p2"], b > a)):
            run[who] = run.get(who, 0) + 1 if won else 0
            longest[who] = max(longest.get(who, 0), run[who])

    # Nail Biter: games decided by CLOSE_MARGIN points or fewer (win or lose)
    close = {}
    for m in matches:
        if abs(m["score"][0] - m["score"][1]) <= CLOSE_MARGIN:
            for who in (m["p1"], m["p2"]):
                close[who] = close.get(who, 0) + 1

    # Peak Performer: highest rating ever reached
    peak = {}
    for m in matches:
        for k in ("p1", "p2"):
            peak[m[k]] = max(peak.get(m[k], START_RATING), m[k + "_after"])

    # Longest Reign: matches a player played themselves while they were number 1 (entering the match
    # with a rating strictly higher than everyone else who has played). Tied leaders don't count.
    latest, reign = {}, {}
    for m in matches:
        before = {m["p1"]: m["p1_before"], m["p2"]: m["p2_before"]}
        field = {**latest, **before}
        for p, r in before.items():
            if all(r > o + 1e-9 for q, o in field.items() if q != p):
                reign[p] = reign.get(p, 0) + 1
        latest[m["p1"]], latest[m["p2"]] = m["p1_after"], m["p2_after"]

    # Toughest Schedule: average rating of opponents at the time they were played
    opp_sum, beaten = {}, {}
    for m in matches:
        a, b = m["score"]
        for me, other, won in (("p1", "p2", a > b), ("p2", "p1", b > a)):
            opp_sum[m[me]] = opp_sum.get(m[me], 0) + m[other + "_before"]
            if won:
                beaten.setdefault(m[me], set()).add(m[other])

    # Player Efficiency Rating (PER): each stat is converted to a z-score against the other qualified players
    # (how many standard deviations above or below the group average), blended with PER_WEIGHTS, then
    # scaled so the group average is exactly 15 and one standard deviation is worth 5 points.
    comps = {p: {"win": d["w"] / d["g"], "diff": (d["f"] - d["a"]) / d["g"], "off": d["f"] / d["g"],
                 "dfn": -d["a"] / d["g"], "sos": opp_sum[p] / d["g"]}
             for p, d in ps.items() if d["g"] >= MIN_GAMES}
    per = {}
    if len(comps) >= 2:
        blend = {p: 0.0 for p in comps}
        for key, weight in PER_WEIGHTS.items():
            vals = [c[key] for c in comps.values()]
            mean = sum(vals) / len(vals)
            sd = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            if sd > 1e-9:
                for p, c in comps.items():
                    blend[p] += weight * (c[key] - mean) / sd
        per = {p: 15 + 5 * z for p, z in blend.items()}

    needs = f"Needs {MIN_GAMES}+ Games"
    cards = [
        ("Player Efficiency Rating", ranked(per, lambda v: f"{v:.1f} PER"), f"Needs 2+ Players With {MIN_GAMES}+ Games"),
        ("Most Dominant", ranked({p: (d["f"] - d["a"]) / d["g"] for p, d in ps.items() if d["g"] >= MIN_GAMES},
                                 lambda v: f"{v:+.1f} Pts Per Game"), needs),
        ("Win Machine", ranked({p: d["w"] for p, d in ps.items()}, lambda v: plural(v, "Win")), ""),
        ("Iron Legs", ranked({p: d["g"] for p, d in ps.items()}, lambda v: plural(v, "Game")), ""),
        ("Giant Killer", ranked(upset, lambda v: f"+{v:.0f} Rating Gap"), "No Upsets Yet"),
        ("Hot Streak", ranked({p: v for p, v in longest.items() if v > 0},
                              lambda v: plural(v, "Win") + " In A Row"), ""),
        ("Brick Wall", ranked({p: -d["a"] / d["g"] for p, d in ps.items() if d["g"] >= MIN_GAMES},
                              lambda v: f"{-v:.1f} Pts Against Per Game"), needs),
        ("Nail Biter", ranked(close, lambda v: plural(v, "Close Game")), "No Close Games Yet"),
        ("Peak Performer", ranked(peak, lambda v: f"{v:.0f} Rating"), ""),
        ("Longest Reign", ranked(reign, lambda v: f"{v} Match{'' if v == 1 else 'es'} At No. 1"), ""),
        ("Toughest Schedule", ranked({p: opp_sum[p] / d["g"] for p, d in ps.items() if d["g"] >= MIN_GAMES},
                                     lambda v: f"{v:.0f} Average Opponent"), needs),
        ("Conqueror", ranked({p: len(v) for p, v in beaten.items()},
                             lambda v: plural(v, "Opponent") + " Beaten"), "No Wins Yet"),
    ]
    if everyone:  # Stats view: also list players who don't qualify for a stat yet, at the bottom
        cards = [(t, rows + [(p, empty if empty.startswith("Needs") else "None", None, None)
                             for p in sorted(ps) if p not in {r[0] for r in rows}], empty) for t, rows, empty in cards]
    return cards


def awards_html(cards, everyone=False):
    """Award name on top, winner on the left with the stat on the same line to the right.
    Tap an award to expand it and see the next two players underneath."""
    ordinal = lambda n: {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")
    out = ["<style>.aw summary{display:block;list-style:none;cursor:pointer;}"
           ".aw summary::-webkit-details-marker{display:none;}"
           ".aw .arrow{display:inline-block;width:7px;height:7px;margin:0 4px 3px 0;border-right:2px solid currentColor;"
           "border-bottom:2px solid currentColor;transform:rotate(45deg);opacity:0.6;}"
           ".aw[open] .arrow{transform:rotate(-135deg);margin-bottom:-2px;}</style>"]
    for title, rows, empty in cards:
        winners = [r for r in rows if r[2] == 1]
        extra = rows[len(winners):] if everyone else rows[len(winners):len(winners) + 2]
        who = " &amp; ".join(html.escape(r[0]) for r in winners) if winners else "No Winner Yet"
        stat = winners[0][1] if winners else empty
        arrow = '<span class="arrow"></span>' if extra else ""
        head = ('<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div style="font-weight:700;font-size:0.95rem;opacity:0.75;">{html.escape(title)}</div>{arrow}</div>'
                '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;">'
                f'<div style="font-weight:700;font-size:1.4rem;">{who}</div>'
                f'<div style="font-weight:700;font-size:1.2rem;color:#2a9d8f;white-space:nowrap;">{html.escape(stat)}</div></div>')
        box = "border-top:1px solid rgba(128,128,128,0.25);padding:12px 4px;"
        if not extra:
            out.append(f'<div style="{box}">{head}</div>')
            continue
        more = "".join(
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;padding:8px 0 0 12px;'
            f'{"opacity:0.55;" if r[2] is None else ""}">'
            f'<div style="font-size:1.05rem;"><span style="display:inline-block;min-width:2.8em;opacity:0.6;">'
            f'{ordinal(r[2]) if r[2] else "-"}</span>{html.escape(r[0])}</div>'
            f'<div style="font-size:1rem;color:#2a9d8f;white-space:nowrap;">{html.escape(r[1])}</div></div>'
            for r in extra)
        out.append(f'<details class="aw" style="{box}"><summary>{head}</summary>{more}</details>')
    st.markdown("".join(out), unsafe_allow_html=True)


def toggle_player(name):
    """Tap a player to pick them, tap again to un-pick. Picking a third drops the earliest pick."""
    sel = st.session_state.get("m_players", [])
    st.session_state.m_players = [p for p in sel if p != name] if name in sel else (sel + [name])[-2:]


RACE_ROW_H = 46
RACE_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--fg:#31333f;--line:rgba(128,128,128,.4);--accent:#ff4b4b;--dur:0ms}
@media (prefers-color-scheme:dark){:root{--fg:#fafafa}}
*{box-sizing:border-box}
html,body{margin:0;padding:0;overflow:hidden;background:transparent;color:var(--fg);
 font-family:"Source Sans Pro","Source Sans 3",-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:15px}
#board{position:relative;width:100%}
.row{position:absolute;left:0;right:0;transition:top var(--dur) ease,opacity .35s ease}
.top{display:flex;align-items:baseline;justify-content:space-between;gap:10px}
.who{display:flex;align-items:baseline;min-width:0}
.rank{width:22px;text-align:right;margin-right:8px;font-weight:700;opacity:.65;flex:none}
.name{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600}
.val{font-weight:700;font-variant-numeric:tabular-nums;flex:none}
.track{height:10px;margin:5px 0 0 30px;border-radius:5px;background:rgba(128,128,128,.16)}
.bar{height:100%;border-radius:5px;transition:width var(--dur) ease}
#cap{min-height:1.4em;margin:10px 0 8px;opacity:.8;font-size:.9rem;text-align:center}
.ctl{display:flex;align-items:center;gap:8px}
.sp{margin-left:auto;display:flex;gap:6px}
button{font:inherit;color:var(--fg);background:transparent;border:1px solid var(--line);border-radius:8px;padding:5px 11px;cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
input[type=range]{width:100%;margin:12px 0 0;accent-color:var(--accent)}
</style></head><body>
<div id="board"></div><div id="cap"></div>
<div class="ctl"><button id="play">Play</button><button id="restart">Restart</button>
<span class="sp"><button data-s="1500">Slow</button><button data-s="850" class="on">Normal</button><button data-s="420">Fast</button></span></div>
<input id="slider" type="range" min="0" value="0">
<script>
const D = __DATA__;
const ROW = __ROW__, N = D.frames.length - 1;
const root = document.documentElement, board = document.getElementById('board'), cap = document.getElementById('cap');
const slider = document.getElementById('slider'), playBtn = document.getElementById('play');
board.style.height = (D.topN * ROW) + 'px';
slider.max = N;
try {  // follow the app's text colour (this frame can read the page it sits in)
  const c = getComputedStyle(parent.document.querySelector('.stApp') || parent.document.body).color;
  if (c) root.style.setProperty('--fg', c);
} catch (e) {}

const rows = {};
D.order.forEach(n => {
  const el = document.createElement('div'); el.className = 'row';
  el.style.height = (ROW - 8) + 'px'; el.style.top = (D.topN * ROW) + 'px'; el.style.opacity = 0;
  const top = document.createElement('div'); top.className = 'top';
  const who = document.createElement('div'); who.className = 'who';
  const rank = document.createElement('div'); rank.className = 'rank';
  const name = document.createElement('div'); name.className = 'name'; name.textContent = n; name.title = n;
  const val = document.createElement('div'); val.className = 'val';
  const track = document.createElement('div'); track.className = 'track';
  const bar = document.createElement('div'); bar.className = 'bar'; bar.style.background = D.colors[n]; bar.style.width = '0%';
  who.append(rank, name); top.append(who, val); track.append(bar); el.append(top, track); board.append(el);
  rows[n] = {el, rank, bar, val, cur: null, raf: 0};
});

function tween(r, to, dur) {
  cancelAnimationFrame(r.raf);
  if (r.cur === null || dur === 0) { r.cur = to; r.val.textContent = Math.round(to); return; }
  const from = r.cur, t0 = performance.now();
  const frame = now => {
    const t = Math.min(1, (now - t0) / dur);
    r.cur = from + (to - from) * t; r.val.textContent = Math.round(r.cur);
    if (t < 1) r.raf = requestAnimationFrame(frame);
  };
  r.raf = requestAnimationFrame(frame);
}
const pct = r => Math.max(3, Math.min(100, (r - D.lo) / (D.hi - D.lo) * 100));

function render(k, dur) {
  root.style.setProperty('--dur', dur + 'ms');
  const f = D.frames[k];
  const ranked = Object.keys(f).sort((a, b) => f[b] - f[a]);
  const pos = {}; ranked.forEach((n, i) => pos[n] = i);
  D.order.forEach(n => {
    const r = rows[n];
    if (n in pos && pos[n] < D.topN) {
      const p = pct(f[n]);
      r.el.style.top = (pos[n] * ROW) + 'px'; r.el.style.opacity = 1;
      r.bar.style.width = p + '%';
      r.rank.textContent = pos[n] + 1; tween(r, f[n], dur);
    } else {
      r.el.style.top = (D.topN * ROW) + 'px'; r.el.style.opacity = 0;
      r.bar.style.width = '0%'; r.cur = null;
    }
  });
  slider.value = k;
  if (k === 0) cap.textContent = 'Start: nobody has played yet';
  else { const m = D.info[k - 1];
    cap.textContent = 'Match ' + k + ' of ' + N + '  |  ' + String(m[0]).slice(0, 10) + '  |  ' + m[1] + ' ' + m[3] + '-' + m[4] + ' ' + m[2]; }
}

let step = N, timer = null, interval = 850;
function stop() { clearInterval(timer); timer = null; playBtn.textContent = 'Play'; playBtn.classList.remove('on'); }
function go(k, dur) { step = k; render(k, dur); }
function tick() { go(step + 1, interval * 0.85); if (step >= N) stop(); }
function start() {
  if (step >= N) go(0, 0);
  playBtn.textContent = 'Pause'; playBtn.classList.add('on');
  clearInterval(timer); timer = setInterval(tick, interval);
}
playBtn.onclick = () => timer ? stop() : start();
document.getElementById('restart').onclick = () => { stop(); go(0, 0); start(); };
slider.oninput = () => { stop(); go(+slider.value, 180); };
document.querySelectorAll('[data-s]').forEach(b => b.onclick = () => {
  document.querySelectorAll('[data-s]').forEach(x => x.classList.remove('on')); b.classList.add('on');
  interval = +b.dataset.s; if (timer) start();
});
render(N, 0);
</script></body></html>"""


def race_html(matches):
    """READ-ONLY: a self-contained animated leaderboard race (plays in the browser, so it's smooth)."""
    order = sorted({p for m in matches for p in (m["p1"], m["p2"])})
    latest, frames = {}, [{}]
    for m in matches:
        latest[m["p1"]], latest[m["p2"]] = round(m["p1_after"], 1), round(m["p2_after"], 1)
        frames.append(dict(latest))
    allr = [r for f in frames for r in f.values()]
    top_n = min(10, len(order))
    data = {"frames": frames, "order": order, "topN": top_n,
            "info": [[m["date"], m["p1"], m["p2"], m["score"][0], m["score"][1]] for m in matches],
            "colors": {p: player_color(p) for p in order},
            "lo": int((min(allr) - 30) // 50 * 50), "hi": int(-(-(max(allr) + 30) // 50) * 50)}
    doc = RACE_TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")).replace("__ROW__", str(RACE_ROW_H))
    return doc, top_n * RACE_ROW_H + 105


CARD_LABELS = ("Attack", "Defence", "Winning", "Form", "Toughness")


def recent_swing(name):
    """Average rating change per game over a player's last 5 games (None if they haven't played)."""
    games = [(m, "p1" if m["p1"] == name else "p2") for m in matches if name in (m["p1"], m["p2"])][-5:]
    return sum(m[me + "_after"] - m[me + "_before"] for m, me in games) / len(games) if games else None


def match_edge(field, opp):
    """How tough one opponent was, from -1 (the softest opponent you could have picked) through 0 (an average
    opponent) to +1 (the toughest). `field` holds the ratings of everyone you could have played at the time."""
    vals = list(field.values())
    lo, hi, mid = min(vals), max(vals), sum(vals) / len(vals)
    if opp >= mid:
        return (opp - mid) / (hi - mid) if hi > mid else 0.0
    return (opp - mid) / (mid - lo) if mid > lo else 0.0


def toughness_edges(matches):
    """READ-ONLY: each player's average edge, match by match. Every opponent is judged against the other players
    who could have been picked at that moment, so the top player isn't marked down for having nobody above him."""
    latest, total, count = {}, {}, {}
    for m in matches:
        for me, other in (("p1", "p2"), ("p2", "p1")):
            who, opp = m[me], m[other]
            field = {q: r for q, r in latest.items() if q not in (who, opp)}
            field[opp] = m[other + "_before"]
            if len(field) >= 2:  # need at least two possible opponents for the comparison to mean anything
                total[who] = total.get(who, 0.0) + match_edge(field, m[other + "_before"])
                count[who] = count.get(who, 0) + 1
        latest[m["p1"]], latest[m["p2"]] = m["p1_after"], m["p2_after"]
    return {p: total[p] / count[p] for p in count}


def card_stats(name):
    """READ-ONLY: a FIFA-style card for one player. Four stats from 0 to 99, and the overall (OVR) is their average.
    Attack  = points scored per game, out of 11 (0 points is 0, 11 or more is 99).
    Defence = points conceded per game: 11 or more is 50, 5 or fewer is 99, in a straight line between.
    Winning = win rate, lifted so 0% is 50 and 100% is 99.
    Form    = recent rating change compared with everyone else: the hottest player gets 99, the coldest 50.
    Toughness = how strong your opponents were compared with who you could have picked (50 = always the softest
                option, about 75 = an average mix, 99 = always the toughest)."""
    games = [(m, "p1" if m["p1"] == name else "p2") for m in matches if name in (m["p1"], m["p2"])]
    if not games:
        return None
    g = len(games)
    f = a = w = 0
    for m, me in games:
        mine, theirs = (m["score"][0], m["score"][1]) if me == "p1" else (m["score"][1], m["score"][0])
        f, a, w = f + mine, a + theirs, w + (mine > theirs)
    swings = {p: recent_swing(p) for p in players}
    swings = [v for v in swings.values() if v is not None]
    lo, hi = min(swings), max(swings)
    form = 75 if hi - lo < 1e-9 else 50 + 49 * (recent_swing(name) - lo) / (hi - lo)  # nobody to compare with: 75
    edge = toughness_edges(matches).get(name)
    tough = 74.5 if edge is None else 50 + 49 * (edge + 1) / 2
    cl = lambda x: max(0, min(99, int(x + 0.5)))  # halves round up
    stats = dict(zip(CARD_LABELS, (
        cl(f / g / 11 * 99),
        cl(50 + 49 * max(0, min(1, (11 - a / g) / 6))),
        cl(50 + 49 * w / g),
        cl(form),
        cl(tough))))
    ranking = sorted(players, key=lambda x: -players[x])
    return {"stats": stats, "ovr": int(sum(stats.values()) / len(stats) + 0.5), "games": g, "wins": w,
            "rating": players[name], "rank": ranking.index(name) + 1}


def card_tier(ovr):
    return ("Elite", "#3fb8af") if ovr >= 85 else ("Gold", "#e6c200") if ovr >= 74 else \
           ("Silver", "#c0c0c0") if ovr >= 62 else ("Bronze", "#cd7f32")


def card_html(name, c):
    tier, colour = card_tier(c["ovr"])
    n = len(c["stats"])
    cells = "".join(
        f'<div style="display:flex;justify-content:space-between;{"grid-column:span 2;" if n % 2 and i == n - 1 else ""}">'
        f'<span>{k}</span><b>{v}</b></div>' for i, (k, v) in enumerate(c["stats"].items()))
    return (f'<div style="max-width:340px;margin:0 auto;border-radius:18px;padding:18px 20px;background:{colour};'
            'color:#1b1b1b;">'
            '<div style="display:flex;justify-content:space-between;align-items:flex-start;">'
            f'<div><div style="font-size:3.2rem;font-weight:800;line-height:1;">{c["ovr"]}</div>'
            '<div style="font-weight:700;letter-spacing:0.1em;font-size:0.8rem;">OVR</div></div>'
            f'<div style="text-align:right;"><div style="font-size:1.5rem;font-weight:800;line-height:1.1;">{html.escape(name)}</div>'
            f'<div style="font-size:0.85rem;font-weight:700;">{tier} Card</div>'
            f'<div style="font-size:0.85rem;">Rating {c["rating"]:.0f}  |  Rank {c["rank"]}</div></div></div>'
            '<div style="border-top:1px solid rgba(0,0,0,0.25);margin:12px 0;"></div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 22px;font-size:1.05rem;">{cells}</div>'
            '<div style="border-top:1px solid rgba(0,0,0,0.25);margin:12px 0;"></div>'
            '<div style="display:flex;justify-content:space-between;font-weight:600;">'
            f'<span>Record {c["wins"]}-{c["games"] - c["wins"]}</span><span>Games {c["games"]}</span></div></div>')


def radar_svg(stats, colour):
    """The same six stats as a radar chart (hexagon): the further out a corner, the higher the stat."""
    import math
    n, cx, cy, R = len(stats), 160, 140, 88

    def pt(i, v):
        ang = -math.pi / 2 + i * 2 * math.pi / n
        return cx + R * v / 99 * math.cos(ang), cy + R * v / 99 * math.sin(ang)
    ring = lambda lvl: " ".join("%.1f,%.1f" % pt(i, lvl) for i in range(n))
    out = ['<svg viewBox="-30 0 380 290" style="width:100%;max-width:360px;display:block;margin:6px auto;" '
           'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="13">']
    out += [f'<polygon points="{ring(l)}" fill="none" stroke="currentColor" stroke-opacity="0.25"/>' for l in (25, 50, 75, 99)]
    out += ['<line x1="%d" y1="%d" x2="%.1f" y2="%.1f" stroke="currentColor" stroke-opacity="0.25"/>' % ((cx, cy) + pt(i, 99))
            for i in range(n)]
    out.append(f'<polygon points="{" ".join("%.1f,%.1f" % pt(i, v) for i, v in enumerate(stats.values()))}" '
               f'fill="{colour}" fill-opacity="0.35" stroke="{colour}" stroke-width="2.5" stroke-linejoin="round"/>')
    for i, (k, v) in enumerate(stats.items()):
        x, y = pt(i, v)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{colour}"/>')
        lx, ly = pt(i, 99 * 1.27)
        anchor = "start" if lx > cx + 8 else "end" if lx < cx - 8 else "middle"
        out.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" fill="currentColor">{k} <tspan font-weight="700">{v}</tspan></text>')
    out.append("</svg>")
    return "".join(out)


STAT_META = {  # tile = button label, short = table heading, fmt = number format, disp = shown value from the internal one
    "Player Efficiency Rating": dict(tile="Efficiency (PER)", short="PER", fmt="%.1f", disp=None,
                                     desc="One number blending win rate, points won and lost, point difference and schedule. 15 is average."),
    "Most Dominant": dict(tile="Most Dominant", short="Dom", fmt="%+.1f", disp=None, desc="Average points won minus average points lost, per game."),
    "Win Machine": dict(tile="Win Machine", short="Wins", fmt="%d", disp=None, desc="Total wins."),
    "Iron Legs": dict(tile="Iron Legs", short="Games", fmt="%d", disp=None, desc="Total games played."),
    "Giant Killer": dict(tile="Giant Killer", short="Giant", fmt="%+.0f", disp=None, desc="Biggest rating gap overcome in a win."),
    "Hot Streak": dict(tile="Hot Streak", short="Streak", fmt="%d", disp=None, desc="Longest run of wins in a row."),
    "Brick Wall": dict(tile="Brick Wall", short="Wall", fmt="%.1f", disp=lambda v: -v, desc="Average points conceded per game. Fewer is better."),
    "Nail Biter": dict(tile="Nail Biter", short="Nail", fmt="%d", disp=None, desc="Games decided by 3 points or fewer, won or lost."),
    "Peak Performer": dict(tile="Peak Performer", short="Peak", fmt="%.0f", disp=None, desc="Highest rating ever reached."),
    "Longest Reign": dict(tile="Longest Reign", short="Reign", fmt="%d", disp=None, desc="Matches played while sitting at number 1."),
    "Toughest Schedule": dict(tile="Tough Schedule", short="Sched", fmt="%.0f", disp=None, desc="Average rating of the opponents faced."),
    "Conqueror": dict(tile="Conqueror", short="Conq", fmt="%d", disp=None, desc="Different opponents beaten."),
}
STAT_VIEWS = ["Cards", "Bars", "Dots", "Heatmap", "Medals", "Radar"]


def stat_meta(title):
    return STAT_META.get(title, dict(tile=title, short=title[:5], fmt="%.1f", disp=None, desc=""))


def goodness(rows):
    """0 (worst) to 1 (best) for every player who has a value for this stat."""
    vals = {r[0]: r[3] for r in rows if r[3] is not None}
    if not vals:
        return {}
    lo, hi = min(vals.values()), max(vals.values())
    return {p: (0.5 if hi - lo < 1e-9 else (v - lo) / (hi - lo)) for p, v in vals.items()}


def set_stat_pick(title):
    st.session_state.stats_pick = title


def toggle_radar(name):
    """Tap to add or remove a player from the radar overlay. At most 3: a fourth tap drops the earliest."""
    sel = st.session_state.get("radar_sel", [])
    st.session_state.radar_sel = [p for p in sel if p != name] if name in sel else (sel + [name])[-3:]


def view_bars(cards):
    """Pick a stat with the tiles, see everyone as a ranked bar chart (the longest bar is always the best)."""
    titles = [t for t, _, _ in cards]
    pick_ = st.session_state.get("stats_pick")
    pick_ = pick_ if pick_ in titles else titles[0]
    button_grid("grid_statpick", [(stat_meta(t)["tile"], f"sp_{i}", t == pick_, set_stat_pick, (t,)) for i, t in enumerate(titles)], cols=3)
    title, rows, empty = next(c for c in cards if c[0] == pick_)
    g = goodness(rows)
    out = [f'<div style="margin:14px 0 4px;font-weight:700;font-size:1.15rem;">{html.escape(title)}</div>',
           f'<div style="opacity:0.7;font-size:0.9rem;margin-bottom:6px;">{html.escape(stat_meta(title)["desc"])}</div>']
    place = 0
    for p, text, pl, v in rows:
        dim = "opacity:0.5;" if v is None else ""
        bar = ""
        if v is not None:
            bar = (f'<div style="height:10px;margin:5px 0 0 30px;border-radius:5px;background:rgba(128,128,128,0.16);">'
                   f'<div style="height:100%;width:{6 + 94 * g[p]:.1f}%;border-radius:5px;background:{player_color(p)};"></div></div>')
        out.append(f'<div style="margin-top:12px;{dim}"><div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px;">'
                   f'<div style="display:flex;align-items:baseline;min-width:0;"><span style="width:22px;text-align:right;margin-right:8px;'
                   f'font-weight:700;opacity:0.65;flex:none;">{pl if pl else "-"}</span>'
                   f'<span style="font-weight:600;">{html.escape(p)}</span></div>'
                   f'<div style="font-weight:700;color:#2a9d8f;flex:none;">{html.escape(text)}</div></div>{bar}</div>')
    st.markdown("".join(out), unsafe_allow_html=True)


def short_codes(names):
    codes = {}
    for n in names:
        w = n.split()
        codes[n] = (w[0][0] + (w[1][0] if len(w) > 1 else w[0][1:2])).upper()
    for c in {c for c in codes.values() if list(codes.values()).count(c) > 1}:
        for n in [n for n in codes if codes[n] == c]:
            codes[n] = n.replace(" ", "")[:3].upper()
    return codes


def view_dots(cards):
    """One line per stat with a dot for every player: dots bunch up where players are close, spread out where they differ."""
    names = sorted({r[0] for _, rows, _ in cards for r in rows})
    codes = short_codes(names)
    out = ['<div style="display:flex;flex-wrap:wrap;gap:4px 14px;margin-bottom:6px;font-size:0.9rem;">' + "".join(
        f'<span style="display:inline-flex;align-items:center;"><span style="width:20px;height:20px;border-radius:50%;background:{player_color(n)};'
        f'color:#fff;font-size:8px;font-weight:700;display:inline-flex;align-items:center;justify-content:center;margin-right:5px;">{codes[n]}</span>'
        f'{html.escape(n)}</span>' for n in names) + '</div>',
        '<div style="opacity:0.65;font-size:0.85rem;margin-bottom:4px;">Further right is better in every stat.</div>']
    for title, rows, empty in cards:
        g = goodness(rows)
        if not g:
            continue
        placed, lane_x, lane_of = [], [], {}
        for p in sorted(g, key=lambda q: g[q]):
            x = 4 + 92 * g[p]
            r = 0
            while r < len(lane_x) and any(abs(x - px) < 8.5 for px in lane_x[r]):
                r += 1
            if r == len(lane_x):
                lane_x.append([])
            lane_x[r].append(x)
            lane_of[p] = (x, r)
        dots = "".join(
            f'<div title="{html.escape(p)}" style="position:absolute;left:calc({x:.1f}% - 11px);bottom:{r * 24 + 6}px;width:22px;height:22px;'
            f'border-radius:50%;background:{player_color(p)};color:#fff;font-size:9px;font-weight:700;display:flex;align-items:center;'
            f'justify-content:center;">{codes[p]}</div>' for p, (x, r) in lane_of.items())
        worst = min((r for r in rows if r[3] is not None), key=lambda r: r[3])
        best = [r for r in rows if r[2] == 1]
        out.append('<div style="border-top:1px solid rgba(128,128,128,0.25);padding:12px 4px 8px;">'
                   '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px;">'
                   f'<div style="font-weight:700;">{html.escape(title)}</div>'
                   f'<div style="color:#2a9d8f;font-weight:700;font-size:0.9rem;text-align:right;">{html.escape(" & ".join(r[0] for r in best))}</div></div>'
                   f'<div style="position:relative;height:{len(lane_x) * 24 + 10}px;margin-top:4px;">'
                   '<div style="position:absolute;left:0;right:0;bottom:0;height:2px;background:rgba(128,128,128,0.35);"></div>' + dots + '</div>'
                   '<div style="display:flex;justify-content:space-between;font-size:0.8rem;opacity:0.7;margin-top:3px;">'
                   f'<span>{html.escape(worst[1])}</span><span>{html.escape(best[0][1] if best else "")}</span></div></div>')
    st.markdown("".join(out), unsafe_allow_html=True)


def view_heatmap(cards):
    """Players down the side, stats across the top. The darker the cell, the better that player is at that stat."""
    names = sorted({r[0] for _, rows, _ in cards for r in rows})
    data, good = {}, {}
    for title, rows, _ in cards:
        m = stat_meta(title)
        conv = m["disp"] or (lambda v: v)
        data[m["short"]] = pd.Series({r[0]: conv(r[3]) for r in rows if r[3] is not None}, dtype=float)
        good[m["short"]] = goodness(rows)
    df = pd.DataFrame({k: v.reindex(names) for k, v in data.items()})
    avg = {p: sum(g.get(p, 0) for g in good.values()) / len(good) for p in names}
    df = df.loc[sorted(names, key=lambda p: -avg[p])]
    df.index.name = "Player"

    def paint(frame):
        css = pd.DataFrame("", index=frame.index, columns=frame.columns)
        for k in frame.columns:
            for p in frame.index:
                g = good.get(k, {}).get(p)
                if g is not None:
                    css.loc[p, k] = f"background-color: rgba(42,157,143,{0.10 + 0.75 * g:.2f});" + ("color: #ffffff;" if g > 0.6 else "")
        return css
    cfg = {stat_meta(t)["short"]: st.column_config.NumberColumn(stat_meta(t)["short"], help=t, format=stat_meta(t)["fmt"])
           for t, _, _ in cards}
    st.dataframe(df.style.apply(paint, axis=None), height=35 * (len(df) + 1) + 14, column_config=cfg)
    st.caption("Darker means better in that column. Best all-rounders are at the top. Swipe sideways for more stats. "
               + "  |  ".join(f"{stat_meta(t)['short']} = {t}" for t, _, _ in cards))


def view_medals(cards):
    """Olympic-style table: how many stats each player leads (gold), is 2nd in (silver) and 3rd in (bronze)."""
    names = sorted({r[0] for _, rows, _ in cards for r in rows})
    tally = {p: [0, 0, 0] for p in names}
    for _, rows, _ in cards:
        for r in rows:
            if r[2] in (1, 2, 3):
                tally[r[0]][r[2] - 1] += 1
    order = sorted(names, key=lambda p: (-(3 * tally[p][0] + 2 * tally[p][1] + tally[p][2]), -tally[p][0], -tally[p][1], p))
    df = pd.DataFrame([{"#": i + 1, "Player": p, "Gold": tally[p][0], "Silver": tally[p][1], "Bronze": tally[p][2],
                        "Points": 3 * tally[p][0] + 2 * tally[p][1] + tally[p][2]} for i, p in enumerate(order)])

    def tint(col, colour):
        return [f"background-color: {colour}; color: #000000;" if v > 0 else "" for v in col]
    styled = (df.style.apply(tint, colour="#FFD700", subset=["Gold"]).apply(tint, colour="#C0C0C0", subset=["Silver"])
              .apply(tint, colour="#CD7F32", subset=["Bronze"]))
    st.dataframe(styled, hide_index=True, height=35 * (len(df) + 1) + 3,
                 column_config={"#": st.column_config.NumberColumn("#", width="small"), "Player": st.column_config.TextColumn("Player", width="medium")})
    st.caption("Gold, silver and bronze count the stats a player is 1st, 2nd or 3rd in. Points: gold 3, silver 2, bronze 1. Ties share a medal.")


def radar_overlay_svg(entries):
    """Several players' card-stat radars drawn on top of each other. entries = [(name, stats, colour), ...]"""
    import math
    n, cx, cy, R = len(CARD_LABELS), 160, 140, 88

    def pt(i, v):
        ang = -math.pi / 2 + i * 2 * math.pi / n
        return cx + R * v / 99 * math.cos(ang), cy + R * v / 99 * math.sin(ang)
    out = ['<svg viewBox="-30 0 380 290" style="width:100%;max-width:380px;display:block;margin:6px auto;" '
           'xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" font-size="13">']
    out += [f'<polygon points="{" ".join("%.1f,%.1f" % pt(i, lvl) for i in range(n))}" fill="none" stroke="currentColor" stroke-opacity="0.25"/>'
            for lvl in (25, 50, 75, 99)]
    out += ['<line x1="%d" y1="%d" x2="%.1f" y2="%.1f" stroke="currentColor" stroke-opacity="0.25"/>' % ((cx, cy) + pt(i, 99)) for i in range(n)]
    for _, stats, colour in entries:
        pts = " ".join("%.1f,%.1f" % pt(i, v) for i, v in enumerate(stats.values()))
        out.append(f'<polygon points="{pts}" fill="{colour}" fill-opacity="0.16" stroke="{colour}" stroke-width="2.5" stroke-linejoin="round"/>')
        out += [f'<circle cx="{pt(i, v)[0]:.1f}" cy="{pt(i, v)[1]:.1f}" r="3.2" fill="{colour}"/>' for i, v in enumerate(stats.values())]
    for i, k in enumerate(CARD_LABELS):
        lx, ly = pt(i, 99 * 1.27)
        anchor = "start" if lx > cx + 8 else "end" if lx < cx - 8 else "middle"
        out.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" fill="currentColor">{k}</text>')
    out.append("</svg>")
    return "".join(out)


def view_radar():
    """Pick up to three players and compare their trading-card stats on one radar chart."""
    cards = {p: card_stats(p) for p in players}
    cards = {p: c for p, c in cards.items() if c}
    if not cards:
        st.info("No games played yet.")
        return
    ranked_ = sorted(cards, key=lambda p: -players[p])
    if "radar_sel" not in st.session_state:
        st.session_state.radar_sel = ranked_[:3]
    sel = [p for p in st.session_state.radar_sel if p in cards]
    st.caption("Tap up to three players to compare their card stats.")
    button_grid("grid_radar", [(p, f"rd_{p}", p in sel, toggle_radar, (p,)) for p in ranked_])
    if not sel:
        st.info("Tap a player to draw their radar.")
        return
    st.markdown(radar_overlay_svg([(p, cards[p]["stats"], player_color(p)) for p in sel]), unsafe_allow_html=True)
    st.markdown('<div style="display:flex;flex-wrap:wrap;gap:4px 18px;justify-content:center;margin-bottom:8px;">' + "".join(
        f'<span style="display:inline-flex;align-items:center;"><span style="width:12px;height:12px;border-radius:50%;background:{player_color(p)};'
        f'display:inline-block;margin-right:6px;"></span>{html.escape(p)} <span style="opacity:0.6;margin-left:6px;">OVR {cards[p]["ovr"]}</span></span>'
        for p in sel) + "</div>", unsafe_allow_html=True)
    table = pd.DataFrame({p: cards[p]["stats"] for p in sel})
    st.dataframe(table, height=35 * (len(table) + 1) + 3)


@st.fragment
def stats_views():
    """MODE 3: the Stats tab, with a row of tabs to switch between different ways of showing the same numbers."""
    view = st.radio("View", STAT_VIEWS, horizontal=True, key="stats_view", label_visibility="collapsed")
    cards = build_awards(matches, everyone=True)
    if view == "Cards":
        awards_html(cards, everyone=True)
    elif view == "Bars":
        view_bars(cards)
    elif view == "Dots":
        view_dots(cards)
    elif view == "Heatmap":
        view_heatmap(cards)
    elif view == "Medals":
        view_medals(cards)
    else:
        view_radar()


def pick(container, label, options, **kw):
    """A dropdown you can only tap, not type into, so no phone keyboard pops up."""
    try:
        return container.selectbox(label, options, filter_mode=None, **kw)
    except TypeError:  # very old Streamlit without filter_mode
        return container.selectbox(label, options, **kw)


def toggle_chart(name):
    """Tap a player's tile to add or remove their line from the rating history graph."""
    sel = st.session_state.get("chart_sel", [])
    st.session_state.chart_sel = [p for p in sel if p != name] if name in sel else sel + [name]


def wide_button(container, label, **kw):
    """A button that fills its column (works across Streamlit versions)."""
    try:
        return container.button(label, width="stretch", **kw)
    except TypeError:
        return container.button(label, use_container_width=True, **kw)


def form_btn(container, label):
    """A submit button that fills its column (works across Streamlit versions)."""
    try:
        return container.form_submit_button(label, width="stretch")
    except TypeError:
        return container.form_submit_button(label, use_container_width=True)


def set_score(slot, n):
    st.session_state[slot] = n


def clear_entry():
    st.session_state.m_players, st.session_state.m_s1, st.session_state.m_s2 = [], None, None


def tap(label, key, selected, on_click, args):
    kw = dict(key=key, type="primary" if selected else "secondary", on_click=on_click, args=args)
    try:
        return st.button(label, width="stretch", **kw)
    except TypeError:  # older Streamlit versions
        return st.button(label, use_container_width=True, **kw)


def button_grid(grid_key, cells, cols=4):
    """Tappable boxes in a grid. Always 4 columns (even on a phone) and rows are added as needed.
    cells = [(label, button_key, selected, on_click, args), ...]"""
    with st.container(key=grid_key):
        for i in range(0, len(cells), cols):
            for col, cell in zip(st.columns(cols), cells[i:i + cols]):
                with col:
                    tap(*cell)


def compare_table(a, b, rows):
    """Side-by-side comparison: names on top, stat label in the middle, each player's number on their side.
    rows = [(label, (shown_a, number_a), (shown_b, number_b)), ...]. The higher number is highlighted."""
    line = "border-top:1px solid rgba(128,128,128,0.25);padding:8px 4px;"
    name_css = "font-weight:700;font-size:1.15rem;padding-bottom:8px;"
    out = ['<div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:baseline;text-align:center;">',
           f'<div style="{name_css}">{html.escape(a)}</div><div></div>',
           f'<div style="{name_css}">{html.escape(b)}</div>']
    for label, (sa, na), (sb, nb) in rows:
        win = "font-weight:700;color:#2a9d8f;"
        out.append(f'<div style="{line}font-size:1.5rem;{win if na > nb else ""}">{sa}</div>'
                   f'<div style="{line}font-size:0.85rem;opacity:0.7;padding:8px 14px;">{html.escape(label)}</div>'
                   f'<div style="{line}font-size:1.5rem;{win if nb > na else ""}">{sb}</div>')
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


# ---------- page ----------
st.set_page_config(page_title="Squash Elo")
st.title("Squash Elo")
data, sha = load()
players, matches = data["players"], data["matches"]
pending = data.setdefault("pending", [])  # results waiting for approval: they change nothing until approved
authed = st.session_state.get("auth", False)

with st.expander("Logged In" if authed else "Login (Needed To Edit)"):
    if authed:
        if st.button("Log Out"):
            st.session_state.auth = False
            st.session_state.page = "Leaderboard"  # don't stay on the Awards tab after logging out
            st.rerun()
    else:
        pw = st.text_input("Password", type="password")
        if st.button("Log In"):
            if pw == st.secrets["APP_PASSWORD"]:
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Wrong password")

if "flash_err" in st.session_state:
    st.error(st.session_state.pop("flash_err"))
if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
if authed and pending:
    st.markdown('<div style="border-left:4px solid #d6a300;background:rgba(214,163,0,0.12);border-radius:4px;'
                f'padding:8px 12px;margin:4px 0 10px 0;"><b>{len(pending)} result{"" if len(pending) == 1 else "s"} '
                f'waiting for approval.</b> Review {"it" if len(pending) == 1 else "them"} in Match History.</div>',
                unsafe_allow_html=True)

# Streamlit's built-in st.tabs jump back to the first tab on reruns, so these tabs are a keyed radio
# (which remembers the selection) styled with CSS to look like normal tabs.
st.markdown("""
<style>
:is(.st-key-page, .st-key-stats_view) div[role="radiogroup"] { gap: 0; flex-wrap: wrap; border-bottom: 1px solid rgba(128,128,128,0.35); }
:is(.st-key-page, .st-key-stats_view) div[role="radiogroup"] label { padding: 8px 14px; margin: 0; cursor: pointer; border-bottom: 2px solid transparent; }
:is(.st-key-page, .st-key-stats_view) div[role="radiogroup"] label > div:first-child { display: none; }
:is(.st-key-page, .st-key-stats_view) div[role="radiogroup"] label:has(input:checked) { border-bottom-color: #ff4b4b; }
:is(.st-key-page, .st-key-stats_view) div[role="radiogroup"] label:has(input:checked) p { color: #ff4b4b; font-weight: 600; }
/* hide the little hover toolbar (download CSV, fullscreen, search) on tables and graphs */
[data-testid="stElementToolbar"], [data-testid="stDataFrameToolbar"], .vega-embed details, .vega-embed summary { display: none !important; }
/* pull the title up closer to the top bar */
[data-testid="stMainBlockContainer"], .block-container { padding-top: 3.75rem !important; }
[data-testid="stHeading"] h1 { padding-top: 0.25rem !important; }
/* tappable tile grids: always 4 columns, even on a phone */
[class*="st-key-grid_"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.4rem !important; }
[class*="st-key-grid_"] [data-testid="stColumn"], [class*="st-key-grid_"] [data-testid="column"] {
    min-width: 0 !important; flex: 1 1 0 !important; width: auto !important; }
[class*="st-key-grid_"] button { min-height: 2.8rem; padding: 0.25rem 0.2rem; }
[class*="st-key-grid_"] button p { font-size: 0.9rem; line-height: 1.1; overflow-wrap: anywhere; }
.st-key-grid_s1 button p, .st-key-grid_s2 button p { font-size: 1.15rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

extra_tabs = {0: [], 1: ["Awards", "Fun"], 2: ["Stats"], 3: ["Stats"]}.get(MODE, [])
pages = ["Leaderboard", "Enter Match", "Match History", "Player History"] + (extra_tabs if authed else [])
if st.session_state.get("page") not in pages:
    st.session_state.page = "Leaderboard"
page = st.radio("Page", pages, horizontal=True, key="page", label_visibility="collapsed")

# ----- leaderboard -----
if page == "Leaderboard":
    stats = {p: [0, 0] for p in players}
    for m in matches:
        loser = m["p2"] if m["winner"] == m["p1"] else m["p1"]
        stats[m["winner"]][0] += 1
        stats[loser][1] += 1
    rows = [{"Rank": i, "Player": p, "Rating": round(players[p]), "Played": sum(stats[p]),
             "W": stats[p][0], "L": stats[p][1],
             "Win %": round(100 * stats[p][0] / sum(stats[p])) if sum(stats[p]) else None}
            for i, p in enumerate(sorted(players, key=lambda x: -players[x]), 1)]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, height=35 * (len(rows) + 1) + 3,
                     column_config={"Win %": st.column_config.NumberColumn("Win %", format="%.0f%%")})
    else:
        st.info("No players yet.")

    # ----- all players' ratings on one graph (read-only) -----
    @st.fragment
    def rating_chart():
        if matches:
            st.subheader("Rating History")
            tl = rating_timeline(matches)
            ranked = [p for p in sorted(players, key=lambda x: -players[x]) if p in set(tl["Player"])]
            if "chart_sel" not in st.session_state:
                st.session_state.chart_sel = ranked[:6]  # start with the top 6
            shown = [p for p in st.session_state.chart_sel if p in ranked]
            st.caption("Tap a player to show or hide their line.")
            button_grid("grid_chart", [(p, f"chart_{p}", p in shown, toggle_chart, (p,)) for p in ranked])
            tl = tl[tl["Player"].isin(shown)]
            if shown:
                # Fixed chart (no .interactive()): can't be dragged or zoomed, stretches to fit the page.
                chart = (alt.Chart(tl).mark_line(point=True)
                         .encode(x=alt.X("Game:Q", axis=alt.Axis(tickMinStep=1, format="d"),
                                         title="Games Played By Each Player"),
                                 y=alt.Y("Rating:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
                                 color=alt.Color("Player:N", legend=None,
                                                 scale=alt.Scale(domain=shown, range=[player_color(p) for p in shown])),
                                 tooltip=["Player:N", "Game:Q", alt.Tooltip("Rating:Q", format=".0f")])
                         .properties(width="container", height=320))
                st.altair_chart(chart)
                # Our own key under the graph: wraps onto more lines as needed, so no name is ever cut off.
                st.markdown('<div style="display:flex;flex-wrap:wrap;gap:4px 16px;margin-top:4px;">' + "".join(
                    f'<span style="display:inline-flex;align-items:center;">'
                    f'<span style="width:12px;height:12px;border-radius:50%;background:{player_color(p)};'
                    f'display:inline-block;margin-right:6px;"></span>{html.escape(p)}</span>' for p in shown) + "</div>",
                    unsafe_allow_html=True)
            else:
                st.info("Tap at least one player to show their rating history.")
    rating_chart()

    if authed:
        with st.form("add", clear_on_submit=True):
            name = st.text_input("New Player")
            if st.form_submit_button("Add Player") and name.strip():
                name = name.strip()
                if name.lower() in [p.lower() for p in players]:
                    st.error("That player already exists.")
                else:
                    players[name] = START_RATING
                    save(data, sha, f"Add player {name}")
                    st.session_state.flash = f"Added {name}"
                    st.rerun()

        if players:
            st.subheader("Manage Players")
            who = pick(st, "Player", sorted(players), key="manage_who")
            with st.form("rename", clear_on_submit=True):
                new = st.text_input("New Name")
                if st.form_submit_button("Rename Player"):
                    new = new.strip()
                    if not new:
                        st.error("Enter a new name.")
                    elif new == who:
                        st.error("That is already their name.")
                    elif new.lower() in [p.lower() for p in players if p != who]:
                        st.error("Another player already has that name.")
                    else:
                        rename_player(players, matches + pending, who, new)
                        save(data, sha, f"Rename {who} to {new}")
                        forget_names()
                        st.session_state.flash = f"Renamed {who} to {new}. All their matches were updated."
                        st.rerun()
            played = sum(who in (m["p1"], m["p2"]) for m in matches + pending)
            if played:
                st.caption(f"{who} is in {played} match{'' if played == 1 else 'es'} (counting any waiting for approval), "
                           "so they can't be removed. Delete or cancel those in Match History first if you really want to remove them.")
            elif st.checkbox(f"I Want To Remove {who}", key=f"rm_ok_{who}") and st.button("Remove Player"):
                del players[who]
                save(data, sha, f"Remove player {who}")
                forget_names()
                st.session_state.flash = f"Removed {who}."
                st.rerun()

# ----- enter match -----
elif page == "Enter Match":
    if len(players) < 2:
        st.info("Add at least two players first." if authed else "There aren't enough players yet. Ask someone who is logged in to add them.")
    else:
        @st.fragment
        def enter_match():
            sel = [p for p in st.session_state.get("m_players", []) if p in players]
            st.session_state.m_players = sel
            st.caption("Games go to 11, win by 2. At 14-14 the next point wins (15-14). "
                       f"Winning 11-0 earns a +{SHUTOUT_BONUS} rating bonus.")
            if not authed:
                st.caption("Results entered here are sent for approval. They count once someone who is logged in approves them.")
            if pending:
                st.caption(f"{len(pending)} result{' is' if len(pending) == 1 else 's are'} already waiting for approval (see Match History).")
            st.markdown("**Tap Two Players**")
            button_grid("grid_players", [(n, f"pick_{n}", n in sel, toggle_player, (n,)) for n in sorted(players)])
            st.caption(f"Player 1: {sel[0] if sel else '-'}   |   Player 2: {sel[1] if len(sel) > 1 else '-'}")

            if len(sel) == 2:
                p1, p2 = sel
                s1, s2 = st.session_state.get("m_s1"), st.session_state.get("m_s2")
                st.markdown(f"**{p1} Score**")
                button_grid("grid_s1", [(str(n), f"sc1_{n}", s1 == n, set_score, ("m_s1", n)) for n in range(16)])
                st.markdown(f"**{p2} Score**")
                button_grid("grid_s2", [(str(n), f"sc2_{n}", s2 == n, set_score, ("m_s2", n)) for n in range(16)])
                st.markdown(f"### {p1} {'-' if s1 is None else s1}  :  {'-' if s2 is None else s2} {p2}")

                go = st.button("Submit Match" if authed else "Submit For Approval", type="primary", key="submit_match")
                st.button("Clear", key="clear_entry", on_click=clear_entry)
                if go:
                    if s1 is None or s2 is None:
                        st.error("Tap a score for both players.")
                    elif not valid_game(s1, s2):
                        st.error(f"{s1}-{s2} isn't valid. {SCORE_RULES}")
                    elif not authed and len(pending) >= MAX_PENDING:
                        st.error("Too many results are waiting for approval right now. Please try again once some have been approved or cancelled.")
                    elif not authed:
                        pending.append({"id": uuid.uuid4().hex[:8], "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                        "p1": p1, "p2": p2, "score": [s1, s2]})
                        save(data, sha, f"Waiting for approval: {p1} {s1}-{s2} {p2}")
                        clear_entry()
                        st.session_state.flash = f"Sent for approval: {p1} {s1}-{s2} {p2}. It counts once someone approves it."
                        st.rerun()
                    else:
                        r1, r2 = players[p1], players[p2]
                        bonus = shutout_bonus(s1, s2)
                        n1, n2 = play(r1, r2, s1, s2, bonus)
                        players[p1], players[p2] = n1, n2
                        matches.append({
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "p1": p1, "p2": p2,
                            "score": [s1, s2], "winner": p1 if s1 > s2 else p2, "bonus": bonus,
                            "p1_before": r1, "p1_after": n1, "p2_before": r2, "p2_after": n2})
                        save(data, sha, f"{p1} {s1}-{s2} {p2}")
                        msg = f"{p1}: {r1:.0f} → {n1:.0f}  |  {p2}: {r2:.0f} → {n2:.0f}"
                        if bonus:
                            msg += f"  |  11-0 bonus: +{bonus} to {p1 if s1 > s2 else p2}"
                        clear_entry()
                        st.session_state.flash = msg
                        st.rerun()
        enter_match()

# ----- match history -----
elif page == "Match History":
    if pending:
        st.subheader(f"Waiting For Approval ({len(pending)})")
        if not authed:
            st.caption("These results count once someone who is logged in approves them.")
            st.dataframe(pd.DataFrame([{"Result": f'{m["p1"]} {m["score"][0]}-{m["score"][1]} {m["p2"]}',
                                        "Submitted": m["date"][5:]} for m in pending]),
                         hide_index=True, height=35 * (len(pending) + 1) + 3)
        else:
            for m in list(pending):
                a, b = m["score"]
                st.markdown(f'<div style="margin-top:6px;"><b>{html.escape(m["p1"])} {a} - {b} {html.escape(m["p2"])}</b><br>'
                            f'<span style="opacity:0.65;font-size:0.85rem;">Submitted {m["date"]}</span></div>',
                            unsafe_allow_html=True)
                with st.container(key=f"grid_pend_{m['id']}"):
                    c1, c2 = st.columns(2)
                    approve = wide_button(c1, "Approve", key=f"approve_{m['id']}", type="primary")
                    cancel = wide_button(c2, "Cancel", key=f"cancel_{m['id']}")
                if approve:
                    p1, p2 = m["p1"], m["p2"]
                    if p1 not in players or p2 not in players or not valid_game(a, b):
                        st.error("This result can't be approved (a player no longer exists, or the score isn't valid). Cancel it instead.")
                    else:
                        r1, r2 = players[p1], players[p2]
                        bonus = shutout_bonus(a, b)
                        n1, n2 = play(r1, r2, a, b, bonus)
                        players[p1], players[p2] = n1, n2
                        matches.append({"date": m["date"], "p1": p1, "p2": p2, "score": [a, b],
                                        "winner": p1 if a > b else p2, "bonus": bonus,
                                        "p1_before": r1, "p1_after": n1, "p2_before": r2, "p2_after": n2})
                        pending.remove(m)
                        save(data, sha, f"Approve {p1} {a}-{b} {p2}")
                        st.session_state.flash = (f"Approved {p1} {a}-{b} {p2}.  {p1}: {r1:.0f} → {n1:.0f}  |  {p2}: {r2:.0f} → {n2:.0f}"
                                                  + (f"  |  11-0 bonus: +{bonus}" if bonus else ""))
                        st.rerun()
                elif cancel:
                    pending.remove(m)
                    save(data, sha, f"Cancel {m['p1']} {a}-{b} {m['p2']}")
                    st.session_state.flash = f"Cancelled {m['p1']} {a}-{b} {m['p2']}."
                    st.rerun()
        st.divider()
    if matches:
        st.dataframe(pd.DataFrame([
            {"#": k, "Date": m["date"], "Player 1": m["p1"], "Player 2": m["p2"],
             "Score": f'{m["score"][0]}-{m["score"][1]}', "Winner": m["winner"]}
            for k, m in zip(range(len(matches), 0, -1), reversed(matches))]), hide_index=True)

        if authed:
            @st.fragment
            def edit_match():
                st.subheader("Edit Or Delete A Match")
                st.caption("Ratings for every later match are recalculated automatically.")
                order = list(range(len(matches) - 1, -1, -1))  # newest first
                i = pick(
                    st, "Match", order, key="edit_idx",
                    format_func=lambda k: f'#{k + 1}  |  {matches[k]["date"]}  |  {matches[k]["p1"]} '
                                          f'{matches[k]["score"][0]}-{matches[k]["score"][1]} {matches[k]["p2"]}')
                m = matches[i]
                names = sorted(players)
                with st.form(f"edit_{i}"):
                    c1, c2 = st.columns(2)
                    e1 = pick(c1, "Player 1", names, index=names.index(m["p1"]), key=f"e1_{i}")
                    e2 = pick(c2, "Player 2", names, index=names.index(m["p2"]), key=f"e2_{i}")
                    s1 = c1.number_input("Player 1 Score", 0, max(15, m["score"][0]), m["score"][0], step=1, key=f"s1_{i}")
                    s2 = c2.number_input("Player 2 Score", 0, max(15, m["score"][1]), m["score"][1], step=1, key=f"s2_{i}")
                    with st.container(key="grid_editbtns"):
                        b1, b2 = st.columns(2)
                        save_clicked = form_btn(b1, "Save Changes")
                        delete_clicked = form_btn(b2, "Delete")
                if save_clicked:
                    if e1 == e2:
                        st.error("Pick two different players.")
                    elif not valid_game(s1, s2):
                        st.error(f"{s1}-{s2} isn't valid. {SCORE_RULES}")
                    else:
                        m.update(p1=e1, p2=e2, score=[s1, s2], bonus=shutout_bonus(s1, s2))
                        recompute(players, matches)
                        save(data, sha, f"Edit match: {e1} {s1}-{s2} {e2}")
                        st.session_state.flash = "Match updated and ratings recalculated."
                        st.rerun()
                elif delete_clicked:
                    matches.pop(i)
                    recompute(players, matches)
                    save(data, sha, "Delete match")
                    st.session_state.flash = "Match deleted and ratings recalculated."
                    st.rerun()
            edit_match()
    else:
        st.info("No matches yet.")

# ----- player history (with head to head) -----
elif page == "Player History":
    @st.fragment
    def player_history():
        if players:
            name = pick(st, "Player", sorted(players), key="hist_player")
            ratings, hist = [START_RATING], []
            for m in matches:
                if name not in (m["p1"], m["p2"]):
                    continue
                me = "p1" if m["p1"] == name else "p2"
                a, b = m["score"]
                before, after = m[me + "_before"], m[me + "_after"]
                ratings.append(after)
                hist.append({"Date": m["date"], "Opponent": m["p2"] if me == "p1" else m["p1"],
                             "Result": "Win" if m["winner"] == name else "Loss",
                             "Score": f"{a}-{b}" if me == "p1" else f"{b}-{a}",
                             "Rating": f"{before:.0f} → {after:.0f}", "Change": f"{after - before:+.1f}"})
            st.metric("Current Rating", round(ratings[-1]))
            if hist:
                last5 = hist[-5:]
                badges = "".join(
                    '<span style="display:inline-block;width:2rem;height:2rem;line-height:2rem;text-align:center;'
                    'border-radius:6px;margin-right:6px;font-weight:700;color:#fff;'
                    f'background:{"#2a9d8f" if h["Result"] == "Win" else "#d1495b"};">{h["Result"][0]}</span>'
                    for h in last5)
                st.markdown(f"**Recent Form** (Last {len(last5)}, Latest On The Right)<div style='margin-top:6px'>{badges}</div>",
                            unsafe_allow_html=True)
            # Fixed chart: no .interactive(), so it can't be dragged or zoomed. It stretches to fit the page
            # and the axes automatically rescale to fit every match.
            chart_df = pd.DataFrame({"Match": range(len(ratings)), "Rating": ratings})
            chart = (alt.Chart(chart_df).mark_line(point=True)
                     .encode(x=alt.X("Match:Q", title="Matches Played", axis=alt.Axis(tickMinStep=1, format="d")),
                             y=alt.Y("Rating:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
                             tooltip=["Match:Q", alt.Tooltip("Rating:Q", format=".0f")])
                     .properties(width="container", height=300))
            st.altair_chart(chart)

            # ----- head to head, right under the graph -----
            st.subheader("Head To Head")
            others = [p for p in sorted(players) if p != name]
            if not others:
                st.info("Add another player to see head to head stats.")
            else:
                played = {}  # opponent -> (games against them, index of the latest one)
                for k, m in enumerate(matches):
                    if name in (m["p1"], m["p2"]):
                        o = m["p2"] if m["p1"] == name else m["p1"]
                        played[o] = (played.get(o, (0, 0))[0] + 1, k)
                fav = max((o for o in played if o in others), key=lambda o: played[o], default=others[0])  # tie: the latest
                opp = pick(st, "Versus", others, index=others.index(fav), key=f"hist_opp_{name}")
                h2h = [m for m in matches if {m["p1"], m["p2"]} == {name, opp}]
                if not h2h:
                    st.info(f"{name} and {opp} haven't played each other yet.")
                else:
                    n = len(h2h)
                    wins, pts, rows = {name: 0, opp: 0}, {name: 0, opp: 0}, []
                    for m in h2h:
                        s = dict(zip((m["p1"], m["p2"]), m["score"]))
                        wins[m["winner"]] += 1
                        pts[name] += s[name]
                        pts[opp] += s[opp]
                        rows.append({"Date": m["date"], name: s[name], opp: s[opp], "Winner": m["winner"]})
                    compare_table(name, opp, [
                        ("Rating", (f"{players[name]:.0f}", players[name]), (f"{players[opp]:.0f}", players[opp])),
                        ("Matches Won", (wins[name], wins[name]), (wins[opp], wins[opp])),
                        ("Points Won", (pts[name], pts[name]), (pts[opp], pts[opp])),
                        ("Win %", (f"{100 * wins[name] / n:.0f}%", wins[name]), (f"{100 * wins[opp] / n:.0f}%", wins[opp])),
                    ])
                    st.markdown(f"**Matches Played Together ({n})**")
                    st.dataframe(pd.DataFrame(rows[::-1]), hide_index=True)

            if hist:
                st.subheader("All Matches")
                st.dataframe(pd.DataFrame(hist[::-1]), hide_index=True)
        else:
            st.info("No players yet.")
    player_history()

# ----- awards -----
elif page in ("Awards", "Stats"):
    if not matches:
        st.info("No matches yet.")
    elif MODE == 3:
        stats_views()
    else:
        awards_html(build_awards(matches, everyone=MODE == 2), everyone=MODE == 2)

# ----- fun (logged in only) -----
else:
    st.subheader("Leaderboard Race")
    if not matches:
        st.info("No matches yet.")
    else:
        doc, height = race_html(matches)
        st.iframe(doc, height=height)

    st.subheader("Player Trading Cards")

    @st.fragment
    def trading_cards():
        if not players:
            st.info("No players yet.")
            return
        who = pick(st, "Player", sorted(players, key=lambda x: -players[x]), key="card_player")
        c = card_stats(who)
        if c is None:
            st.info(f"{who} hasn't played yet, so there's no card to show.")
            return
        st.html(card_html(who, c))
        st.markdown(radar_svg(c["stats"], player_color(who)), unsafe_allow_html=True)
    trading_cards()
