"""Squash Elo - web version (Streamlit).
Anyone can VIEW. Only people with the password can add players / enter, edit or delete results.
Data lives in squash_data.json inside a separate private GitHub repo, so it survives restarts.
"""
import base64
import html
import json
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st

START_RATING, K_FACTOR = 1000, 32
MIN_GAMES = 3  # games a player needs before average-based awards (Most Dominant, Brick Wall) count
CLOSE_MARGIN = 3  # a game decided by this many points or fewer counts as a close game (Nail Biter)
URL = f"https://api.github.com/repos/{st.secrets['DATA_REPO']}/contents/squash_data.json"
HEAD = {"Authorization": f"Bearer {st.secrets['GITHUB_TOKEN']}", "Accept": "application/vnd.github+json"}


# ---------- storage ----------
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
    requests.put(URL, headers=HEAD, json=body, timeout=15).raise_for_status()


# ---------- elo ----------
def elo(r1, r2, s1):
    e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    return r1 + K_FACTOR * (s1 - e1), r2 + K_FACTOR * ((1 - s1) - (1 - e1))


def valid_game(a, b):
    hi, lo = max(a, b), min(a, b)
    return (hi == 11 and lo <= 9) or (hi > 11 and hi - lo == 2)


def recompute(players, matches):
    """Replay every match in order from the starting rating. Used after editing or deleting a match."""
    for p in players:
        players[p] = START_RATING
    for m in matches:
        a, b = m["score"]
        r1, r2 = players[m["p1"]], players[m["p2"]]
        n1, n2 = elo(r1, r2, 1 if a > b else 0)
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


def build_awards(matches):
    """READ-ONLY: returns [(award name, [winner names], stat text), ...]. Tied players share an award."""
    ps = player_stats(matches)

    def top(scores):
        if not scores:
            return [], None
        best = max(scores.values())
        return [p for p, v in scores.items() if abs(v - best) < 1e-9], best

    # Most Dominant: biggest gap between average points won and average points lost per game
    names, best = top({p: (d["f"] - d["a"]) / d["g"] for p, d in ps.items() if d["g"] >= MIN_GAMES})
    cards = [("Most Dominant", names, f"{best:+.1f} Pts Per Game" if names else f"Needs {MIN_GAMES}+ Games")]
    # Win Machine: most wins
    names, best = top({p: d["w"] for p, d in ps.items()})
    cards.append(("Win Machine", names, f"{best} Win{'' if best == 1 else 's'}" if names else ""))
    # Iron Legs: most games played
    names, best = top({p: d["g"] for p, d in ps.items()})
    cards.append(("Iron Legs", names, f"{best} Game{'' if best == 1 else 's'}" if names else ""))

    # Giant Killer: beat an opponent rated the most points higher than you (rating before the match)
    upset = {}
    for m in matches:
        win, lose = ("p1", "p2") if m["score"][0] > m["score"][1] else ("p2", "p1")
        gap = m[lose + "_before"] - m[win + "_before"]
        if gap > 0:
            upset[m[win]] = max(upset.get(m[win], 0), gap)
    names, best = top(upset)
    cards.append(("Giant Killer", names, f"+{best:.0f} Rating Gap" if names else "No Upsets Yet"))

    # Hot Streak: longest run of consecutive wins
    run, longest = {}, {}
    for m in matches:
        a, b = m["score"]
        for who, won in ((m["p1"], a > b), (m["p2"], b > a)):
            run[who] = run.get(who, 0) + 1 if won else 0
            longest[who] = max(longest.get(who, 0), run[who])
    names, best = top({p: v for p, v in longest.items() if v > 0})
    cards.append(("Hot Streak", names, f"{best} Win{'' if best == 1 else 's'} In A Row" if names else ""))

    # Brick Wall: fewest average points conceded per game
    names, best = top({p: -d["a"] / d["g"] for p, d in ps.items() if d["g"] >= MIN_GAMES})
    cards.append(("Brick Wall", names, f"{-best:.1f} Pts Against Per Game" if names else f"Needs {MIN_GAMES}+ Games"))

    # Nail Biter: most games decided by CLOSE_MARGIN points or fewer (win or lose)
    close = {}
    for m in matches:
        if abs(m["score"][0] - m["score"][1]) <= CLOSE_MARGIN:
            for who in (m["p1"], m["p2"]):
                close[who] = close.get(who, 0) + 1
    names, best = top(close)
    cards.append(("Nail Biter", names, f"{best} Close Game{'' if best == 1 else 's'}" if names else "No Close Games Yet"))

    # Peak Performer: highest rating ever reached
    peak = {}
    for m in matches:
        for k in ("p1", "p2"):
            peak[m[k]] = max(peak.get(m[k], START_RATING), m[k + "_after"])
    names, best = top(peak)
    cards.append(("Peak Performer", names, f"{best:.0f} Rating" if names else ""))
    return cards


def awards_html(cards):
    """Award name on top, winner underneath on the left, stat on the same line to the right."""
    out = []
    for title, winners, stat in cards:
        who = " &amp; ".join(html.escape(w) for w in winners) if winners else "No Winner Yet"
        out.append('<div style="border-top:1px solid rgba(128,128,128,0.25);padding:12px 4px;">'
                   f'<div style="font-weight:700;font-size:0.95rem;opacity:0.75;">{html.escape(title)}</div>'
                   '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;">'
                   f'<div style="font-weight:700;font-size:1.4rem;">{who}</div>'
                   f'<div style="font-weight:700;font-size:1.2rem;color:#2a9d8f;white-space:nowrap;">{html.escape(stat)}</div>'
                   '</div></div>')
    st.markdown("".join(out), unsafe_allow_html=True)


def compare_table(a, b, rows):
    """Side-by-side comparison: names on top, stat label in the middle, each player's number on their side.
    rows = [(label, (shown_a, number_a), (shown_b, number_b)), ...]. The higher number is highlighted."""
    line = "border-top:1px solid rgba(128,128,128,0.25);padding:8px 4px;"
    name_css = "font-weight:700;font-size:1.15rem;padding-bottom:8px;"
    out = ['<div style="display:grid;grid-template-columns:1fr auto 1fr;align-items:center;text-align:center;">',
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

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))

# Streamlit's built-in st.tabs jump back to the first tab on reruns, so these tabs are a keyed radio
# (which remembers the selection) styled with CSS to look like normal tabs.
st.markdown("""
<style>
.st-key-page div[role="radiogroup"] { gap: 0; flex-wrap: wrap; border-bottom: 1px solid rgba(128,128,128,0.35); }
.st-key-page div[role="radiogroup"] label { padding: 8px 14px; margin: 0; cursor: pointer; border-bottom: 2px solid transparent; }
.st-key-page div[role="radiogroup"] label > div:first-child { display: none; }
.st-key-page div[role="radiogroup"] label:has(input:checked) { border-bottom-color: #ff4b4b; }
.st-key-page div[role="radiogroup"] label:has(input:checked) p { color: #ff4b4b; font-weight: 600; }
</style>
""", unsafe_allow_html=True)
pages = ["Leaderboard", "Enter Match", "Match History", "Player History"] + (["Awards"] if authed else [])
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
        podium = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}  # flat gold, silver, bronze
        styled = pd.DataFrame(rows).style.apply(
            lambda r: [f"background-color: {podium[r['Rank']]}; color: #000000" if r["Rank"] in podium else ""] * len(r),
            axis=1)
        st.dataframe(styled, hide_index=True,
                     column_config={"Win %": st.column_config.NumberColumn("Win %", format="%.0f%%")})
    else:
        st.info("No players yet.")

    # ----- all players' ratings on one graph (read-only) -----
    if matches:
        st.subheader("Rating History")
        tl = rating_timeline(matches)
        ranked = [p for p in sorted(players, key=lambda x: -players[x]) if p in set(tl["Player"])]
        shown = st.multiselect("Players Shown", ranked, default=ranked[:6], key="chart_players")
        tl = tl[tl["Player"].isin(shown)]
        if shown:
            # Fixed chart (no .interactive()): can't be dragged or zoomed, stretches to fit the page.
            chart = (alt.Chart(tl).mark_line(point=True)
                     .encode(x=alt.X("Game:Q", title="Games Played By Each Player",
                                     axis=alt.Axis(tickMinStep=1, format="d")),
                             y=alt.Y("Rating:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
                             color=alt.Color("Player:N", legend=alt.Legend(orient="bottom", title=None)),
                             tooltip=["Player:N", "Game:Q", alt.Tooltip("Rating:Q", format=".0f")])
                     .properties(width="container", height=320))
            st.altair_chart(chart)
        else:
            st.info("Pick at least one player to show.")

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

# ----- enter match -----
elif page == "Enter Match":
    if not authed:
        st.info("Log in (top of page) to enter results.")
    elif len(players) < 2:
        st.info("Add at least two players first.")
    else:
        names = sorted(players)
        st.caption("Single game to 11, win by 2.")
        with st.form("match", clear_on_submit=True):
            c1, c2 = st.columns(2)
            p1 = c1.selectbox("Player 1", names)
            p2 = c2.selectbox("Player 2", names, index=1)
            a = c1.number_input("Player 1 Score", min_value=0, max_value=99, value=None, step=1, placeholder="Score")
            b = c2.number_input("Player 2 Score", min_value=0, max_value=99, value=None, step=1, placeholder="Score")
            if st.form_submit_button("Submit Match"):
                if p1 == p2:
                    st.error("Pick two different players.")
                elif a is None or b is None:
                    st.error("Enter both scores.")
                elif not valid_game(a, b):
                    st.error(f"{a}-{b} isn't valid. Games go to 11, win by 2 (e.g. 11-9, 12-10, 15-13).")
                else:
                    r1, r2 = players[p1], players[p2]
                    n1, n2 = elo(r1, r2, 1 if a > b else 0)
                    players[p1], players[p2] = n1, n2
                    matches.append({
                        "date": datetime.now().strftime("%Y-%m-%d %H:%M"), "p1": p1, "p2": p2,
                        "score": [a, b], "winner": p1 if a > b else p2,
                        "p1_before": r1, "p1_after": n1, "p2_before": r2, "p2_after": n2})
                    save(data, sha, f"{p1} {a}-{b} {p2}")
                    st.session_state.flash = f"{p1}: {r1:.0f} → {n1:.0f}  |  {p2}: {r2:.0f} → {n2:.0f}"
                    st.rerun()

# ----- match history -----
elif page == "Match History":
    if matches:
        st.dataframe(pd.DataFrame([
            {"#": k, "Date": m["date"], "Player 1": m["p1"], "Player 2": m["p2"],
             "Score": f'{m["score"][0]}-{m["score"][1]}', "Winner": m["winner"]}
            for k, m in zip(range(len(matches), 0, -1), reversed(matches))]), hide_index=True)

        if authed:
            st.subheader("Edit Or Delete A Match")
            st.caption("Ratings for every later match are recalculated automatically.")
            order = list(range(len(matches) - 1, -1, -1))  # newest first
            i = st.selectbox(
                "Match", order, key="edit_idx",
                format_func=lambda k: f'#{k + 1}  |  {matches[k]["date"]}  |  {matches[k]["p1"]} '
                                      f'{matches[k]["score"][0]}-{matches[k]["score"][1]} {matches[k]["p2"]}')
            m = matches[i]
            names = sorted(players)
            with st.form(f"edit_{i}"):
                c1, c2 = st.columns(2)
                e1 = c1.selectbox("Player 1", names, index=names.index(m["p1"]), key=f"e1_{i}")
                e2 = c2.selectbox("Player 2", names, index=names.index(m["p2"]), key=f"e2_{i}")
                s1 = c1.number_input("Player 1 Score", 0, 99, m["score"][0], step=1, key=f"s1_{i}")
                s2 = c2.number_input("Player 2 Score", 0, 99, m["score"][1], step=1, key=f"s2_{i}")
                if st.form_submit_button("Save Changes"):
                    if e1 == e2:
                        st.error("Pick two different players.")
                    elif not valid_game(s1, s2):
                        st.error(f"{s1}-{s2} isn't valid. Games go to 11, win by 2 (e.g. 11-9, 12-10, 15-13).")
                    else:
                        m.update(p1=e1, p2=e2, score=[s1, s2])
                        recompute(players, matches)
                        save(data, sha, f"Edit match: {e1} {s1}-{s2} {e2}")
                        st.session_state.flash = "Match updated and ratings recalculated."
                        st.rerun()
            if st.checkbox("I Want To Delete This Match", key=f"del_ok_{i}") and st.button("Delete Match"):
                matches.pop(i)
                recompute(players, matches)
                save(data, sha, "Delete match")
                st.session_state.flash = "Match deleted and ratings recalculated."
                st.rerun()
    else:
        st.info("No matches yet.")

# ----- player history (with head to head) -----
elif page == "Player History":
    if players:
        name = st.selectbox("Player", sorted(players), key="hist_player")
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
            opp = st.selectbox("Versus", others, key="hist_opp")
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

# ----- awards -----
else:
    if not matches:
        st.info("No matches yet.")
    else:
        awards_html(build_awards(matches))
