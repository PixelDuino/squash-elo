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
st.set_page_config(page_title="Squash Elo", page_icon="🎾")
st.title("Squash Elo")
data, sha = load()
players, matches = data["players"], data["matches"]
authed = st.session_state.get("auth", False)

with st.expander("Logged in ✅" if authed else "Login (needed to edit)"):
    if authed:
        if st.button("Log out"):
            st.session_state.auth = False
            st.rerun()
    else:
        pw = st.text_input("Password", type="password")
        if st.button("Log in"):
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
page = st.radio("Page", ["Leaderboard", "Enter Match", "Match History", "Player History"],
                horizontal=True, key="page", label_visibility="collapsed")

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
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"Win %": st.column_config.NumberColumn("Win %", format="%.0f%%")})
    else:
        st.info("No players yet.")
    if authed:
        with st.form("add", clear_on_submit=True):
            name = st.text_input("New player")
            if st.form_submit_button("Add player") and name.strip():
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
            a = c1.number_input("Player 1 score", min_value=0, max_value=99, value=None, step=1, placeholder="Score")
            b = c2.number_input("Player 2 score", min_value=0, max_value=99, value=None, step=1, placeholder="Score")
            if st.form_submit_button("Submit match"):
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
            {"Date": m["date"], "Player 1": m["p1"], "Player 2": m["p2"],
             "Score": f'{m["score"][0]}-{m["score"][1]}', "Winner": m["winner"]}
            for m in reversed(matches)]), hide_index=True)

        if authed:
            st.subheader("Edit or delete a match")
            st.caption("Ratings for every later match are recalculated automatically.")
            order = list(range(len(matches) - 1, -1, -1))  # newest first
            i = st.selectbox(
                "Match", order, key="edit_idx",
                format_func=lambda k: f'{matches[k]["date"]}  |  {matches[k]["p1"]} '
                                      f'{matches[k]["score"][0]}-{matches[k]["score"][1]} {matches[k]["p2"]}')
            m = matches[i]
            names = sorted(players)
            with st.form(f"edit_{i}"):
                c1, c2 = st.columns(2)
                e1 = c1.selectbox("Player 1", names, index=names.index(m["p1"]), key=f"e1_{i}")
                e2 = c2.selectbox("Player 2", names, index=names.index(m["p2"]), key=f"e2_{i}")
                s1 = c1.number_input("Player 1 score", 0, 99, m["score"][0], step=1, key=f"s1_{i}")
                s2 = c2.number_input("Player 2 score", 0, 99, m["score"][1], step=1, key=f"s2_{i}")
                if st.form_submit_button("Save changes"):
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
            if st.checkbox("I want to delete this match", key=f"del_ok_{i}") and st.button("Delete match"):
                matches.pop(i)
                recompute(players, matches)
                save(data, sha, "Delete match")
                st.session_state.flash = "Match deleted and ratings recalculated."
                st.rerun()
    else:
        st.info("No matches yet.")

# ----- player history (with head to head) -----
else:
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
        st.metric("Current rating", round(ratings[-1]))
        # Fixed chart: no .interactive(), so it can't be dragged or zoomed. It stretches to fit the page
        # and the axes automatically rescale to fit every match.
        chart_df = pd.DataFrame({"Match": range(len(ratings)), "Rating": ratings})
        chart = (alt.Chart(chart_df).mark_line(point=True)
                 .encode(x=alt.X("Match:Q", title="Matches played", axis=alt.Axis(tickMinStep=1, format="d")),
                         y=alt.Y("Rating:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format="d")),
                         tooltip=["Match:Q", alt.Tooltip("Rating:Q", format=".0f")])
                 .properties(width="container", height=300))
        st.altair_chart(chart)

        # ----- head to head, right under the graph -----
        st.subheader("Head to head")
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
                    ("Matches won", (wins[name], wins[name]), (wins[opp], wins[opp])),
                    ("Points won", (pts[name], pts[name]), (pts[opp], pts[opp])),
                    ("Win %", (f"{100 * wins[name] / n:.0f}%", wins[name]), (f"{100 * wins[opp] / n:.0f}%", wins[opp])),
                ])
                st.markdown(f"**Matches played together ({n})**")
                st.dataframe(pd.DataFrame(rows[::-1]), hide_index=True)

        if hist:
            st.subheader("All matches")
            st.dataframe(pd.DataFrame(hist[::-1]), hide_index=True)
    else:
        st.info("No players yet.")
