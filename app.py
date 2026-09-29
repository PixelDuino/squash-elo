"""Squash Elo - web version (Streamlit).
Anyone can VIEW. Only people with the password can add players / enter results.
Data lives in squash_data.json inside a separate private GitHub repo, so it survives restarts.
"""
import base64
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

# A keyed radio (instead of st.tabs) so the page you're on is remembered when anything changes.
page = st.radio("Page", ["Leaderboard", "Enter Match", "Match History", "Player History", "Head to Head"],
                horizontal=True, key="page", label_visibility="collapsed")

# ----- leaderboard -----
if page == "Leaderboard":
    stats = {p: [0, 0] for p in players}
    for m in matches:
        loser = m["p2"] if m["winner"] == m["p1"] else m["p1"]
        stats[m["winner"]][0] += 1
        stats[loser][1] += 1
    rows = [{"Rank": i, "Player": p, "Rating": round(players[p]), "Played": sum(stats[p]),
             "W": stats[p][0], "L": stats[p][1]}
            for i, p in enumerate(sorted(players, key=lambda x: -players[x]), 1)]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
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
        if authed and st.checkbox("Enable undo") and st.button("Undo last match"):
            m = matches.pop()
            players[m["p1"]], players[m["p2"]] = m["p1_before"], m["p2_before"]
            save(data, sha, f"Undo {m['p1']} vs {m['p2']}")
            st.session_state.flash = "Last match removed"
            st.rerun()
    else:
        st.info("No matches yet.")

# ----- player history -----
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
        if hist:
            st.dataframe(pd.DataFrame(hist[::-1]), hide_index=True)
    else:
        st.info("No players yet.")

# ----- head to head -----
else:
    if len(players) < 2:
        st.info("Add at least two players first.")
    else:
        names = sorted(players)
        c1, c2 = st.columns(2)
        A = c1.selectbox("Player 1", names, key="h2h_a")
        B = c2.selectbox("Player 2", names, index=1, key="h2h_b")
        if A == B:
            st.info("Pick two different players.")
        else:
            h2h = [m for m in matches if {m["p1"], m["p2"]} == {A, B}]
            if not h2h:
                st.info(f"{A} and {B} haven't played each other yet.")
            else:
                wins, pts, rows = {A: 0, B: 0}, {A: 0, B: 0}, []
                for m in h2h:
                    s = dict(zip((m["p1"], m["p2"]), m["score"]))
                    wins[m["winner"]] += 1
                    pts[A] += s[A]
                    pts[B] += s[B]
                    rows.append({"Date": m["date"], A: s[A], B: s[B], "Winner": m["winner"]})
                st.subheader("Matches won")
                x1, x2 = st.columns(2)
                x1.metric(A, wins[A])
                x2.metric(B, wins[B])
                st.subheader("Points won")
                y1, y2 = st.columns(2)
                y1.metric(A, pts[A])
                y2.metric(B, pts[B])
                st.subheader(f"Matches played together ({len(h2h)})")
                st.dataframe(pd.DataFrame(rows[::-1]), hide_index=True)
