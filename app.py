import streamlit as st
import chromadb
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import plotly.express as px

# Works locally (.env) AND on Streamlit Cloud (secrets)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Page config (must be first Streamlit command) ---
st.set_page_config(
    page_title="IPL Cricket Analytics",
    page_icon="🏏",
    layout="wide"
)

# --- Load everything once and cache it ---
@st.cache_resource
def load_model():
    return genai.GenerativeModel('gemini-2.5-flash-lite')

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="data/embeddings")
    return client.get_collection("ipl_matches")

@st.cache_data
def load_dataframe():
    return pd.read_csv("data/processed/matches_cleaned.csv")

model    = load_model()
collection = load_collection()
df       = load_dataframe()

# --- Core functions ---
def ask_cricket_rag(question, n_results=5):
    results = collection.query(query_texts=[question], n_results=n_results)
    context = ""
    for i, doc in enumerate(results['documents'][0], 1):
        context += f"Match {i}:\n{doc}\n\n"

    prompt = f"""You are an expert IPL cricket analyst.
Use ONLY the match data below to answer the question accurately.
If the answer isn't in the data, say so honestly.

MATCH DATA:
{context}

QUESTION: {question}

Give a clear, detailed answer with specific stats and match details:"""

    response = model.generate_content(prompt)
    return response.text

def cricket_analytics(question):
    q = question.lower()

    if "player of the match" in q or "potm" in q or "most awards" in q:
        potm = df[df['player_of_match'] != 'Not Awarded']['player_of_match'].value_counts().head(5)
        return f"🏆 Most Player of the Match Awards:\n{potm.to_string()}"

    if "biggest win" in q or "largest margin" in q or "biggest margin" in q:
        biggest = df[df['result'] == 'runs'].nlargest(3, 'result_margin')[
            ['team1','team2','winner','result_margin','season','venue']
        ]
        return f"💥 Biggest Wins by Runs:\n{biggest.to_string(index=False)}"

    if "most wins" in q or "most successful" in q or "best team" in q:
        wins = df[df['winner'] != 'No Result']['winner'].value_counts().head(5)
        return f"🏏 Most Wins in IPL History:\n{wins.to_string()}"

    if ("vs" in q or "head to head" in q or "against" in q):
        teams = [t for t in df['team1'].unique() if t.lower() in q]
        if len(teams) >= 2:
            h2h = df[
                ((df['team1']==teams[0]) & (df['team2']==teams[1])) |
                ((df['team1']==teams[1]) & (df['team2']==teams[0]))
            ]
            wins = h2h['winner'].value_counts()
            return f"⚔️ Head to Head:\n{wins.to_string()}\nTotal matches: {len(h2h)}"

    return None

def smart_answer(question):
    analytics = cricket_analytics(question)
    if analytics:
        return analytics, "📊 Analytics"
    return ask_cricket_rag(question), "🤖 RAG + Gemini"

# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.title("🏏 IPL Analytics")
    st.markdown("---")

    page = st.radio("Navigate", [
        "💬 Ask Anything",
        "📊 Team Stats",
        "⚔️ Head to Head",
        "🏆 Records"
    ])

    st.markdown("---")
    st.markdown("**Try asking:**")
    example_questions = [
        "How did MI perform at Wankhede?",
        "Who won the most POTM awards?",
        "Biggest win margin in IPL history?",
        "Most successful team in IPL?"
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.question = q

# ─── PAGE: ASK ANYTHING ────────────────────────────────────
if page == "💬 Ask Anything":
    st.title("💬 Ask Anything About IPL")
    st.markdown("Powered by RAG + Gemini AI over 1095 real IPL matches")

    question = st.text_input(
        "Your question:",
        value=st.session_state.get("question", ""),
        placeholder="e.g. How many times did CSK win the toss and choose to bat?"
    )

    if st.button("🔍 Get Answer", type="primary") and question:
        with st.spinner("Searching 1095 matches..."):
            answer, source = smart_answer(question)

        st.markdown(f"*Source: {source}*")
        st.markdown("### Answer")
        st.markdown(answer)

# ─── PAGE: TEAM STATS ──────────────────────────────────────
elif page == "📊 Team Stats":
    st.title("📊 Team Statistics")

    team = st.selectbox("Select Team", sorted(df['team1'].unique()))

    team_matches = df[(df['team1'] == team) | (df['team2'] == team)]
    wins         = team_matches[team_matches['winner'] == team]
    losses       = team_matches[
        (team_matches['winner'] != team) &
        (team_matches['winner'] != 'No Result')
    ]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Matches", len(team_matches))
    col2.metric("Wins",          len(wins))
    col3.metric("Losses",        len(losses))
    col4.metric("Win Rate",      f"{len(wins)/len(team_matches)*100:.1f}%")

    st.markdown("---")

    # Wins by season chart
    wins_by_season = wins.groupby('season').size().reset_index(name='wins')
    fig = px.bar(
        wins_by_season, x='season', y='wins',
        title=f"{team} — Wins Per Season",
        color='wins', color_continuous_scale='Greens'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top POTM players
    st.markdown("### 🏆 Top Player of the Match Winners")
    potm = wins['player_of_match'].value_counts().head(8).reset_index()
    potm.columns = ['Player', 'Awards']
    st.dataframe(potm, use_container_width=True)

# ─── PAGE: HEAD TO HEAD ────────────────────────────────────
elif page == "⚔️ Head to Head":
    st.title("⚔️ Head to Head Comparison")

    teams = sorted(df['team1'].unique())
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams, index=0)
    with col2:
        team2 = st.selectbox("Team 2", teams, index=1)

    if st.button("Compare", type="primary"):
        h2h = df[
            ((df['team1']==team1) & (df['team2']==team2)) |
            ((df['team1']==team2) & (df['team2']==team1))
        ]

        if len(h2h) == 0:
            st.warning("No matches found between these teams.")
        else:
            t1_wins = len(h2h[h2h['winner'] == team1])
            t2_wins = len(h2h[h2h['winner'] == team2])

            col1, col2, col3 = st.columns(3)
            col1.metric(f"{team1} Wins", t1_wins)
            col2.metric("Total Matches", len(h2h))
            col3.metric(f"{team2} Wins", t2_wins)

            # Recent matches
            st.markdown("### Recent Matches")
            recent = h2h.sort_values('date', ascending=False).head(5)[
                ['date','venue','winner','player_of_match','result_margin','result']
            ]
            st.dataframe(recent, use_container_width=True)

# ─── PAGE: RECORDS ─────────────────────────────────────────
elif page == "🏆 Records":
    st.title("🏆 IPL All-Time Records")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Most Wins")
        wins_df = df[df['winner'] != 'No Result']['winner'].value_counts().head(8).reset_index()
        wins_df.columns = ['Team', 'Wins']
        fig = px.bar(wins_df, x='Wins', y='Team', orientation='h',
                     color='Wins', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Most POTM Awards")
        potm_df = df[df['player_of_match'] != 'Not Awarded']['player_of_match'].value_counts().head(8).reset_index()
        potm_df.columns = ['Player', 'Awards']
        fig2 = px.bar(potm_df, x='Awards', y='Player', orientation='h',
                      color='Awards', color_continuous_scale='Oranges')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 💥 Biggest Wins by Runs")
    biggest = df[df['result']=='runs'].nlargest(5,'result_margin')[
        ['season','team1','team2','winner','result_margin','venue']
    ]
    st.dataframe(biggest, use_container_width=True)

    st.markdown("### 🎯 Biggest Wins by Wickets")
    biggest_w = df[df['result']=='wickets'].nlargest(5,'result_margin')[
        ['season','team1','team2','winner','result_margin','venue']
    ]
    st.dataframe(biggest_w, use_container_width=True)