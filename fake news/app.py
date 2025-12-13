import streamlit as st
from openai_service import check_with_openai

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 Fake News Detector")
st.markdown("Enter a statement or news headline to verify its authenticity.")

# Input
statement = st.text_area(
    "Statement to check:",
    placeholder="Enter the news or statement you want to fact-check...",
    height=150
)

@st.cache_data(ttl=86400)
def cached_check(text: str):
    return check_with_openai(text)

# Check button
if st.button("Check Statement", type="primary", use_container_width=True):
    if not statement or len(statement.strip()) < 10:
        st.error("Please enter a statement with at least 10 characters.")
    else:
        with st.spinner("Analyzing statement... This may take a few seconds."):
            result = cached_check(statement)

            verdict = result.get("verdict", "UNSURE")
            confidence = result.get("confidence", "Low")
            explanation = result.get("explanation", "")
            sources = result.get("sources", [])

            verdict_colors = {
                "TRUE": "green",
                "FALSE": "red",
                "MISLEADING": "orange",
                "UNSURE": "gray"
            }

            verdict_color = verdict_colors.get(verdict, "gray")

            st.markdown("---")
            st.markdown(f"### Verdict: :{verdict_color}[{verdict}]")
            st.markdown(f"**Confidence:** {confidence}")

            st.markdown("### Explanation")
            st.write(explanation)

            if sources:
                st.markdown("### Sources")
                for source in sources:
                    st.markdown(f"- {source}")
            else:
                st.info("No sources available.")

# Sidebar info
with st.sidebar:
    st.markdown("### About")
    st.info(
        """
        This tool uses OpenAI to analyze statements.

        **Verdict Types:**
        - 🟢 TRUE
        - 🔴 FALSE
        - 🟠 MISLEADING
        - ⚫ UNSURE

        Results are cached for 24 hours per statement.
        """
    )
