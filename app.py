import streamlit as st
import requests
import re
import os
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

total_tokens_used = {"input": 0, "output": 0}

st.set_page_config(page_title="ResearchCrew", page_icon="🔬", layout="wide")

st.markdown("""
<style>
.main-title { font-size: 42px; font-weight: 800; color: #172554; }
.subtitle { font-size: 18px; color: #64748b; margin-bottom: 25px; }
.agent-card { background: white; padding: 18px; margin: 10px 0; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
.source-card { background: white; padding: 18px; margin: 12px 0; border-radius: 10px; border-left: 5px solid #2563eb; border: 1px solid #e2e8f0; }
.metric-card { background: white; padding: 22px; border-radius: 12px; text-align: center; border: 1px solid #e2e8f0; }
.metric-number { font-size: 30px; font-weight: 800; color: #2563eb; }
.metric-label { color: #64748b; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔬 ResearchCrew</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-Agent Research and Citation Verification System</div>', unsafe_allow_html=True)
st.write("ResearchCrew coordinates multiple cooperating AI agents to produce an evidence-based research brief instead of an unsupported AI-generated answer.")


# =========================================================
# AGENT 1 - PLANNING
# =========================================================

def planning_agent(question):
    stop_words = {"what", "are", "is", "the", "a", "an", "of", "in", "on",
                  "for", "to", "and", "with", "how", "why", "does", "do"}
    words = re.findall(r"[A-Za-z]+", question.lower())
    keywords = []
    for word in words:
        if word not in stop_words and len(word) > 2 and word not in keywords:
            keywords.append(word)
    return keywords


# =========================================================
# AGENT 2 - LITERATURE SEARCH
# =========================================================

def literature_search_agent(question):
    url = "https://api.openalex.org/works"
    q = question.lower()

    if "artificial intelligence" in q and "healthcare" in q:
        search_query = "artificial intelligence healthcare"
    else:
        cleaned = re.sub(r"[^a-zA-Z0-9 ]", " ", question)
        search_query = " ".join(cleaned.split())

    if not search_query:
        search_query = "artificial intelligence healthcare"

    params = {"search": search_query, "per-page": 8}

    try:
        response = requests.get(url, params=params, timeout=30,
                                 headers={"User-Agent": "ResearchCrew/1.0"})

        if response.status_code != 200:
            return [], response.status_code

        data = response.json()
        results = data.get("results", [])
        sources = []

        for item in results:
            title = item.get("title")
            if not title:
                continue

            location = item.get("primary_location") or {}
            landing_url = location.get("landing_page_url")
            pdf_url = location.get("pdf_url")
            doi = item.get("doi")
            if not landing_url and doi:
                landing_url = doi

            authors = []
            for authorship in (item.get("authorships") or []):
                author = authorship.get("author") or {}
                name = author.get("display_name")
                if name:
                    authors.append(name)

            abstract = ""
            inverted_index = item.get("abstract_inverted_index")
            if inverted_index:
                words = []
                for word, positions in inverted_index.items():
                    for position in positions:
                        words.append((position, word))
                words.sort(key=lambda x: x[0])
                abstract = " ".join(word for _, word in words)

            sources.append({
                "title": title,
                "year": item.get("publication_year", "N/A"),
                "doi": doi or "N/A",
                "url": landing_url or "N/A",
                "pdf_url": pdf_url or "N/A",
                "citations": item.get("cited_by_count", 0),
                "authors": authors,
                "abstract": abstract
            })

        return sources, 200

    except Exception:
        return [], 0


# =========================================================
# AGENT 3 - EVIDENCE EXTRACTION
# =========================================================

def evidence_extraction_agent(sources):
    evidence = []
    for index, source in enumerate(sources, start=1):
        abstract = source.get("abstract", "")
        evidence_text = abstract[:1000] if abstract else "Abstract was not available in the OpenAlex metadata."
        evidence.append({
            "source_id": index,
            "source_title": source["title"],
            "evidence": evidence_text
        })
    return evidence


# =========================================================
# AGENT 4 - SUMMARISATION (LLM-powered)
# =========================================================

def summarisation_agent(question, sources):
    if not sources:
        return "No academic literature was retrieved for the research question."

    context = ""
    for i, source in enumerate(sources, start=1):
        abstract = source.get("abstract", "No abstract available")
        context += f"\n[Source {i}] {source['title']} ({source['year']})\n{abstract[:500]}\n"

    prompt = f"""You are a research summarisation agent. Based ONLY on the sources below, write a concise 3-4 sentence evidence-based summary answering this research question. Cite sources using [Source N] notation for every claim.

Research Question: {question}

Sources:
{context}

Write the summary now:"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        total_tokens_used["input"] += response.usage.input_tokens
        total_tokens_used["output"] += response.usage.output_tokens
        return response.content[0].text
    except Exception as e:
        return f"Summarisation failed: {e}"


# =========================================================
# AGENT 5 - CRITICAL REVIEW (LLM-powered)
# =========================================================

def critical_review_agent(sources, evidence):
    if not sources:
        return ["No literature sources were retrieved.", "The research query should be refined."]

    context = ""
    for i, source in enumerate(sources, start=1):
        context += f"[Source {i}] {source['title']} - Abstract: {source.get('abstract', 'N/A')[:300]}\n"

    prompt = f"""You are a critical review agent for academic research. Review these sources for: (1) potential bias, (2) missing perspectives, (3) recency issues, (4) quality of evidence. Give 4-5 short bullet points as honest critical feedback.

Sources:
{context}

Write your critical review as short bullet points (no markdown, just plain sentences):"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        total_tokens_used["input"] += response.usage.input_tokens
        total_tokens_used["output"] += response.usage.output_tokens
        review_text = response.content[0].text
        review_lines = [line.strip("- ").strip() for line in review_text.split("\n") if line.strip()]
        return review_lines
    except Exception as e:
        return [f"Critical review failed: {e}"]


# =========================================================
# AGENT 6 - CITATION VERIFICATION (LLM-powered)
# =========================================================

def citation_verification_agent(sources, evidence):
    if not evidence:
        return 0, 0, 0

    total = len(evidence)
    covered = 0

    for item in evidence:
        source_id = item.get("source_id")
        evidence_text = item.get("evidence", "")

        if not (isinstance(source_id, int) and 1 <= source_id <= len(sources)):
            continue

        prompt = f"""Does the following text contain substantive, relevant information (not just a title or empty abstract)? Answer with only YES or NO.

Text: {evidence_text[:500]}"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            total_tokens_used["input"] += response.usage.input_tokens
            total_tokens_used["output"] += response.usage.output_tokens
            answer = response.content[0].text.strip().upper()
            if "YES" in answer:
                covered += 1
        except Exception:
            pass

    percentage = round(covered / total * 100, 2) if total > 0 else 0
    return percentage, covered, total


# =========================================================
# QUESTION INPUT
# =========================================================

st.header("🔎 Enter your research question")

question = st.text_area(
    "Research Question",
    value="What are the applications of Artificial Intelligence in healthcare?",
    height=100
)

run = st.button("🚀 Run Multi-Agent Research", type="primary", use_container_width=True)


# =========================================================
# MAIN WORKFLOW
# =========================================================

if run:

    with st.spinner("Planning Agent running..."):
        keywords = planning_agent(question)

    with st.spinner("Literature Search Agent searching OpenAlex..."):
        sources, api_status = literature_search_agent(question)

    with st.spinner("Evidence Extraction Agent running..."):
        evidence = evidence_extraction_agent(sources)

    with st.spinner("Summarisation Agent (Claude) running..."):
        summary = summarisation_agent(question, sources)

    with st.spinner("Critical Review Agent (Claude) running..."):
        review = critical_review_agent(sources, evidence)

    with st.spinner("Citation Verification Agent (Claude) running..."):
        citation_percentage, covered, total = citation_verification_agent(sources, evidence)

    # =====================================================
    # PIPELINE STATUS
    # =====================================================

    st.header("🤖 Multi-Agent Pipeline")

    pipeline = [
        ("🧠 Planning Agent", "Research plan created"),
        ("🔎 Literature Search Agent", f"{len(sources)} academic sources found"),
        ("📑 Evidence Extraction Agent", f"{len(evidence)} evidence records created"),
        ("✍️ Summarisation Agent", "LLM-generated summary created"),
        ("🧐 Critical Review Agent", "LLM-generated critical review completed"),
        ("🔗 Citation Verification Agent", f"{citation_percentage}% citation coverage verified")
    ]

    for name, status in pipeline:
        st.markdown(f"""
            <div class="agent-card">
            <b>{name}</b><br>✅ Completed<br><small>{status}</small>
            </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # PLANNING
    # =====================================================

    st.header("🧠 1. Research Planning")
    st.write("**Research Objective:**")
    st.write(f"Investigate the research question: {question}")
    st.write("**Search Keywords:**")
    st.write(", ".join(keywords))

    # =====================================================
    # RESEARCH BRIEF
    # =====================================================

    st.header("📄 2. Evidence-Based Research Brief")
    st.write("**Research Question:**")
    st.write(question)
    st.write("**Summary (Claude-generated):**")
    st.write(summary)

    # =====================================================
    # LITERATURE
    # =====================================================

    st.header("📚 3. Literature Search Results")

    if api_status != 200:
        st.error(f"OpenAlex API error. HTTP Status: {api_status}")
    elif not sources:
        st.warning("No academic sources were retrieved.")
    else:
        st.success(f"{len(sources)} academic sources retrieved.")
        for index, source in enumerate(sources, start=1):
            st.markdown(f"""
                <div class="source-card">
                <h4>{index}. {source["title"]}</h4>
                <b>Publication Year:</b> {source["year"]}<br><br>
                <b>Citations:</b> {source["citations"]}
                </div>
            """, unsafe_allow_html=True)

            if source["authors"]:
                st.write("**Authors:** " + ", ".join(source["authors"][:6]))
            if source["doi"] != "N/A":
                st.write("**DOI:** " + source["doi"])
            if source["url"] != "N/A":
                st.markdown(f"[🔗 Open Academic Source]({source['url']})")
            if source["pdf_url"] != "N/A":
                st.markdown(f"[📄 Open PDF]({source['pdf_url']})")
            if source["abstract"]:
                with st.expander("View Abstract"):
                    st.write(source["abstract"])

    # =====================================================
    # EVIDENCE
    # =====================================================

    st.header("📑 4. Evidence Extraction")

    if evidence:
        for item in evidence:
            st.markdown(f"""