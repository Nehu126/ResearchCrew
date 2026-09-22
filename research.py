import requests


OPENALEX_URL = "https://api.openalex.org/works"


# ============================================================
# 1. PLANNING AGENT
# ============================================================

def planning_agent(question):
    question = question.strip()

    keywords = question.lower()

    return {
        "question": question,
        "keywords": keywords,
        "plan": (
            "1. Identify the main research topic. "
            "2. Search academic literature. "
            "3. Extract evidence from research abstracts. "
            "4. Summarise the collected evidence. "
            "5. Perform a critical review. "
            "6. Verify citation metadata."
        )
    }


# ============================================================
# 2. OPENALEX LITERATURE SEARCH AGENT
# ============================================================

def search_openalex(question, max_results=5):

    # Improve search for common AI + healthcare query
    q = question.strip()

    lower_q = q.lower()

    if "artificial intelligence" in lower_q and "healthcare" in lower_q:
        q = "artificial intelligence healthcare"

    params = {
        "search": q,
        "per-page": max_results,
        "mailto": "researchcrew@example.com"
    }

    try:

        response = requests.get(
            OPENALEX_URL,
            params=params,
            timeout=30
        )

        print("OpenAlex HTTP Status:", response.status_code)

        response.raise_for_status()

        data = response.json()

        records = data.get("results", [])

        print("OpenAlex Records Found:", len(records))

        sources = []

        for record in records:

            title = record.get(
                "display_name",
                "Untitled"
            )

            publication_year = record.get(
                "publication_year",
                "Unknown"
            )

            authors_list = []

            for authorship in record.get(
                "authorships",
                []
            ):

                author = authorship.get(
                    "author",
                    {}
                )

                author_name = author.get(
                    "display_name"
                )

                if author_name:
                    authors_list.append(
                        author_name
                    )

            authors = ", ".join(
                authors_list
            )

            primary_location = record.get(
                "primary_location"
            ) or {}

            source_info = primary_location.get(
                "source"
            ) or {}

            journal = source_info.get(
                "display_name",
                "Unknown"
            )

            doi = record.get(
                "doi"
            )

            landing_page = primary_location.get(
                "landing_page_url"
            )

            pdf_url = None

            pdf_info = primary_location.get(
                "pdf"
            ) or {}

            if pdf_info:
                pdf_url = pdf_info.get(
                    "url"
                )

            url = (
                pdf_url
                or landing_page
                or doi
            )

            abstract = extract_abstract(
                record.get(
                    "abstract_inverted_index"
                )
            )

            cited_by_count = record.get(
                "cited_by_count",
                0
            )

            source = {
                "title": title,
                "authors": authors
                if authors
                else "Unknown",
                "year": publication_year,
                "journal": journal,
                "doi": doi
                if doi
                else "Not available",
                "url": url
                if url
                else "Not available",
                "abstract": abstract,
                "cited_by_count": cited_by_count
            }

            sources.append(source)

        return sources

    except requests.exceptions.RequestException as error:

        print(
            "OpenAlex request failed:",
            error
        )

        return []


# ============================================================
# 3. ABSTRACT EXTRACTION
# ============================================================

def extract_abstract(inverted_index):

    if not inverted_index:
        return ""

    words = []

    try:

        for word, positions in inverted_index.items():

            for position in positions:

                words.append(
                    (
                        position,
                        word
                    )
                )

        words.sort(
            key=lambda x: x[0]
        )

        abstract = " ".join(
            word
            for _, word in words
        )

        return abstract.strip()

    except Exception:

        return ""


# ============================================================
# 4. EVIDENCE EXTRACTION AGENT
# ============================================================

def extract_evidence(sources):

    evidence_items = []

    for source in sources:

        title = source.get(
            "title",
            "Untitled"
        )

        abstract = source.get(
            "abstract",
            ""
        )

        if abstract:

            evidence = abstract[:1200]

        else:

            evidence = (
                "Evidence extracted from "
                "academic metadata. "
                f"The study is titled '{title}' "
                f"and was published in "
                f"{source.get('year', 'an unknown year')}."
            )

        evidence_items.append(
            {
                "source": title,
                "evidence": evidence
            }
        )

    return evidence_items


# ============================================================
# 5. SUMMARISATION AGENT
# ============================================================

def summarise_research(
    question,
    sources
):

    if not sources:

        return (
            "No academic sources were found "
            "for the research question."
        )

    summary_parts = []

    summary_parts.append(
        f"Research question: {question}"
    )

    summary_parts.append(
        f"\nResearchCrew retrieved "
        f"{len(sources)} academic sources "
        f"from OpenAlex."
    )

    summary_parts.append(
        "\nThe retrieved literature includes "
        "research studies related to the "
        "given research topic."
    )

    years = []

    for source in sources:

        year = source.get(
            "year"
        )

        if isinstance(
            year,
            int
        ):

            years.append(year)

    if years:

        summary_parts.append(
            f"\nPublication years range from "
            f"{min(years)} to {max(years)}."
        )

    summary_parts.append(
        "\nThe evidence extraction stage "
        "uses available research abstracts "
        "to identify supporting information."
    )

    return "\n".join(
        summary_parts
    )


# ============================================================
# 6. CRITICAL REVIEW AGENT
# ============================================================

def critical_review(
    sources,
    evidence_items
):

    if not sources:

        return (
            "Critical review could not be "
            "performed because no sources "
            "were retrieved."
        )

    review = []

    review.append(
        f"ResearchCrew retrieved {len(sources)} "
        "academic sources."
    )

    if evidence_items:

        review.append(
            f"Evidence was extracted from "
            f"{len(evidence_items)} source(s)."
        )

    journals = set()

    for source in sources:

        journal = source.get(
            "journal"
        )

        if journal and journal != "Unknown":

            journals.add(journal)

    if journals:

        review.append(
            f"The sources cover "
            f"{len(journals)} distinct "
            "publication venue(s)."
        )

    review.append(
        "A limitation is that the system "
        "primarily uses bibliographic metadata "
        "and abstracts rather than complete "
        "full-text papers."
    )

    review.append(
        "Therefore, the generated research "
        "brief should be manually reviewed "
        "before academic publication."
    )

    return "\n\n".join(
        review
    )


# ============================================================
# 7. CITATION VERIFICATION AGENT
# ============================================================

def verify_citations(sources):

    citation_results = []

    for source in sources:

        title = source.get(
            "title",
            "Untitled"
        )

        doi = source.get(
            "doi"
        )

        url = source.get(
            "url"
        )

        valid = bool(
            title
            and title != "Untitled"
            and (
                doi
                or (
                    url
                    and url != "Not available"
                )
            )
        )

        citation_results.append(
            {
                "source": title,
                "valid": valid
            }
        )

    return citation_results


# ============================================================
# 8. MAIN MULTI-AGENT WORKFLOW
# ============================================================

def run_research(
    question,
    max_results=5
):

    # -------------------------
    # Planning Agent
    # -------------------------

    planning = planning_agent(
        question
    )

    # -------------------------
    # Literature Search Agent
    # -------------------------

    sources = search_openalex(
        question,
        max_results
    )

    # -------------------------
    # Evidence Extraction Agent
    # -------------------------

    evidence_items = extract_evidence(
        sources
    )

    # -------------------------
    # Summarisation Agent
    # -------------------------

    summary_text = summarise_research(
        question,
        sources
    )

    # -------------------------
    # Critical Review Agent
    # -------------------------

    review_text = critical_review(
        sources,
        evidence_items
    )

    # -------------------------
    # Citation Verification Agent
    # -------------------------

    citation_results = verify_citations(
        sources
    )

    # -------------------------
    # Final Result
    # -------------------------

    return {

        "planning": True,

        "literature": bool(
            sources
        ),

        "evidence": bool(
            evidence_items
        ),

        "summary": bool(
            summary_text
        ),

        "review": bool(
            review_text
        ),

        "citation": bool(
            citation_results
        ),

        "plan_text": planning["plan"],

        "sources": sources,

        "evidence_items": evidence_items,

        "summary_text": summary_text,

        "review_text": review_text,

        "citation_results": citation_results
    }