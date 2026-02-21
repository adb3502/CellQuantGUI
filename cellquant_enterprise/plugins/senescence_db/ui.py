"""
Gradio UI components for the SenescenceDB plugin.

Three sub-tabs: Papers, Protocols, Findings.
"""

from typing import Dict, List
import gradio as gr
import pandas as pd

from cellquant_enterprise.plugins.senescence_db.models import Paper, Protocol, Finding
from cellquant_enterprise.plugins.senescence_db.storage import SenescenceDBStore

PROTOCOL_CATEGORIES = ["staining", "treatment", "imaging", "analysis", "cell culture", "other"]
FINDING_CATEGORIES = ["mechanism", "biomarker", "pathway", "phenotype", "therapeutic", "other"]
CONFIDENCE_LEVELS = ["low", "medium", "high", "confirmed"]


# ── DataFrame helpers ────────────────────────────────────────────────

def _papers_df(papers: List[Paper]) -> pd.DataFrame:
    if not papers:
        return pd.DataFrame(columns=["ID", "Title", "Authors", "Year", "Journal", "Tags"])
    return pd.DataFrame([{
        "ID": p.id,
        "Title": p.title,
        "Authors": p.authors[:40] + ("\u2026" if len(p.authors) > 40 else ""),
        "Year": p.year if p.year else "",
        "Journal": p.journal,
        "Tags": ", ".join(p.tags),
    } for p in papers])


def _protocols_df(protocols: List[Protocol]) -> pd.DataFrame:
    if not protocols:
        return pd.DataFrame(columns=["ID", "Title", "Category", "Tags"])
    return pd.DataFrame([{
        "ID": p.id,
        "Title": p.title,
        "Category": p.category,
        "Tags": ", ".join(p.tags),
    } for p in protocols])


def _findings_df(findings: List[Finding]) -> pd.DataFrame:
    if not findings:
        return pd.DataFrame(columns=["ID", "Title", "Category", "Confidence", "Tags"])
    return pd.DataFrame([{
        "ID": f.id,
        "Title": f.title,
        "Category": f.category,
        "Confidence": f.confidence,
        "Tags": ", ".join(f.tags),
    } for f in findings])


def _stats_html(store: SenescenceDBStore) -> str:
    s = store.stats()
    return f"""
    <div class="stats-grid" style="grid-template-columns: repeat(4, 1fr);">
        <div class="stat-card">
            <div class="stat-value">{s['papers']}</div>
            <div class="stat-label">Papers</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{s['protocols']}</div>
            <div class="stat-label">Protocols</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{s['findings']}</div>
            <div class="stat-label">Findings</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{s['tags']}</div>
            <div class="stat-label">Unique Tags</div>
        </div>
    </div>
    """


# ── Main tab builder ────────────────────────────────────────────────

def create_senescence_db_tab(store: SenescenceDBStore) -> Dict:
    """Build the SenescenceDB Gradio tab contents."""

    # Global search bar + stats
    gr.HTML('<div class="section-header">Knowledge Base</div>')
    stats_display = gr.HTML(value=_stats_html(store))

    with gr.Row():
        global_search = gr.Textbox(label="Search everything", placeholder="Search papers, protocols, findings\u2026", scale=4)
        search_btn = gr.Button("Search", variant="secondary", scale=1)

    # Sub-tabs
    with gr.Tabs():
        # ═══════════════════════════════════════════════════════════
        # Papers sub-tab
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("Papers"):
            papers_selected_id = gr.State(value="")

            papers_table = gr.Dataframe(
                value=_papers_df(store.list_papers()),
                interactive=False,
            )

            with gr.Row():
                new_paper_btn = gr.Button("New Paper", variant="primary")
                delete_paper_btn = gr.Button("Delete Selected", variant="secondary")

            with gr.Group():
                paper_title = gr.Textbox(label="Title")
                with gr.Row():
                    paper_authors = gr.Textbox(label="Authors", scale=3)
                    paper_year = gr.Number(label="Year", value=2024, precision=0, scale=1)
                with gr.Row():
                    paper_journal = gr.Textbox(label="Journal", scale=2)
                    paper_doi = gr.Textbox(label="DOI", scale=2)
                    paper_pmid = gr.Textbox(label="PMID", scale=1)
                paper_abstract = gr.Textbox(label="Abstract", lines=3)
                paper_tags = gr.Textbox(label="Tags (comma-separated)")
                paper_notes = gr.Textbox(label="Notes", lines=2)

            with gr.Row():
                save_paper_btn = gr.Button("Save Paper", variant="primary")
                paper_status = gr.Textbox(label="", interactive=False, scale=2)

        # ═══════════════════════════════════════════════════════════
        # Protocols sub-tab
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("Protocols"):
            protocols_selected_id = gr.State(value="")

            protocols_table = gr.Dataframe(
                value=_protocols_df(store.list_protocols()),
                interactive=False,
            )

            with gr.Row():
                new_protocol_btn = gr.Button("New Protocol", variant="primary")
                delete_protocol_btn = gr.Button("Delete Selected", variant="secondary")

            with gr.Group():
                protocol_title = gr.Textbox(label="Title")
                protocol_category = gr.Dropdown(label="Category", choices=PROTOCOL_CATEGORIES, value="staining")
                protocol_steps = gr.Textbox(label="Steps (Markdown)", lines=6, placeholder="1. Step one\n2. Step two\n...")
                protocol_reagents = gr.Textbox(label="Reagents", lines=2)
                protocol_tips = gr.Textbox(label="Tips & Troubleshooting", lines=2)
                protocol_tags = gr.Textbox(label="Tags (comma-separated)")

            with gr.Row():
                save_protocol_btn = gr.Button("Save Protocol", variant="primary")
                protocol_status = gr.Textbox(label="", interactive=False, scale=2)

        # ═══════════════════════════════════════════════════════════
        # Findings sub-tab
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("Findings"):
            findings_selected_id = gr.State(value="")

            findings_table = gr.Dataframe(
                value=_findings_df(store.list_findings()),
                interactive=False,
            )

            with gr.Row():
                new_finding_btn = gr.Button("New Finding", variant="primary")
                delete_finding_btn = gr.Button("Delete Selected", variant="secondary")

            with gr.Group():
                finding_title = gr.Textbox(label="Title")
                finding_summary = gr.Textbox(label="Summary", lines=2)
                finding_details = gr.Textbox(label="Details (Markdown)", lines=4)
                with gr.Row():
                    finding_category = gr.Dropdown(label="Category", choices=FINDING_CATEGORIES, value="mechanism")
                    finding_confidence = gr.Dropdown(label="Confidence", choices=CONFIDENCE_LEVELS, value="medium")
                finding_tags = gr.Textbox(label="Tags (comma-separated)")

            with gr.Row():
                save_finding_btn = gr.Button("Save Finding", variant="primary")
                finding_status = gr.Textbox(label="", interactive=False, scale=2)

    # ── Handlers ─────────────────────────────────────────────────────

    # -- Global search --
    def _global_search(query):
        if not query.strip():
            return (
                _papers_df(store.list_papers()),
                _protocols_df(store.list_protocols()),
                _findings_df(store.list_findings()),
                _stats_html(store),
            )
        results = store.search(query)
        return (
            _papers_df(results["papers"]),
            _protocols_df(results["protocols"]),
            _findings_df(results["findings"]),
            _stats_html(store),
        )

    search_btn.click(
        fn=_global_search,
        inputs=[global_search],
        outputs=[papers_table, protocols_table, findings_table, stats_display],
    )
    global_search.submit(
        fn=_global_search,
        inputs=[global_search],
        outputs=[papers_table, protocols_table, findings_table, stats_display],
    )

    # -- Papers --
    def _new_paper():
        p = Paper()
        return p.id, "", "", 2024, "", "", "", "", "", "", ""

    def _select_paper(table, evt: gr.SelectData):
        if table is None or table.empty:
            return [gr.update()] * 11
        pid = str(table.iloc[evt.index[0]]["ID"])
        p = store.get_paper(pid)
        if p is None:
            return [gr.update()] * 11
        return (
            p.id, p.title, p.authors, p.year, p.journal,
            p.doi, p.pmid, p.abstract, ", ".join(p.tags), p.notes, "",
        )

    def _save_paper(sel_id, title, authors, year, journal, doi, pmid, abstract, tags_str, notes):
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        existing = store.get_paper(sel_id)
        p = existing if existing else Paper(id=sel_id)
        p.title = title
        p.authors = authors
        p.year = int(year) if year else 0
        p.journal = journal
        p.doi = doi
        p.pmid = pmid
        p.abstract = abstract
        p.tags = tags
        p.notes = notes
        if existing:
            store.update_paper(p)
        else:
            store.add_paper(p)
        return _papers_df(store.list_papers()), f"Saved \u2014 {p.title}", _stats_html(store)

    def _delete_paper(sel_id):
        store.delete_paper(sel_id)
        return _papers_df(store.list_papers()), "Deleted", _stats_html(store)

    new_paper_btn.click(fn=_new_paper, outputs=[
        papers_selected_id, paper_title, paper_authors, paper_year, paper_journal,
        paper_doi, paper_pmid, paper_abstract, paper_tags, paper_notes, paper_status,
    ])
    papers_table.select(fn=_select_paper, inputs=[papers_table], outputs=[
        papers_selected_id, paper_title, paper_authors, paper_year, paper_journal,
        paper_doi, paper_pmid, paper_abstract, paper_tags, paper_notes, paper_status,
    ])
    save_paper_btn.click(fn=_save_paper, inputs=[
        papers_selected_id, paper_title, paper_authors, paper_year, paper_journal,
        paper_doi, paper_pmid, paper_abstract, paper_tags, paper_notes,
    ], outputs=[papers_table, paper_status, stats_display])
    delete_paper_btn.click(fn=_delete_paper, inputs=[papers_selected_id],
                           outputs=[papers_table, paper_status, stats_display])

    # -- Protocols --
    def _new_protocol():
        p = Protocol()
        return p.id, "", "staining", "", "", "", "", ""

    def _select_protocol(table, evt: gr.SelectData):
        if table is None or table.empty:
            return [gr.update()] * 8
        pid = str(table.iloc[evt.index[0]]["ID"])
        p = store.get_protocol(pid)
        if p is None:
            return [gr.update()] * 8
        return p.id, p.title, p.category, p.steps, p.reagents, p.tips, ", ".join(p.tags), ""

    def _save_protocol(sel_id, title, category, steps, reagents, tips, tags_str):
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        existing = store.get_protocol(sel_id)
        p = existing if existing else Protocol(id=sel_id)
        p.title = title
        p.category = category
        p.steps = steps
        p.reagents = reagents
        p.tips = tips
        p.tags = tags
        if existing:
            store.update_protocol(p)
        else:
            store.add_protocol(p)
        return _protocols_df(store.list_protocols()), f"Saved \u2014 {p.title}", _stats_html(store)

    def _delete_protocol(sel_id):
        store.delete_protocol(sel_id)
        return _protocols_df(store.list_protocols()), "Deleted", _stats_html(store)

    new_protocol_btn.click(fn=_new_protocol, outputs=[
        protocols_selected_id, protocol_title, protocol_category,
        protocol_steps, protocol_reagents, protocol_tips, protocol_tags, protocol_status,
    ])
    protocols_table.select(fn=_select_protocol, inputs=[protocols_table], outputs=[
        protocols_selected_id, protocol_title, protocol_category,
        protocol_steps, protocol_reagents, protocol_tips, protocol_tags, protocol_status,
    ])
    save_protocol_btn.click(fn=_save_protocol, inputs=[
        protocols_selected_id, protocol_title, protocol_category,
        protocol_steps, protocol_reagents, protocol_tips, protocol_tags,
    ], outputs=[protocols_table, protocol_status, stats_display])
    delete_protocol_btn.click(fn=_delete_protocol, inputs=[protocols_selected_id],
                              outputs=[protocols_table, protocol_status, stats_display])

    # -- Findings --
    def _new_finding():
        f = Finding()
        return f.id, "", "", "", "mechanism", "medium", "", ""

    def _select_finding(table, evt: gr.SelectData):
        if table is None or table.empty:
            return [gr.update()] * 8
        fid = str(table.iloc[evt.index[0]]["ID"])
        f = store.get_finding(fid)
        if f is None:
            return [gr.update()] * 8
        return f.id, f.title, f.summary, f.details, f.category, f.confidence, ", ".join(f.tags), ""

    def _save_finding(sel_id, title, summary, details, category, confidence, tags_str):
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        existing = store.get_finding(sel_id)
        f = existing if existing else Finding(id=sel_id)
        f.title = title
        f.summary = summary
        f.details = details
        f.category = category
        f.confidence = confidence
        f.tags = tags
        if existing:
            store.update_finding(f)
        else:
            store.add_finding(f)
        return _findings_df(store.list_findings()), f"Saved \u2014 {f.title}", _stats_html(store)

    def _delete_finding(sel_id):
        store.delete_finding(sel_id)
        return _findings_df(store.list_findings()), "Deleted", _stats_html(store)

    new_finding_btn.click(fn=_new_finding, outputs=[
        findings_selected_id, finding_title, finding_summary,
        finding_details, finding_category, finding_confidence, finding_tags, finding_status,
    ])
    findings_table.select(fn=_select_finding, inputs=[findings_table], outputs=[
        findings_selected_id, finding_title, finding_summary,
        finding_details, finding_category, finding_confidence, finding_tags, finding_status,
    ])
    save_finding_btn.click(fn=_save_finding, inputs=[
        findings_selected_id, finding_title, finding_summary,
        finding_details, finding_category, finding_confidence, finding_tags,
    ], outputs=[findings_table, finding_status, stats_display])
    delete_finding_btn.click(fn=_delete_finding, inputs=[findings_selected_id],
                             outputs=[findings_table, finding_status, stats_display])

    return {
        "papers_table": papers_table,
        "protocols_table": protocols_table,
        "findings_table": findings_table,
        "stats_display": stats_display,
    }
