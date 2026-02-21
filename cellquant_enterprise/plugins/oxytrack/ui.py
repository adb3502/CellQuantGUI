"""
Gradio UI components for the OxyTrack plugin.
"""

from typing import Dict, List, Optional, Tuple
import gradio as gr
import pandas as pd
from datetime import datetime

from cellquant_enterprise.plugins.oxytrack.models import (
    Experiment, OxysterolTreatment, SenescenceMarker,
)
from cellquant_enterprise.plugins.oxytrack.storage import OxyTrackStore

# Common oxysterol compounds for the dropdown
OXYSTEROL_COMPOUNDS = [
    "7-ketocholesterol",
    "7\u03b1-hydroxycholesterol",
    "7\u03b2-hydroxycholesterol",
    "25-hydroxycholesterol",
    "27-hydroxycholesterol",
    "24S-hydroxycholesterol",
    "cholesterol-5\u03b1,6\u03b1-epoxide",
    "cholestane-3\u03b2,5\u03b1,6\u03b2-triol",
    "Other",
]

SENESCENCE_MARKERS = [
    "SA-\u03b2-gal",
    "p21 (CDKN1A)",
    "p16 (CDKN2A)",
    "\u03b3H2AX",
    "Lamin B1",
    "IL-6",
    "IL-8",
    "Ki-67",
    "53BP1",
    "Other",
]

STATUS_CHOICES = ["draft", "in_progress", "analysed", "archived"]


def _experiments_to_dataframe(experiments: List[Experiment]) -> pd.DataFrame:
    """Convert a list of experiments into a summary DataFrame."""
    if not experiments:
        return pd.DataFrame(columns=["ID", "Name", "Cell Line", "Treatments", "Status", "Created"])
    rows = []
    for e in experiments:
        treatments_str = ", ".join(t.label() for t in e.treatments) if e.treatments else "\u2014"
        rows.append({
            "ID": e.id,
            "Name": e.name,
            "Cell Line": e.cell_line,
            "Treatments": treatments_str,
            "Status": e.status,
            "Created": e.date_created[:10] if e.date_created else "",
        })
    return pd.DataFrame(rows)


def create_oxytrack_tab(store: OxyTrackStore) -> Dict:
    """Build the OxyTrack Gradio tab contents."""

    # ── State ────────────────────────────────────────────────────────
    selected_id = gr.State(value="")

    # ── Left: Experiment List ────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Experiments</div>')

            with gr.Row():
                search_box = gr.Textbox(
                    label="Search",
                    placeholder="Filter experiments\u2026",
                    scale=3,
                )
                refresh_btn = gr.Button("Refresh", variant="secondary", scale=1)

            experiments_table = gr.Dataframe(
                headers=["ID", "Name", "Cell Line", "Treatments", "Status", "Created"],
                datatype=["str", "str", "str", "str", "str", "str"],
                value=_experiments_to_dataframe(store.list_experiments()),
                interactive=False,
            )

            with gr.Row():
                new_btn = gr.Button("New Experiment", variant="primary")
                delete_btn = gr.Button("Delete Selected", variant="secondary")

        # ── Right: Experiment Detail / Editor ────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<div class="section-header">Experiment Details</div>')

            with gr.Group():
                exp_name = gr.Textbox(label="Experiment Name", placeholder="e.g. 7KC Dose-Response in IMR90")
                with gr.Row():
                    cell_line = gr.Textbox(label="Cell Line", placeholder="e.g. IMR90, BJ, HCA2")
                    passage = gr.Textbox(label="Passage", placeholder="e.g. P12")
                    status = gr.Dropdown(label="Status", choices=STATUS_CHOICES, value="draft")
                exp_desc = gr.Textbox(label="Description", placeholder="Brief description of the experiment", lines=2)
                tags_input = gr.Textbox(label="Tags (comma-separated)", placeholder="e.g. dose-response, 7KC, IMR90")

            gr.HTML('<div class="section-header">Oxysterol Treatments</div>')

            with gr.Row():
                compound_select = gr.Dropdown(
                    label="Compound",
                    choices=OXYSTEROL_COMPOUNDS,
                    value="7-ketocholesterol",
                    scale=2,
                )
                conc_input = gr.Number(label="Conc. (\u00b5M)", value=10.0, scale=1)
                duration_input = gr.Number(label="Duration (h)", value=24.0, scale=1)
                add_treatment_btn = gr.Button("Add", variant="secondary", scale=1)

            treatments_table = gr.Dataframe(
                headers=["Compound", "Conc. (\u00b5M)", "Duration (h)", "Vehicle"],
                datatype=["str", "number", "number", "str"],
                value=[],
                interactive=False,
            )

            gr.HTML('<div class="section-header">Senescence Markers</div>')

            with gr.Row():
                marker_select = gr.Dropdown(
                    label="Marker",
                    choices=SENESCENCE_MARKERS,
                    value="SA-\u03b2-gal",
                    scale=2,
                )
                channel_input = gr.Textbox(label="Channel Suffix", value="C2", scale=1)
                add_marker_btn = gr.Button("Add", variant="secondary", scale=1)

            markers_table = gr.Dataframe(
                headers=["Marker", "Channel", "Type"],
                datatype=["str", "str", "str"],
                value=[],
                interactive=False,
            )

            gr.HTML('<div class="section-header">Data Link</div>')

            with gr.Row():
                data_folder = gr.Textbox(label="CellQuant Data Folder", placeholder="Path to image folder\u2026", scale=3)
                results_path = gr.Textbox(label="Results CSV", placeholder="Path to results CSV\u2026", scale=3)

            notes_input = gr.Textbox(label="Notes", placeholder="Free-form notes\u2026", lines=3)

            with gr.Row():
                save_btn = gr.Button("Save Experiment", variant="primary")
                save_status = gr.Textbox(label="", interactive=False, scale=2)

    # ── Handlers ─────────────────────────────────────────────────────

    # Transient state for treatments / markers being assembled
    treatments_state = gr.State(value=[])
    markers_state = gr.State(value=[])

    def _refresh(query: str):
        exps = store.search(query) if query.strip() else store.list_experiments()
        return _experiments_to_dataframe(exps)

    def _new_experiment():
        exp = Experiment(name="New Experiment")
        return (
            exp.id,          # selected_id
            exp.name,
            "",              # cell_line
            "",              # passage
            "draft",
            "",              # desc
            "",              # tags
            [],              # treatments_state
            pd.DataFrame(columns=["Compound", "Conc. (\u00b5M)", "Duration (h)", "Vehicle"]),
            [],              # markers_state
            pd.DataFrame(columns=["Marker", "Channel", "Type"]),
            "",              # data_folder
            "",              # results_path
            "",              # notes
            "",              # save_status
        )

    def _select_experiment(table: pd.DataFrame, evt: gr.SelectData):
        if table is None or table.empty:
            return [gr.update()] * 15
        row_idx = evt.index[0]
        exp_id = str(table.iloc[row_idx]["ID"])
        exp = store.get(exp_id)
        if exp is None:
            return [gr.update()] * 15

        t_rows = [[t.compound, t.concentration_um, t.duration_hours, t.vehicle] for t in exp.treatments]
        t_df = pd.DataFrame(t_rows, columns=["Compound", "Conc. (\u00b5M)", "Duration (h)", "Vehicle"]) if t_rows else pd.DataFrame(columns=["Compound", "Conc. (\u00b5M)", "Duration (h)", "Vehicle"])

        m_rows = [[m.name, m.channel_suffix, m.marker_type] for m in exp.markers]
        m_df = pd.DataFrame(m_rows, columns=["Marker", "Channel", "Type"]) if m_rows else pd.DataFrame(columns=["Marker", "Channel", "Type"])

        t_state = [t.__dict__ for t in exp.treatments]
        m_state = [m.__dict__ for m in exp.markers]

        return (
            exp.id,
            exp.name,
            exp.cell_line,
            exp.passage,
            exp.status,
            exp.description,
            ", ".join(exp.tags),
            t_state,
            t_df,
            m_state,
            m_df,
            exp.data_folder,
            exp.results_path,
            exp.notes,
            "",
        )

    def _add_treatment(compound, conc, duration, current_treatments):
        t = {"compound": compound, "concentration_um": conc, "duration_hours": duration, "vehicle": "ethanol", "vehicle_pct": 0.1}
        current_treatments = list(current_treatments) + [t]
        rows = [[d["compound"], d["concentration_um"], d["duration_hours"], d["vehicle"]] for d in current_treatments]
        df = pd.DataFrame(rows, columns=["Compound", "Conc. (\u00b5M)", "Duration (h)", "Vehicle"])
        return current_treatments, df

    def _add_marker(marker_name, channel, current_markers):
        m = {"name": marker_name, "channel_suffix": channel, "marker_type": "fluorescence"}
        current_markers = list(current_markers) + [m]
        rows = [[d["name"], d["channel_suffix"], d["marker_type"]] for d in current_markers]
        df = pd.DataFrame(rows, columns=["Marker", "Channel", "Type"])
        return current_markers, df

    def _save(
        sel_id, name, cl, pas, stat, desc, tags_str,
        treatments_list, markers_list, df, rp, notes
    ):
        treatments = [OxysterolTreatment(**t) for t in treatments_list]
        markers = [SenescenceMarker(**m) for m in markers_list]
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]

        existing = store.get(sel_id)
        exp = existing if existing else Experiment(id=sel_id)
        exp.name = name
        exp.cell_line = cl
        exp.passage = pas
        exp.status = stat
        exp.description = desc
        exp.tags = tags
        exp.treatments = treatments
        exp.markers = markers
        exp.data_folder = df
        exp.results_path = rp
        exp.notes = notes

        if existing:
            store.update(exp)
        else:
            store.add(exp)

        table = _experiments_to_dataframe(store.list_experiments())
        return table, f"Saved \u2014 {exp.name} ({exp.id})"

    def _delete(sel_id):
        if sel_id and store.delete(sel_id):
            return _experiments_to_dataframe(store.list_experiments()), f"Deleted {sel_id}"
        return _experiments_to_dataframe(store.list_experiments()), "Nothing to delete"

    # Wire events
    refresh_btn.click(fn=_refresh, inputs=[search_box], outputs=[experiments_table])
    search_box.change(fn=_refresh, inputs=[search_box], outputs=[experiments_table])

    new_btn.click(
        fn=_new_experiment,
        outputs=[
            selected_id, exp_name, cell_line, passage, status, exp_desc, tags_input,
            treatments_state, treatments_table, markers_state, markers_table,
            data_folder, results_path, notes_input, save_status,
        ],
    )

    experiments_table.select(
        fn=_select_experiment,
        inputs=[experiments_table],
        outputs=[
            selected_id, exp_name, cell_line, passage, status, exp_desc, tags_input,
            treatments_state, treatments_table, markers_state, markers_table,
            data_folder, results_path, notes_input, save_status,
        ],
    )

    add_treatment_btn.click(
        fn=_add_treatment,
        inputs=[compound_select, conc_input, duration_input, treatments_state],
        outputs=[treatments_state, treatments_table],
    )

    add_marker_btn.click(
        fn=_add_marker,
        inputs=[marker_select, channel_input, markers_state],
        outputs=[markers_state, markers_table],
    )

    save_btn.click(
        fn=_save,
        inputs=[
            selected_id, exp_name, cell_line, passage, status, exp_desc, tags_input,
            treatments_state, markers_state, data_folder, results_path, notes_input,
        ],
        outputs=[experiments_table, save_status],
    )

    delete_btn.click(
        fn=_delete,
        inputs=[selected_id],
        outputs=[experiments_table, save_status],
    )

    return {
        "experiments_table": experiments_table,
        "selected_id": selected_id,
        "save_btn": save_btn,
    }
