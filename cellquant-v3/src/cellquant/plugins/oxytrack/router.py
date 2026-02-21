"""FastAPI router for OxyTrack endpoints."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import List

from fastapi import APIRouter, HTTPException

from cellquant.plugins.oxytrack.schemas import (
    DoseResponsePoint,
    DoseResponseResponse,
    DoseResponseSeries,
    ExperimentCreate,
    ExperimentDetail,
    ExperimentSummary,
    ExperimentUpdate,
    ImportFromSessionRequest,
    MarkerPanelSchema,
    ObservationCreate,
    ObservationResponse,
    TimeCourseSeries,
    TimeCourseResponse,
    TimePointSchema,
    TreatmentArmSchema,
)
from cellquant.plugins.oxytrack.store import OxyTrackStore

# Initialised by the plugin's ``activate()``
_store: OxyTrackStore | None = None


def set_store(store: OxyTrackStore) -> None:
    global _store
    _store = store


def _get_store() -> OxyTrackStore:
    if _store is None:
        raise RuntimeError("OxyTrack store not initialised")
    return _store


def build_router() -> APIRouter:
    router = APIRouter()

    # -- experiment CRUD -----------------------------------------------------

    @router.get("/experiments", response_model=List[ExperimentSummary])
    async def list_experiments():
        store = _get_store()
        return [
            ExperimentSummary(
                id=e.id,
                name=e.name,
                cell_line=e.cell_line,
                n_treatments=len(e.treatments),
                n_timepoints=len(e.timepoints),
                n_observations=len(e.observations),
                created_at=e.created_at,
                tags=e.tags,
            )
            for e in store.list_experiments()
        ]

    @router.post("/experiments", response_model=ExperimentDetail, status_code=201)
    async def create_experiment(req: ExperimentCreate):
        store = _get_store()
        exp = store.create_experiment(
            name=req.name,
            cell_line=req.cell_line,
            passage=req.passage,
            description=req.description,
            treatments=[t.model_dump() for t in req.treatments],
            timepoints=[tp.model_dump() for tp in req.timepoints],
            marker_panel=req.marker_panel.model_dump(),
            tags=req.tags,
        )
        return _experiment_to_detail(exp)

    @router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
    async def get_experiment(experiment_id: str):
        store = _get_store()
        exp = store.get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(404, "Experiment not found")
        return _experiment_to_detail(exp)

    @router.patch("/experiments/{experiment_id}", response_model=ExperimentDetail)
    async def update_experiment(experiment_id: str, req: ExperimentUpdate):
        store = _get_store()
        updates = req.model_dump(exclude_unset=True)
        if "treatments" in updates and updates["treatments"] is not None:
            updates["treatments"] = [t.model_dump() for t in req.treatments]
        if "timepoints" in updates and updates["timepoints"] is not None:
            updates["timepoints"] = [tp.model_dump() for tp in req.timepoints]
        if "marker_panel" in updates and updates["marker_panel"] is not None:
            updates["marker_panel"] = req.marker_panel.model_dump()

        exp = store.update_experiment(experiment_id, updates)
        if exp is None:
            raise HTTPException(404, "Experiment not found")
        return _experiment_to_detail(exp)

    @router.delete("/experiments/{experiment_id}")
    async def delete_experiment(experiment_id: str):
        store = _get_store()
        if not store.delete_experiment(experiment_id):
            raise HTTPException(404, "Experiment not found")
        return {"status": "deleted"}

    # -- observations --------------------------------------------------------

    @router.post(
        "/experiments/{experiment_id}/observations",
        response_model=ObservationResponse,
        status_code=201,
    )
    async def add_observation(experiment_id: str, req: ObservationCreate):
        store = _get_store()
        obs = store.add_observation(experiment_id, req.model_dump())
        if obs is None:
            raise HTTPException(404, "Experiment not found")
        return ObservationResponse(
            id=obs.id,
            experiment_id=obs.experiment_id,
            treatment_id=obs.treatment_id,
            timepoint_hours=obs.timepoint_hours,
            marker=obs.marker,
            value=obs.value,
            unit=obs.unit,
            n_cells=obs.n_cells,
            std_dev=obs.std_dev,
            cellquant_session_id=obs.cellquant_session_id,
            notes=obs.notes,
            recorded_at=obs.recorded_at,
        )

    @router.delete("/experiments/{experiment_id}/observations/{observation_id}")
    async def delete_observation(experiment_id: str, observation_id: str):
        store = _get_store()
        if not store.delete_observation(experiment_id, observation_id):
            raise HTTPException(404, "Observation not found")
        return {"status": "deleted"}

    # -- CellQuant import ----------------------------------------------------

    @router.post(
        "/experiments/{experiment_id}/import",
        response_model=List[ObservationResponse],
    )
    async def import_from_session(experiment_id: str, req: ImportFromSessionRequest):
        """Import quantification results from a CellQuant session."""
        store = _get_store()
        exp = store.get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(404, "Experiment not found")

        from cellquant.api.dependencies import get_session

        session = get_session(req.cellquant_session_id)
        if session.results_df is None:
            session.load_results()
        if session.results_df is None or len(session.results_df) == 0:
            raise HTTPException(400, "CellQuant session has no results")

        df = session.results_df
        ctcf_cols = [c for c in df.columns if c.endswith("_CTCF")]
        created: list[ObservationResponse] = []

        for col in ctcf_cols:
            marker = col.replace("_CTCF", "")
            mean_val = float(df[col].mean())
            std_val = float(df[col].std())
            n = len(df)

            obs = store.add_observation(experiment_id, {
                "treatment_id": req.treatment_id,
                "timepoint_hours": req.timepoint_hours,
                "marker": marker,
                "value": mean_val,
                "unit": "CTCF",
                "n_cells": n,
                "std_dev": std_val,
                "cellquant_session_id": req.cellquant_session_id,
            })
            if obs:
                created.append(ObservationResponse(
                    id=obs.id,
                    experiment_id=obs.experiment_id,
                    treatment_id=obs.treatment_id,
                    timepoint_hours=obs.timepoint_hours,
                    marker=obs.marker,
                    value=obs.value,
                    unit=obs.unit,
                    n_cells=obs.n_cells,
                    std_dev=obs.std_dev,
                    cellquant_session_id=obs.cellquant_session_id,
                    notes=obs.notes,
                    recorded_at=obs.recorded_at,
                ))

        return created

    # -- analysis views ------------------------------------------------------

    @router.get(
        "/experiments/{experiment_id}/timecourse",
        response_model=TimeCourseResponse,
    )
    async def get_timecourse(experiment_id: str, marker: str | None = None):
        """Get time-course data grouped by treatment and marker."""
        store = _get_store()
        exp = store.get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(404, "Experiment not found")

        treatment_map = {t.id: t.name for t in exp.treatments}

        # Group: (treatment_id, marker) -> list of (hours, value, std)
        groups: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
        for obs in exp.observations:
            if marker and obs.marker != marker:
                continue
            groups[(obs.treatment_id, obs.marker)].append(
                (obs.timepoint_hours, obs.value, obs.std_dev)
            )

        series = []
        for (tid, mkr), pts in sorted(groups.items()):
            pts.sort(key=lambda x: x[0])
            series.append(TimeCourseSeries(
                treatment=treatment_map.get(tid, tid),
                marker=mkr,
                hours=[p[0] for p in pts],
                means=[p[1] for p in pts],
                std_devs=[p[2] for p in pts],
            ))

        return TimeCourseResponse(experiment_id=experiment_id, series=series)

    @router.get(
        "/experiments/{experiment_id}/dose-response",
        response_model=DoseResponseResponse,
    )
    async def get_dose_response(experiment_id: str, timepoint_hours: float | None = None):
        """Get dose-response data grouped by oxysterol and marker."""
        store = _get_store()
        exp = store.get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(404, "Experiment not found")

        treatment_info = {t.id: t for t in exp.treatments}

        # Group: (oxysterol, marker) -> list of (conc, value, std, n)
        groups: dict[tuple[str, str], list[tuple[float, float, float, int]]] = defaultdict(list)
        for obs in exp.observations:
            if timepoint_hours is not None and obs.timepoint_hours != timepoint_hours:
                continue
            t = treatment_info.get(obs.treatment_id)
            if t is None:
                continue
            groups[(t.oxysterol, obs.marker)].append(
                (t.concentration_uM, obs.value, obs.std_dev, obs.n_cells)
            )

        series = []
        for (oxy, mkr), pts in sorted(groups.items()):
            pts.sort(key=lambda x: x[0])
            series.append(DoseResponseSeries(
                oxysterol=oxy,
                marker=mkr,
                points=[
                    DoseResponsePoint(
                        concentration_uM=p[0], mean=p[1], std_dev=p[2], n_cells=p[3]
                    )
                    for p in pts
                ],
            ))

        return DoseResponseResponse(experiment_id=experiment_id, series=series)

    return router


# -- helpers -----------------------------------------------------------------

def _experiment_to_detail(exp):
    from dataclasses import asdict

    return ExperimentDetail(
        id=exp.id,
        name=exp.name,
        cell_line=exp.cell_line,
        passage=exp.passage,
        description=exp.description,
        treatments=[
            TreatmentArmSchema(
                name=t.name,
                oxysterol=t.oxysterol,
                concentration_uM=t.concentration_uM,
                vehicle=t.vehicle,
                vehicle_pct=t.vehicle_pct,
                notes=t.notes,
            )
            for t in exp.treatments
        ],
        timepoints=[
            TimePointSchema(
                hours=tp.hours,
                label=tp.label,
                cellquant_session_id=tp.cellquant_session_id,
            )
            for tp in exp.timepoints
        ],
        marker_panel=MarkerPanelSchema(
            markers=exp.marker_panel.markers,
            channel_mapping=exp.marker_panel.channel_mapping,
        ),
        observations=[
            ObservationResponse(
                id=o.id,
                experiment_id=o.experiment_id,
                treatment_id=o.treatment_id,
                timepoint_hours=o.timepoint_hours,
                marker=o.marker,
                value=o.value,
                unit=o.unit,
                n_cells=o.n_cells,
                std_dev=o.std_dev,
                cellquant_session_id=o.cellquant_session_id,
                notes=o.notes,
                recorded_at=o.recorded_at,
            )
            for o in exp.observations
        ],
        created_at=exp.created_at,
        updated_at=exp.updated_at,
        tags=exp.tags,
    )
