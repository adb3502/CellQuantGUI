"""
BHARAT Cohort Analytics API router.

Provides endpoints for:
  - Cohort import and listing
  - Demographics and biomarker profiling
  - AgingClock India biological-age analysis
  - Demo data generation
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cellquant.bharat.cohort_manager import get_cohort_store
from cellquant.bharat.demo_data import generate_demo_cohort

router = APIRouter(prefix="/bharat", tags=["bharat"])


# ── Request / Response helpers ────────────────────────────

class ImportRequest(BaseModel):
    file_path: str


class ImportResponse(BaseModel):
    cohort_id: str
    name: str
    n_subjects: int


class DemoRequest(BaseModel):
    n: int = 500
    seed: int = 42


# ── Cohort management ────────────────────────────────────

@router.get("/cohorts")
async def list_cohorts():
    """List all imported cohorts."""
    store = get_cohort_store()
    return store.list_cohorts()


@router.post("/cohorts/import", response_model=ImportResponse)
async def import_cohort(req: ImportRequest):
    """Import cohort data from a CSV or XLSX file."""
    store = get_cohort_store()
    try:
        cohort_id = store.import_file(req.file_path)
    except FileNotFoundError:
        raise HTTPException(404, f"File not found: {req.file_path}")
    except Exception as e:
        raise HTTPException(400, f"Import failed: {e}")

    df = store.get_df(cohort_id)
    return ImportResponse(
        cohort_id=cohort_id,
        name=req.file_path.rsplit("/", 1)[-1],
        n_subjects=len(df),
    )


@router.post("/cohorts/demo", response_model=ImportResponse)
async def load_demo_cohort(req: DemoRequest):
    """Generate and load a synthetic demo cohort."""
    store = get_cohort_store()
    df = generate_demo_cohort(n=req.n, seed=req.seed)

    # Write to a temp CSV and import through the normal path
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="bharat_demo_"
    )
    df.to_csv(tmp, index=False)
    tmp.close()
    try:
        cohort_id = store.import_file(tmp.name, name="BHARAT Demo Cohort")
    finally:
        os.unlink(tmp.name)

    return ImportResponse(
        cohort_id=cohort_id,
        name="BHARAT Demo Cohort",
        n_subjects=len(df),
    )


# ── Demographics ──────────────────────────────────────────

@router.get("/cohorts/{cohort_id}/demographics")
async def get_demographics(cohort_id: str):
    """Get demographic breakdown for a cohort."""
    store = get_cohort_store()
    try:
        return store.demographics(cohort_id)
    except KeyError:
        raise HTTPException(404, f"Cohort not found: {cohort_id}")


# ── Biomarker profile ────────────────────────────────────

@router.get("/cohorts/{cohort_id}/biomarkers")
async def get_biomarker_profile(cohort_id: str):
    """Get population-level biomarker statistics."""
    store = get_cohort_store()
    try:
        return store.biomarker_profile(cohort_id)
    except KeyError:
        raise HTTPException(404, f"Cohort not found: {cohort_id}")


# ── AgingClock India ──────────────────────────────────────

@router.post("/cohorts/{cohort_id}/aging-clock/run")
async def run_aging_clock(cohort_id: str):
    """Run the AgingClock India analysis on the cohort."""
    store = get_cohort_store()
    try:
        summary = store.run_aging_clock(cohort_id)
    except KeyError:
        raise HTTPException(404, f"Cohort not found: {cohort_id}")
    return summary


@router.get("/cohorts/{cohort_id}/aging-clock/results")
async def get_aging_clock_results(cohort_id: str, page: int = 0, per_page: int = 50):
    """Get paginated individual AgingClock predictions."""
    store = get_cohort_store()
    predictions = store.get_predictions(cohort_id)
    if not predictions:
        raise HTTPException(404, "No AgingClock results. Run analysis first.")

    total = len(predictions)
    start = page * per_page
    end = min(start + per_page, total)
    page_items = predictions[start:end]

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page,
        "results": [
            {
                "subject_id": p.subject_id,
                "chronological_age": p.chronological_age,
                "biological_age": p.biological_age,
                "age_gap": p.age_gap,
                "percentile": p.percentile,
                "confidence": p.confidence,
                "top_accelerators": [
                    {
                        "name": c.name,
                        "display_name": c.display_name,
                        "value": c.value,
                        "contribution": c.contribution,
                        "reference_range": c.reference_range,
                        "status": c.status,
                    }
                    for c in p.top_accelerators
                ],
                "top_decelerators": [
                    {
                        "name": c.name,
                        "display_name": c.display_name,
                        "value": c.value,
                        "contribution": c.contribution,
                        "reference_range": c.reference_range,
                        "status": c.status,
                    }
                    for c in p.top_decelerators
                ],
            }
            for p in page_items
        ],
    }


@router.get("/cohorts/{cohort_id}/aging-clock/subject/{subject_id}")
async def get_subject_aging(cohort_id: str, subject_id: str):
    """Get AgingClock result for a single subject."""
    store = get_cohort_store()
    pred = store.get_subject_prediction(cohort_id, subject_id)
    if pred is None:
        raise HTTPException(404, f"No prediction for subject: {subject_id}")
    return {
        "subject_id": pred.subject_id,
        "chronological_age": pred.chronological_age,
        "biological_age": pred.biological_age,
        "age_gap": pred.age_gap,
        "percentile": pred.percentile,
        "confidence": pred.confidence,
        "top_accelerators": [
            {
                "name": c.name,
                "display_name": c.display_name,
                "value": c.value,
                "contribution": c.contribution,
                "reference_range": c.reference_range,
                "status": c.status,
            }
            for c in pred.top_accelerators
        ],
        "top_decelerators": [
            {
                "name": c.name,
                "display_name": c.display_name,
                "value": c.value,
                "contribution": c.contribution,
                "reference_range": c.reference_range,
                "status": c.status,
            }
            for c in pred.top_decelerators
        ],
    }


# ── Cohort data table ────────────────────────────────────

@router.get("/cohorts/{cohort_id}/data")
async def get_cohort_data(cohort_id: str, page: int = 0, per_page: int = 50):
    """Get paginated raw cohort data."""
    store = get_cohort_store()
    try:
        df = store.get_df(cohort_id)
    except KeyError:
        raise HTTPException(404, f"Cohort not found: {cohort_id}")

    total = len(df)
    start = page * per_page
    end = min(start + per_page, total)
    page_df = df.iloc[start:end]

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": (total + per_page - 1) // per_page,
        "columns": list(df.columns),
        "data": page_df.where(page_df.notna(), None).to_dict(orient="records"),
    }
