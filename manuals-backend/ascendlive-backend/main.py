from typing import List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DiagnoseRequest(BaseModel):
    complaint: str
    modelNumber: str
    productSubtype: Optional[str] = None
    errorCode: Optional[str] = None
    observedSymptoms: List[str] = []


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/diagnose")
def diagnose(payload: DiagnoseRequest):
    complaint = payload.complaint.lower()
    error_code = (payload.errorCode or "").upper()

    diagnosis = "General troubleshooting guidance"
    confidence = "0.60"
    recommended_actions = [
        "Verify the model number and symptom details.",
        "Check the user manual troubleshooting section.",
    ]
    escalation = "Try self-service steps first, then escalate if issue persists."

    if "not cooling" in complaint or "warm" in complaint:
        diagnosis = "Cooling performance issue"
        confidence = "0.82"
        recommended_actions = [
            "Verify temperature settings are correct.",
            "Check vents for blockage.",
            "Inspect door seals and condenser coils.",
        ]
        escalation = "Escalate if cabinet temperature remains unsafe after cleanup and reset."

    if error_code in {"22E", "22C"}:
        diagnosis = "Refrigerator fan / frost issue"
        confidence = "0.91"
        recommended_actions = [
            "Power cycle the refrigerator for 5 minutes.",
            "Inspect for frost buildup around the evaporator area.",
            "Ensure doors are sealing properly.",
        ]
        escalation = "Dispatch service if the code returns after defrost and restart."

    return {
        "status": "ok",
        "diagnosis": diagnosis,
        "confidence": confidence,
        "recommendedActions": recommended_actions,
        "escalationRecommendation": escalation,
        "citations": [],
    }


@app.get("/manuals/snippets")
def manuals_snippets(
    modelNumber: str = Query(...),
    issue: Optional[str] = Query(None),
    errorCode: Optional[str] = Query(None),
):
    return {
        "status": "ok",
        "modelNumber": modelNumber,
        "snippets": [
            {
                "sectionTitle": "Cooling performance and airflow",
                "excerpt": "Do not block cold air vents. Verify recommended temperature settings and allow time for stabilization.",
                "citationLabel": f"{modelNumber} manual (pp. 14-15)",
            },
            {
                "sectionTitle": "Door seal inspection",
                "excerpt": "If doors do not close completely, cooling performance may drop and frost may form.",
                "citationLabel": f"{modelNumber} manual (p. 18)",
            },
        ],
    }
