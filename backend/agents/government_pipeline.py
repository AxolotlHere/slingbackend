"""
backend/agents/government_pipeline.py

Agent 7 — Autonomous Government Pipeline

Auto-drafts PMGSY (Pradhan Mantri Gram Sadak Yojana) funding applications
from road assessment data. The engineer's job: read → press send.

LLM: Qwen2.5-7B via Ollama (local, no API key, no cloud, no billing)
PDF: ReportLab (self-contained)

References:
  - PMGSY Application Format (MoRTH circular 2023)
  - IRC:SP:20-2002 thresholds
  - MoRTH Schedule of Rates 2024
"""

import datetime
import logging
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# MoRTH Schedule of Rates 2024 (₹/km)
SOR_2024 = {
    ("BC",       "Routine"):        300_000,
    ("BC",       "Preventive"):   1_200_000,
    ("BC",       "Rehabilitation"):5_000_000,
    ("BC",       "Reconstruction"):12_000_000,
    ("WBM",      "Routine"):        150_000,
    ("WBM",      "Preventive"):     600_000,
    ("WBM",      "Rehabilitation"):3_000_000,
    ("WBM",      "Reconstruction"):8_000_000,
    ("Granular", "Routine"):         80_000,
    ("Granular", "Preventive"):     300_000,
    ("Granular", "Rehabilitation"):1_500_000,
    ("Granular", "Reconstruction"):5_000_000,
    ("Concrete", "Routine"):        200_000,
    ("Concrete", "Preventive"):     800_000,
    ("Concrete", "Rehabilitation"):4_000_000,
    ("Concrete", "Reconstruction"):10_000_000,
}

# IRC:SP:20 intervention → SOR intervention mapping
IRI_TO_INTERVENTION = [
    (2.0, "Routine"),
    (4.0, "Preventive"),
    (6.0, "Rehabilitation"),
    (float("inf"), "Reconstruction"),
]


def _get_intervention(iri: float) -> str:
    for threshold, intervention in IRI_TO_INTERVENTION:
        if iri < threshold:
            return intervention
    return "Reconstruction"


def _get_unit_cost(surface: str, intervention: str) -> int:
    key = (surface, intervention)
    if key in SOR_2024:
        return SOR_2024[key]
    # Fallback to WBM
    return SOR_2024.get(("WBM", intervention), 3_000_000)


class GovernmentPipelineAgent:
    """
    Generates complete PMGSY funding applications from road assessment data.
    """

    def __init__(
        self,
        gemini_model: str = "gemini-3-flash-preview",
        gemini_api_key: Optional[str] = None,
    ):
        self.model = gemini_model
        self.api_key = gemini_api_key
        # If API key is missing or empty, fallback to template
        self._llm_ok = bool(self.api_key and self.api_key.strip())
        if not self._llm_ok:
            logger.warning("GEMINI_API_KEY not found — using template-based application text.")

    def draft_pmgsy_application(
        self,
        road_data: dict,
        economic_data: dict,
        district_info: dict,
    ) -> dict:
        """
        Generate complete PMGSY funding application.

        Args:
            road_data:     Fused segment dict (iri_value, surface_type, length_km, etc.)
            economic_data: Output of EconomicCascadeEngine.compute_cascade()
            district_info: dict with keys: district, state, block, road_name, village

        Returns:
            Application dict with text, budget, metadata, status.
        """
        iri       = road_data.get("iri_value")
        iri_cond  = road_data.get("iri_condition", "Unknown")
        length_km = road_data.get("length_km", 1.0)
        surface   = road_data.get("surface_type", "WBM")

        # Determine intervention from IRI
        intervention = _get_intervention(iri) if iri else "Rehabilitation"

        # Budget calculation
        unit_cost    = _get_unit_cost(surface, intervention)
        total_budget = unit_cost * length_km

        # Generate application text
        application_text = self._generate_text(
            road_data=road_data,
            economic_data=economic_data,
            district_info=district_info,
            intervention=intervention,
            total_budget=total_budget,
        )

        return {
            "application_text":     application_text,
            "intervention_type":    intervention,
            "surface_type":         surface,
            "road_length_km":       round(length_km, 2),
            "iri_value":            iri,
            "iri_condition":        iri_cond,
            "unit_cost_per_km_lakh": round(unit_cost / 100_000, 1),
            "total_budget_lakh":    round(total_budget / 100_000, 1),
            "irc_standard_cited":   "IRC:SP:20-2002",
            "sor_year":             "MoRTH SOR 2024",
            "beneficiary_population": economic_data.get("population_affected", "N/A"),
            "annual_economic_loss_lakh": economic_data.get("total_annual_economic_loss_lakh", "N/A"),
            "district":             district_info.get("district", ""),
            "state":                district_info.get("state", ""),
            "road_name":            district_info.get("road_name", "Road under assessment"),
            "status":               "DRAFT — Ready for Engineer Review",
            "generated_at":         datetime.datetime.now().isoformat(),
            "model_used":           self.model if self._llm_ok else "template",
        }

    def _generate_text(
        self,
        road_data: dict,
        economic_data: dict,
        district_info: dict,
        intervention: str,
        total_budget: float,
    ) -> str:
        """Generate 4-paragraph application text via Gemini or template fallback."""

        if self._llm_ok:
            return self._llm_generate(road_data, economic_data, district_info,
                                       intervention, total_budget)
        return self._template_generate(road_data, economic_data, district_info,
                                        intervention, total_budget)

    def _llm_generate(self, road_data, economic_data, district_info, intervention, total_budget):
        prompt = f"""You are drafting a PMGSY (Pradhan Mantri Gram Sadak Yojana) \
funding application for road rehabilitation. Write a formal government application \
with these exact details:

Road Name:           {district_info.get('road_name', 'Road under assessment')}
Village/Block:       {district_info.get('village', '')} / {district_info.get('block', '')}
District/State:      {district_info.get('district', '')} / {district_info.get('state', '')}
Length:              {road_data.get('length_km', 1.0):.2f} km
IRI (measured):      {road_data.get('iri_value', 'N/A')} m/km
IRC:SP:20 condition: {road_data.get('iri_condition', 'N/A')}
Surface type:        {road_data.get('surface_type', 'WBM')}
Proposed work:       {intervention}
Estimated cost:      ₹{total_budget/100_000:.1f} Lakh
Beneficiary pop.:    {economic_data.get('population_affected', 'N/A')}
Annual econ. loss:   ₹{economic_data.get('total_annual_economic_loss_lakh', 'N/A')} Lakh

Write EXACTLY 4 paragraphs:
1. Background and current condition — cite IRI value and IRC:SP:20 standard.
2. Economic justification — use the loss figures and community impact data.
3. Proposed intervention and technical specifications per MoRTH SOR 2024.
4. Budget summary and formal request for sanction.

Write in formal government English. Reference IRC:SP:20-2002 and MoRTH SOR 2024. \
No bullet points. No headers within the paragraphs."""

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 700
                }
            }
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                logger.error(f"Gemini API error ({response.status_code}): {response.text}")
        except Exception as exc:
            logger.warning(f"LLM generation failed: {exc}. Using template.")

        return self._template_generate(road_data, economic_data, district_info,
                                        intervention, total_budget)

    def _template_generate(self, road_data, economic_data, district_info, intervention, total_budget):
        """Deterministic template when Ollama is unavailable."""
        road_name = district_info.get("road_name", "Road under assessment")
        district  = district_info.get("district", "the district")
        state     = district_info.get("state", "")
        length    = road_data.get("length_km", 1.0)
        iri       = road_data.get("iri_value", "N/A")
        condition = road_data.get("iri_condition", "N/A")
        surface   = road_data.get("surface_type", "WBM")
        pop       = economic_data.get("population_affected", "N/A")
        econ_loss = economic_data.get("total_annual_economic_loss_lakh", "N/A")

        return f"""BACKGROUND AND CURRENT CONDITION: {road_name} in {district}, {state}, \
spanning {length:.2f} km of {surface} surface, has been assessed under the PULSE \
multi-channel road survey system. The International Roughness Index (IRI) measurement, \
conducted in accordance with IRC:SP:20-2002 standards, recorded a value of {iri} m/km, \
classifying the road as {condition}. This measurement was obtained via a calibrated \
accelerometer-based quarter-car model, consistent with World Bank and IRC methodology.

ECONOMIC JUSTIFICATION: The deteriorated condition of this road imposes an estimated \
₹{econ_loss} Lakh annual economic burden on the {pop} residents dependent upon it. \
This burden arises from excess vehicle operating costs, post-harvest agricultural \
produce losses due to transport damage, extended journey times for students and \
healthcare seekers, and delayed emergency response. Immediate intervention will prevent \
further structural deterioration and eliminate this recurring annual economic drain.

PROPOSED INTERVENTION AND TECHNICAL SPECIFICATIONS: The recommended intervention is \
{intervention} of the {surface} surface as per MoRTH Schedule of Rates 2024. \
The work shall be carried out in accordance with IRC:SP:20-2002 specifications for \
rural road rehabilitation, including appropriate drainage restoration, edge repair, \
and surface treatment as determined by the executing engineer following site inspection.

BUDGET AND REQUEST FOR SANCTION: The estimated project cost is ₹{total_budget/100_000:.1f} Lakh \
for {length:.2f} km at the MoRTH SOR 2024 unit rate for {surface} {intervention}. \
It is hereby requested that the competent authority sanction the above project under \
PMGSY Phase III for early execution, prioritising the welfare of the {pop} beneficiaries \
who have been subjected to substandard road conditions for an extended period."""
