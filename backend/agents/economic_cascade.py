"""
backend/agents/economic_cascade.py

Agent 5 — Economic Cascade Engine

Computes the TRUE economic cost of road deterioration to communities.

Every IRI unit increase above baseline causes cascading economic effects:
  1. Vehicle Operating Cost (VOC) — World Bank HDM-4 curves, India calibration
  2. Agricultural produce loss in transit (post-harvest damage)
  3. School attendance reduction (longer journey time)
  4. Healthcare access deterioration (ambulance delay)

Data sources (all free/open):
  - WorldPop API: population counts per grid cell
  - OpenStreetMap (Overpass API): schools, PHCs, markets, agricultural land
  - World Bank HDM-4: VOC curves calibrated for India (CRRI 2024)

LLM narrative generation via Ollama (local, no cloud required).
"""

import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

# Default AADT + VOC parameters (India, mixed rural road)
VOC_BASELINE_INR_PER_KM = 12.0   # ₹/km baseline vehicle operating cost (Good road)
VOC_INCREASE_PCT_PER_IRI = 2.5   # % increase per IRI unit above baseline
ASSUMED_DAILY_VEHICLES = 200      # Typical rural road AADT

# Baseline IRI for Good condition
IRI_BASELINE = 2.0

# Agricultural production assumptions (India mixed crop, NABARD 2024)
AVG_PRODUCE_VALUE_PER_HA_INR = 80_000   # ₹/year/ha
POST_HARVEST_LOSS_SLOPE = 1.5           # % extra loss per IRI unit above 3.0
IRI_AGRICULTURAL_THRESHOLD = 3.0

# School attendance model
CYCLING_SPEED_GOOD_KMH = 12.0     # On Good road
ATTENDANCE_DROP_PER_10MIN = 5.0   # % drop per 10 extra minutes journey

# Healthcare: ambulance base speed
AMBULANCE_BASE_SPEED_KMH = 40.0


class EconomicCascadeEngine:
    """
    Computes family-level economic impact of road deterioration.
    """

    def __init__(
        self, 
        gemini_model: str = "gemini-3-flash-preview", 
        gemini_api_key: Optional[str] = None
    ):
        self.model = gemini_model
        self.api_key = gemini_api_key
        self._llm_available = bool(self.api_key and self.api_key.strip())
        if not self._llm_available:
            logger.warning("GEMINI_API_KEY not found — narrative generation disabled (using template fallback).")

    def compute_cascade(
        self,
        segment: dict,
        osm_context: dict,
        population: int = 500,
    ) -> dict:
        """
        Full economic cascade from IRI value to family-level rupee impact.

        Args:
            segment:     Fused segment dict (must have iri_value, length_km, surface_type).
            osm_context: OSM data dict with keys:
                         schools (list), health_facilities (list),
                         agricultural_land_ha (float), markets (list).
            population:  Estimated population in the dependent area.

        Returns:
            Economic cascade dict with all impact components + LLM narrative.
        """
        iri       = segment.get("iri_value")
        length_km = segment.get("length_km", 0.1)
        surface   = segment.get("surface_type", "WBM")

        if iri is None:
            return {"error": "no_iri_value", "narrative": "IRI data unavailable for economic analysis."}

        # ── 1. Vehicle Operating Cost (VOC) ───────────────────────────────
        voc_increase_pct = max(0.0, (iri - IRI_BASELINE) * VOC_INCREASE_PCT_PER_IRI)
        daily_voc_increase = (
            ASSUMED_DAILY_VEHICLES
            * VOC_BASELINE_INR_PER_KM
            * (voc_increase_pct / 100.0)
            * length_km
        )
        annual_voc_cost = daily_voc_increase * 365

        # ── 2. Agricultural Loss ──────────────────────────────────────────
        produce_loss_pct = 0.0
        if iri > IRI_AGRICULTURAL_THRESHOLD:
            produce_loss_pct = (iri - IRI_AGRICULTURAL_THRESHOLD) * POST_HARVEST_LOSS_SLOPE

        nearby_farms_ha = osm_context.get("agricultural_land_ha", 50)
        agricultural_loss_annual = (
            nearby_farms_ha * AVG_PRODUCE_VALUE_PER_HA_INR * (produce_loss_pct / 100.0)
        )

        # ── 3. School Attendance Impact ───────────────────────────────────
        schools = osm_context.get("schools", [])
        attendance_impacts = []

        for school in schools[:5]:  # Cap at 5 nearest schools
            dist_km = school.get("distance_km", 1.0)
            students = school.get("student_count", 200)

            # Speed reduction on rough road (capped at 40% reduction)
            speed_reduction = max(0.4, 1.0 - (iri - IRI_BASELINE) * 0.12)
            actual_speed = CYCLING_SPEED_GOOD_KMH * speed_reduction

            journey_baseline_min = (dist_km / CYCLING_SPEED_GOOD_KMH) * 60
            journey_actual_min   = (dist_km / actual_speed) * 60
            extra_min = max(0.0, journey_actual_min - journey_baseline_min)

            # Attendance drop: 5% per 10 extra minutes
            attendance_drop_pct = min(30.0, extra_min * (ATTENDANCE_DROP_PER_10MIN / 10.0))

            attendance_impacts.append({
                "school":               school.get("name", "Nearby School"),
                "students_affected":    students,
                "extra_travel_minutes": round(extra_min, 1),
                "attendance_drop_pct":  round(attendance_drop_pct, 1),
                "distance_km":          dist_km,
            })

        # ── 4. Healthcare Access / Ambulance Delay ────────────────────────
        phcs = osm_context.get("health_facilities", [])
        max_ambulance_delay_min = 0.0

        for phc in phcs[:2]:
            dist_km = phc.get("distance_km", 5.0)
            speed_reduction = max(0.3, 1.0 - (iri - IRI_BASELINE) * 0.15)
            base_time_min   = (dist_km / AMBULANCE_BASE_SPEED_KMH) * 60
            actual_time_min = (dist_km / (AMBULANCE_BASE_SPEED_KMH * speed_reduction)) * 60
            delay = max(0.0, actual_time_min - base_time_min)
            max_ambulance_delay_min = max(max_ambulance_delay_min, delay)

        # ── 5. Totals ─────────────────────────────────────────────────────
        total_annual_loss = annual_voc_cost + agricultural_loss_annual

        cascade_summary = {
            "segment_id":           segment.get("segment_id"),
            "iri":                  iri,
            "population_affected":  population,
            "length_km":            length_km,

            # VOC
            "voc_increase_pct":     round(voc_increase_pct, 1),
            "annual_voc_cost_lakh": round(annual_voc_cost / 100_000, 2),

            # Agriculture
            "agricultural_loss_pct":          round(produce_loss_pct, 1),
            "agricultural_loss_annual_lakh":  round(agricultural_loss_annual / 100_000, 2),

            # School
            "schools_affected":               attendance_impacts,
            "total_students_affected":        sum(s["students_affected"] for s in attendance_impacts),

            # Healthcare
            "health_facilities_nearby":       len(phcs),
            "ambulance_delay_minutes":        round(max_ambulance_delay_min, 1),

            # Summary
            "total_annual_economic_loss_lakh": round(total_annual_loss / 100_000, 2),
            "monthly_loss_lakh":               round(total_annual_loss / 100_000 / 12, 2),
        }

        # ── 6. LLM Narrative ──────────────────────────────────────────────
        cascade_summary["narrative"] = self._generate_narrative(cascade_summary)

        return cascade_summary

    def _generate_narrative(self, data: dict) -> str:
        """
        Generate plain-language economic impact paragraph via Gemini API.
        Falls back to a template if API key is unavailable or request fails.
        """
        if not self._llm_available:
            return self._template_narrative(data)

        prompt = f"""You are a development economist writing a one-paragraph impact summary \
for a district engineer reviewing road conditions in rural India.

Road data:
- IRI: {data['iri']} m/km
- Population affected: {data['population_affected']}
- Annual economic loss: ₹{data['total_annual_economic_loss_lakh']} Lakh
- Vehicle operating cost increase: ₹{data['annual_voc_cost_lakh']} Lakh/year
- Agricultural loss: ₹{data['agricultural_loss_annual_lakh']} Lakh/year
- Students affected: {data['total_students_affected']}
- Ambulance delay: {data['ambulance_delay_minutes']} minutes extra

Write exactly 3 sentences:
1. State the total annual economic loss in rupees.
2. Name the specific human impacts (farmers, students, patients) with the actual numbers.
3. Make the case for urgent intervention using the numbers.

Be specific. Use actual numbers. Write in formal English. No markdown."""

        try:
            import requests as req
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 250
                }
            }
            response = req.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                resp_data = response.json()
                return resp_data["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                logger.error(f"Gemini API error ({response.status_code}): {response.text}")
        except Exception as exc:
            logger.warning(f"Gemini narrative generation failed: {exc}")

        return self._template_narrative(data)

    def _template_narrative(self, data: dict) -> str:
        """Fallback narrative template when Ollama is unavailable."""
        students = data.get("total_students_affected", 0)
        return (
            f"The deteriorated road (IRI {data['iri']:.1f} m/km) imposes an estimated "
            f"₹{data['total_annual_economic_loss_lakh']:.1f} Lakh annual economic burden "
            f"on the dependent community of {data['population_affected']:,} residents. "
            f"This includes ₹{data['annual_voc_cost_lakh']:.1f} Lakh in excess vehicle "
            f"operating costs, ₹{data['agricultural_loss_annual_lakh']:.1f} Lakh in "
            f"post-harvest agricultural losses, and {students:,} students facing extended "
            f"journey times of up to {max((s['extra_travel_minutes'] for s in data.get('schools_affected', [])), default=0):.0f} "
            f"extra minutes daily; immediate intervention is warranted before structural "
            f"failure compounds these costs further."
        )

    def fetch_osm_context(self, lat: float, lng: float, radius_m: int = 3000) -> dict:
        """
        Query Overpass API for schools, health facilities, and land use near coordinates.

        Args:
            lat, lng:  Coordinates of the road segment midpoint.
            radius_m:  Search radius in metres.

        Returns:
            osm_context dict ready for compute_cascade().
        """
        try:
            import overpy
            api = overpy.Overpass()

            query = f"""
            [out:json][timeout:10];
            (
              node["amenity"="school"](around:{radius_m},{lat},{lng});
              node["amenity"="hospital"](around:{radius_m},{lat},{lng});
              node["amenity"="clinic"](around:{radius_m},{lat},{lng});
              node["amenity"="health_post"](around:{radius_m},{lat},{lng});
              way["landuse"="farmland"](around:{radius_m},{lat},{lng});
            );
            out body;
            """
            result = api.query(query)

            schools = []
            for node in result.nodes:
                if node.tags.get("amenity") == "school":
                    # Approximate distance (Haversine not worth it at small scales)
                    schools.append({
                        "name":         node.tags.get("name", "Unnamed School"),
                        "distance_km":  radius_m / 2000.0,  # Rough midpoint estimate
                        "student_count": 200,  # Default — WorldPop doesn't give this
                    })

            phcs = []
            for node in result.nodes:
                if node.tags.get("amenity") in ("hospital", "clinic", "health_post"):
                    phcs.append({
                        "name":        node.tags.get("name", "Health Facility"),
                        "distance_km": radius_m / 3000.0,
                    })

            # Agricultural land area (rough estimate from farmland ways)
            farm_ha = len(result.ways) * 5.0  # Very rough: each farmland way ≈ 5 ha

            return {
                "schools":             schools,
                "health_facilities":   phcs,
                "agricultural_land_ha": max(farm_ha, 10.0),
            }

        except Exception as exc:
            logger.warning(f"OSM query failed: {exc}. Using default context.")
            return {
                "schools":             [{"name": "Nearby School", "distance_km": 1.5, "student_count": 200}],
                "health_facilities":   [{"name": "PHC", "distance_km": 5.0}],
                "agricultural_land_ha": 50.0,
            }
