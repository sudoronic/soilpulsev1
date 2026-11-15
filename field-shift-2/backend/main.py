import csv
import io
import math
import os
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gtts import gTTS
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="SoilPulse — NASA Field Shift Backend", version="2.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
NRCS_SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
CDL_POINT_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLValue"
BASE_DIR = Path(__file__).resolve().parent
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]


class FieldAnalyzeRequest(BaseModel):
    polygon: Dict[str, Any]
    lat: float
    lon: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class KissanAdvisorRequest(BaseModel):
    lat: float
    lon: float
    area_acres: float = 10.0
    rain_mm: float
    peak_temp_c: float
    soil: str
    ph: Any
    water_source: str = "Canal / Mixed"
    current_crop: str
    recommended_rotation: str
    question: str
    language: str = "Urdu (اردو)"


class KissanVoiceRequest(BaseModel):
    lat: float
    lon: float
    area_acres: float = 10.0
    rain_mm: float
    peak_temp_c: float
    soil: str
    ph: Any
    water_source: str = "Canal / Mixed"
    current_crop: str
    recommended_rotation: str
    audio_b64: str
    mime_type: str = "audio/webm"
    language: str = "Urdu (اردو)"


class TtsRequest(BaseModel):
    text: str
    lang: str = "ur"


def wgs84_to_epsg5070(lon: float, lat: float) -> tuple[float, float]:
    a, f = 6378137.0, 1.0 / 298.257222101
    e2 = 2 * f - f * f
    e = math.sqrt(e2)
    phi1, phi2 = math.radians(29.5), math.radians(45.5)
    phi0, lam0 = math.radians(23.0), math.radians(-96.0)
    phi, lam = math.radians(lat), math.radians(lon)

    def m_fn(p: float) -> float:
        return math.cos(p) / math.sqrt(1.0 - e2 * math.sin(p) ** 2)

    def q_fn(p: float) -> float:
        sp = math.sin(p)
        return (1.0 - e2) * (sp / (1.0 - e2 * sp * sp) - (1.0 / (2.0 * e)) * math.log((1.0 - e * sp) / (1.0 + e * sp)))

    m1, m2 = m_fn(phi1), m_fn(phi2)
    q0, q1, q2, q = q_fn(phi0), q_fn(phi1), q_fn(phi2), q_fn(phi)
    n = (m1 * m1 - m2 * m2) / (q2 - q1)
    c = m1 * m1 + n * q1
    rho0 = a * math.sqrt(c - n * q0) / n
    rho = a * math.sqrt(c - n * q) / n
    theta = n * (lam - lam0)
    return rho * math.sin(theta), rho0 - rho * math.cos(theta)


@app.get("/")
async def healthcheck():
    return {
        "status": "ok",
        "service": "SoilPulse Backend",
        "earthdata_configured": bool(os.getenv("EARTHDATA_USER") and os.getenv("EARTHDATA_PASS")),
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "snic_csv_present": (BASE_DIR / "fields_report.csv").exists(),
    }


@app.get("/api/punjab/bare-fields")
async def get_punjab_bare_fields():
    csv_path = BASE_DIR / "fields_report.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="fields_report.csv not found inside backend/ folder.")

    fields, bare_count, cult_count, bare_ha, cult_ha = [], 0, 0, 0.0, 0.0
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = row.get("status", "")
            area = float(row.get("area_ha") or 0.0)
            ndvi = float(row["mean_ndvi"]) if row.get("mean_ndvi") else None
            if status == "Bare":
                bare_count += 1
                bare_ha += area
            elif status == "Cultivated":
                cult_count += 1
                cult_ha += area
            fields.append({
                "id": int(row["field_id"]),
                "lon": float(row["center_lon"]),
                "lat": float(row["center_lat"]),
                "area_ha": round(area, 3),
                "area_acres": round(area * 2.471, 2),
                "ndvi": ndvi,
                "status": status,
                "bare_months": row.get("typical_bare_months", "Rarely bare"),
                "may_pct": float(row["May"]) if row.get("May") else 0.0,
                "jun_pct": float(row["Jun"]) if row.get("Jun") else 0.0,
            })

    return {
        "total_fields": len(fields),
        "bare_count": bare_count,
        "cultivated_count": cult_count,
        "bare_ha": round(bare_ha, 1),
        "cultivated_ha": round(cult_ha, 1),
        "fields": fields,
    }


async def call_gemini(parts: list[Dict[str, Any]]) -> tuple[str, str]:
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY missing in backend/.env")

    payload = {"contents": [{"role": "user", "parts": parts}]}
    endpoints = []
    if gemini_key.startswith("AQ."):
        for m in GEMINI_MODELS:
            endpoints.append((f"vertex:{m}", f"https://aiplatform.googleapis.com/v1/publishers/google/models/{m}:generateContent?key={gemini_key}"))
    else:
        for m in GEMINI_MODELS:
            endpoints.append((f"aistudio:{m}", f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={gemini_key}"))

    last_err = ""
    async with httpx.AsyncClient() as client:
        for label, url in endpoints:
            try:
                resp = await client.post(url, json=payload, timeout=30.0)
                if resp.status_code == 200:
                    cands = resp.json().get("candidates") or []
                    if cands:
                        text_parts = [p.get("text", "") for p in cands[0].get("content", {}).get("parts", []) if p.get("text")]
                        if text_parts:
                            return "\n".join(text_parts).strip(), label
                last_err = f"{label} HTTP {resp.status_code}: {resp.text[:350]}"
            except Exception as exc:
                last_err = f"{label} error: {str(exc)}"
    raise HTTPException(status_code=502, detail=f"Advisor request failed: {last_err}")


@app.post("/api/ai/kissan-advisor")
async def kissan_ai_advisor(req: KissanAdvisorRequest):
    prompt = (
        f"You are a warm, practical Pakistani agricultural extension officer speaking to a local farmer. "
        f"Reply ONLY in simple, everyday {req.language} (3 short sentences, zero technical jargon like NDVI or ET0). "
        f"Field telemetry ({req.lat:.3f}, {req.lon:.3f}, {req.area_acres:.1f} acres): "
        f"Water source: {req.water_source}, Rainfall: {req.rain_mm} mm/yr, Peak heat: {req.peak_temp_c}°C, "
        f"Soil: {req.soil} (pH {req.ph}), Current crop: {req.current_crop}, Recommended rotation: {req.recommended_rotation}. "
        f"Farmer question: {req.question}. "
        f"End with one brief reminder in {req.language} to check seed variety and water turn with a local agriculture officer."
    )
    reply, model = await call_gemini([{"text": prompt}])
    return {"reply": reply, "model": model}


@app.post("/api/ai/kissan-voice")
async def kissan_ai_voice(req: KissanVoiceRequest):
    prompt = (
        f"Listen to this farmer's spoken question (Urdu, Punjabi, Sindhi, Pashto, or English). "
        f"Reply ONLY in simple, everyday {req.language} (3 short sentences, zero technical jargon). "
        f"Field telemetry ({req.lat:.3f}, {req.lon:.3f}, {req.area_acres:.1f} acres): "
        f"Water source: {req.water_source}, Rainfall: {req.rain_mm} mm/yr, Peak heat: {req.peak_temp_c}°C, "
        f"Soil: {req.soil} (pH {req.ph}), Current crop: {req.current_crop}, Recommended rotation: {req.recommended_rotation}. "
        f"Answer their spoken question directly and remind them to verify seed variety locally."
    )
    parts = [
        {"text": prompt},
        {"inline_data": {"mime_type": req.mime_type.split(";")[0], "data": req.audio_b64}},
    ]
    reply, model = await call_gemini(parts)
    return {"reply": reply, "model": model}


@app.post("/api/tts")
async def text_to_speech(req: TtsRequest):
    lang_map = {"ur": "ur", "pa": "ur", "sd": "ur", "ps": "ur", "en": "en"}
    tts_lang = lang_map.get(req.lang, "ur")
    clean_text = req.text.replace("*", "").replace("#", "").strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Empty text")
    try:
        buf = io.BytesIO()
        gTTS(text=clean_text, lang=tts_lang, slow=False).write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/mpeg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS error: {exc}")


async def get_appeears_token(client: httpx.AsyncClient) -> Optional[str]:
    user, password = os.getenv("EARTHDATA_USER"), os.getenv("EARTHDATA_PASS")
    if not user or not password:
        return None
    resp = await client.post(f"{APPEEARS_BASE}/login", auth=(user, password), timeout=20.0)
    return resp.json().get("token") if resp.status_code == 200 else None


async def submit_appeears_area_task(
    client: httpx.AsyncClient, token: str, polygon_feature: Dict[str, Any], start_date: str, end_date: str
) -> Dict[str, Any]:
    layer_sets = [
        [{"product": "MOD13Q1.061", "layer": "_250m_16_days_NDVI"}, {"product": "SPL4SMGP.007", "layer": "Geophysical_Data_sm_rootzone"}],
        [{"product": "MOD13Q1.061", "layer": "_250m_16_days_NDVI"}],
    ]
    last_error = ""
    for layers in layer_sets:
        payload = {
            "task_type": "area",
            "task_name": f"soilpulse_{start_date}_{end_date}",
            "params": {
                "dates": [{"startDate": start_date, "endDate": end_date}],
                "layers": layers,
                "output": {"format": {"type": "geotiff"}, "projection": "geographic"},
                "geo": {"type": "FeatureCollection", "features": [polygon_feature]},
            },
        }
        resp = await client.post(f"{APPEEARS_BASE}/task", headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=30.0)
        if resp.status_code in (200, 202):
            return {"status": "submitted", "task_id": resp.json().get("task_id")}
        last_error = f"HTTP {resp.status_code}: {resp.text}"
    return {"status": "appeears_queued_notice", "detail": last_error}


async def query_usda_nrcs_soil(client: httpx.AsyncClient, lat: float, lon: float) -> Optional[Dict[str, Any]]:
    if not (24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0):
        return None
    sql = f"""
    SELECT TOP 1 mu.muname, c.compname, c.drainagecl, ch.sandtotal_r, ch.silttotal_r, ch.claytotal_r, ch.om_r, ch.ph1to1h2o_r
    FROM mapunit mu INNER JOIN component c ON c.mukey = mu.mukey AND c.majcompflag = 'Yes'
    INNER JOIN chorizon ch ON ch.cokey = c.cokey AND ch.hzdept_r <= 15
    WHERE mu.mukey IN (SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT({lon} {lat})'))
    ORDER BY c.comppct_r DESC, ch.hzdept_r ASC
    """
    try:
        resp = await client.post(NRCS_SDA_URL, json={"query": sql, "format": "JSON+COLUMNNAME"}, timeout=10.0)
        table = resp.json().get("Table") if resp.status_code == 200 else None
        if not table or len(table) < 2:
            return None
        row = dict(zip(table[0], table[1]))
        clay, sand = float(row.get("claytotal_r") or 20), float(row.get("sandtotal_r") or 40)
        return {
            "map_unit": row.get("muname"),
            "component_name": row.get("compname"),
            "drainage_class": row.get("drainagecl"),
            "organic_matter_pct": row.get("om_r"),
            "ph": row.get("ph1to1h2o_r"),
            "texture_label": "Clay Loam" if clay >= 27 else "Sandy Loam" if sand >= 52 else "Silt Loam",
        }
    except Exception:
        return None


async def query_cdl_crop_history(client: httpx.AsyncClient, lat: float, lon: float) -> Optional[Dict[str, Any]]:
    if not (24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0):
        return None
    x_alb, y_alb = wgs84_to_epsg5070(lon, lat)
    history = {}
    for yr in (2023, 2022, 2021):
        try:
            resp = await client.get(CDL_POINT_URL, params={"year": yr, "x": round(x_alb, 2), "y": round(y_alb, 2)}, timeout=8.0)
            if resp.status_code == 200 and "<Result>" in resp.text:
                root = ET.fromstring(resp.text)
                res_el = root.find(".//Result")
                if res_el is not None and res_el.text:
                    history[str(yr)] = res_el.text.strip()
        except Exception:
            continue
    return {"history_by_year": history, "recent_crop": next(iter(history.values()))} if history else None


@app.post("/api/field/analyze")
async def analyze_field(req: FieldAnalyzeRequest):
    end_dt = req.end_date or date.today().strftime("%m-%d-%Y")
    start_dt = req.start_date or (date.today() - timedelta(days=365)).strftime("%m-%d-%Y")
    async with httpx.AsyncClient() as client:
        nrcs_soil = await query_usda_nrcs_soil(client, req.lat, req.lon)
        cdl_history = await query_cdl_crop_history(client, req.lat, req.lon)
        try:
            token = await get_appeears_token(client)
            appeears_result = await submit_appeears_area_task(client, token, req.polygon, start_dt, end_dt) if token else {"status": "skipped_no_earthdata_credentials"}
        except Exception as exc:
            appeears_result = {"status": "appeears_error", "detail": str(exc)}
    return {"coordinates": {"lat": req.lat, "lon": req.lon}, "nrcs_soil": nrcs_soil, "cdl_crop_history": cdl_history, "appeears": appeears_result}