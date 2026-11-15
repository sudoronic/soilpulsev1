import os
from datetime import date, timedelta
from typing import Any, Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Field Shift NASA & Soil Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

APPEEARS_BASE = "https://appeears.earthdatacloud.nasa.gov/api"
NRCS_SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"


class FieldAnalyzeRequest(BaseModel):
    polygon: Dict[str, Any]  # GeoJSON Feature from Leaflet.draw
    lat: float
    lon: float
    start_date: Optional[str] = None
    end_date: Optional[str] = None


async def get_appeears_token(client: httpx.AsyncClient) -> Optional[str]:
    """Authenticate to NASA AppEEARS using Earthdata Login credentials in .env."""
    user = os.getenv("EARTHDATA_USER")
    password = os.getenv("EARTHDATA_PASS")
    if not user or not password:
        return None

    resp = await client.post(
        f"{APPEEARS_BASE}/login",
        auth=(user, password),
        timeout=20.0,
    )
    if resp.status_code == 200:
        return resp.json().get("token")
    return None


async def submit_appeears_area_task(
    client: httpx.AsyncClient,
    token: str,
    polygon_feature: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Submit a multi-sensor SMAP + ECOSTRESS + MODIS NDVI polygon task to AppEEARS."""
    payload = {
        "task_type": "area",
        "task_name": f"field_shift_{start_date}_{end_date}",
        "params": {
            "dates": [{"startDate": start_date, "endDate": end_date}],
            "layers": [
                {"product": "MOD13Q1.061", "layer": "_250m_16_days_NDVI"},
                {"product": "SPL4SMGP.007", "layer": "sm_rootzone"},
                {"product": "ECO_L2_LSTE.002", "layer": "LST"},
            ],
            "output": {
                "format": {"type": "geotiff"},
                "projection": "geographic",
            },
            "geo": {
                "type": "FeatureCollection",
                "features": [polygon_feature],
            },
        },
    }

    resp = await client.post(
        f"{APPEEARS_BASE}/task",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=30.0,
    )
    if resp.status_code not in (200, 202):
        raise HTTPException(status_code=502, detail=f"AppEEARS error: {resp.text}")
    return resp.json()


async def query_usda_nrcs_soil(client: httpx.AsyncClient, lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Query USDA-NRCS SSURGO Soil Data Access by point (US coverage)."""
    sql = f"""
    SELECT TOP 1
        mu.muname,
        c.compname,
        c.drainagecl,
        ch.sandtotal_r,
        ch.silttotal_r,
        ch.claytotal_r,
        ch.om_r,
        ch.ph1to1h2o_r
    FROM mapunit mu
    INNER JOIN component c ON c.mukey = mu.mukey AND c.majcompflag = 'Yes'
    INNER JOIN chorizon ch ON ch.cokey = c.cokey AND ch.hzdept_r <= 15
    WHERE mu.mukey IN (
        SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT({lon} {lat})')
    )
    ORDER BY c.comppct_r DESC, ch.hzdept_r ASC
    """
    try:
        resp = await client.post(
            NRCS_SDA_URL,
            json={"query": sql, "format": "JSON+COLUMNNAME"},
            timeout=15.0,
        )
        if resp.status_code != 200:
            return None
        table = resp.json().get("Table")
        if not table or len(table) < 2:
            return None

        cols = table[0]
        row = dict(zip(cols, table[1]))
        clay = float(row.get("claytotal_r") or 20)
        sand = float(row.get("sandtotal_r") or 40)
        texture_label = "Clay Loam" if clay >= 27 else "Sandy Loam" if sand >= 52 else "Silt Loam"

        return {
            "map_unit": row.get("muname"),
            "component_name": row.get("compname"),
            "drainage_class": row.get("drainagecl"),
            "sand_pct": row.get("sandtotal_r"),
            "silt_pct": row.get("silttotal_r"),
            "clay_pct": row.get("claytotal_r"),
            "organic_matter_pct": row.get("om_r"),
            "ph": row.get("ph1to1h2o_r"),
            "texture_label": texture_label,
        }
    except Exception:
        return None


@app.post("/api/field/analyze")
async def analyze_field(req: FieldAnalyzeRequest):
    end_dt = req.end_date or date.today().strftime("%m-%d-%Y")
    start_dt = req.start_date or (date.today() - timedelta(days=365)).strftime("%m-%d-%Y")

    async with httpx.AsyncClient() as client:
        nrcs_soil = await query_usda_nrcs_soil(client, req.lat, req.lon)

        token = await get_appeears_token(client)
        if token:
            task = await submit_appeears_area_task(client, token, req.polygon, start_dt, end_dt)
            appeears_result = {
                "status": "submitted",
                "task_id": task.get("task_id"),
                "poll_url": f"/api/appeears/status/{task.get('task_id')}",
            }
        else:
            appeears_result = {
                "status": "skipped_no_earthdata_credentials",
                "note": "Set EARTHDATA_USER and EARTHDATA_PASS in backend/.env to submit live AppEEARS tasks.",
            }

    return {
        "coordinates": {"lat": req.lat, "lon": req.lon},
        "nrcs_soil": nrcs_soil,
        "appeears": appeears_result,
    }


@app.get("/api/appeears/status/{task_id}")
async def check_appeears_status(task_id: str):
    async with httpx.AsyncClient() as client:
        token = await get_appeears_token(client)
        if not token:
            raise HTTPException(status_code=401, detail="Earthdata credentials not configured.")
        resp = await client.get(
            f"{APPEEARS_BASE}/task/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=20.0,
        )
        return resp.json()