import base64

from fastapi import Cookie, FastAPI, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth import create_session_token, verify_password, verify_session_token
from app.export import series_to_csv, series_to_excel
from app.models import ExportRequest, LoginRequest, SeriesResponse, UploadResponse
from app.pipeline.pipeline import run_pipeline

app = FastAPI(title="Chart Digitizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/login")
def login(request: LoginRequest, response: Response):
    if not verify_password(request.password):
        raise HTTPException(status_code=401, detail="Incorrect password")

    token = create_session_token()
    response.set_cookie("session", token, httponly=True, samesite="none", secure=True, max_age=7 * 24 * 60 * 60)
    return {"status": "ok"}


def _require_session(session: str | None) -> None:
    if session is None or not verify_session_token(session):
        raise HTTPException(status_code=401, detail="Not authenticated")


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile, session: str | None = Cookie(default=None)):
    _require_session(session)

    image_bytes = await file.read()
    try:
        result = run_pipeline(image_bytes)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))

    return UploadResponse(
        chart_type=result.chart_type,
        series=[
            SeriesResponse(
                name=s.name, color_bgr=s.color_bgr, points=s.points, censoring_marks=s.censoring_marks
            )
            for s in result.series
        ],
        overlay_image_base64=base64.b64encode(result.overlay_image_png).decode("ascii"),
        x_axis_calibrated_from_ocr=result.x_axis_calibrated_from_ocr,
        y_axis_calibrated_from_ocr=result.y_axis_calibrated_from_ocr,
        x_reference_points=result.x_reference_points,
        y_reference_points=result.y_reference_points,
    )


@app.post("/api/export/csv")
def export_csv(request: ExportRequest, session: str | None = Cookie(default=None)):
    _require_session(session)
    csv_text = series_to_csv([s.model_dump() for s in request.series])
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=chart_data.csv"},
    )


@app.post("/api/export/excel")
def export_excel(request: ExportRequest, session: str | None = Cookie(default=None)):
    _require_session(session)
    excel_bytes = series_to_excel([s.model_dump() for s in request.series])
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=chart_data.xlsx"},
    )
