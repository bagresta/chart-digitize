import base64

from fastapi import Cookie, FastAPI, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.auth import create_session_token, verify_password, verify_session_token
from app.export import series_to_csv, series_to_excel
from app.models import ExportRequest, LoginRequest, SeriesResponse, UploadResponse
from app.pipeline.pipeline import AxisCalibrationError, run_pipeline

app = FastAPI(title="Chart Digitizer API")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB - generous for a chart screenshot/photo

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chart-digitize.vercel.app"],
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
async def upload(
    file: UploadFile,
    session: str | None = Cookie(default=None),
    manual_x_min: float | None = Form(default=None),
    manual_x_max: float | None = Form(default=None),
    manual_y_min: float | None = Form(default=None),
    manual_y_max: float | None = Form(default=None),
):
    _require_session(session)

    manual_x_range = (manual_x_min, manual_x_max) if manual_x_min is not None and manual_x_max is not None else None
    manual_y_range = (manual_y_min, manual_y_max) if manual_y_min is not None and manual_y_max is not None else None

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": "Uploaded file exceeds the 10MB size limit.",
            },
        )

    try:
        result = run_pipeline(image_bytes, manual_x_range=manual_x_range, manual_y_range=manual_y_range)
    except AxisCalibrationError:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "axis_calibration_failed",
                "message": "Couldn't read axis labels automatically. Enter axis min/max values to continue.",
            },
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail={"error": "processing_failed", "message": str(error)})

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
