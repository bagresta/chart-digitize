from pydantic import BaseModel


class LoginRequest(BaseModel):
    password: str


class SeriesResponse(BaseModel):
    name: str
    color_bgr: tuple[int, int, int]
    points: list[tuple[float, float]]
    censoring_marks: list[tuple[float, float]]


class UploadResponse(BaseModel):
    chart_type: str
    series: list[SeriesResponse]
    overlay_image_base64: str
    x_axis_calibrated_from_ocr: bool
    y_axis_calibrated_from_ocr: bool
    x_reference_points: list[tuple[float, float]]
    y_reference_points: list[tuple[float, float]]


class SeriesInput(BaseModel):
    name: str
    points: list[tuple[float, float]]


class ExportRequest(BaseModel):
    series: list[SeriesInput]
