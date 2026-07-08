"""Converts corrected series data (as sent back from the frontend after
review/editing) into downloadable CSV or Excel files."""
import io

import openpyxl


def series_to_csv(series: list[dict]) -> str:
    lines = ["series,x,y"]
    for s in series:
        for x, y in s["points"]:
            lines.append(f"{s['name']},{x},{y}")
    return "\n".join(lines) + "\n"


def series_to_excel(series: list[dict]) -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["series", "x", "y"])
    for s in series:
        for x, y in s["points"]:
            sheet.append([s["name"], x, y])

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
