"""Converts corrected series data (as sent back from the frontend after
review/editing) into downloadable CSV or Excel files."""
import csv
import io

import openpyxl


def _sanitize_for_spreadsheet(name: str) -> str:
    # Prefix formula-triggering characters with a single quote so
    # spreadsheet apps treat the value as literal text, not a formula.
    # Also normalize newlines so a crafted name can't break CSV row structure.
    sanitized = name.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    if sanitized[:1] in ("=", "+", "-", "@", "\t"):
        sanitized = "'" + sanitized
    return sanitized


def series_to_csv(series: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    # \r\n is the CSV-spec default line terminator, but the pre-existing
    # output (and its tests) used bare \n, so match that instead of
    # introducing \r\n into every row.
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["series", "x", "y"])
    for s in series:
        name = _sanitize_for_spreadsheet(s["name"])
        for x, y in s["points"]:
            writer.writerow([name, x, y])
    return buffer.getvalue()


def series_to_excel(series: list[dict]) -> bytes:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["series", "x", "y"])
    for s in series:
        name = _sanitize_for_spreadsheet(s["name"])
        for x, y in s["points"]:
            sheet.append([name, x, y])

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()
