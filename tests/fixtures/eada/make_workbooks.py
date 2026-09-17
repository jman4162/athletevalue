"""Regenerate the two synthetic EADA workbooks used by tests/unit/test_eada.py.

The schools and figures are invented. Run from the repository root:

    uvx --with xlwt --with xlsxwriter python tests/fixtures/eada/make_workbooks.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
HEADER = [
    "unitid",
    "institution_name",
    "state_cd",
    "classification_name",
    "ClassificationOther",
    "Sports",
    "PARTIC_MEN",
    "REV_MEN",
    "EXP_MEN",
]
SCHOOL = ("Example State University", "WA", "NCAA Division I-FBS", "")
COLLEGE = ("Sample College", "OR", "NCAA Division I without football", "")
ROWS = [
    [900001, *SCHOOL, "Basketball", 15, 12500000, 9800000],
    [900001, *SCHOOL, "Soccer", 28, 900000, 1200000],
    [900002, *COLLEGE, "Basketball", 14, "", 2100000],
]


def main() -> None:
    import xlsxwriter
    import xlwt

    book = xlwt.Workbook()
    sheet = book.add_sheet("Schools")
    for r, row in enumerate([HEADER, *ROWS]):
        for c, value in enumerate(row):
            sheet.write(r, c, value)
    book.save(str(XLS_PATH))

    workbook = xlsxwriter.Workbook(str(XLSX_PATH))
    sheet = workbook.add_worksheet("schools")
    for r, row in enumerate([HEADER, *ROWS]):
        for c, value in enumerate(row):
            if value != "":
                sheet.write(r, c, value)
    workbook.close()


XLS_PATH = HERE / "Schools.xls"
XLSX_PATH = HERE / "schools.xlsx"

if __name__ == "__main__":
    main()
