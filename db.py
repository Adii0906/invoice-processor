"""
DB interface: save_invoice(data), query_invoices(filters) -> dataframe

Search is case-insensitive and matches partial text, works with missing
dates, and supports a combined free-text search across vendor and
description so users don't need to know which field a term is in.
"""
import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "invoices.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT,
            invoice_date TEXT,
            total_amount REAL,
            tax_amount REAL,
            category TEXT,
            description TEXT,
            confidence REAL,
            raw_text TEXT,
            file_path TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_invoice(data: dict, raw_text: str, file_path: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO invoices (vendor_name, invoice_date, total_amount, tax_amount,
                               category, description, confidence, raw_text, file_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["vendor_name"], data.get("invoice_date"), data["total_amount"],
        data.get("tax_amount", 0.0), data["category"], data["description"],
        data.get("confidence", 0.0), raw_text, file_path, datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()


def query_invoices(
    search_text: str = None,
    category: str = None,
    date_from: str = None,
    date_to: str = None,
    amount_min: float = None,
    amount_max: float = None,
    sort_by: str = "created_at",
    sort_desc: bool = True,
) -> pd.DataFrame:
    """
    search_text matches vendor_name OR description, case-insensitively,
    so a user typing "amazon" or "AMAZON" or "office chair" finds results
    regardless of how the original text was cased or which field it's in.
    Rows with a missing invoice_date are still included unless a date
    filter is explicitly applied.
    """
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM invoices WHERE 1=1"
    params = []

    if search_text:
        query += " AND (LOWER(vendor_name) LIKE ? OR LOWER(description) LIKE ?)"
        term = f"%{search_text.lower()}%"
        params.extend([term, term])

    if category:
        query += " AND category = ?"
        params.append(category)

    if date_from:
        query += " AND invoice_date IS NOT NULL AND invoice_date >= ?"
        params.append(date_from)

    if date_to:
        query += " AND invoice_date IS NOT NULL AND invoice_date <= ?"
        params.append(date_to)

    if amount_min is not None:
        query += " AND total_amount >= ?"
        params.append(amount_min)

    if amount_max is not None:
        query += " AND total_amount <= ?"
        params.append(amount_max)

    allowed_sort_cols = {"created_at", "invoice_date", "total_amount", "vendor_name"}
    if sort_by not in allowed_sort_cols:
        sort_by = "created_at"
    query += f" ORDER BY {sort_by} {'DESC' if sort_desc else 'ASC'}"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_categories_in_use() -> list:
    """Returns only categories that actually have saved invoices, so the
    filter dropdown doesn't show empty options."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT DISTINCT category FROM invoices ORDER BY category").fetchall()
    conn.close()
    return [r[0] for r in rows]