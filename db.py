"""
DB interface: save_invoice(data), query_invoices(filters) -> dataframe
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


def query_invoices(vendor: str = None, category: str = None,
                    date_from: str = None, date_to: str = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM invoices WHERE 1=1"
    params = []
    if vendor:
        query += " AND vendor_name LIKE ?"
        params.append(f"%{vendor}%")
    if category:
        query += " AND category = ?"
        params.append(category)
    if date_from:
        query += " AND invoice_date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND invoice_date <= ?"
        params.append(date_to)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df