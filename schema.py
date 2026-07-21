"""
Strict schema for invoice extraction.
This is the contract the LLM output must satisfy. No regex fallback —
validation failure triggers the repair loop in extractor.py.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date


class InvoiceExtraction(BaseModel):
    vendor_name: str = Field(..., min_length=1, description="Name of the vendor/business on the invoice")
    invoice_date: Optional[str] = Field(None, description="Invoice date in YYYY-MM-DD format, null if not found")
    total_amount: float = Field(..., ge=0, description="Total amount on the invoice")
    tax_amount: Optional[float] = Field(0.0, ge=0, description="Tax amount, 0 if not present")
    category: str = Field(..., description="One of: Office, IT, Travel, Food, Utilities, Other")
    description: str = Field(..., description="Short description of what was purchased")
    confidence: float = Field(..., ge=0, le=1, description="Model's self-reported confidence 0-1")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        allowed = {"Office", "IT", "Travel", "Food", "Utilities", "Other"}
        if v not in allowed:
            # don't silently coerce with regex-style guessing — raise so repair loop handles it
            raise ValueError(f"category must be one of {allowed}, got '{v}'")
        return v

    @field_validator("invoice_date")
    @classmethod
    def validate_date(cls, v):
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"invoice_date must be YYYY-MM-DD, got '{v}'")
        return v