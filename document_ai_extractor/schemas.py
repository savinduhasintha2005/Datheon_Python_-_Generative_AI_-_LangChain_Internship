from datetime import date
from typing import Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from google.genai import types


class InvoiceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    quantity: float = Field(ge=0)
    unit_price: float = Field(ge=0)
    amount: float = Field(ge=0)

    @field_validator("quantity")
    @classmethod
    def validate_quantity(cls, value):
        if value <= 0:
            raise ValueError("Quantity must be greater than zero.")
        return value


class InvoiceData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    vendor: Optional[str] = None
    customer: Optional[str] = None
    currency: Optional[str] = None

    items: list[InvoiceItem] = Field(default_factory=list)

    subtotal: Optional[float] = Field(default=None, ge=0)
    tax: Optional[float] = Field(default=None, ge=0)
    total: Optional[float] = Field(default=None, ge=0)

    @field_validator("invoice_date")
    @classmethod
    def validate_invoice_date(cls, value):
        if value is not None:
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(
                    "Invoice date must use YYYY-MM-DD format."
                ) from exc

        return value

    @model_validator(mode="after")
    def validate_totals(self):
        if (
            self.subtotal is not None
            and self.tax is not None
            and self.total is not None
            and abs(self.subtotal + self.tax - self.total) > 0.02
        ):
            raise ValueError(
                "Subtotal plus tax must equal total."
            )

        return self


def nullable_string():
    return types.Schema(
        type=types.Type.STRING,
        nullable=True,
    )


def nullable_number():
    return types.Schema(
        type=types.Type.NUMBER,
        nullable=True,
    )


GEMINI_INVOICE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "invoice_number": nullable_string(),
        "invoice_date": nullable_string(),
        "vendor": nullable_string(),
        "customer": nullable_string(),
        "currency": nullable_string(),

        "items": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "description": types.Schema(
                        type=types.Type.STRING
                    ),
                    "quantity": types.Schema(
                        type=types.Type.NUMBER
                    ),
                    "unit_price": types.Schema(
                        type=types.Type.NUMBER
                    ),
                    "amount": types.Schema(
                        type=types.Type.NUMBER
                    ),
                },
                required=[
                    "description",
                    "quantity",
                    "unit_price",
                    "amount",
                ],
            ),
        ),

        "subtotal": nullable_number(),
        "tax": nullable_number(),
        "total": nullable_number(),
    },
    required=[
        "invoice_number",
        "invoice_date",
        "vendor",
        "customer",
        "currency",
        "items",
        "subtotal",
        "tax",
        "total",
    ],
)
