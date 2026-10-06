"""
Structured private data retrieval query filter model.
Enforces strict Pydantic validation, bounds checking, and type safety.
"""
import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

ALLOWED_STATUSES = {"Processing", "Shipped", "Out for delivery", "Delivered"}
_TIME_REGEX = re.compile(r"^\d{2}:\d{2}$")
_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class OrderQueryFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_name: Optional[str] = None
    customer_id: Optional[int] = None
    product: Optional[str] = None
    status: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        # Check against allowed statuses (case-insensitive mapping to canonical representation)
        for allowed in ALLOWED_STATUSES:
            if v.strip().lower() == allowed.lower():
                return allowed
        raise ValueError(
            f"Invalid status '{v}'. Allowed statuses are: {', '.join(sorted(ALLOWED_STATUSES))}"
        )

    @field_validator("date_from", "date_to")
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not _DATE_REGEX.match(v):
            raise ValueError(f"Invalid date format '{v}'. Expected YYYY-MM-DD.")
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid calendar date '{v}'.")
        return v

    @field_validator("time_from", "time_to")
    @classmethod
    def validate_time(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not _TIME_REGEX.match(v):
            raise ValueError(f"Invalid time format '{v}'. Expected HH:MM in 24-hour format.")
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError(f"Invalid 24-hour time '{v}'.")
        return v

    @model_validator(mode="after")
    def validate_ranges(self) -> "OrderQueryFilter":
        if self.date_from is not None and self.date_to is not None:
            if self.date_from > self.date_to:
                raise ValueError("date_from cannot be after date_to")

        if self.amount_min is not None and self.amount_max is not None:
            if self.amount_min > self.amount_max:
                raise ValueError("amount_min cannot be greater than amount_max")

        if self.time_from is not None and self.time_to is not None:
            if self.time_from > self.time_to:
                raise ValueError(
                    "Cross-midnight time ranges aren't supported yet — please split into two separate queries."
                )

        return self

    def is_empty(self) -> bool:
        fields = [
            self.customer_name,
            self.customer_id,
            self.product,
            self.status,
            self.date_from,
            self.date_to,
            self.time_from,
            self.time_to,
            self.amount_min,
            self.amount_max,
        ]
        return all(v is None for v in fields)
