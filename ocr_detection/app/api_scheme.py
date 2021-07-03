#!/usr/bin/env python3
# -*- coding: utf-8 -*

from pydantic import BaseModel


class ModelInput(BaseModel):
    TransactionAmt: float
    ProductCD: int
    card1: int
    C1: float
    C2: float
    C3: float
    C4: float
    C5: float
    C6: float
    C7: float
    C8: float
    C9: float
    C10: float
    C11: float
    C12: float
    C13: float
    C14: float


class ModelOutput(BaseModel):
    esun_uuid: str  # 玉山 client 傳給開發者的 task id
    server_uuid: str  # 由開發者自行產生的 uuid
    answer: str  # 圖片辨識出的文字，例如: "陳"
    server_timestamp: int  # Inference API 回傳的時間 (Unix Epoch Time)


class HealthCheckOutput(BaseModel):
    health: bool


