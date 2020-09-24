# !/usr/bin/env python3
# -*- coding: utf-8 -*-


def fail(msg: str, code: int = -1) -> dict:
    return {
        "msg": msg,
        "code": code,
        "data": None
    }


def success(data=None, code: int = 0, msg: str = '') -> dict:
    return {
        "msg": msg,
        "code": code,
        "data": data
    }
