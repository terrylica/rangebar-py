"""Notification integrations for rangebar-py.

This package provides notification backends for the hooks system.
Currently supported: Telegram.

Usage
-----
>>> from rangebar.notify.telegram import enable_telegram_notifications
>>> enable_telegram_notifications()
>>> # Now all hook events will be sent to Telegram
"""

from __future__ import annotations

__all__: list[str] = []
