"""
Bridges PyDeck HTML output to Qt WebEngine (inject data, reload, callbacks).
"""

import logging
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pydeck as pdk
from PySide6.QtCore import QTemporaryDir, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView

from uq_desktop_processor.gui.map_view.constants import DECK_VIEW_STATE_JS
from uq_desktop_processor.gui.map_view.html import (
    expose_deck_instance_on_window,
    inject_mapbox_gl_css,
    parse_view_state_json,
    sanitize_pydeck_inline_json_html,
)


class DeckWebController:
    """
    DeckWebController UI helper class.
    """

    def __init__(self, web: QWebEngineView) -> None:
        """
        Run   init  .

        :param web: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: __init__(web)
            Out: UI/application state updated as intended.
        """
        self._web = web
        self._deck_html_dir = QTemporaryDir()
        if not self._deck_html_dir.isValid():
            logging.getLogger(__name__).warning("QTemporaryDir for map HTML is invalid; using setHtml.")

        self.map_commit_generation = 0
        self.deck_surface_ready = False
        self._deck_expecting_html_load = False
        self._deck_html_gen = 0
        self._deck_finish_queue: deque[int] = deque()

        self._web.loadFinished.connect(self._on_deck_html_load_finished)

    def _on_deck_html_load_finished(self, ok: bool) -> None:
        """
        Run  on deck html load finished.

        :param ok: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _on_deck_html_load_finished(ok)
            Out: UI/application state updated as intended.
        """
        if not self._deck_expecting_html_load or not self._deck_finish_queue:
            return
        finished_gen = self._deck_finish_queue.popleft()
        if finished_gen != self._deck_html_gen:
            return
        self._deck_expecting_html_load = False
        self.deck_surface_ready = bool(ok)
        if not ok:
            logging.getLogger(__name__).warning("WebEngine load failed.")

    def set_deck_page_html(self, deck: pdk.Deck) -> None:
        """
        Run set deck page html.

        :param deck: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: set_deck_page_html(deck)
            Out: UI/application state updated as intended.
        """
        self._deck_html_gen += 1
        self._deck_finish_queue.append(self._deck_html_gen)
        self._deck_expecting_html_load = True
        self.deck_surface_ready = False

        raw = deck.to_html(as_string=True, css_background_color="#0d0d0d")
        raw = sanitize_pydeck_inline_json_html(raw)
        html = expose_deck_instance_on_window(inject_mapbox_gl_css(raw))
        if self._deck_html_dir.isValid():
            path = Path(self._deck_html_dir.path()) / "deck.html"
            path.write_text(html, encoding="utf-8")
            self._web.load(QUrl.fromLocalFile(str(path.resolve())))
        else:
            self._web.setHtml(html, QUrl("https://cdn.jsdelivr.net/"))

    def clear_pending(self) -> None:
        """
        Run clear pending.

        :return: Result of this step or updated UI/application state.

        Example::
            In: clear_pending()
            Out: UI/application state updated as intended.
        """
        self.deck_surface_ready = False
        self._deck_expecting_html_load = False
        self._deck_finish_queue.clear()

    def schedule_reload_preserving_view(
        self,
        *,
        on_view: Callable[[pdk.ViewState | None, int], None],
    ) -> None:
        """
        Run schedule reload preserving view.

        :return: Result of this step or updated UI/application state.

        Example::
            In: schedule_reload_preserving_view()
            Out: UI/application state updated as intended.
        """
        self.map_commit_generation += 1
        gen = self.map_commit_generation
        self._web.page().runJavaScript(
            DECK_VIEW_STATE_JS,
            lambda res, g=gen: self._on_deck_view_captured(res, g, on_view),
        )

    @staticmethod
    def _on_deck_view_captured(
        result: Any,
        scheduled_gen: int,
        on_view: Callable[[pdk.ViewState | None, int], None],
    ) -> None:
        """
        Run  on deck view captured.

        :param result: See caller/context.
        :param scheduled_gen: See caller/context.
        :param on_view: See caller/context.
        :return: Result of this step or updated UI/application state.

        Example::
            In: _on_deck_view_captured(result, scheduled_gen, on_view)
            Out: UI/application state updated as intended.
        """
        view: pdk.ViewState | None = None
        if isinstance(result, dict):
            parsed = parse_view_state_json(result)
            if parsed is not None:
                try:
                    view = pdk.ViewState(
                        latitude=float(cast(Any, parsed["latitude"])),
                        longitude=float(cast(Any, parsed["longitude"])),
                        zoom=float(cast(Any, parsed["zoom"])),
                        pitch=float(cast(Any, parsed.get("pitch", 0))),
                        bearing=float(cast(Any, parsed.get("bearing", 0))),
                    )
                except (KeyError, TypeError, ValueError):
                    view = None
        on_view(view, scheduled_gen)
