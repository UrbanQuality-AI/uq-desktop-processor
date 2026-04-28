"""
HTML templates and embedding glue for standalone or Qt-embedded map pages.
"""

import json
import logging
import re

from PySide6.QtCore import QJsonValue


def sanitize_pydeck_inline_json_html(deck_html: str) -> str:
    """
    Escape ``</script>`` sequences inside the deck JSON block so Chromium does not
    terminate the script tag early (e.g. OSM tags in string data).
    """
    marker = "const jsonInput = "
    if marker not in deck_html:
        return deck_html

    parts = deck_html.split(marker)
    if len(parts) < 2:
        return deck_html

    json_and_rest = parts[1].split("</script>", 1)
    if len(json_and_rest) < 2:
        return deck_html

    json_content = json_and_rest[0]
    rest_of_page = json_and_rest[1]

    safe_json = re.sub(r"</script", r"<\\/script", json_content, flags=re.IGNORECASE)

    return parts[0] + marker + safe_json + "</script>" + rest_of_page


def expose_deck_instance_on_window(deck_html: str) -> str:
    """Make the deck wrapper reachable from ``runJavaScript`` (``const`` is not a window property)."""
    needle = "const deckInstance = createDeck("
    if needle in deck_html:
        return deck_html.replace(needle, "window.cityLensDeck = createDeck(", 1)
    logging.getLogger(__name__).warning(
        "pydeck HTML has no expected deckInstance line; map view may not preserve pan/zoom on layer toggles."
    )
    return deck_html


def inject_mapbox_gl_css(deck_html: str) -> str:
    """Pydeck template loads mapbox-gl.js but not mapbox-gl.css; add it to silence warnings / layout glitches."""
    if "mapbox-gl.css" in deck_html:
        return deck_html
    marker = "mapbox-gl-js/v1.13.0/mapbox-gl.js"
    script_marker_index = deck_html.find(marker)
    if script_marker_index < 0:
        return deck_html
    script_close_index = deck_html.find("</script>", script_marker_index)
    if script_close_index < 0:
        return deck_html
    script_close_index += len("</script>")
    link = '\n    <link rel="stylesheet" ' 'href="https://api.tiles.mapbox.com/mapbox-gl-js/v1.13.0/mapbox-gl.css" />'
    return deck_html[:script_close_index] + link + deck_html[script_close_index:]


def coerce_webengine_js_json(result: object) -> object:
    """QWebEnginePage.runJavaScript often delivers a QJsonValue; normalize to Python types."""
    if isinstance(result, QJsonValue):
        if result.isNull() or result.isUndefined():
            return None
        return result.toVariant()
    return result


def parse_view_state_json(result: object) -> dict[str, object] | None:
    """Parse JS-returned view state string/dict into a plain dict."""
    result = coerce_webengine_js_json(result)
    if isinstance(result, str) and result.strip() not in ("", "null", "undefined"):
        try:
            parsed = json.loads(result)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    if isinstance(result, dict):
        return result
    return None
