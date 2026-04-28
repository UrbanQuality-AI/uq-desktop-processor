"""
Map layer IDs, color palettes, and display limits for PyDeck-based map views.
"""

EULER_LINE_COLORS: list[list[int]] = [
    [0, 242, 255],
    [255, 0, 170],
    [0, 255, 136],
    [255, 214, 0],
    [136, 86, 255],
    [255, 128, 0],
    [120, 220, 255],
    [255, 80, 80],
    [180, 255, 120],
    [200, 120, 255],
]

# QListWidget: first row = drawn on top of the map (deck.gl last in stack).
MAP_LAYER_IDS_TOP_FIRST: tuple[str, ...] = ("clip", "points", "euler", "roads")
MAP_LAYER_LABELS: dict[str, str] = {
    "roads": "Road network",
    "points": "Sampling points",
    "euler": "Euler routes (GPX)",
    "clip": "Estimation results (avg)",
}

MAP_REDRAW_DEBOUNCE_MS = 48
ROADS_MAP_SIMPLIFY_METERS_BASE = 12.0
ROADS_MAP_MAX_FEATURES_BASE = 25000
POINTS_MAP_MAX_BASE = 120_000
EULER_VERTEX_BUDGET_BASE = 250_000

# Read live camera before setHtml so toggling layers does not reset pan/zoom.
DECK_VIEW_STATE_JS = r"""
(function() {
  try {
    function normalizeVs(vs) {
      if (!vs) return null;
      if (vs.latitude != null && vs.longitude != null && vs.zoom != null) return vs;
      return null;
    }
    function deepPickVs(obj, depth) {
      if (!obj || depth > 6) return null;
      var n = normalizeVs(obj);
      if (n) return n;
      if (typeof obj !== "object") return null;
      var keys = Object.keys(obj);
      for (var i = 0; i < keys.length; i++) {
        var inner = obj[keys[i]];
        var p = deepPickVs(inner, depth + 1);
        if (p) return p;
      }
      return null;
    }
    var di = (typeof window.cityLensDeck !== "undefined") ? window.cityLensDeck : null;
    if (!di) return null;
    var root = di.deck || di;
    var vs = normalizeVs(root.viewState);
    if (!vs) vs = deepPickVs(root.viewState, 0);
    if (!vs && root.viewManager) {
      var m = root.viewManager;
      vs = normalizeVs(m.viewState) || deepPickVs(m.viewState, 0);
      if (!vs && m.viewStates) vs = deepPickVs(m.viewStates, 0);
      if (!vs && m._viewports && m._viewports.length) {
        var vp = m._viewports[0];
        vs = normalizeVs(vp.viewState) || deepPickVs(vp.viewState, 0);
        if (!vs && vp.viewport) vs = normalizeVs(vp.viewport) || deepPickVs(vp.viewport, 0);
      }
    }
    if (!vs && typeof root.getViewports === "function") {
      try {
        var vps = root.getViewports(root.width && root.height ? [0, 0, root.width, root.height] : undefined);
        if (vps && vps.length) {
          var vp0 = vps[0];
          vs = normalizeVs(vp0) || deepPickVs(vp0, 0);
        }
      } catch (e2) {}
    }
    if (!vs || vs.latitude == null || vs.longitude == null || vs.zoom == null) return null;
    return JSON.stringify({
      latitude: vs.latitude,
      longitude: vs.longitude,
      zoom: vs.zoom,
      pitch: vs.pitch != null ? vs.pitch : 0,
      bearing: vs.bearing != null ? vs.bearing : 0
    });
  } catch (e) { return null; }
})()
"""
