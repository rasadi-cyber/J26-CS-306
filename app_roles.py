"""Role colours for the dashboard: poisoned drones red, the rare drone green, regular drones unchanged (neutral).

One place for every role colour, so drone names, role labels and role fields look the same on every page:
  color(role)          hex for the current theme (lighter red / green in dark mode); None for regular / unknown
  span(text, role)     HTML <span> in the role colour (escaped text); plain escaped text when neutral
  md(text, role)       Streamlit Markdown :red[...] / :green[...] (metric and widget labels, captions, headers)
  mark(text)           Markdown prose with the role phrases ("poisoned drones", "rare drone") coloured
  rows(df, roles, ..)  pandas Styler: colour the cells of one row per drone by that drone's role
  columns(df, ..)      pandas Styler: colour whole columns that hold a role's numbers ("Poison share", "Rare share")
  label_color(map)     Altair axis / legend labelColor that colours each listed drone label by its role
  toggle(), server_on()  sidebar switch: may the server-only pages colour drones by their TRUE role?

Only colours change: no text, layout or logic. Role keys are the ones in rounds.jsonl / meta.json
("general", "rare", "poisoned"); display names ("Regular", "Rare (De-auth)", "Poisoned") are accepted too.
"""
import html
import json
import math
import numbers
import re

import altair as alt
import pandas as pd
import streamlit as st

LIGHT = {"poisoned": "#c62828", "rare": "#2e7d32"}
DARK = {"poisoned": "#ef5350", "rare": "#66bb6a"}  # lighter variants: readable on the near-black dark theme
TINT = {False: {"poisoned": "rgba(198,40,40,.07)", "rare": "rgba(46,125,50,.07)"},
        True: {"poisoned": "rgba(239,83,80,.13)", "rare": "rgba(102,187,106,.13)"}}
MD = {"poisoned": "red", "rare": "green"}  # Streamlit's Markdown colour names (metric/widget labels take no hex)
# Streamlit's own chart label colour (axis and legend labels) per theme, kept for the uncoloured labels.
LABEL_DEFAULT = {False: "#808495", True: "#e6eaf1"}
TOGGLE_KEY = "role_colors"
TOGGLE_HELP = ("Poisoned drones in red, the rare drone in green. **On**: every page colours drone names by their true "
               "role (simulation view). **Off**: the server-only pages (Trust scores, Audit log) show drone names "
               "uncoloured, as the real aggregation server would see them: it never knows the true roles. Pages that "
               "already showed the roles (Overview, Swarm monitor, Decisions, Results, Drone Sites) always colour them.")


def dark():
    return bool(st.session_state.get("ui_dark"))


def norm(role):
    """'poisoned' / 'rare' / None (regular or unknown) from a role key or a display name."""
    s = str(role or "").strip().lower()
    if s.startswith("poison"):
        return "poisoned"
    if s.startswith("rare"):
        return "rare"
    return None


def color(role):
    return (DARK if dark() else LIGHT).get(norm(role))


def server_on():
    """True when the server-only pages may colour drones by their true role (sidebar switch, default on)."""
    return bool(st.session_state.get(TOGGLE_KEY, True))


def toggle():
    st.sidebar.toggle("Colour drones by true role (simulation view)", True, key=TOGGLE_KEY, help=TOGGLE_HELP)


# ---------- text ----------
def span(text, role, bold=False):
    c = color(role)
    t = html.escape(str(text))
    if not c:
        return t
    return f'<span style="color:{c}{";font-weight:700" if bold else ""}">{t}</span>'


def md(text, role):
    c = MD.get(norm(role))
    return f":{c}[{text}]" if c else str(text)


_PHRASES = [(re.compile(r"\b[Pp]oisoned(?: drones?(?:'s)?| drone\(s\))?(?!\w)"), "poisoned"),
            (re.compile(r"\b[Rr]are drones?(?:'s)?(?!\w)"), "rare")]


def mark(text):
    """Colour the role phrases in Markdown prose ('the poisoned drones', 'the rare drone') without changing the text."""
    for rx, role in _PHRASES:
        text = rx.sub(lambda m: md(m.group(0), role), text)
    return text


# ---------- tables ----------
def css(role, tint=False, bold=True):
    c = color(role)
    if not c:
        return ""
    out = f"color:{c}" + (";font-weight:600" if bold else "")
    return out + (f";background-color:{TINT[dark()][norm(role)]}" if tint else "")


def _plain(v):
    """Same text Streamlit shows for an unstyled cell (a Styler shows its own text in every cell): integers as is,
    floats with up to 4 decimals and no trailing zeros."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "None"
    if isinstance(v, bool) or not isinstance(v, numbers.Number):
        return str(v)
    if isinstance(v, numbers.Integral):
        return str(int(v))
    return f"{float(v):.4f}".rstrip("0").rstrip(".") or "0"


def styler(data):
    """A Styler for `data`; a plain DataFrame keeps the text Streamlit would show for it."""
    return data.style.format(_plain) if isinstance(data, pd.DataFrame) else data


def rows(data, roles, cols, tint=True):
    """Colour the `cols` cells of each row by that row's role (`roles`: one entry per row, in row order)."""
    sty = styler(data)
    cols = [c for c in cols if c in sty.data.columns]
    roles = list(roles)
    if not cols or not any(norm(r) for r in roles):
        return sty

    def paint(sub):
        return pd.DataFrame([[css(r, tint)] * sub.shape[1] for r in roles[:len(sub)]], index=sub.index,
                            columns=sub.columns)
    return sty.apply(paint, axis=None, subset=cols)


def column_role(name):
    """Role whose numbers a column holds, from its header ('Poison share', 'Rare voted back in', 'Poisoned ...')."""
    return norm(str(name).split()[0]) if str(name).strip() else None


def columns(data, roles_by_col=None):
    """Colour whole columns by role: given {column: role}, or detected from the headers (column_role)."""
    sty = styler(data)
    cols = sty.data.columns
    pairs = (roles_by_col or {c: column_role(c) for c in cols}).items()
    for col, role in pairs:
        if col in cols and norm(role):
            sty = sty.map(lambda _, s=css(role, bold=False): s, subset=[col])
    return sty


# ---------- charts ----------
def label_color(label_roles):
    """labelColor for an Altair axis or legend: each label in {label: role} drawn in its role colour, all other labels
    in Streamlit's usual label colour. Undefined (theme unchanged) when no label has a role colour."""
    red = [str(k) for k, r in label_roles.items() if norm(r) == "poisoned"]
    green = [str(k) for k, r in label_roles.items() if norm(r) == "rare"]
    if not red and not green:
        return alt.Undefined
    pal = DARK if dark() else LIGHT
    expr = (f"indexof({json.dumps(red)}, datum.label) >= 0 ? '{pal['poisoned']}' : "
            f"indexof({json.dumps(green)}, datum.label) >= 0 ? '{pal['rare']}' : '{LABEL_DEFAULT[dark()]}'")
    return alt.ExprRef(expr)
