"""
DFA String Acceptance Simulator — Automata Theory
A complete interactive simulator for Deterministic Finite Automata (DFA).

Run:
    pip install -r requirements.txt
    streamlit run dfa_simulator_dashboard.py
"""

from collections import defaultdict, deque
import html
import json
import math
import random
import re
import time
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ==============================================================================
# 1. CORE DFA DATA STRUCTURES & SIMULATION ENGINE
# ==============================================================================

class DFA:
    """Represents a Deterministic Finite Automaton (Q, Sigma, delta, q0, F)."""

    def __init__(self, states, alphabet, start_state, accept_states, transitions):
        self.states = [str(s).strip() for s in states if str(s).strip()]
        self.alphabet = [str(a).strip() for a in alphabet if str(a).strip()]
        self.start_state = str(start_state).strip()
        self.accept_states = set(str(s).strip() for s in accept_states if str(s).strip())
        # transitions: dict of (state, symbol) -> next_state
        self.transitions = {
            (str(k[0]).strip(), str(k[1]).strip()): str(v).strip()
            for k, v in transitions.items()
        }

    def validate(self):
        """Validates the formal consistency of the DFA."""
        errors = []
        warnings = []

        if not self.states:
            errors.append("States set Q cannot be empty.")
        if not self.alphabet:
            errors.append("Alphabet Sigma cannot be empty.")
        if self.start_state not in self.states:
            errors.append(f"Start state '{self.start_state}' is not in states set Q.")

        for acc in self.accept_states:
            if acc not in self.states:
                errors.append(f"Accept state '{acc}' is not in states set Q.")

        # Check transition targets
        for (st, sym), nxt in self.transitions.items():
            if st not in self.states:
                errors.append(f"Transition source '{st}' is not in Q.")
            if sym not in self.alphabet:
                errors.append(f"Transition symbol '{sym}' is not in alphabet Sigma.")
            if nxt not in self.states:
                errors.append(f"Transition target '{nxt}' is not in Q.")

        # Check completeness (is delta total?)
        missing = []
        for st in self.states:
            for sym in self.alphabet:
                if (st, sym) not in self.transitions:
                    missing.append((st, sym))
        if missing:
            if len(missing) <= 6:
                missing_str = ", ".join(f"delta({s}, '{a}')" for s, a in missing)
                warnings.append(f"Transition function delta is incomplete. Missing: {missing_str}.")
            else:
                warnings.append(f"DFA is incomplete. {len(missing)} transitions are missing.")

        return errors, warnings

    def complete_with_trap(self, trap_name="q_trap"):
        """Returns a new DFA with all undefined transitions routed to a dead/trap state."""
        new_states = list(self.states)
        needed = any((s, a) not in self.transitions for s in self.states for a in self.alphabet)
        if not needed:
            return self

        if trap_name not in new_states:
            new_states.append(trap_name)

        new_trans = dict(self.transitions)
        for s in new_states:
            for a in self.alphabet:
                if (s, a) not in new_trans:
                    new_trans[(s, a)] = trap_name

        return DFA(new_states, self.alphabet, self.start_state, self.accept_states, new_trans)

    def analyze_states(self):
        """Analyzes states for reachability and dead/trap conditions."""
        # Reachable states from start_state using BFS
        reachable = set([self.start_state]) if self.start_state in self.states else set()
        queue = deque([self.start_state]) if self.start_state in self.states else deque()
        while queue:
            u = queue.popleft()
            for sym in self.alphabet:
                v = self.transitions.get((u, sym))
                if v and v not in reachable and v in self.states:
                    reachable.add(v)
                    queue.append(v)

        # Productive states (states that can reach at least one accept state in F)
        productive = set(s for s in self.accept_states if s in self.states)
        p_queue = deque(list(productive))
        rev = defaultdict(list)
        for (u, sym), v in self.transitions.items():
            rev[v].append(u)
        while p_queue:
            v = p_queue.popleft()
            for u in rev[v]:
                if u not in productive and u in self.states:
                    productive.add(u)
                    p_queue.append(u)

        unreachable = [s for s in self.states if s not in reachable]
        dead = [s for s in self.states if s not in productive]

        return {
            "reachable": sorted(list(reachable)),
            "unreachable": sorted(unreachable),
            "productive": sorted(list(productive)),
            "dead": sorted(dead)
        }

    def simulate(self, input_string):
        """
        Simulates the DFA on the given input string.
        Returns a simulation record containing full step trace and verdict.
        """
        raw_str = "" if input_string is None else str(input_string)
        # Normalize representations of empty string
        if raw_str.strip() in {"ε", "eps", "epsilon", '""', "''"}:
            raw_str = ""

        trace = []
        curr = self.start_state

        # Initial step 0
        trace.append({
            "step": 0,
            "char": "",
            "char_idx": -1,
            "from_state": None,
            "to_state": curr,
            "is_accepting": curr in self.accept_states,
            "rule": "Initial State",
            "explanation": f"Automaton initialized at start state {curr}."
        })

        if len(raw_str) == 0:
            is_acc = curr in self.accept_states
            reason = (
                f"Empty string epsilon processed. Halted at start state {curr}, which IS an accepting state."
                if is_acc else
                f"Empty string epsilon processed. Halted at start state {curr}, which is NOT an accepting state."
            )
            return {
                "input_string": "",
                "characters": [],
                "accepted": is_acc,
                "final_state": curr,
                "error": None,
                "halt_reason": reason,
                "trace": trace,
                "state_path": [curr]
            }

        characters = list(raw_str)
        state_path = [curr]

        for idx, ch in enumerate(characters):
            # Check symbol validity
            if ch not in self.alphabet:
                return {
                    "input_string": raw_str,
                    "characters": characters,
                    "accepted": False,
                    "final_state": curr,
                    "error": f"Symbol '{ch}' at position {idx} is not in the alphabet Sigma = {{{', '.join(self.alphabet)}}}.",
                    "halt_reason": f"Rejected due to invalid alphabet symbol '{ch}' at position {idx}.",
                    "trace": trace,
                    "state_path": state_path
                }

            if (curr, ch) not in self.transitions:
                return {
                    "input_string": raw_str,
                    "characters": characters,
                    "accepted": False,
                    "final_state": curr,
                    "error": f"Missing transition: delta({curr}, '{ch}') is undefined.",
                    "halt_reason": f"Rejected: No transition defined from state {curr} on symbol '{ch}'.",
                    "trace": trace,
                    "state_path": state_path
                }

            nxt = self.transitions[(curr, ch)]
            is_acc = nxt in self.accept_states
            trace.append({
                "step": idx + 1,
                "char": ch,
                "char_idx": idx,
                "from_state": curr,
                "to_state": nxt,
                "is_accepting": is_acc,
                "rule": f"delta({curr}, '{ch}') = {nxt}",
                "explanation": f"Read '{ch}' at index {idx}: transitioned from {curr} -> {nxt} ({'accepting' if is_acc else 'non-accepting'})."
            })
            curr = nxt
            state_path.append(curr)

        accepted = curr in self.accept_states
        if accepted:
            reason = f"String fully processed. Halted in state {curr} in F (Accepting State)."
        else:
            reason = f"String fully processed. Halted in state {curr} not in F (Non-Accepting State)."

        return {
            "input_string": raw_str,
            "characters": characters,
            "accepted": accepted,
            "final_state": curr,
            "error": None,
            "halt_reason": reason,
            "trace": trace,
            "state_path": state_path
        }

    def get_transition_matrix(self):
        """Builds a formatted pandas DataFrame representing the transition table."""
        rows = []
        for s in self.states:
            prefix = ""
            if s == self.start_state:
                prefix += "-> "
            if s in self.accept_states:
                prefix += "* "
            state_label = f"{prefix}{s}".strip()

            row = {"State": state_label}
            for sym in self.alphabet:
                row[sym] = self.transitions.get((s, sym), "—")
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self):
        return {
            "states": self.states,
            "alphabet": self.alphabet,
            "start_state": self.start_state,
            "accept_states": list(self.accept_states),
            "transitions": {f"{s},{a}": n for (s, a), n in self.transitions.items()}
        }

    @classmethod
    def from_dict(cls, data):
        trans = {}
        for key, val in data.get("transitions", {}).items():
            parts = [p.strip() for p in key.split(",", 1)]
            if len(parts) == 2:
                trans[(parts[0], parts[1])] = val
        return cls(
            states=data.get("states", []),
            alphabet=data.get("alphabet", []),
            start_state=data.get("start_state", ""),
            accept_states=data.get("accept_states", []),
            transitions=trans
        )


# ==============================================================================
# 2. PRESET AUTOMATA LIBRARY
# ==============================================================================

PRESETS = {
    "Ends with '01'": {
        "description": "Accepts all binary strings that end with the suffix '01'.",
        "alphabet": ["0", "1"],
        "states": ["q0", "q1", "q2"],
        "start_state": "q0",
        "accept_states": ["q2"],
        "transitions": {
            ("q0", "0"): "q1", ("q0", "1"): "q0",
            ("q1", "0"): "q1", ("q1", "1"): "q2",
            ("q2", "0"): "q1", ("q2", "1"): "q0",
        },
        "sample_input": "1001",
        "test_cases": ["01", "1001", "1101", "1010", "111", "0", ""]
    },
    "Even 0s and Even 1s (Parity)": {
        "description": "Accepts binary strings containing an even number of '0's and an even number of '1's.",
        "alphabet": ["0", "1"],
        "states": ["q_ee", "q_oe", "q_eo", "q_oo"],
        "start_state": "q_ee",
        "accept_states": ["q_ee"],
        "transitions": {
            ("q_ee", "0"): "q_oe", ("q_ee", "1"): "q_eo",
            ("q_oe", "0"): "q_ee", ("q_oe", "1"): "q_oo",
            ("q_eo", "0"): "q_oo", ("q_eo", "1"): "q_ee",
            ("q_oo", "0"): "q_eo", ("q_oo", "1"): "q_oe",
        },
        "sample_input": "0101",
        "test_cases": ["", "00", "11", "0101", "100110", "01", "000", "1110"]
    },
    "Contains Substring 'aba'": {
        "description": "Accepts strings over {a, b} that contain 'aba' anywhere as a contiguous substring.",
        "alphabet": ["a", "b"],
        "states": ["q0", "q1", "q2", "q3"],
        "start_state": "q0",
        "accept_states": ["q3"],
        "transitions": {
            ("q0", "a"): "q1", ("q0", "b"): "q0",
            ("q1", "a"): "q1", ("q1", "b"): "q2",
            ("q2", "a"): "q3", ("q2", "b"): "q0",
            ("q3", "a"): "q3", ("q3", "b"): "q3",
        },
        "sample_input": "babab",
        "test_cases": ["aba", "babab", "aaba", "bbabb", "ab", "a", ""]
    },
    "Binary Divisible by 3": {
        "description": "Accepts binary representations of integers divisible by 3 (q0: rem 0, q1: rem 1, q2: rem 2).",
        "alphabet": ["0", "1"],
        "states": ["q0", "q1", "q2"],
        "start_state": "q0",
        "accept_states": ["q0"],
        "transitions": {
            ("q0", "0"): "q0", ("q0", "1"): "q1",
            ("q1", "0"): "q2", ("q1", "1"): "q0",
            ("q2", "0"): "q1", ("q2", "1"): "q2",
        },
        "sample_input": "110",
        "test_cases": ["0", "11", "110", "1001", "1100", "1", "10", "100", "101"]
    },
    "Starts with 'a' and Ends with 'b'": {
        "description": "Accepts strings that begin with 'a' and terminate with 'b', rejecting any other sequence into a trap state.",
        "alphabet": ["a", "b"],
        "states": ["q0", "q1", "q2", "q_trap"],
        "start_state": "q0",
        "accept_states": ["q2"],
        "transitions": {
            ("q0", "a"): "q1", ("q0", "b"): "q_trap",
            ("q1", "a"): "q1", ("q1", "b"): "q2",
            ("q2", "a"): "q1", ("q2", "b"): "q2",
            ("q_trap", "a"): "q_trap", ("q_trap", "b"): "q_trap",
        },
        "sample_input": "aab",
        "test_cases": ["ab", "aab", "abb", "ababb", "a", "ba", "bb", "aba", ""]
    },
    "Decimal Floating Point Lexer": {
        "description": "Lexical analyzer for decimal numbers with optional integer part, dot '.', and digits 'd'.",
        "alphabet": ["d", "."],
        "states": ["q0", "q1", "q2", "q3", "q_trap"],
        "start_state": "q0",
        "accept_states": ["q1", "q3"],
        "transitions": {
            ("q0", "d"): "q1", ("q0", "."): "q2",
            ("q1", "d"): "q1", ("q1", "."): "q3",
            ("q2", "d"): "q3", ("q2", "."): "q_trap",
            ("q3", "d"): "q3", ("q3", "."): "q_trap",
            ("q_trap", "d"): "q_trap", ("q_trap", "."): "q_trap",
        },
        "sample_input": "d.dd",
        "test_cases": ["d", "dd", ".d", "d.d", "d.dd", ".", "d..d", ""]
    }
}


# ==============================================================================
# 3. TEXT PARSER FOR CUSTOM DFA INPUT
# ==============================================================================

def parse_custom_dfa(states_raw, alphabet_raw, start_raw, accept_raw, transitions_raw):
    """Parses user input strings into a validated DFA instance."""
    def clean_tokens(text):
        if not text:
            return []
        items = [t.strip() for t in re.split(r"[\s,]+", text.strip())]
        return [i for i in items if i]

    states = clean_tokens(states_raw)
    alphabet = clean_tokens(alphabet_raw)
    start_state = start_raw.strip()
    accept_states = clean_tokens(accept_raw)

    transitions = {}
    lines = [line.strip() for line in transitions_raw.splitlines() if line.strip() and not line.startswith("#")]

    for line_no, line in enumerate(lines, 1):
        if "->" in line:
            src_part, tgt = line.split("->", 1)
        elif "→" in line:
            src_part, tgt = line.split("→", 1)
        elif "=" in line:
            src_part, tgt = line.split("=", 1)
        else:
            tokens = line.split()
            if len(tokens) == 3:
                src_part = f"{tokens[0]},{tokens[1]}"
                tgt = tokens[2]
            else:
                raise ValueError(f"Line {line_no}: Unable to parse transition '{line}'. Expected format 'state, symbol -> next_state'.")

        tgt = tgt.strip()
        src_tokens = clean_tokens(src_part)
        if len(src_tokens) < 2:
            raise ValueError(f"Line {line_no}: Expected source state and symbol in '{src_part}'.")

        src_state = src_tokens[0]
        symbols = src_tokens[1:]
        for sym in symbols:
            clean_sym = sym.replace("|", "").strip()
            if clean_sym:
                transitions[(src_state, clean_sym)] = tgt

    return DFA(states, alphabet, start_state, accept_states, transitions)


def format_transitions_for_editor(dfa):
    """Formats transitions into clean multi-line text for the custom input box."""
    lines = []
    groups = defaultdict(list)
    for (src, sym), tgt in sorted(dfa.transitions.items()):
        groups[(src, tgt)].append(sym)

    for (src, tgt), syms in groups.items():
        sym_str = " | ".join(sorted(syms))
        lines.append(f"{src}, {sym_str} -> {tgt}")
    return "\n".join(lines)


# ==============================================================================
# 4. HIGH-RESOLUTION SVG STATE MACHINE DIAGRAM GENERATOR
# ==============================================================================

def generate_dfa_svg(dfa, active_state=None, active_transition=None, layout_mode="Circular"):
    """
    Generates a crisp, responsive SVG diagram of the DFA.
    Highlights the current active state and traversed transition during simulation.
    """
    states = dfa.states
    n = len(states)
    if n == 0:
        return "<svg width='100%' height='200'><text x='50%' y='50%'>No states defined</text></svg>"

    width = 720
    height = 370
    cx = width / 2
    cy = height / 2

    coords = {}
    if layout_mode == "Linear" or n <= 2:
        spacing = (width - 160) / max(1, n - 1) if n > 1 else 0
        start_x = 100
        for i, s in enumerate(states):
            coords[s] = (start_x + i * spacing, cy)
    else:
        radius = min(140 + min(n, 8) * 12, 165)
        for i, s in enumerate(states):
            angle = math.pi + (2 * math.pi * i / n)
            coords[s] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

    grouped_transitions = defaultdict(list)
    for (src, sym), tgt in dfa.transitions.items():
        if src in coords and tgt in coords:
            grouped_transitions[(src, tgt)].append(sym)

    svg_parts = []
    svg_parts.append(f"""<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" style="width: 100%; height: auto; max-height: 420px; font-family: system-ui, -apple-system, sans-serif;">
    <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#64748b" />
        </marker>
        <marker id="arrow-active" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 1 L 9 5 L 0 9 z" fill="#f59e0b" />
        </marker>
        <marker id="arrow-start" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#2563eb" />
        </marker>
        <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
    </defs>
    <rect width="{width}" height="{height}" fill="#f8fafc" rx="10" stroke="#e2e8f0" stroke-width="1.5"/>
    """)

    # 1. Draw Transition Edges
    for (src, tgt), syms in grouped_transitions.items():
        label = ", ".join(sorted(syms))
        p1 = coords[src]
        p2 = coords[tgt]

        is_edge_active = False
        if active_transition:
            if active_transition[0] == src and active_transition[1] == tgt and active_transition[2] in syms:
                is_edge_active = True

        edge_color = "#f59e0b" if is_edge_active else "#64748b"
        stroke_width = "3.5" if is_edge_active else "1.8"
        marker = "url(#arrow-active)" if is_edge_active else "url(#arrow)"

        if src == tgt:
            x, y = p1
            rot_deg = 0 if (layout_mode == "Linear" or n <= 2) else (math.degrees(math.atan2(y - cy, x - cx)) + 90)

            svg_parts.append(f"""
            <g transform="translate({x:.1f}, {y:.1f}) rotate({rot_deg:.1f})">
                <path d="M -12,-22 C -32,-66 32,-66 12,-22" fill="none" stroke="{edge_color}" stroke-width="{stroke_width}" marker-end="{marker}"/>
                <rect x="-16" y="-62" width="{max(32, len(label)*11)}" height="18" rx="4" fill="{('#fef3c7' if is_edge_active else '#ffffff')}" stroke="{edge_color}" stroke-width="1"/>
                <text x="0" y="-50" text-anchor="middle" font-size="11.5" font-weight="bold" fill="{('#b45309' if is_edge_active else '#1e293b')}">{html.escape(label)}</text>
            </g>
            """)
        else:
            x1, y1 = p1
            x2, y2 = p2
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue

            ux = dx / dist
            uy = dy / dist
            nx = -uy
            ny = ux

            has_reverse = (tgt, src) in grouped_transitions
            curve_offset = 26 if has_reverse else (12 if layout_mode == "Circular" else 14)

            r = 26
            start_x = x1 + ux * r
            start_y = y1 + uy * r
            end_x = x2 - ux * (r + 4)
            end_y = y2 - uy * (r + 4)

            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            ctrl_x = mid_x + nx * curve_offset
            ctrl_y = mid_y + ny * curve_offset

            lbl_x = (start_x + 2 * ctrl_x + end_x) / 4
            lbl_y = (start_y + 2 * ctrl_y + end_y) / 4
            lbl_w = max(24, len(label) * 11)

            svg_parts.append(f"""
            <path d="M {start_x:.1f},{start_y:.1f} Q {ctrl_x:.1f},{ctrl_y:.1f} {end_x:.1f},{end_y:.1f}"
                  fill="none" stroke="{edge_color}" stroke-width="{stroke_width}" marker-end="{marker}"/>
            <rect x="{lbl_x - lbl_w/2:.1f}" y="{lbl_y - 9:.1f}" width="{lbl_w}" height="18" rx="4"
                  fill="{('#fef3c7' if is_edge_active else '#ffffff')}" stroke="{edge_color}" stroke-width="1"/>
            <text x="{lbl_x:.1f}" y="{lbl_y + 4:.1f}" text-anchor="middle" font-size="11.5" font-weight="bold"
                  fill="{('#b45309' if is_edge_active else '#1e293b')}">{html.escape(label)}</text>
            """)

    # 2. Draw Start Indicator Arrow
    if dfa.start_state in coords:
        sx, sy = coords[dfa.start_state]
        arrow_len = 45
        in_x = sx - 26 - 4
        src_x = in_x - arrow_len
        svg_parts.append(f"""
        <g>
            <path d="M {src_x:.1f},{sy:.1f} L {in_x:.1f},{sy:.1f}" fill="none" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrow-start)"/>
            <text x="{src_x - 6:.1f}" y="{sy + 4:.1f}" text-anchor="end" font-size="12" font-weight="bold" fill="#2563eb">Start</text>
        </g>
        """)

    # 3. Draw State Nodes
    for s in states:
        x, y = coords[s]
        is_active = (s == active_state)
        is_accept = (s in dfa.accept_states)
        is_trap = ("trap" in s.lower() or "dead" in s.lower())

        node_fill = "#fef3c7" if is_active else ("#f8fafc" if is_trap else "#ffffff")
        stroke_color = "#f59e0b" if is_active else ("#94a3b8" if is_trap else "#2563eb")
        stroke_width = "3.5" if is_active else "2"
        text_color = "#b45309" if is_active else ("#64748b" if is_trap else "#0f172a")

        if is_active:
            svg_parts.append(f"""
            <circle cx="{x:.1f}" cy="{y:.1f}" r="34" fill="none" stroke="#f59e0b" stroke-width="4" opacity="0.4" filter="url(#glow)"/>
            <circle cx="{x:.1f}" cy="{y:.1f}" r="31" fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4 3"/>
            """)

        svg_parts.append(f"""
        <circle cx="{x:.1f}" cy="{y:.1f}" r="26" fill="{node_fill}" stroke="{stroke_color}" stroke-width="{stroke_width}"/>
        """)

        if is_accept:
            svg_parts.append(f"""
            <circle cx="{x:.1f}" cy="{y:.1f}" r="21" fill="none" stroke="{stroke_color}" stroke-width="1.8"/>
            """)

        svg_parts.append(f"""
        <text x="{x:.1f}" y="{y + 5:.1f}" text-anchor="middle" font-size="13.5" font-weight="700" fill="{text_color}">{html.escape(s)}</text>
        """)

        if is_active:
            svg_parts.append(f"""
            <g transform="translate({x:.1f}, {y - 36:.1f})">
                <rect x="-34" y="-12" width="68" height="18" rx="4" fill="#f59e0b"/>
                <text x="0" y="1" text-anchor="middle" font-size="10" font-weight="bold" fill="#ffffff">CURRENT</text>
            </g>
            """)

    svg_parts.append("</svg>")
    return "".join(svg_parts)


# ==============================================================================
# 5. VISUAL TAPE & STATE FLOW HTML RENDERERS (EMOJI-FREE)
# ==============================================================================

def render_tape_html(characters, current_char_idx, is_empty_string=False):
    """Renders a clean visual input tape with pointer indicator."""
    if is_empty_string or not characters:
        return """
        <div style="background: #f8fafc; border: 1.5px dashed #cbd5e1; border-radius: 8px; padding: 0.9rem; text-align: center; margin-bottom: 0.8rem; font-family: system-ui, sans-serif;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem;">Visual Input Tape (w)</div>
            <div style="display: inline-block; padding: 0.35rem 1rem; border-radius: 6px; background: #e2e8f0; color: #1e293b; font-weight: 700; font-family: monospace; font-size: 1.05rem;">
                epsilon (Empty String — Length 0)
            </div>
            <div style="font-size: 0.78rem; color: #64748b; margin-top: 0.3rem;">Evaluation evaluates whether initial state q0 is in F.</div>
        </div>
        """

    cells_html = []
    pointers_html = []
    indices_html = []

    for idx, ch in enumerate(characters):
        if idx < current_char_idx:
            # Consumed
            bg = "#ecfdf5"
            border = "#a7f3d0"
            color = "#047857"
            weight = "600"
            shadow = "none"
            ptr = ""
        elif idx == current_char_idx:
            # Active pointer
            bg = "#2563eb"
            border = "#1d4ed8"
            color = "#ffffff"
            weight = "800"
            shadow = "0 3px 8px rgba(37, 99, 235, 0.3)"
            ptr = """<div style="width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 8px solid #2563eb; margin: 4px auto 0 auto;"></div>"""
        else:
            # Upcoming
            bg = "#ffffff"
            border = "#e2e8f0"
            color = "#64748b"
            weight = "600"
            shadow = "none"
            ptr = ""

        pointers_html.append(f"""
        <div style="width: 44px; text-align: center; height: 14px; display: inline-block; vertical-align: bottom;">
            {ptr}
        </div>
        """)

        cells_html.append(f"""
        <div style="width: 44px; height: 42px; line-height: 40px; text-align: center; background: {bg};
                    border: 2px solid {border}; border-radius: 6px; color: {color}; font-weight: {weight};
                    font-size: 1.15rem; font-family: monospace; box-shadow: {shadow}; display: inline-block;
                    margin: 0 2px;">
            {html.escape(ch)}
        </div>
        """)

        indices_html.append(f"""
        <div style="width: 44px; text-align: center; font-size: 0.72rem; color: #94a3b8; font-weight: 600; display: inline-block; margin-top: 3px;">
            [{idx}]
        </div>
        """)

    all_pointers = "".join(pointers_html)
    all_cells = "".join(cells_html)
    all_indices = "".join(indices_html)

    return f"""
    <div style="background: #f8fafc; border: 1.5px solid #e2e8f0; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.8rem; overflow-x: auto; font-family: system-ui, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
            <span style="font-size: 0.75rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;">Visual Input Tape (w)</span>
            <span style="font-size: 0.78rem; font-weight: 600; color: #2563eb; background: #eff6ff; padding: 2px 8px; border-radius: 4px;">
                Tape Pointer: {current_char_idx if current_char_idx < len(characters) else "Finished"}
            </span>
        </div>
        <div style="white-space: nowrap; padding-bottom: 2px;">
            <div>{all_pointers}</div>
            <div>{all_cells}</div>
            <div>{all_indices}</div>
        </div>
    </div>
    """


def render_state_flow_html(trace, current_step):
    """Renders a clean pipeline of states and transitions."""
    if not trace:
        return ""
    items = []
    for idx, row in enumerate(trace):
        is_cur = (idx == current_step)
        is_past = (idx < current_step)
        st_name = row["to_state"]
        is_acc = row["is_accepting"]

        if is_cur:
            node_bg = "#fef3c7"
            node_border = "#f59e0b"
            node_color = "#b45309"
            node_shadow = "0 0 8px rgba(245, 158, 11, 0.4)"
            badge = "<div style='font-size: 9px; color: #b45309; font-weight: bold;'>ACTIVE</div>"
        elif is_past:
            node_bg = "#ecfdf5"
            node_border = "#10b981"
            node_color = "#047857"
            node_shadow = "none"
            badge = "<div style='font-size: 9px; color: #047857;'>VISITED</div>"
        else:
            node_bg = "#f8fafc"
            node_border = "#cbd5e1"
            node_color = "#64748b"
            node_shadow = "none"
            badge = "<div style='font-size: 9px; color: #94a3b8;'>NEXT</div>"

        acc_tag = " (F)" if is_acc else ""
        node_html = f"""
        <div style="display: inline-block; text-align: center; vertical-align: middle;">
            {badge}
            <div style="min-width: 42px; height: 38px; line-height: 34px; padding: 0 8px; border-radius: 6px;
                        background: {node_bg}; border: 2px solid {node_border}; color: {node_color};
                        font-weight: bold; font-size: 13.5px; box-shadow: {node_shadow}; display: inline-block;">
                {html.escape(st_name)}{acc_tag}
            </div>
        </div>
        """

        if idx > 0:
            char = row["char"]
            arr_color = "#f59e0b" if idx == current_step else ("#10b981" if idx < current_step else "#94a3b8")
            arr_html = f"""
            <div style="display: inline-block; vertical-align: middle; margin: 0 4px; text-align: center;">
                <div style="font-size: 10px; font-weight: bold; color: {arr_color}; background: #fff; padding: 1px 5px; border-radius: 3px; border: 1px solid {arr_color}; margin-bottom: 2px;">
                    '{char}'
                </div>
                <div style="color: {arr_color}; font-size: 14px; line-height: 1;">-></div>
            </div>
            """
            items.append(arr_html)

        items.append(node_html)

    all_flow = "".join(items)
    return f"""
    <div style="background: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-bottom: 0.8rem; overflow-x: auto; white-space: nowrap; font-family: system-ui, sans-serif;">
        <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 6px;">
            Sequential Transition Path (q0 -> q_final)
        </div>
        <div style="display: flex; align-items: center;">
            {all_flow}
        </div>
    </div>
    """


# ==============================================================================
# 6. STREAMLIT APPLICATION INTERFACE
# ==============================================================================

def main():
    st.set_page_config(
        page_title="DFA String Acceptance Simulator",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling injection (clean, professional, emoji-free)
    st.markdown("""
    <style>
    .main {background: #f8fafc;}
    .block-container {padding-top: 1.5rem; max-width: 1240px;}
    .hero {
        padding: 1.4rem 1.8rem;
        border-radius: 10px;
        color: white;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.15);
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; color: white; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.01em;}
    .hero p {margin: 0.35rem 0 0 0; color: #dbeafe; font-size: 0.98rem;}
    .badge-bar {margin-top: 0.7rem; display: flex; gap: 0.5rem; flex-wrap: wrap;}
    .hero-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #ffffff;
    }
    .verdict-box {
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .verdict-accept {
        background: #ecfdf5;
        border-left: 5px solid #10b981;
        color: #065f46;
    }
    .verdict-reject {
        background: #fef2f2;
        border-left: 5px solid #ef4444;
        color: #991b1b;
    }
    .step-banner {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .sim-panel {
        background: #ffffff;
        border: 1.5px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Banner
    st.markdown("""
    <div class="hero">
        <h1>DFA String Acceptance Simulator</h1>
        <p>Formal Languages & Automata Theory — Interactive Step-by-Step String Recognition Engine</p>
        <div class="badge-bar">
            <span class="hero-badge">Deterministic Finite Automaton</span>
            <span class="hero-badge">5-Tuple: (Q, Sigma, delta, q0, F)</span>
            <span class="hero-badge">Tape Head & Vector State Graph</span>
            <span class="hero-badge">Extended Transition Trace</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==========================================================================
    # SIDEBAR: PRESET SELECTION & CUSTOM DFA CONFIGURATION
    # ==========================================================================
    with st.sidebar:
        st.subheader("Automaton Configuration")
        mode = st.radio("Configuration Mode:", ["Preconfigured Examples", "Custom DFA Specification"], index=0)

        if mode == "Preconfigured Examples":
            selected_preset_name = st.selectbox("Select Classic DFA Example:", list(PRESETS.keys()), index=0)
            preset_data = PRESETS[selected_preset_name]
            st.info(f"**Description:** {preset_data['description']}")

            dfa = DFA(
                states=preset_data["states"],
                alphabet=preset_data["alphabet"],
                start_state=preset_data["start_state"],
                accept_states=preset_data["accept_states"],
                transitions=preset_data["transitions"]
            )
            default_test_input = preset_data["sample_input"]

        else:
            st.markdown("Define formal 5-tuple components:")
            c_states = st.text_input("States Q (comma-separated):", value="q0, q1, q2")
            c_alphabet = st.text_input("Alphabet Sigma (comma-separated):", value="0, 1")
            c_start = st.text_input("Start State q0:", value="q0")
            c_accept = st.text_input("Accept States F (comma-separated):", value="q2")
            c_transitions = st.text_area(
                "Transitions delta (format: state, symbol -> next_state):",
                value="q0, 0 -> q1\nq0, 1 -> q0\nq1, 0 -> q1\nq1, 1 -> q2\nq2, 0 -> q1\nq2, 1 -> q0",
                height=150
            )

            try:
                dfa = parse_custom_dfa(c_states, c_alphabet, c_start, c_accept, c_transitions)
                default_test_input = "1001"
            except Exception as e:
                st.error(f"Error parsing DFA: {e}")
                st.stop()

        # DFA Validation & Completeness Check
        errors, warnings = dfa.validate()
        if errors:
            for err in errors:
                st.error(f"Error: {err}")
            st.stop()

        if warnings:
            for w in warnings:
                st.warning(f"Warning: {w}")
            if st.button("Auto-Complete with Trap State (q_trap)"):
                dfa = dfa.complete_with_trap()
                st.rerun()

        st.divider()
        st.subheader("5-Tuple Specification")
        st.markdown(f"""
        - **States (Q):** `{", ".join(dfa.states)}`
        - **Alphabet (Sigma):** `{", ".join(dfa.alphabet)}`
        - **Start State (q0):** `{dfa.start_state}`
        - **Accept States (F):** `{", ".join(dfa.accept_states) if dfa.accept_states else "(None)"}`
        - **Total Transitions (|delta|):** `{len(dfa.transitions)}`
        """)

        layout_choice = st.selectbox("Diagram Layout:", ["Circular", "Linear"], index=0)

    # ==========================================================================
    # MAIN WORKSPACE: STRING INPUT & LIVE SIMULATION (FRONT & CENTER)
    # ==========================================================================
    st.markdown("### String Processing & Interactive Simulation")

    col_input, col_presets = st.columns([3, 2])
    with col_input:
        user_input_str = st.text_input(
            "Enter input string to simulate (w):",
            value=st.session_state.get("current_string", default_test_input),
            help="Type your input string. To test the empty string, leave blank or enter 'epsilon' or 'eps'."
        )
        st.session_state["current_string"] = user_input_str

    with col_presets:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        if mode == "Preconfigured Examples":
            st.caption("Quick test buttons from example:")
            btn_cols = st.columns(min(len(preset_data["test_cases"]), 5))
            for i, test_val in enumerate(preset_data["test_cases"][:5]):
                with btn_cols[i]:
                    label = test_val if test_val != "" else "epsilon"
                    if st.button(label, key=f"btn_quick_{i}", use_container_width=True):
                        st.session_state["current_string"] = test_val
                        st.session_state["step_idx"] = 0
                        st.rerun()

    # Run Simulation Engine
    sim_result = dfa.simulate(user_input_str)
    total_steps = len(sim_result["trace"]) - 1 if sim_result["trace"] else 0

    if "step_idx" not in st.session_state:
        st.session_state["step_idx"] = 0
    st.session_state["step_idx"] = max(0, min(st.session_state["step_idx"], total_steps))
    current_step = st.session_state["step_idx"]

    # Simulation Controls Bar
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_play, ctrl_speed, ctrl_slider = st.columns([1, 1, 1, 1, 1.4, 1.3, 3])
    with ctrl_c1:
        if st.button("Reset", use_container_width=True):
            st.session_state["step_idx"] = 0
            st.rerun()
    with ctrl_c2:
        if st.button("Prev", use_container_width=True, disabled=(current_step == 0)):
            st.session_state["step_idx"] = max(0, current_step - 1)
            st.rerun()
    with ctrl_c3:
        if st.button("Next", use_container_width=True, disabled=(current_step == total_steps)):
            st.session_state["step_idx"] = min(total_steps, current_step + 1)
            st.rerun()
    with ctrl_c4:
        if st.button("End", use_container_width=True):
            st.session_state["step_idx"] = total_steps
            st.rerun()
    with ctrl_play:
        play_btn = st.button("Auto-Play", use_container_width=True, type="primary")
    with ctrl_speed:
        speed_val = st.selectbox("Speed:", ["0.4s (Fast)", "0.8s (Normal)", "1.2s (Slow)"], index=1, label_visibility="collapsed")
        delay = float(speed_val.split("s")[0])
    with ctrl_slider:
        new_step = st.slider(
            "Stepper:",
            min_value=0,
            max_value=total_steps,
            value=current_step,
            key="slider_step"
        )
        if new_step != current_step:
            st.session_state["step_idx"] = new_step
            st.rerun()

    # Handle Live Auto-Play Animation
    if play_btn:
        anim_placeholder = st.empty()
        for s_idx in range(total_steps + 1):
            t_entry = sim_result["trace"][s_idx]
            c_ptr = s_idx if s_idx < len(sim_result["characters"]) else len(sim_result["characters"])
            t_html = render_tape_html(
                sim_result["characters"],
                c_ptr,
                is_empty_string=(len(sim_result["characters"]) == 0)
            )
            f_html = render_state_flow_html(sim_result["trace"], s_idx)
            c_st = t_entry["to_state"]
            is_acc = t_entry["is_accepting"]
            rule_text = f"Step {s_idx} of {total_steps}: {t_entry['rule']}" if s_idx > 0 else f"Step 0 of {total_steps}: Start at {c_st}"
            act_edge = (t_entry["from_state"], t_entry["to_state"], t_entry["char"]) if s_idx > 0 else None
            s_svg = generate_dfa_svg(dfa, active_state=c_st, active_transition=act_edge, layout_mode=layout_choice)

            with anim_placeholder.container():
                components.html(t_html, height=125, scrolling=False)
                components.html(f_html, height=95, scrolling=False)
                st.markdown(f"""
                <div class="step-banner">
                    <div>
                        <span style="font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase;">ANIMATION:</span>
                        <span style="font-size: 1.05rem; margin-left: 0.5rem; color: #1e293b;">{rule_text}</span>
                    </div>
                    <div>
                        <span style="padding: 0.3rem 0.8rem; border-radius: 4px; font-weight: 700; font-size: 0.82rem;
                                     background: {('#ecfdf5' if is_acc else '#f1f5f9')};
                                     color: {('#065f46' if is_acc else '#475569')};
                                     border: 1px solid {('#a7f3d0' if is_acc else '#cbd5e1')};">
                            {('[Accepting State in F]' if is_acc else '[Non-Accepting State not in F]')}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                components.html(s_svg, height=390, scrolling=False)
            time.sleep(delay)
        st.session_state["step_idx"] = total_steps
        st.rerun()

    # Active Step Display
    current_trace_entry = sim_result["trace"][current_step]
    char_pointer = current_step if current_step < len(sim_result["characters"]) else len(sim_result["characters"])

    # 1. Visual Input Tape
    tape_html = render_tape_html(
        sim_result["characters"],
        char_pointer,
        is_empty_string=(len(sim_result["characters"]) == 0)
    )
    components.html(tape_html, height=125, scrolling=False)

    # 2. Sequential State Transition Pathway
    flow_html = render_state_flow_html(sim_result["trace"], current_step)
    components.html(flow_html, height=95, scrolling=False)

    # 3. Two-Column Live Simulation Canvas
    cur_state = current_trace_entry["to_state"]
    is_cur_accepting = current_trace_entry["is_accepting"]

    if current_step == 0:
        spotlight_msg = f"Initial Start State: Automaton begins at {cur_state}."
    else:
        prev_entry = sim_result["trace"][current_step - 1]
        read_char = current_trace_entry["char"]
        spotlight_msg = f"Transition Rule: delta({prev_entry['to_state']}, '{read_char}') -> {cur_state}"

    st.markdown(f"""
    <div class="step-banner">
        <div>
            <span style="font-size: 0.85rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Step {current_step} of {total_steps}:</span>
            <span style="font-size: 1.05rem; margin-left: 0.5rem; color: #1e293b; font-weight: 600;">{spotlight_msg}</span>
        </div>
        <div>
            <span style="padding: 0.3rem 0.8rem; border-radius: 4px; font-weight: 700; font-size: 0.82rem;
                         background: {('#ecfdf5' if is_cur_accepting else '#f1f5f9')};
                         color: {('#065f46' if is_cur_accepting else '#475569')};
                         border: 1px solid {('#a7f3d0' if is_cur_accepting else '#cbd5e1')};">
                {('[Accepting State in F]' if is_cur_accepting else '[Non-Accepting State not in F]')}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    active_edge = None
    if current_step > 0:
        active_edge = (
            current_trace_entry["from_state"],
            current_trace_entry["to_state"],
            current_trace_entry["char"]
        )

    col_graph, col_info = st.columns([3, 2])
    with col_graph:
        svg_code = generate_dfa_svg(
            dfa,
            active_state=cur_state,
            active_transition=active_edge,
            layout_mode=layout_choice
        )
        components.html(svg_code, height=390, scrolling=False)

    with col_info:
        st.markdown("#### State & Decision Status")
        st.markdown(f"""
        - **Active State:** `{cur_state}`
        - **Is Initial State:** `{'Yes' if cur_state == dfa.start_state else 'No'}`
        - **Is Final / Accepting:** `{'Yes (in F)' if cur_state in dfa.accept_states else 'No (not in F)'}`
        - **Remaining Symbols:** `{max(0, len(sim_result['characters']) - current_step)}`
        - **State Sequence:**
        """)
        st.code(" -> ".join(sim_result["state_path"][:current_step + 1]), language="text")

        # Final verdict display upon reaching completion
        if current_step == total_steps:
            if sim_result["accepted"]:
                st.markdown(f"""
                <div class="verdict-box verdict-accept">
                    <h4 style="margin: 0; color: #065f46;">STRING ACCEPTED</h4>
                    <p style="margin: 0.4rem 0 0 0; font-size: 0.92rem;">
                        The input string <code>"{sim_result['input_string']}"</code> belongs to the language <b>L(M)</b>.
                        The computation successfully consumed all input symbols and terminated in accepting state <b>{sim_result['final_state']} in F</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-box verdict-reject">
                    <h4 style="margin: 0; color: #991b1b;">STRING REJECTED</h4>
                    <p style="margin: 0.4rem 0 0 0; font-size: 0.92rem;">
                        The input string <code>"{sim_result['input_string']}"</code> is NOT accepted. {sim_result['halt_reason']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # ==========================================================================
    # DETAILED ANALYSIS TABS (EMOJI-FREE)
    # ==========================================================================
    tab_log, tab_math, tab_batch, tab_docs = st.tabs([
        "Step-by-Step Trace Log",
        "Formal Definition & Extended Trace",
        "Batch Test Suite",
        "Export & Documentation"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: TRANSITION LOG TABLE
    # --------------------------------------------------------------------------
    with tab_log:
        st.subheader("Step-by-Step Transition Log")
        log_rows = []
        for row in sim_result["trace"]:
            log_rows.append({
                "Step": row["step"],
                "Read Symbol": f"'{row['char']}'" if row["char"] else "epsilon (Initial)",
                "From State": row["from_state"] or "—",
                "To State": row["to_state"],
                "Is Final?": "Yes" if row["is_accepting"] else "No",
                "Transition Rule": row["rule"],
                "Explanation": row["explanation"]
            })
        trace_df = pd.DataFrame(log_rows)

        def highlight_active_row(df_row):
            if df_row["Step"] == current_step:
                return ["background-color: #fef3c7; font-weight: bold;"] * len(df_row)
            return [""] * len(df_row)

        st.dataframe(
            trace_df.style.apply(highlight_active_row, axis=1),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------------------------
    # TAB 2: FORMAL MATHEMATICAL TRACE
    # --------------------------------------------------------------------------
    with tab_math:
        st.subheader("Formal Automata Definition & Extended Transition Trace")

        st.markdown("""
        In Automata Theory, a Deterministic Finite Automaton is formally defined as a 5-tuple:
        $$M = (Q, \\Sigma, \\delta, q_0, F)$$
        """)

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown(f"""
            - **Set of States ($Q$):** $\\{{ {', '.join(dfa.states)} \\}}$
            - **Input Alphabet ($\\Sigma$):** $\\{{ {', '.join(dfa.alphabet)} \\}}$
            - **Start State ($q_0$):** ${dfa.start_state}$
            - **Set of Accepting States ($F$):** $\\{{ {', '.join(dfa.accept_states) if dfa.accept_states else '\\emptyset'} \\}}$
            """)

        with f_col2:
            st.markdown("#### Transition Table ($\\delta: Q \\times \\Sigma \\to Q$)")
            matrix_df = dfa.get_transition_matrix()
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
            st.caption("Legend: `->` denotes initial state; `*` denotes accepting state.")

        st.divider()
        st.subheader("State Classification & Reachability Diagnostics")
        state_diag = dfa.analyze_states()
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown(f"**Reachable States ($Q_{{reach}}$):** `{{{', '.join(state_diag['reachable'])}}}`")
            if state_diag["unreachable"]:
                st.warning(f"Unreachable states detected: `{{{', '.join(state_diag['unreachable'])}}}` (cannot be reached from start state q0).")
            else:
                st.success("All states are reachable from the start state q0.")
        with d_col2:
            st.markdown(f"**Productive / Live States ($Q_{{live}}$):** `{{{', '.join(state_diag['productive'])}}}`")
            if state_diag["dead"]:
                st.info(f"Dead / Trap states detected: `{{{', '.join(state_diag['dead'])}}}` (cannot reach any accepting state in F).")
            else:
                st.caption("No dead/trap states detected.")

        st.divider()
        st.subheader("Inductive Derivation with Extended Transition Function $\\hat{\\delta}$")
        st.markdown("""
        The extended transition function $\\hat{\\delta}: Q \\times \\Sigma^* \\to Q$ processes an entire string $w$:
        - **Base Case:** $\\hat{\\delta}(q, \\varepsilon) = q$
        - **Inductive Step:** For string $w = x a$ where $x \\in \\Sigma^*$ and $a \\in \\Sigma$:
          $$\\hat{\\delta}(q, w) = \\delta(\\hat{\\delta}(q, x), a)$$
        """)

        trace_math_lines = []
        trace_math_lines.append(f"\\hat{{\\delta}}({dfa.start_state}, \\varepsilon) = {dfa.start_state}")

        accum_str = ""
        cur_q = dfa.start_state
        for idx, ch in enumerate(sim_result["characters"]):
            if (cur_q, ch) in dfa.transitions:
                nxt_q = dfa.transitions[(cur_q, ch)]
                accum_str += ch
                trace_math_lines.append(
                    f"\\hat{{\\delta}}({dfa.start_state}, \\text{{{accum_str}}}) = "
                    f"\\delta(\\hat{{\\delta}}({dfa.start_state}, \\text{{{accum_str[:-1] if accum_str[:-1] else '\\varepsilon'}}}), '{ch}') = "
                    f"\\delta({cur_q}, '{ch}') = {nxt_q}"
                )
                cur_q = nxt_q
            else:
                trace_math_lines.append(f"\\text{{Transition undefined for }} \\delta({cur_q}, '{ch}')")
                break

        st.markdown("**Step-by-step mathematical derivation:**")
        st.latex("\n\\\\\n".join(trace_math_lines))

        st.markdown(f"""
        **Final Acceptance Condition:**
        $$\\hat{{\\delta}}({dfa.start_state}, \\text{{{sim_result['input_string'] if sim_result['input_string'] else '\\varepsilon'}}}) = {sim_result['final_state']} {'\\in' if sim_result['accepted'] else '\\notin'} F \\implies \\mathbf{{{('ACCEPTED' if sim_result['accepted'] else 'REJECTED')}}}$$
        """)

    # --------------------------------------------------------------------------
    # TAB 3: BATCH TESTING SUITE
    # --------------------------------------------------------------------------
    with tab_batch:
        st.subheader("Batch Testing & Random String Generator")
        st.markdown("Test multiple strings simultaneously against the active DFA.")

        default_batch_text = "\n".join(preset_data["test_cases"]) if mode == "Preconfigured Examples" else "1001\n01\n111\n1010\n0\n"
        b_input = st.text_area("Enter batch test strings (one string per line):", value=default_batch_text, height=130)

        gen_col1, gen_col2 = st.columns([1, 4])
        with gen_col1:
            if st.button("Generate 10 Random Strings", use_container_width=True):
                random_strings = []
                for _ in range(10):
                    length = random.randint(0, 7)
                    rnd_s = "".join(random.choice(dfa.alphabet) for _ in range(length))
                    random_strings.append(rnd_s)
                b_input = "\n".join(random_strings)
                st.session_state["batch_custom_input"] = b_input
                st.rerun()

        raw_lines = b_input.splitlines()
        test_strings = [line.strip() for line in raw_lines]

        results = []
        for s in test_strings:
            res = dfa.simulate(s)
            disp_str = s if s != "" else "epsilon"
            results.append({
                "String (w)": disp_str,
                "Length |w|": len(s),
                "Final State": res["final_state"],
                "Verdict": "ACCEPTED" if res["accepted"] else "REJECTED",
                "State Path": " -> ".join(res["state_path"]),
                "Reason": res["halt_reason"]
            })

        batch_df = pd.DataFrame(results)

        acc_count = sum(1 for r in results if r["Verdict"] == "ACCEPTED")
        rej_count = len(results) - acc_count
        acc_rate = (acc_count / len(results) * 100) if results else 0

        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("Total Tested", len(results))
        bm2.metric("Accepted", acc_count)
        bm3.metric("Rejected", rej_count)
        bm4.metric("Acceptance Rate", f"{acc_rate:.1f}%")

        def color_verdict(val):
            color = "#10b981" if val == "ACCEPTED" else "#ef4444"
            return f"color: {color}; font-weight: bold;"

        styler = batch_df.style
        if hasattr(styler, "map"):
            styled_batch_df = styler.map(color_verdict, subset=["Verdict"])
        else:
            styled_batch_df = styler.applymap(color_verdict, subset=["Verdict"])

        st.dataframe(
            styled_batch_df,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------------------------
    # TAB 4: EXPORT & DOCUMENTATION
    # --------------------------------------------------------------------------
    with tab_docs:
        st.subheader("Export DFA and Simulation Trace")

        down_col1, down_col2, down_col3 = st.columns(3)
        with down_col1:
            trace_report = [
                "=" * 60,
                "DFA STRING ACCEPTANCE SIMULATION REPORT",
                "=" * 60,
                f"Input String: {sim_result['input_string'] if sim_result['input_string'] else 'epsilon'}",
                f"Length: {len(sim_result['characters'])}",
                f"Final State: {sim_result['final_state']}",
                f"Verdict: {'ACCEPTED' if sim_result['accepted'] else 'REJECTED'}",
                f"Reason: {sim_result['halt_reason']}",
                "-" * 60,
                "STEP-BY-STEP TRACE:",
                "-" * 60,
            ]
            for row in sim_result["trace"]:
                trace_report.append(
                    f"Step {row['step']:02d} | Read: {row['char'] or 'eps':3s} | "
                    f"State: {row['to_state']:6s} | Final: {'Yes' if row['is_accepting'] else 'No':3s} | "
                    f"Rule: {row['rule']}"
                )
            trace_report.append("=" * 60)

            st.download_button(
                "Download Trace Report (.txt)",
                data="\n".join(trace_report),
                file_name="dfa_simulation_trace.txt",
                mime="text/plain",
                use_container_width=True
            )

        with down_col2:
            st.download_button(
                "Download DFA Definition (.json)",
                data=json.dumps(dfa.to_dict(), indent=2),
                file_name="dfa_definition.json",
                mime="application/json",
                use_container_width=True
            )

        with down_col3:
            st.download_button(
                "Download Transition Table (.csv)",
                data=dfa.get_transition_matrix().to_csv(index=False),
                file_name="dfa_transition_table.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.divider()
        st.subheader("Automata Theory Reference Guide")
        with st.expander("1. Definition of a Deterministic Finite Automaton (DFA)", expanded=True):
            st.markdown("""
            - **Deterministic:** For every state $q \\in Q$ and input symbol $a \\in \\Sigma$, there is **strictly one** uniquely defined next state $\\delta(q, a)$.
            - **No Epsilon Transitions:** A DFA cannot change states without consuming an input symbol from $\\Sigma$.
            - **Total Transition Function:** The transition function $\\delta$ is a total function mapping every pair $(q, a) \\in Q \\times \\Sigma$ to some state in $Q$.
            """)

        with st.expander("2. Language Acceptance Criteria"):
            st.markdown("""
            A string $w = a_1 a_2 \\dots a_n \\in \\Sigma^*$ is accepted by $M = (Q, \\Sigma, \\delta, q_0, F)$ if and only if there exists a sequence of states $r_0, r_1, \\dots, r_n \\in Q$ such that:
            1. $r_0 = q_0$ (Computation begins in initial state).
            2. $r_{i+1} = \\delta(r_i, a_{i+1})$ for all $0 \\le i < n$ (Every step follows $\\delta$).
            3. $r_n \\in F$ (Computation terminates in an accepting/final state).
            """)

        with st.expander("3. DFA vs. NFA Comparison"):
            st.markdown("""
            | Property | DFA (Deterministic) | NFA (Non-Deterministic) |
            | :--- | :--- | :--- |
            | **Next state for $(q, a)$** | Exactly one: $\\delta(q, a) \\in Q$ | Set of states: $\\delta(q, a) \\subseteq Q$ |
            | **Epsilon transitions** | Not permitted | Allowed |
            | **Processing complexity** | $O(|w|)$ time | $O(|Q|^2 \\cdot |w|)$ time |
            | **Computational power** | Recognizes Regular Languages | Recognizes Regular Languages (Equivalent) |
            | **Conversion** | Baseline representation | Convertible via Powerset Construction ($2^{|Q|}$) |
            """)

    st.divider()
    st.caption("DFA String Acceptance Simulator — Formal Languages & Automata Theory")


if __name__ == "__main__":
    main()
