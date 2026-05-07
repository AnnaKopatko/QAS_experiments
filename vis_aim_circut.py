"""
draw_circuit.py  —  Custom quantum circuit SVG renderer for QAS thesis
=======================================================================
Parses the architecture txt file produced by train_search.py and renders
a clean, publication-quality SVG (and optionally PDF via cairosvg).

Usage
-----
    python draw_circuit.py --input arch.txt --output circuit.svg
    python draw_circuit.py --input arch.txt --output circuit.pdf   # needs cairosvg
    python draw_circuit.py --input arch.txt --output circuit.svg \
        --n_qubits 6 --wire_spacing 60 --layer_width 130 \
        --rz_color "#378ADD" --ry_color "#639922" \
        --show_layer_idx --title "Final Optimized Circuit (LiH)"

Input format (one block per layer, any number of layers)
---------------------------------------------------------
Layer 0 (idx=2575):
  Rotations: RZ(wire=0), RZ(wire=1), RZ(wire=4)
  CNOTs:     CNOT(2->3)
Layer 1 (idx=2502):
  ...

Output
------
A standalone SVG file.  Import into LaTeX with:
    \\usepackage{svg}
    \\includesvg[width=\\linewidth]{circuit}
Or convert to PDF:
    inkscape circuit.svg --export-pdf=circuit.pdf
    # or: python draw_circuit.py ... --output circuit.pdf  (requires cairosvg)
"""

import re
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Layer:
    index: int
    search_idx: Optional[int]
    rotations: list   # list of (gate_type, wire)  e.g. ("RZ", 0)
    cnots: list       # list of (ctrl, tgt)         e.g. (2, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_architecture(text: str) -> list[Layer]:
    """Parse the architecture txt produced by train_search.py."""
    layers = []
    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        # ── Layer header ──────────────────────────────────────────────────────
        m = re.match(r"Layer\s+(\d+)\s*(?:\(idx=(\d+)\))?", line, re.I)
        if m:
            if current is not None:
                layers.append(current)
            idx = int(m.group(1))
            search_idx = int(m.group(2)) if m.group(2) else None
            current = Layer(index=idx, search_idx=search_idx,
                            rotations=[], cnots=[])
            continue

        if current is None:
            continue

        # ── Rotations line ────────────────────────────────────────────────────
        if re.match(r"Rotations\s*:", line, re.I):
            gate_part = re.sub(r"^Rotations\s*:\s*", "", line, flags=re.I)
            for gm in re.finditer(r"(R[XYZ])\s*\(\s*wire\s*=\s*(\d+)\s*\)", gate_part, re.I):
                current.rotations.append((gm.group(1).upper(), int(gm.group(2))))
            continue

        # ── CNOTs line ────────────────────────────────────────────────────────
        if re.match(r"CNOTs?\s*:", line, re.I):
            cnot_part = re.sub(r"^CNOTs?\s*:\s*", "", line, flags=re.I)
            for cm in re.finditer(r"CNOT\s*\(\s*(\d+)\s*->\s*(\d+)\s*\)", cnot_part, re.I):
                current.cnots.append((int(cm.group(1)), int(cm.group(2))))
            continue

    if current is not None:
        layers.append(current)

    return layers


# ──────────────────────────────────────────────────────────────────────────────
# SVG renderer
# ──────────────────────────────────────────────────────────────────────────────

def render_svg(
    layers: list[Layer],
    n_qubits: int,
    # geometry
    wire_spacing: int = 60,
    layer_width: int = 130,
    left_margin: int = 60,
    right_margin: int = 40,
    top_margin: int = 50,
    bottom_margin: int = 70,
    gate_w: int = 44,
    gate_h: int = 26,
    # colours (light-mode defaults; dark-mode via CSS media query)
    rz_fill: str = "#E6F1FB",
    rz_stroke: str = "#378ADD",
    rz_text: str = "#0C447C",
    ry_fill: str = "#EAF3DE",
    ry_stroke: str = "#639922",
    ry_text: str = "#27500A",
    rx_fill: str = "#FAEEDA",
    rx_stroke: str = "#BA7517",
    rx_text: str = "#633806",
    cnot_color: str = "#2C2C2A",
    wire_color: str = "#888780",
    layer_bg: str = "#F8F8F6",
    layer_stroke: str = "#D3D1C7",
    qubit_label_color: str = "#5F5E5A",
    # options
    title: str = "",
    show_layer_idx: bool = True,
    dark_mode_support: bool = True,
) -> str:

    n_layers = len(layers)
    total_w = left_margin + n_layers * layer_width + right_margin
    wire_y = [top_margin + i * wire_spacing for i in range(n_qubits)]
    total_h = top_margin + (n_qubits - 1) * wire_spacing + bottom_margin + (30 if title else 0)

    title_offset = 30 if title else 0

    lines = []

    def x(line_idx: float) -> float:
        return left_margin + line_idx * layer_width

    def emit(s): lines.append(s)

    # ── SVG header ────────────────────────────────────────────────────────────
    emit(f'<svg xmlns="http://www.w3.org/2000/svg" '
         f'width="{total_w}" height="{total_h}" '
         f'viewBox="0 0 {total_w} {total_h}" '
         f'role="img">')
    emit(f'  <title>{title or "Quantum circuit diagram"}</title>')
    emit(f'  <desc>Quantum circuit with {n_qubits} qubits and {n_layers} layers</desc>')

    # ── Styles ────────────────────────────────────────────────────────────────
    emit('  <style>')
    emit('    .wire  { stroke-width: 1.2; fill: none; }')
    emit('    .gate  { rx: 4; stroke-width: 1; }')
    emit('    .gt    { font-family: monospace; font-size: 11px; '
         'text-anchor: middle; dominant-baseline: central; font-weight: 600; }')
    emit('    .ql    { font-family: monospace; font-size: 13px; '
         'dominant-baseline: central; text-anchor: middle; }')
    emit('    .ll    { font-family: sans-serif; font-size: 10px; text-anchor: middle; }')
    emit('    .llidx { font-family: monospace; font-size: 9px; text-anchor: middle; }')
    emit('    .tl    { font-family: sans-serif; font-size: 14px; '
         'text-anchor: middle; font-weight: 500; dominant-baseline: central; }')
    emit(f'    .rz-r  {{ fill: {rz_fill}; stroke: {rz_stroke}; }}')
    emit(f'    .ry-r  {{ fill: {ry_fill}; stroke: {ry_stroke}; }}')
    emit(f'    .rx-r  {{ fill: {rx_fill}; stroke: {rx_stroke}; }}')
    emit(f'    .rz-t  {{ fill: {rz_text}; }}')
    emit(f'    .ry-t  {{ fill: {ry_text}; }}')
    emit(f'    .rx-t  {{ fill: {rx_text}; }}')
    if dark_mode_support:
        emit('    @media (prefers-color-scheme: dark) {')
        emit('      .rz-r { fill: #0C447C; stroke: #85B7EB; }')
        emit('      .ry-r { fill: #27500A; stroke: #97C459; }')
        emit('      .rx-r { fill: #633806; stroke: #FAC775; }')
        emit('      .rz-t { fill: #B5D4F4; }')
        emit('      .ry-t { fill: #C0DD97; }')
        emit('      .rx-t { fill: #FAC775; }')
        emit(f'      .ql   {{ fill: #888780; }}')
        emit(f'      .ll   {{ fill: #888780; }}')
        emit(f'      .llidx{{ fill: #5F5E5A; }}')
        emit(f'      .tl   {{ fill: #c2c0b6; }}')
        emit(f'      .wire {{ stroke: #5F5E5A; }}')
        emit(f'      .cnot-line {{ stroke: #c2c0b6; }}')
        emit(f'      .cnot-ctrl {{ fill: #c2c0b6; }}')
        emit(f'      .cnot-tgt  {{ stroke: #c2c0b6; fill: none; }}')
        emit(f'      .layer-bg  {{ fill: #2C2C2A; stroke: #444441; }}')
        emit('    }')
    emit('  </style>')

    # ── Arrow marker (not used but kept for compatibility) ────────────────────
    emit('  <defs>')
    emit('    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" '
         'markerWidth="5" markerHeight="5" orient="auto-start-reverse">')
    emit('      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" '
         'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>')
    emit('    </marker>')
    emit('  </defs>')

    y_offset = title_offset

    # ── Optional title ────────────────────────────────────────────────────────
    if title:
        cx = total_w / 2
        emit(f'  <text class="tl" x="{cx:.1f}" y="20" '
             f'fill="{qubit_label_color}">{title}</text>')

    # ── Layer background bands ────────────────────────────────────────────────
    band_top = y_offset + top_margin - wire_spacing // 2
    band_h = (n_qubits - 1) * wire_spacing + wire_spacing
    for i in range(n_layers):
        bx = x(i) + 8
        bw = layer_width - 16
        emit(f'  <rect class="layer-bg" x="{bx:.1f}" y="{band_top:.1f}" '
             f'width="{bw:.1f}" height="{band_h:.1f}" rx="6" '
             f'fill="{layer_bg}" stroke="{layer_stroke}" stroke-width="0.5"/>')

    # ── Qubit wires ───────────────────────────────────────────────────────────
    wire_x_start = left_margin - 20
    wire_x_end = left_margin + n_layers * layer_width + 20
    for q in range(n_qubits):
        wy = y_offset + wire_y[q]
        emit(f'  <line class="wire" x1="{wire_x_start}" y1="{wy}" '
             f'x2="{wire_x_end}" y2="{wy}" stroke="{wire_color}"/>')

    # ── Qubit labels ──────────────────────────────────────────────────────────
    for q in range(n_qubits):
        wy = y_offset + wire_y[q]
        emit(f'  <text class="ql" x="{left_margin - 30:.1f}" y="{wy:.1f}" '
             f'fill="{qubit_label_color}">q{q}</text>')

    # ── Gates per layer ───────────────────────────────────────────────────────
    for li, layer in enumerate(layers):
        layer_cx = x(li) + layer_width / 2

        # Layer label below
        label_y = y_offset + wire_y[-1] + wire_spacing // 2 + 16
        emit(f'  <text class="ll" x="{layer_cx:.1f}" y="{label_y:.1f}" '
             f'fill="{qubit_label_color}">Layer {layer.index}</text>')
        if show_layer_idx and layer.search_idx is not None:
            emit(f'  <text class="llidx" x="{layer_cx:.1f}" y="{label_y + 13:.1f}" '
                 f'fill="{wire_color}">idx {layer.search_idx}</text>')

        # --- Rotation gates --------------------------------------------------
        # Each layer is split into two zones:
        #   rot_zone_cx  = left 55% of the layer band  (rotation gates)
        #   cnot_zone_cx = right 45% of the layer band (CNOT gates)
        # When a layer has only rotations or only CNOTs, the element is
        # centred in the full band instead.
        has_rotations = bool(layer.rotations)
        has_cnots     = bool(layer.cnots)

        if has_rotations and has_cnots:
            rot_zone_cx  = x(li) + layer_width * 0.38
            cnot_zone_cx = x(li) + layer_width * 0.72
        else:
            rot_zone_cx  = layer_cx
            cnot_zone_cx = layer_cx

        wire_gate_count: dict[int, int] = {}
        for gtype, gwire in layer.rotations:
            wire_gate_count[gwire] = wire_gate_count.get(gwire, 0) + 1

        wire_gate_placed: dict[int, int] = {}
        gate_slot_w = gate_w + 4

        for gtype, gwire in layer.rotations:
            slot = wire_gate_placed.get(gwire, 0)
            wire_gate_placed[gwire] = slot + 1
            total_slots = wire_gate_count[gwire]

            group_w = total_slots * gate_slot_w + (total_slots - 1) * 2
            start_x = rot_zone_cx - group_w / 2 + slot * (gate_slot_w + 2)

            gx = start_x
            gy = y_offset + wire_y[gwire] - gate_h / 2

            css_r = f"{gtype.lower()}-r"
            css_t = f"{gtype.lower()}-t"

            emit(f'  <rect class="gate {css_r}" '
                 f'x="{gx:.1f}" y="{gy:.1f}" '
                 f'width="{gate_w}" height="{gate_h}" rx="4"/>')
            emit(f'  <text class="gt {css_t}" '
                 f'x="{gx + gate_w/2:.1f}" y="{y_offset + wire_y[gwire]:.1f}">'
                 f'{gtype}</text>')

        # --- CNOT gates -------------------------------------------------------
        for ctrl, tgt in layer.cnots:
            cnot_x = cnot_zone_cx

            ctrl_y = y_offset + wire_y[ctrl]
            tgt_y  = y_offset + wire_y[tgt]

            # Vertical line
            line_y1 = min(ctrl_y, tgt_y)
            line_y2 = max(ctrl_y, tgt_y)
            emit(f'  <line class="cnot-line" '
                 f'x1="{cnot_x:.1f}" y1="{line_y1:.1f}" '
                 f'x2="{cnot_x:.1f}" y2="{line_y2:.1f}" '
                 f'stroke="{cnot_color}" stroke-width="1.5"/>')

            # Control dot
            emit(f'  <circle class="cnot-ctrl" '
                 f'cx="{cnot_x:.1f}" cy="{ctrl_y:.1f}" r="5" fill="{cnot_color}"/>')

            # Target circle with cross
            r = 11
            emit(f'  <circle class="cnot-tgt" '
                 f'cx="{cnot_x:.1f}" cy="{tgt_y:.1f}" r="{r}" '
                 f'fill="none" stroke="{cnot_color}" stroke-width="1.5"/>')
            emit(f'  <line class="cnot-line" '
                 f'x1="{cnot_x - r:.1f}" y1="{tgt_y:.1f}" '
                 f'x2="{cnot_x + r:.1f}" y2="{tgt_y:.1f}" '
                 f'stroke="{cnot_color}" stroke-width="1.5"/>')
            emit(f'  <line class="cnot-line" '
                 f'x1="{cnot_x:.1f}" y1="{tgt_y - r:.1f}" '
                 f'x2="{cnot_x:.1f}" y2="{tgt_y + r:.1f}" '
                 f'stroke="{cnot_color}" stroke-width="1.5"/>')

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_y = y_offset + wire_y[-1] + wire_spacing // 2 + 42
    lx = left_margin
    items = [
        ("RZ", "rz-r", "rz-t"),
        ("RY", "ry-r", "ry-t"),
        ("RX", "rx-r", "rx-t"),
    ]
    for label, rcls, tcls in items:
        emit(f'  <rect class="gate {rcls}" x="{lx:.1f}" y="{legend_y - 8:.1f}" '
             f'width="32" height="18" rx="3"/>')
        emit(f'  <text class="gt {tcls}" x="{lx + 16:.1f}" y="{legend_y + 1:.1f}">'
             f'{label}</text>')
        emit(f'  <text class="ll" x="{lx + 40:.1f}" y="{legend_y + 1:.1f}" '
             f'dominant-baseline="central" fill="{qubit_label_color}">'
             f'rotation gate</text>')
        lx += 140

    # CNOT legend
    emit(f'  <circle cx="{lx + 5:.1f}" cy="{legend_y - 2:.1f}" r="4" fill="{cnot_color}"/>')
    emit(f'  <line x1="{lx + 5:.1f}" y1="{legend_y + 2:.1f}" '
         f'x2="{lx + 5:.1f}" y2="{legend_y + 12:.1f}" '
         f'stroke="{cnot_color}" stroke-width="1.5"/>')
    emit(f'  <circle cx="{lx + 5:.1f}" cy="{legend_y + 18:.1f}" r="7" '
         f'fill="none" stroke="{cnot_color}" stroke-width="1.5"/>')
    emit(f'  <text class="ll" x="{lx + 18:.1f}" y="{legend_y + 1:.1f}" '
         f'dominant-baseline="central" fill="{qubit_label_color}">CNOT</text>')

    emit('</svg>')
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Render a quantum circuit SVG from the QAS architecture txt file."
    )
    p.add_argument("--input",  "-i", required=True,
                   help="Path to architecture txt (e.g. arch.txt)")
    p.add_argument("--output", "-o", default="circuit.svg",
                   help="Output path (.svg or .pdf)")
    p.add_argument("--n_qubits", type=int, default=None,
                   help="Number of qubits (auto-detected if omitted)")
    p.add_argument("--title", type=str, default="",
                   help='Title shown above the circuit, e.g. "Final circuit (LiH)"')
    p.add_argument("--wire_spacing",  type=int, default=60)
    p.add_argument("--layer_width",   type=int, default=130)
    p.add_argument("--left_margin",   type=int, default=60)
    p.add_argument("--right_margin",  type=int, default=40)
    p.add_argument("--top_margin",    type=int, default=50)
    p.add_argument("--bottom_margin", type=int, default=70)
    p.add_argument("--gate_w", type=int, default=44)
    p.add_argument("--gate_h", type=int, default=26)
    p.add_argument("--rz_fill",   default="#E6F1FB")
    p.add_argument("--rz_stroke", default="#378ADD")
    p.add_argument("--rz_text",   default="#0C447C")
    p.add_argument("--ry_fill",   default="#EAF3DE")
    p.add_argument("--ry_stroke", default="#639922")
    p.add_argument("--ry_text",   default="#27500A")
    p.add_argument("--rx_fill",   default="#FAEEDA")
    p.add_argument("--rx_stroke", default="#BA7517")
    p.add_argument("--rx_text",   default="#633806")
    p.add_argument("--cnot_color",  default="#2C2C2A")
    p.add_argument("--wire_color",  default="#888780")
    p.add_argument("--layer_bg",    default="#F8F8F6")
    p.add_argument("--layer_stroke",default="#D3D1C7")
    p.add_argument("--show_layer_idx", action="store_true", default=True)
    p.add_argument("--no_layer_idx",   action="store_true",
                   help="Hide layer search-space index labels")
    p.add_argument("--no_dark_mode",   action="store_true",
                   help="Omit dark-mode CSS media query (for PDF export)")
    args = p.parse_args()

    with open(args.input, "r") as f:
        text = f.read()

    layers = parse_architecture(text)
    if not layers:
        print("ERROR: No layers found. Check your input file format.")
        return

    # Auto-detect n_qubits
    if args.n_qubits is None:
        all_wires = set()
        for l in layers:
            for _, w in l.rotations:
                all_wires.add(w)
            for ctrl, tgt in l.cnots:
                all_wires.add(ctrl); all_wires.add(tgt)
        n_qubits = max(all_wires) + 1 if all_wires else 1
        print(f"Auto-detected {n_qubits} qubits.")
    else:
        n_qubits = args.n_qubits

    show_idx = args.show_layer_idx and not args.no_layer_idx
    dark = not args.no_dark_mode

    svg = render_svg(
        layers=layers,
        n_qubits=n_qubits,
        wire_spacing=args.wire_spacing,
        layer_width=args.layer_width,
        left_margin=args.left_margin,
        right_margin=args.right_margin,
        top_margin=args.top_margin,
        bottom_margin=args.bottom_margin,
        gate_w=args.gate_w,
        gate_h=args.gate_h,
        rz_fill=args.rz_fill,
        rz_stroke=args.rz_stroke,
        rz_text=args.rz_text,
        ry_fill=args.ry_fill,
        ry_stroke=args.ry_stroke,
        ry_text=args.ry_text,
        rx_fill=args.rx_fill,
        rx_stroke=args.rx_stroke,
        rx_text=args.rx_text,
        cnot_color=args.cnot_color,
        wire_color=args.wire_color,
        layer_bg=args.layer_bg,
        layer_stroke=args.layer_stroke,
        title=args.title,
        show_layer_idx=show_idx,
        dark_mode_support=dark,
    )

    out = args.output
    if out.endswith(".pdf"):
        try:
            import cairosvg
            cairosvg.svg2pdf(bytestring=svg.encode(), write_to=out)
            print(f"PDF written to: {out}")
        except ImportError:
            print("cairosvg not found — writing SVG instead.")
            out = out.replace(".pdf", ".svg")
            with open(out, "w") as f:
                f.write(svg)
            print(f"SVG written to: {out}")
    else:
        with open(out, "w") as f:
            f.write(svg)
        print(f"SVG written to: {out}")

    print(f"  Layers : {len(layers)}")
    print(f"  Qubits : {n_qubits}")
    for l in layers:
        print(f"  Layer {l.index} (idx={l.search_idx}): "
              f"{len(l.rotations)} rotation(s), {len(l.cnots)} CNOT(s)")


if __name__ == "__main__":
    main()