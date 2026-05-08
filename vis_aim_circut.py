"""
draw_circuit.py  —  Custom quantum circuit SVG renderer for QAS thesis
=======================================================================
Usage
-----
    python draw_circuit.py --input arch.txt --output circuit.svg
    python draw_circuit.py --input arch.txt --output circuit.svg \
        --n_qubits 6 --wire_spacing 60 --layer_width 160 \
        --title "Final Optimized Circuit (LiH)"
    python draw_circuit.py --input arch.txt --output circuit.pdf  # needs cairosvg

Input format
------------
Layer 0 (idx=2575):
  Rotations: RZ(wire=0), RZ(wire=1), RZ(wire=4)
  CNOTs:     CNOT(2->3), CNOT(5->0)
Layer 1 (idx=2502):
  ...

Output: standalone SVG (or PDF via cairosvg / inkscape).
LaTeX:  \\usepackage{svg}  then  \\includesvg[width=\\linewidth]{circuit}
"""

import re
import argparse
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Layer:
    index: int
    search_idx: Optional[int]
    rotations: list   # [(gate_type, wire), ...]
    cnots: list       # [(ctrl, tgt), ...]


def parse_architecture(text: str) -> list:
    layers = []
    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        m = re.match(r"Layer\s+(\d+)\s*(?:\(idx=(\d+)\))?", line, re.I)
        if m:
            if current is not None:
                layers.append(current)
            idx = int(m.group(1))
            search_idx = int(m.group(2)) if m.group(2) else None
            current = Layer(index=idx, search_idx=search_idx, rotations=[], cnots=[])
            continue

        if current is None:
            continue

        if re.match(r"Rotations\s*[^:]*:", line, re.I):
            gate_part = re.sub(r"^Rotations\s*[^:]*:\s*", "", line, flags=re.I)
            for gm in re.finditer(r"(R[XYZ])\s*\(\s*wire\s*=\s*(\d+)\s*\)", gate_part, re.I):
                current.rotations.append((gm.group(1).upper(), int(gm.group(2))))
            continue

        if re.match(r"CNOTs?\s*:", line, re.I):
            cnot_part = re.sub(r"^CNOTs?\s*:\s*", "", line, flags=re.I)
            for cm in re.finditer(r"CNOT\s*\(\s*(\d+)\s*->\s*(\d+)\s*\)", cnot_part, re.I):
                current.cnots.append((int(cm.group(1)), int(cm.group(2))))
            continue

    if current is not None:
        layers.append(current)

    return layers


def render_svg(
    layers,
    n_qubits,
    wire_spacing=60,
    layer_width=160,
    left_margin=60,
    right_margin=40,
    top_margin=50,
    bottom_margin=45,
    gate_w=44,
    gate_h=26,
    rz_fill="#E6F1FB", rz_stroke="#378ADD", rz_text="#0C447C",
    ry_fill="#EAF3DE", ry_stroke="#639922", ry_text="#27500A",
    rx_fill="#FAEEDA", rx_stroke="#BA7517", rx_text="#633806",
    cnot_color="#2C2C2A",
    wire_color="#888780",
    layer_bg="#F8F8F6",
    layer_stroke="#D3D1C7",
    qubit_label_color="#5F5E5A",
    title="",
    show_layer_idx=True,
    dark_mode_support=True,
):
    n_layers = len(layers)
    total_w = left_margin + n_layers * layer_width + right_margin
    wire_y = [top_margin + i * wire_spacing for i in range(n_qubits)]
    total_h = top_margin + (n_qubits - 1) * wire_spacing + bottom_margin + (30 if title else 0)
    title_offset = 30 if title else 0

    lines = []
    def x(li): return left_margin + li * layer_width
    def emit(s): lines.append(s)

    emit(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="{total_h}" '
         f'viewBox="0 0 {total_w} {total_h}" role="img">')
    emit(f'  <title>{title or "Quantum circuit diagram"}</title>')
    emit(f'  <desc>Quantum circuit with {n_qubits} qubits and {n_layers} layers</desc>')

    emit('  <style>')
    emit('    .wire  { stroke-width: 1.2; fill: none; }')
    emit('    .gate  { stroke-width: 1; }')
    emit('    .gt    { font-family: monospace; font-size: 11px; text-anchor: middle; '
         'dominant-baseline: central; font-weight: 600; }')
    emit('    .ql    { font-family: monospace; font-size: 13px; dominant-baseline: central; text-anchor: middle; }')
    emit('    .ll    { font-family: sans-serif; font-size: 10px; text-anchor: middle; }')
    emit('    .llidx { font-family: monospace; font-size: 9px; text-anchor: middle; }')
    emit('    .tl    { font-family: sans-serif; font-size: 14px; text-anchor: middle; '
         'font-weight: 500; dominant-baseline: central; }')
    emit(f'    .rz-r {{ fill: {rz_fill}; stroke: {rz_stroke}; }}')
    emit(f'    .ry-r {{ fill: {ry_fill}; stroke: {ry_stroke}; }}')
    emit(f'    .rx-r {{ fill: {rx_fill}; stroke: {rx_stroke}; }}')
    emit(f'    .rz-t {{ fill: {rz_text}; }}')
    emit(f'    .ry-t {{ fill: {ry_text}; }}')
    emit(f'    .rx-t {{ fill: {rx_text}; }}')
    if dark_mode_support:
        emit('    @media (prefers-color-scheme: dark) {')
        emit('      .rz-r { fill: #0C447C; stroke: #85B7EB; }')
        emit('      .ry-r { fill: #27500A; stroke: #97C459; }')
        emit('      .rx-r { fill: #633806; stroke: #FAC775; }')
        emit('      .rz-t { fill: #B5D4F4; } .ry-t { fill: #C0DD97; } .rx-t { fill: #FAC775; }')
        emit(f'      .ql, .ll, .llidx, .tl {{ fill: #888780; }}')
        emit(f'      .wire {{ stroke: #5F5E5A; }}')
        emit(f'      .cnot-line {{ stroke: #c2c0b6; }}')
        emit(f'      .cnot-ctrl {{ fill: #c2c0b6; }}')
        emit(f'      .cnot-tgt  {{ stroke: #c2c0b6; fill: none; }}')
        emit(f'      .layer-bg  {{ fill: #2C2C2A; stroke: #444441; }}')
        emit('    }')
    emit('  </style>')

    y_offset = title_offset

    if title:
        emit(f'  <text class="tl" x="{total_w / 2:.1f}" y="20" fill="{qubit_label_color}">{title}</text>')

    # Layer background bands
    band_top = y_offset + top_margin - wire_spacing // 2
    band_h   = (n_qubits - 1) * wire_spacing + wire_spacing
    for i in range(n_layers):
        emit(f'  <rect class="layer-bg" x="{x(i) + 8:.1f}" y="{band_top:.1f}" '
             f'width="{layer_width - 16:.1f}" height="{band_h:.1f}" rx="6" '
             f'fill="{layer_bg}" stroke="{layer_stroke}" stroke-width="0.5"/>')

    # Qubit wires
    wx0, wx1 = left_margin - 20, left_margin + n_layers * layer_width + 20
    for q in range(n_qubits):
        wy = y_offset + wire_y[q]
        emit(f'  <line class="wire" x1="{wx0}" y1="{wy}" x2="{wx1}" y2="{wy}" stroke="{wire_color}"/>')

    # Qubit labels
    for q in range(n_qubits):
        wy = y_offset + wire_y[q]
        emit(f'  <text class="ql" x="{left_margin - 30:.1f}" y="{wy:.1f}" fill="{qubit_label_color}">q{q}</text>')

    # Gates
    CNOT_SPACING = 24   # horizontal pixels between adjacent CNOTs in same layer

    for li, layer in enumerate(layers):
        layer_cx  = x(li) + layer_width / 2
        has_rots  = bool(layer.rotations)
        has_cnots = bool(layer.cnots)
        n_cnots   = len(layer.cnots)

        # ── Layer labels ──────────────────────────────────────────────────────
        label_y = y_offset + wire_y[-1] + wire_spacing // 2 + 16
        emit(f'  <text class="ll" x="{layer_cx:.1f}" y="{label_y:.1f}" '
             f'fill="{qubit_label_color}">Layer {layer.index}</text>')
        if show_layer_idx and layer.search_idx is not None:
            emit(f'  <text class="llidx" x="{layer_cx:.1f}" y="{label_y + 13:.1f}" '
                 f'fill="{wire_color}">idx {layer.search_idx}</text>')

        # ── X-zone allocation ─────────────────────────────────────────────────
        # Rotations: left zone.  CNOTs: right zone, spread horizontally.
        if has_rots and has_cnots:
            rot_zone_cx    = x(li) + layer_width * 0.32
            cnot_group_cx  = x(li) + layer_width * 0.72
        else:
            rot_zone_cx    = layer_cx
            cnot_group_cx  = layer_cx

        # Centre the CNOT group around cnot_group_cx
        cnot_x0 = cnot_group_cx - (n_cnots - 1) * CNOT_SPACING / 2

        # ── Rotation gates ────────────────────────────────────────────────────
        wire_gate_count = {}
        for _, gwire in layer.rotations:
            wire_gate_count[gwire] = wire_gate_count.get(gwire, 0) + 1

        wire_gate_placed = {}
        gate_slot_w = gate_w + 4

        for gtype, gwire in layer.rotations:
            slot        = wire_gate_placed.get(gwire, 0)
            wire_gate_placed[gwire] = slot + 1
            total_slots = wire_gate_count[gwire]

            group_w = total_slots * gate_slot_w + (total_slots - 1) * 2
            gx = rot_zone_cx - group_w / 2 + slot * (gate_slot_w + 2)
            gy = y_offset + wire_y[gwire] - gate_h / 2

            emit(f'  <rect class="gate {gtype.lower()}-r" x="{gx:.1f}" y="{gy:.1f}" '
                 f'width="{gate_w}" height="{gate_h}" rx="4"/>')
            emit(f'  <text class="gt {gtype.lower()}-t" '
                 f'x="{gx + gate_w / 2:.1f}" y="{y_offset + wire_y[gwire]:.1f}">{gtype}</text>')

        # ── CNOT gates — one x-column per CNOT, no overlap ───────────────────
        for ci, (ctrl, tgt) in enumerate(layer.cnots):
            cnot_x = cnot_x0 + ci * CNOT_SPACING
            ctrl_y = y_offset + wire_y[ctrl]
            tgt_y  = y_offset + wire_y[tgt]
            r = 11

            # Vertical spine
            emit(f'  <line class="cnot-line" '
                 f'x1="{cnot_x:.1f}" y1="{min(ctrl_y, tgt_y):.1f}" '
                 f'x2="{cnot_x:.1f}" y2="{max(ctrl_y, tgt_y):.1f}" '
                 f'stroke="{cnot_color}" stroke-width="1.5"/>')
            # Control dot
            emit(f'  <circle class="cnot-ctrl" cx="{cnot_x:.1f}" cy="{ctrl_y:.1f}" '
                 f'r="5" fill="{cnot_color}"/>')
            # Target ⊕
            emit(f'  <circle class="cnot-tgt" cx="{cnot_x:.1f}" cy="{tgt_y:.1f}" r="{r}" '
                 f'fill="none" stroke="{cnot_color}" stroke-width="1.5"/>')
            emit(f'  <line class="cnot-line" '
                 f'x1="{cnot_x - r:.1f}" y1="{tgt_y:.1f}" '
                 f'x2="{cnot_x + r:.1f}" y2="{tgt_y:.1f}" stroke="{cnot_color}" stroke-width="1.5"/>')
            emit(f'  <line class="cnot-line" '
                 f'x1="{cnot_x:.1f}" y1="{tgt_y - r:.1f}" '
                 f'x2="{cnot_x:.1f}" y2="{tgt_y + r:.1f}" stroke="{cnot_color}" stroke-width="1.5"/>')

    emit('</svg>')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description="Render a quantum circuit SVG from a QAS architecture txt.")
    p.add_argument("--input",  "-i", required=True, help="Path to architecture txt")
    p.add_argument("--output", "-o", default="circuit.svg", help="Output .svg or .pdf")
    p.add_argument("--n_qubits",      type=int,   default=None)
    p.add_argument("--title",         type=str,   default="")
    p.add_argument("--wire_spacing",  type=int,   default=60)
    p.add_argument("--layer_width",   type=int,   default=160)
    p.add_argument("--left_margin",   type=int,   default=60)
    p.add_argument("--right_margin",  type=int,   default=40)
    p.add_argument("--top_margin",    type=int,   default=50)
    p.add_argument("--bottom_margin", type=int,   default=45)
    p.add_argument("--gate_w",        type=int,   default=44)
    p.add_argument("--gate_h",        type=int,   default=26)
    p.add_argument("--rz_fill",    default="#E6F1FB")
    p.add_argument("--rz_stroke",  default="#378ADD")
    p.add_argument("--rz_text",    default="#0C447C")
    p.add_argument("--ry_fill",    default="#EAF3DE")
    p.add_argument("--ry_stroke",  default="#639922")
    p.add_argument("--ry_text",    default="#27500A")
    p.add_argument("--rx_fill",    default="#FAEEDA")
    p.add_argument("--rx_stroke",  default="#BA7517")
    p.add_argument("--rx_text",    default="#633806")
    p.add_argument("--cnot_color",    default="#2C2C2A")
    p.add_argument("--wire_color",    default="#888780")
    p.add_argument("--layer_bg",      default="#F8F8F6")
    p.add_argument("--layer_stroke",  default="#D3D1C7")
    p.add_argument("--no_layer_idx",  action="store_true")
    p.add_argument("--no_dark_mode",  action="store_true")
    args = p.parse_args()

    with open(args.input, "r") as f:
        text = f.read()

    layers = parse_architecture(text)
    if not layers:
        print("ERROR: No layers found.")
        return

    if args.n_qubits is None:
        all_wires = set()
        for layer in layers:
            for _, w in layer.rotations: all_wires.add(w)
            for c, t in layer.cnots:    all_wires.add(c); all_wires.add(t)
        n_qubits = max(all_wires) + 1 if all_wires else 1
        print(f"Auto-detected {n_qubits} qubits.")
    else:
        n_qubits = args.n_qubits

    svg = render_svg(
        layers=layers, n_qubits=n_qubits,
        wire_spacing=args.wire_spacing, layer_width=args.layer_width,
        left_margin=args.left_margin, right_margin=args.right_margin,
        top_margin=args.top_margin, bottom_margin=args.bottom_margin,
        gate_w=args.gate_w, gate_h=args.gate_h,
        rz_fill=args.rz_fill, rz_stroke=args.rz_stroke, rz_text=args.rz_text,
        ry_fill=args.ry_fill, ry_stroke=args.ry_stroke, ry_text=args.ry_text,
        rx_fill=args.rx_fill, rx_stroke=args.rx_stroke, rx_text=args.rx_text,
        cnot_color=args.cnot_color, wire_color=args.wire_color,
        layer_bg=args.layer_bg, layer_stroke=args.layer_stroke,
        title=args.title,
        show_layer_idx=not args.no_layer_idx,
        dark_mode_support=not args.no_dark_mode,
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
            with open(out, "w") as f: f.write(svg)
            print(f"SVG written to: {out}")
    else:
        with open(out, "w") as f: f.write(svg)
        print(f"SVG written to: {out}")

    print(f"  Layers: {len(layers)}, Qubits: {n_qubits}")
    for layer in layers:
        print(f"  Layer {layer.index} (idx={layer.search_idx}): "
              f"{len(layer.rotations)} rot(s), {len(layer.cnots)} CNOT(s)")


if __name__ == "__main__":
    main()