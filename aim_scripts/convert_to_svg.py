import os
import sys
from pathlib import Path
import cairosvg

def convert_folder(input_folder: str) -> None:
    input_path = Path(input_folder)
    output_path = Path(str(input_path) + "_png")
    output_path.mkdir(exist_ok=True)

    svg_files = list(input_path.glob("*.svg"))
    if not svg_files:
        print(f"No SVG files found in {input_path}")
        return

    for svg_file in svg_files:
        output_file = output_path / (svg_file.stem + ".png")
        cairosvg.svg2png(url=str(svg_file), write_to=str(output_file))
        print(f"Converted: {svg_file.name} -> {output_file.name}")

    print(f"\nDone. {len(svg_files)} files saved to {output_path}/")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "circuit_images"
    convert_folder(folder)