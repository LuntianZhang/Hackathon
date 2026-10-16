from pathlib import Path

from PIL import Image, ImageDraw


def contact_sheet(items: list[tuple[str, Image.Image]], output: Path, columns: int = 4) -> None:
    if not items:
        raise ValueError("缩略图清单不能为空")
    cell_width, cell_height = 256, 216
    sheet = Image.new("RGB", (columns * cell_width, ((len(items) + columns - 1) // columns) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, original) in enumerate(items):
        image = original.copy()
        image.thumbnail((cell_width - 8, cell_height - 28))
        left, top = (index % columns) * cell_width, (index // columns) * cell_height
        sheet.paste(image, (left + 4, top + 24))
        draw.text((left + 4, top + 4), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
