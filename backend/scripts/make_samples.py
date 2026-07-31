"""Generate the synthetic sample documents committed under samples/.

Produces three fictional, PII-free receipt/invoice images with PIL,
varied in vendor, layout, and legibility so extraction confidence varies
across them (per T7):

  1. coffee-receipt.png    -- clean, simple, single line item.
  2. hardware-invoice.png  -- clean, several line items.
  3. grocery-receipt-skewed.png -- gray background, smaller font, slight
     rotation -- a rougher "phone photo of a receipt" case.

This is a one-off generator, not part of the pytest suite (no DB/network
calls). Run it from `backend/` with the venv active:

    cd backend
    python scripts/make_samples.py

Requires pillow (dev dependency, see pyproject.toml
`[project.optional-dependencies].dev`). Adapted from the image-drawing
approach in scripts/smoke.py.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# samples/ lives at the repo root, two levels up from backend/scripts/.
SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "samples"


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Prefer a real TTF for legibility; fall back to PIL's bitmap default
    font if none is found (keeps the script portable off Windows)."""
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[tuple[str, ImageFont.ImageFont | ImageFont.FreeTypeFont]],
    x: int,
    y: int,
    line_height: int,
    fill: str = "black",
) -> None:
    for text, font in lines:
        draw.text((x, y), text, fill=fill, font=font)
        y += line_height


def make_coffee_receipt(path: Path) -> None:
    """Clean receipt, one vendor, one line item."""
    width, height = 420, 340
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    font = _load_font(16)
    bold_font = _load_font(20)

    lines = [
        ("Cascade Coffee Roasters", bold_font),
        ("456 Pine St, Portland OR", font),
        ("", font),
        ("Date: 2026-02-18", font),
        ("", font),
        ("Item              Qty  Price   Total", font),
        ("Espresso Blend 12oz 1  $16.00  $16.00", font),
        ("", font),
        ("Subtotal:              $16.00", font),
        ("Tax:                    $1.44", font),
        ("Total:                 $17.44", font),
        ("", font),
        ("Currency: USD", font),
    ]
    _draw_lines(draw, lines, x=20, y=20, line_height=24)
    image.save(path)


def make_hardware_invoice(path: Path) -> None:
    """Clean invoice, several line items."""
    width, height = 460, 460
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    font = _load_font(15)
    bold_font = _load_font(20)

    lines = [
        ("Ironclad Hardware Supply", bold_font),
        ("78 Foundry Ave, Detroit MI", font),
        ("Invoice", font),
        ("", font),
        ("Date: 2026-01-05", font),
        ("", font),
        ("Item                    Qty  Price   Total", font),
        ("Wood Screws 1in (box)     3  $4.50   $13.50", font),
        ("Claw Hammer 16oz          1  $22.00  $22.00", font),
        ("Duct Tape 3-pack          2  $9.75   $19.50", font),
        ("Safety Goggles            1  $11.25  $11.25", font),
        ("", font),
        ("Subtotal:                     $66.25", font),
        ("Tax:                            $5.30", font),
        ("Total:                        $71.55", font),
        ("", font),
        ("Currency: USD", font),
    ]
    _draw_lines(draw, lines, x=20, y=20, line_height=24)
    image.save(path)


def make_grocery_receipt_skewed(path: Path) -> None:
    """Rougher case: gray background, smaller font, slightly rotated --
    approximates a hastily snapped phone photo, meant to pull extraction
    confidence down on at least a few fields."""
    width, height = 380, 320
    background = (210, 210, 210)
    image = Image.new("RGB", (width, height), color=background)
    draw = ImageDraw.Draw(image)
    font = _load_font(11)
    bold_font = _load_font(14)

    lines = [
        ("Green Valley Grocery", bold_font),
        ("12 Elm Court, Unit B, Boise ID", font),
        ("", font),
        ("Date: 2026-03-02", font),
        ("", font),
        ("Item          Qty  Price  Total", font),
        ("Milk 1gal       1  $4.29  $4.29", font),
        ("Bread Loaf      2  $3.10  $6.20", font),
        ("Eggs Dozen      1  $3.85  $3.85", font),
        ("", font),
        ("Subtotal:            $14.34", font),
        ("Tax:                  $0.86", font),
        ("Total:                $15.20", font),
        ("", font),
        ("Currency: USD", font),
    ]
    _draw_lines(draw, lines, x=16, y=16, line_height=18, fill=(40, 40, 40))

    # Slight rotation to simulate a not-quite-square scan/photo. expand=True
    # so nothing gets cropped; fillcolor matches the page background so the
    # corners created by the rotation don't show up as black.
    image = image.rotate(2.2, expand=True, fillcolor=background, resample=Image.BICUBIC)
    image.save(path)


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        (make_coffee_receipt, SAMPLES_DIR / "coffee-receipt.png"),
        (make_hardware_invoice, SAMPLES_DIR / "hardware-invoice.png"),
        (make_grocery_receipt_skewed, SAMPLES_DIR / "grocery-receipt-skewed.png"),
    ]
    for generator, path in targets:
        generator(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
