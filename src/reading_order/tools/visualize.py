# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/


from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
import traceback


from reading_order.order.parse_xml import (
    LINE_TYPE_TEXT,
    LINE_TYPE_TOCHU,
    LINE_TYPE_WARICHU,
    LINE_TYPE_COLORS,
    parse_xml,
)
from reading_order.utils.file import collect_files
from reading_order.utils.logger import get_logger
from reading_order.utils.time import TimeKeeper


def visualize(xml_path, logger, output_dir=None, shrink=True, text_block=True,
              show_order=True, image_paths=None):

    dat = parse_xml(xml_path)
    image_paths = image_paths or list(
        (xml_path.parent.parent / "img").glob("*"))
    ratio = 5 if shrink else 1

    given_odir = False
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        given_odir = True
    else:
        output_dir = xml_path.parent.parent / "img"
        if not output_dir.exists():
            logger.error(
                "Output dir was not be detected. Specify with --output option.")
            return

    for page in dat["pages"]:
        name = Path(page["image"]).name
        impaths = [impath for impath in image_paths if name in impath.name]
        if 1 != len(impaths):
            logger.warning(
                "%d images found for xml: %s (%s page %d)" %
                (len(impaths), name, xml_path.name, page["index"])
            )
            impath = None
        else:
            impath = impaths[0]

        w = page["width"] // ratio
        h = page["height"] // ratio

        # Name of output image file.
        if given_odir:
            name = xml_path.with_suffix(".%d.order.jpg" % page["index"]).name
        else:
            if impath is not None:
                name = impath.with_suffix(".order.jpg").name
            else:
                name = xml_path.with_suffix(".order.jpg").name

        with Image.new("RGBA", size=(w, h)) as im:
            draw = ImageDraw.Draw(im)

            # Text block.
            if text_block:
                for text_block in page["text_blocks"]:
                    polygon = [x//ratio for x in text_block["polygon"]]
                    if len(polygon) < 4:
                        logger.warning(
                            "Detected illegal polygon with %d coordinates: %s" %
                            (len(polygon) // 2, xml_path)
                        )
                        continue
                    draw.polygon(polygon, fill=(0, 0, 255, 40))

            reading_order = list()
            order = list()

            for line in page["lines"]:
                # Line bbox.
                color = LINE_TYPE_COLORS[line["type"]]
                bbox = [x//ratio for x in line["bbox"]]
                draw.rectangle(bbox, outline=(*color, 255), width=5//ratio)

                # Reading order.
                if line["type"] in [
                        LINE_TYPE_TEXT,
                        LINE_TYPE_TOCHU,
                        LINE_TYPE_WARICHU]:
                    x0, y0, x1, y1 = line["bbox"]
                    x = (x0 + x1) / 2 // ratio
                    y = (y0 + y1) / 2 // ratio
                    reading_order.append((x, y))
                    order.append(line["order"])

            if not text_block:
                # Sort reading_order with order val. Only ORDER attributes are
                # used to draw the reading order, without correction by xml
                # structure such as text block.
                if order:
                    zipped = [(o, ro,) for o, ro in zip(order, reading_order)]
                    def first(x): return x[0]
                    _, reading_order = zip(*sorted(zipped, key=first))
            if show_order:
                draw.line(reading_order, width=10 //
                          ratio, fill=(0, 0, 255, 255))

            # Mark start position.
            if reading_order:
                x, y = reading_order[0]
                r = 50 // ratio
                if show_order:
                    draw.arc((x-r, y-r, x+r, y+r), 0, 360,
                             fill=(0, 255, 0, 255), width=10//ratio)

            # Save image.
            if impath and impath.exists():
                with Image.open(impath) as bg:
                    bg = bg.resize((w, h))
                    bg.paste(im, mask=im.getchannel("A"))
                    bg.save(output_dir / name, "JPEG")
            else:
                with Image.new("RGB", size=(w, h), color=(255, 255, 255)) as bg:
                    bg.paste(im, mask=im.getchannel("A"))
                    bg.save(output_dir / name, "JPEG")


def get_args():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("input", metavar="XML/DIR", type=Path)
    ap.add_argument("--image-dir", metavar="DIR", type=Path)
    ap.add_argument("--only", metavar="PATTERN")
    ap.add_argument("--skip", metavar="PATTERN")
    ap.add_argument("-o", "--output", metavar="DIR", type=Path)
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--no-text-block", action="store_true")
    ap.add_argument("--no-order", action="store_true")
    return ap.parse_args()


def main():
    time_keeper = TimeKeeper()
    with time_keeper.measure_time("total process"):
        logger = get_logger("ReadingOrder")
        args = get_args()

        # Get the list of unsorted xmls
        xml_paths = collect_files(
            args.input, ext=".xml", only=args.only, skip=args.skip)
        img_paths = collect_files(
            args.image_dir, ext=".jpg|.JPG|.tif|.TIF") if args.image_dir else None
        logger.info("Number of xml files found: %d" % len(xml_paths))

        # Main loop
        for xml_path in tqdm(xml_paths):
            try:
                visualize(
                    xml_path, logger, output_dir=args.output, shrink=not
                    args.no_shrink, text_block=not args.no_text_block,
                    show_order=not args.no_order, image_paths=img_paths,
                )
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.error("Failed to visualize: %s" % xml_path)
                logger.error(traceback.format_exc())

    time_keeper.print(logger=logger)


if "__main__" == __name__:
    main()
