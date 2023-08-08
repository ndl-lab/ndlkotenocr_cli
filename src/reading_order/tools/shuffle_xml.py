# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/


from argparse import ArgumentParser
from tqdm import tqdm
import traceback


from reading_order.utils.file import collect_files
from reading_order.utils.xml import shuffle_xml_file
from reading_order.utils.logger import get_logger


def main():
    ap = ArgumentParser()
    ap.add_argument("input", metavar="XML/DIR")
    ap.add_argument("--only", metavar="PATTERN")
    ap.add_argument("--skip", metavar="PATTERN")
    args = ap.parse_args()

    logger = get_logger("ReadingOrder")
    xml_paths = collect_files(args.input, ext=".xml",
                              only=args.only, skip=args.skip)
    for xml_path in tqdm(xml_paths):
        try:
            o_fname = xml_path.with_suffix(".shuf.xml")
            shuffle_xml_file(xml_path, o_fname)
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.error("Failed to shuffle: %s" % xml_path)
            logger.error(traceback.format_exc())


if "__main__" == __name__:
    main()
