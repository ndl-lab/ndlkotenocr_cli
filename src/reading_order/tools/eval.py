# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/


from pathlib import Path
from tqdm import tqdm
import copy
import traceback


from reading_order.utils.file import collect_files
from reading_order.utils.logger import get_logger
from reading_order.utils.time import TimeKeeper
from reading_order.xy_cut.eval import eval_path as xy_cut_eval_path


def get_args():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("input", metavar="XML/DIR", type=Path)
    ap.add_argument("--only", metavar="PATTERN")
    ap.add_argument("--skip", metavar="PATTERN")
    ap.add_argument("--method", choices=["xy-cut"], default="xy-cut")

    xy_cut_options = ap.add_argument_group("options for method `xy-cut`")
    xy_cut_options.add_argument("--plot-partition", action="store_true")
    xy_cut_options.add_argument(
        "--line-width-scale", metavar="FLOAT", type=float, default=1.0)
    xy_cut_options.add_argument("--smoothing", action="store_true")

    return ap.parse_args()


def main():
    time_keeper = TimeKeeper()

    with time_keeper.measure_time("total process"):
        args = get_args()
        logger = get_logger("ReadingOrder")

        # Switch the solver according to `--method` option
        solver = {
            "xy-cut": xy_cut_eval_path,
        }.get(args.method)

        # Extra arguments to be passed to the solver
        extra = {
            "xy-cut": {
                "plot_partition": args.plot_partition,
                "line_width_scale": args.line_width_scale,
                "smoothing": args.smoothing,
            },
        }.get(args.method)

        # Get the list of unsorted xmls
        xml_paths = collect_files(
            args.input, ext=".xml", only=args.only, skip=args.skip)
        logger.info("Number of xml files found: %d" % len(xml_paths))

        # Main loop
        num_pages = 0
        for xml_path in tqdm(xml_paths):
            try:
                out_path = xml_path.with_suffix(".sorted.xml")
                num_pages += solver(
                    xml_path, out_path, looger=logger, time_keeper=time_keeper,
                    **extra,
                )
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.error("Failed to eval: %s" % xml_path)
                logger.error(traceback.format_exc())

    logger.info("Number of sorted pages: %d" % num_pages)
    time_keeper.print(logger=logger)


def infer_with_cli(input_data):
    from reading_order.xy_cut.eval import eval_xml
    output_data = copy.deepcopy(input_data)
    logger = get_logger("ReadingOrder")

    try:
        root = output_data["xml"].getroot()
        eval_xml(root, logger=logger)
    except KeyboardInterrupt:
        raise
    except Exception:
        logger.error("Failed to eval")
        logger.error(traceback.format_exc())

    return output_data


if "__main__" == __name__:
    main()
