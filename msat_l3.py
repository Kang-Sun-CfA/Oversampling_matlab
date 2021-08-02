import argparse
import datetime
import dateutil.parser
import logging
import matplotlib.pyplot as plt
from popy import F_wrapper_l3
import sys
import os


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Level 3 Processing')

    group = parser.add_argument(
        '-i', '--l2-input-dir', required=True, help='input directory')
    group = parser.add_argument(
        '-o', '--output-dir', required=True, help='output directory')
    group = parser.add_argument(
        '--start-time', required=True, help='ISO-8601 start time')
    group = parser.add_argument(
        '--end-time', required=True, help='ISO-8601 end time')
    group = parser.add_argument(
        '-w', '--west', required=True, type=float, help='grid west boundary')
    group = parser.add_argument(
        '-e', '--east', required=True, type=float, help='grid east boundary')
    group = parser.add_argument(
        '-n', '--north', required=True, type=float, help='grid north boundary')
    group = parser.add_argument(
        '-s', '--south', required=True, type=float, help='grid south boundary')
    group = parser.add_argument(
        '--grid-size', type=float, default=0.001, help='grid size in degree')
    group = parser.add_argument(
        '--num-cores', type=int, default=8, help='number of cores to use')
    group = parser.add_argument(
        '--block-length', type=int, default=300, help='block length')

    args = parser.parse_args()

    start_time = dateutil.parser.parse(args.start_time)
    end_time = dateutil.parser.parse(args.end_time)

    l3_data = F_wrapper_l3(instrum='MethaneSAT', product='CH4',grid_size=args.grid_size,
            l2_path_pattern=args.l2_input_dir+'/*O2-CH4_%Y%m%dT*.nc',
            start_date_array=[start_time], end_date_array=[end_time], if_use_presaved_l2g=False,
            west=args.west, east=args.east, south=args.south, north=args.north, ncores=8, block_length=300)
    l3_data.plot(plot_field='XCH4')
    plt.savefig(os.path.join(args.output_dir, 'l3.png'))


if __name__ == "__main__":
    main()
