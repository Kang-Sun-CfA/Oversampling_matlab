import argparse
import datetime
import dateutil.parser
import logging
import matplotlib.pyplot as plt
import numpy as np
from popy import popy
import sys
import os


def run_popy(l2_dir: str, output_dir: str, m: popy, ncores: int = 8, block_length: int = 300):
    # read level 2 data in RAM
    m.F_subset_MethaneSAT(l2_dir)
    # optionally save level 2 data subset
    m.F_save_l2g_to_mat(os.path.join(output_dir, 'test_l2g.mat'))
    # regrid level 2 to level 3. The 0.001 degree grid defined here is 5000 x 5000
    # block_length = 300 divides the grid into 256 blocks, which can be separately regridded in parallel
    # min(ncores, max_cpu) will be used
    l3_data = m.F_parallel_regrid(block_length=block_length, ncores=ncores)
    # save level 3 data to a quick .mat format. likely will be netcdf in future
    m.F_save_l3_to_mat(os.path.join(output_dir, 'test_l3.mat'), l3_data)
    m.F_plot_l3_cartopy(l3_data=l3_data, plot_field='XCH4')
    plt.savefig(os.path.join(output_dir, 'l3.png'))


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

    # define spatial/temporal constraints and grid size (in degree)
    m = popy(instrum='MethaneSAT', product='CH4', west=args.west, east=args.east, south=args.south, north=args.north,
             start_year=start_time.year, start_month=start_time.month, start_day=start_time.day,
             start_hour=start_time.hour, start_minute=start_time.minute, start_second=start_time.second,
             end_year=end_time.year, end_month=end_time.month, end_day=end_time.day,
             end_hour=end_time.hour, end_minute=end_time.minute, end_second=end_time.second,
             grid_size=args.grid_size)

    run_popy(l2_dir=args.l2_input_dir, output_dir=args.output_dir, m=m,
             ncores=args.num_cores, block_length=args.block_length)


if __name__ == "__main__":
    main()
