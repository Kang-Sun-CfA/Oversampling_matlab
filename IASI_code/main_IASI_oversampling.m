% This script calls two important functions, F_subset_IASI.m and
% F_regrid_IASI.m. The first subsets the cumbersome L2 data within a lat 
% lon box (contiguous US, for example), performs initial quality control,
% and saves the subsetted data, which I called "L2g" data. The second
% function loads the L2g data, further subsets them (e.g., over Colorado),
% and regrids L2g pixels into L3 grids.

% updated by Kang Sun from testbed_IASI_oversampling.m on 2017/07/08
% known problems: 
% much many NaNs in L2 data in 2009
% a few files cannot be read in 2014
% many cases fcloud == true, but CLcov is NaN

clear
clc
%% You have to update these directories, VERY IMPORTANT
L2dir = 'E:\metopa\'; % raw L2 data directory, nh3fix_yyyymmdd.mat
L2gdir = 'D:\Research_CfA\IASI\L2g\'; % intermediate data, or L2g
L3dir = 'D:\Research_CfA\IASI\L3\'; % regridded data, or L3
% file containing the pixel size lookup table
pixel_size_file = '.\daysss.mat';
%% subsetting L2 data, using F_subset_IASI.m. Save the results every year
pixel_shape = load(pixel_size_file);

inp_subset = [];
% CONUS
inp_subset.MinLat = 25;
inp_subset.MaxLat = 50;
inp_subset.MinLon = -130;
inp_subset.MaxLon = -63;

% inp_subset.MaxCF = 0.25;
inp_subset.Min_pos_TC = 0.5;
inp_subset.Max_neg_TC = -0.5;
inp_subset.MinTskin = 263.15;
inp_subset.MaxSand = 0.5;
inp_subset.day_or_night = 'day';

inp_subset.L2dir = L2dir;
inp_subset.showfilter = false;

for iyear = 2008:2016
    inp_subset.Startdate = [iyear 1 1];
    inp_subset.Enddate = [iyear 12 31];
    output_subset = F_subset_IASI(inp_subset);
    % calculate pixel shape
    [output_subset.u, output_subset.v, output_subset.t] =...
        F_define_IASI_pixel(output_subset.lat,output_subset.ifov,...
        pixel_shape.uuu4,pixel_shape.vvv4,pixel_shape.ttt4);
    L2g_fn = ['CONUS_',num2str(iyear),'.mat'];
    save([L2gdir,L2g_fn],'inp_subset','output_subset')
end
%% regrid subsetted L2 data (or L2g data) into L3 data, save L3 monthly
clc
for iyear = 2008:2016
    L2g_fn = ['CONUS_',num2str(iyear),'.mat'];
    load([L2gdir,L2g_fn],'inp_subset','output_subset')
    inp_regrid = [];
    % Resolution of oversampled L3 data?
    inp_regrid.Res = 0.02; % in degree
    % Colorado
    inp_regrid.MinLon = -109;
    inp_regrid.MaxLon = -102;
    inp_regrid.MinLat = 37;
    inp_regrid.MaxLat = 41;
    
    end_date_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    if iyear == 2008 || iyear == 2016
        end_date_month = [31 29 31 30 31 30 31 31 30 31 30 31];
    end
    for imonth = 1:12
        inp_regrid.Startdate = [iyear imonth 1];
        inp_regrid.Enddate = [iyear imonth end_date_month(imonth)];
        output_regrid = F_regrid_IASI(inp_regrid,output_subset);
        L3_fn = ['Colorado_',num2str(iyear),'_',num2str(imonth),'.mat'];
        save([L3dir,L3_fn],'inp_regrid','output_regrid')
    end
end
