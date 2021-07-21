% this script subsets SAO OMHCHO into more managable files, calling
% F_subset_OMHCHO.m. the OMHCHO files are already there, do not need
% download

% written by Kang Sun on 2017/07/16
%%
clear;clc
if ispc
    L2dir = 'D:\Research_CfA\OMHCHO\L2\';
    L2gdir = 'D:\Research_CfA\OMHCHO\L2g\';
else
    L2dir = '/data/tempo1/Shared/OMHCHO/';
    L2gdir = '/data/tempo1/Shared/kangsun/OMHCHO/L2g/';
end
inp_subset = [];
% CONUS
inp_subset.MinLat = 25;
inp_subset.MaxLat = 50;
inp_subset.MinLon = -130;
inp_subset.MaxLon = -63;

inp_subset.MaxCF = 0.3;
inp_subset.MaxSZA = 75;

inp_subset.usextrack = 1:60;

inp_subset.L2dir = L2dir;
inp_subset.filelist = dir(L2dir);
for iyear = 2004:2017
    inp_subset.Startdate = [iyear 1 1];
    inp_subset.Enddate = [iyear 12 31];
    output_subset = F_subset_OMHCHO(inp_subset);
    
    L2g_fn = ['CONUS_',num2str(iyear),'.mat'];
    save([L2gdir,L2g_fn],'inp_subset','output_subset')
end
