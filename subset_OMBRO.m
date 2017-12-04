% this script subsets SAO OMBRO into more managable files, calling
% F_subset_OMBRO.m. the OMBRO files are already there, do not need
% download

% written by Kang Sun on 2017/12/03
%%
clear;clc
if ispc
    L2dir = 'D:\Research_CfA\OMBRO\L2\';
    L2gdir = 'D:\Research_CfA\OMBRO\L2g\';
else
    L2dir = '/data/tempo1/Shared/OMBRO/';
    L2gdir = '/data/tempo1/Shared/kangsun/OMBRO/L2g/';
end
if ~exist(L2gdir,'dir')
    mkdir(L2gdir)
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
for iyear = 2005:2017
    inp_subset.Startdate = [iyear 1 1];
    inp_subset.Enddate = [iyear 12 31];
    output_subset = F_subset_OMBRO(inp_subset);
    
    L2g_fn = ['CONUS_',num2str(iyear),'.mat'];
    save([L2gdir,L2g_fn],'inp_subset','output_subset')
end
% %%
% clc
% inp_subset.Startdate = [2005 7 6];
%     inp_subset.Enddate = [2005 7 9];
%     output_subset = F_subset_OMBRO(inp_subset);
