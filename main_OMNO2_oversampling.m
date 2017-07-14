clear
clc
%% You have to update these directories, VERY IMPORTANT
% I had to jump back and forth between my PC and unix workstation
if ispc
    L2dir = 'D:\Research_CfA\OMNO2\L2\';
    L2gdir = 'D:\Research_CfA\OMNO2\L2g\';
    L3dir = 'D:\Research_CfA\OMNO2\L3\';
else
L2dir = '/data/tempo1/Shared/kangsun/OMNO2/L2_data/'; % raw L2 data directory, 
L2gdir = '/data/tempo1/Shared/kangsun/OMNO2/L2g/'; % intermediate data, or L2g
L3dir = '/data/tempo1/Shared/kangsun/OMNO2/L3/'; % regridded data, or L3
end
if ~exist(L2gdir,'dir')
    mkdir(L2gdir)
end
if ~exist(L3dir,'dir')
    mkdir(L3dir)
end
if_download = false;
if_subset = false;
if_regrid = true;
%% download OMNO2
if if_download
inp_download = [];
% CONUS
inp_download.MinLat = 25;
inp_download.MaxLat = 50;
inp_download.MinLon = -130;
inp_download.MaxLon = -63;

inp_download.if_download_xml = true;
inp_download.if_download_he5 = true;

inp_download.swath_BDR_fn = '/data/tempo1/Shared/kangsun/OMNO2/Important_constant/OMI_BDR.mat';
inp_download.url0 = 'https://aura.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level2/OMNO2.003/';
inp_download.L2dir = L2dir;

for iyear = 2004:2017
    inp_download.Startdate = [iyear 1 1];
    inp_download.Enddate = [iyear 12 31];
    inp_download.if_parallel = false;
    % parallel download is fast, but
    output_download = F_download_OMI(inp_download);
    save([L2dir,num2str(iyear),'output_download.mat'],'output_download')
end
end
%%
if if_subset
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

for iyear = 2004:2017
    inp_subset.Startdate = [iyear 1 1];
    inp_subset.Enddate = [iyear 12 31];
    output_subset = F_subset_OMNO2(inp_subset);

    L2g_fn = ['CONUS_',num2str(iyear),'.mat'];
    save([L2gdir,L2g_fn],'inp_subset','output_subset')
end
    
end
%% regrid subsetted L2 data (or L2g data) into L3 data, save L3 monthly
clc
for iyear = 2008:2008
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
    inp_regrid.MaxCF = 0.3;
    inp_regrid.MaxSZA = 60;
    inp_regrid.vcdname = 'colno2';
    inp_regrid.vcderrorname = 'colno2error';
    inp_regrid.if_parallel = false;
    
    end_date_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    if iyear == 2004 || iyear == 2008 || iyear == 2016
        end_date_month = [31 29 31 30 31 30 31 31 30 31 30 31];
    end
    for imonth = 1:12
        inp_regrid.Startdate = [iyear imonth 1];
        inp_regrid.Enddate = [iyear imonth end_date_month(imonth)];
        output_regrid = F_regrid_OMI(inp_regrid,output_subset);
        L3_fn = ['Colorado_',num2str(iyear),'_',num2str(imonth),'.mat'];
        save([L3dir,L3_fn],'inp_regrid','output_regrid')
    end
end

