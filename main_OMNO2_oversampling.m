clear
clc
%% You have to update these directories, VERY IMPORTANT
L2dir = '/data/tempo1/Shared/kangsun/OMNO2/L2_data/'; % raw L2 data directory, 
L2gdir = '/data/tempo1/Shared/kangsun/OMNO2/L2g/'; % intermediate data, or L2g
L3dir = '/data/tempo1/Shared/kangsun/OMNO2/L3/'; % regridded data, or L3
if ~exist(L2gdir,'dir')
    mkdir(L2gdir)
end
if ~exist(L3dir,'dir')
    mkdir(L3dir)
end
% %% Kang Sun's PC
% L2dir = 'd:\Research_CfA\OMNO2\L2\';
%% download OMNO2
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
% %%
% inp_subset = [];
% % CONUS
% inp_subset.MinLat = 25;
% inp_subset.MaxLat = 50;
% inp_subset.MinLon = -130;
% inp_subset.MaxLon = -63;
% 
% inp_subset.MaxCF = 0.3;
% inp_subset.MaxSZA = 75;
% 
% inp_subset.if_download_xml = true;
% inp_subset.if_download_he5 = true;
% inp_subset.if_delete_he5 = false;
% 
% inp_subset.swath_BDR_fn = '/data/tempo1/Shared/kangsun/OMNO2/Important_constant/OMI_BDR.mat';
% inp_subset.url0 = 'https://aura.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level2/OMNO2.003/';
% inp_subset.L2dir = L2dir;
% 
% for iyear = 2008:2008
%     inp_subset.Startdate = [iyear 1 1];
%     inp_subset.Enddate = [iyear 12 31];
%     output_subset = F_download_subset_OMNO2(inp_subset);
% 
%     L2g_fn = ['CONUS_',num2str(iyear),'.mat'];
%     save([L2gdir,L2g_fn],'inp_subset','output_subset')
% end
% 
