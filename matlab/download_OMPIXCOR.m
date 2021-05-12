% this script downloads OMPIXCOR dataset from NASA DISC, calling the
% F_download_OMI.m function

% written by Kang Sun on 2017/07/16
%%
L2dir = '/data/tempo1/Shared/kangsun/OMPIXCOR/L2_data/'; % raw L2 data directory,
inp_download = [];
% CONUS
inp_download.MinLat = 25;
inp_download.MaxLat = 50;
inp_download.MinLon = -130;
inp_download.MaxLon = -63;

inp_download.if_download_xml = true;
inp_download.if_download_he5 = true;

inp_download.swath_BDR_fn = '/data/tempo1/Shared/kangsun/OMNO2/Important_constant/OMI_BDR.mat';
inp_download.url0 = 'https://aura.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level2/OMPIXCOR.003/';
inp_download.L2dir = L2dir;

for iyear = 2004:2017
    inp_download.Startdate = [iyear 1 1];
    inp_download.Enddate = [iyear 12 31];
    inp_download.if_parallel = false;
    % parallel download is fast, but DOES NOT WORK!!!!!!!!!!!!!!!!!!!!!
    output_download = F_download_OMI(inp_download);
    save([L2dir,num2str(iyear),'output_download.mat'],'output_download')
end