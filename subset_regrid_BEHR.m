% subset and regrid BEHR data
% written by Kang Sun on 2018/09/13
clear;clc

% directory containing regriding functions, available at https://github.com/Kang-Sun-CfA/Oversampling_matlab
code_dir = '~/OMI/Oversampling_matlab/';
addpath(code_dir)

% BEHR data directory
BEHR_data_dir = '/mnt/Data2/BEHR/';

% raw L2 data directory. BEHR L2 can be downloaded using https://github.com/CohenBerkeleyLab/BEHRDownloader
L2dir = [BEHR_data_dir,'L2/']; 
L2gdir = [BEHR_data_dir,'L2g/']; % intermediate data, or L2g
L3dir = [BEHR_data_dir,'L3/']; % regridded data, or L3

if ~exist(L2gdir,'dir')
    mkdir(L2gdir)
end
if ~exist(L3dir,'dir')
    mkdir(L3dir)
end
%%
clc
inp_subset = [];
% CONUS
inp_subset.MinLat = 24;
inp_subset.MaxLat = 50;
inp_subset.MinLon = -130;
inp_subset.MaxLon = -63;

% only cloud fraction <= 0.2 are kept. fiddle with BEHRQualityFlags if you
% want to keep higher cloud fraction data
inp_subset.MaxCF = 0.3;
inp_subset.MaxSZA = 75;

inp_subset.usextrack = 1:60;
inp_subset.L2dir = L2dir;

for iyear = 2005:2008
    inp_subset.Startdate = [iyear 1 1];
    inp_subset.Enddate = [iyear 12 31];
    output_subset = F_subset_BEHR(inp_subset);
    
    L2g_fn = ['BEHR_',num2str(iyear),'.mat'];
    save([L2gdir,L2g_fn],'inp_subset','output_subset')
    disp([num2str(iyear),' subset done'])
end

%% regrid subsetted L2 data (or L2g data) into L3 data, save L3 monthly
clc
for iyear = [2005 2007 2008]
    L2g_fn = ['BEHR_',num2str(iyear),'.mat'];
    load([L2gdir,L2g_fn],'inp_subset','output_subset')
    inp_regrid = [];
    % Resolution of oversampled L3 data
    inp_regrid.Res = 0.02; % in degree
    % CONUS
    inp_regrid.MinLat = 24;
    inp_regrid.MaxLat = 50;
    inp_regrid.MinLon = -129;
    inp_regrid.MaxLon = -64;
    
    inp_regrid.MaxCF = 0.2;
    inp_regrid.MaxSZA = 60;
    inp_regrid.usextrack = 6:55;
    inp_regrid.errorpower = 1;
    inp_regrid.vcdname = 'colno2';
    inp_regrid.vcderrorname = 'colno2error';
    % if use parallel for loop. Can be slow for large domain
    inp_regrid.if_parallel = true;
    
    end_date_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    if iyear == 2008 || iyear == 2012 || iyear == 2016 || iyear == 2020
        end_date_month = [31 29 31 30 31 30 31 31 30 31 30 31];
    end
    for imonth = 1:12
        disp( ['Working on ',num2str(iyear),', ',num2str(imonth)])
        inp_regrid.Startdate = [iyear imonth 1];
        inp_regrid.Enddate = [iyear imonth end_date_month(imonth)];
        output_regrid = F_regrid_OMI(inp_regrid,output_subset);
        L3_fn = ['BEHR_',num2str(iyear),'_',num2str(imonth),'.mat'];
        save([L3dir,L3_fn],'inp_regrid','output_regrid')
        disp([num2str(iyear),'/',num2str(imonth),' regridding done'])
    end
end
%% plot
clc
usemonth = 5:9;
useyear = 2008;
L3_fn = ['BEHR_',num2str(useyear(1)),'_',num2str(usemonth(1)),'.mat'];
load([L3dir,L3_fn],'inp_regrid','output_regrid')
nx = length(output_regrid.xgrid);
ny = length(output_regrid.ygrid);
nt = length(usemonth);
above = zeros(ny,nx,nt,'single');
below = above;
for iyear = 1:length(useyear)
    for imonth = 1:length(usemonth)
        disp( ['Working on ',num2str(useyear(iyear)),', ',...
            num2str(usemonth(imonth))])
        L3_fn = ['BEHR_',num2str(useyear(iyear)),'_',...
            num2str(usemonth(imonth)),'.mat'];
        tmp = load([L3dir,L3_fn],'inp_regrid','output_regrid');
        above(:,:,imonth) = above(:,:,imonth)+tmp.output_regrid.A;
        below(:,:,imonth) = below(:,:,imonth)+tmp.output_regrid.B;
    end
end
nlat = output_regrid.ygrid;
nlon = output_regrid.xgrid;
nvcd = double(squeeze(sum(above,3)./sum(below,3)));
h = pcolor(nlon,nlat,nvcd);set(h,'edgecolor','none')