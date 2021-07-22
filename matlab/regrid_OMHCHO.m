clear
clc
%% You have to update these directories, VERY IMPORTANT
% I had to jump back and forth between my PC and unix workstation
if ispc
    L2dir = 'D:\Research_CfA\OMHCHO\L2\';
    L2gdir = 'D:\Research_CfA\OMHCHO\L2g\';
    L3dir = 'D:\Research_CfA\OMHCHO\L3\';
else
    L2dir = '/data/tempo1/Shared/OMHCHO/'; % raw L2 data directory,
    L2gdir = '/data/tempo1/Shared/kangsun/OMHCHO/L2g/'; % intermediate data, or L2g
    L3dir = '/data/tempo1/Shared/kangsun/OMHCHO/L3/'; % regridded data, or L3
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
if_plot = false;
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
        output_subset = F_subset_OMHCHO(inp_subset);
        
        L2g_fn = ['CONUS_',num2str(iyear),'.mat'];
        save([L2gdir,L2g_fn],'inp_subset','output_subset')
    end
    
end
%% regrid subsetted L2 data (or L2g data) into L3 data, save L3 monthly
if if_regrid
    clc
    for iyear = 2004:2016
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
        inp_regrid.vcdname = 'colhcho';
        inp_regrid.vcderrorname = 'colhchoerror';
        inp_regrid.if_parallel = true;
        
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
end
%%
if if_plot
    clear output_regrid    
    for iyear = 2008:2008
        for imonth = 1:12
            L3_fn = ['Colorado_',num2str(iyear),'_',num2str(imonth),'.mat'];
            if ~exist('output_regrid','var')
                load([L3dir,L3_fn],'inp_regrid','output_regrid')
                A = output_regrid.A;
                B = output_regrid.B;
                D = output_regrid.D;
            else
                load([L3dir,L3_fn],'inp_regrid','output_regrid')
                A = A+output_regrid.A;
                B = B+output_regrid.B;
                D = D+output_regrid.D;
            end
        end
    end
    
    C = A./B;
    Clim = [0 .5e16];% min/max plotted nh3 column
% shape file for US states
US_states = shaperead('D:\GIS data\US_state\cb_2016_us_state_20m.shp');
opengl software
close all
figure('unit','inch','color','w','position',[0 1 8 6])

h = pcolor(output_regrid.xgrid,output_regrid.ygrid,...
    double(C/1e16));
set(h,'edgecolor','none');
colormap('parula')
caxis(Clim/1e16)

hc = colorbar('south');
set(hc,'position',[0.35 0.03 0.3 0.02])
set(get(hc,'xlabel'),'string','Ammonia column [10^{16} molec/cm2]')
hold on
for istate = 1:length(US_states)
    plot(US_states(istate).X,US_states(istate).Y,'color','w')
end
alpha(h,0.8)
addpath('c:\Users\Kang Sun\dropbox\matlab functions\export_fig\')
plot_google_map('MapType','terrain')
xlim([inp_regrid.MinLon inp_regrid.MaxLon])
ylim([inp_regrid.MinLat inp_regrid.MaxLat])
pos = get(gca,'position');
set(gca,'position',[pos(1) pos(2)+0.05 pos(3:4)])

end
