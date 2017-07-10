clear
clc
%% You have to update these directories, VERY IMPORTANT
L2dir = 'E:\metopa\'; % raw L2 data directory, nh3fix_yyyymmdd.mat
L2gdir = 'D:\Research_CfA\IASI\L2g\'; % intermediate data
L3dir = 'D:\Research_CfA\IASI\L3\'; % regridded data
% file containing the pixel size lookup table
pixel_size_file = 'daysss.mat';
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

for iyear = 2014:2016
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
%%
clc
iyear = 2014;
inp_subset.Startdate = [iyear 6 27];
inp_subset.Enddate = [iyear 6 29];
inp_subset.showfilter = true;
output_subset = F_subset_IASI(inp_subset);

%% regrid subsetted L2 data (or L2g data) into L3 data, save L3 monthly
clc
for iyear = 2014:2016
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
%%
close all
step = 3;
C = cell(12/step,1);
count = 0;
for imonth = 1:step:12
    clear output_regrid
    for iyear = 2008:2015
        for istep = 0:step-1
        L3_fn = ['Colorado_',num2str(iyear),'_',num2str(imonth+istep),'.mat'];
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
    count = count+1;
    C{count} = A./B;
    figure;
    h = pcolor(output_regrid.xgrid,output_regrid.ygrid,...
    double(C{count}/1e16));
    set(h,'edgecolor','none');
    caxis([0 1])
end

%%
clear output_regrid

for iyear = 2008:2015
    for imonth = 6:8
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
%%
clc
Clim = [15 16.2];% min/max plotted nh3 column
% shape file for US states
US_states = shaperead('D:\GIS data\US_state\cb_2016_us_state_20m.shp');
opengl software
close all
figure('unit','inch','color','w','position',[0 1 12 8])
for i = 1:4
    subplot(2,2,i)
    plotmat = C{i};
    plotmat(plotmat <= 0) = nan;
plotmat = double(log10(plotmat));
plotmat(plotmat < Clim(1)) = nan;
h = pcolor(output_regrid.xgrid,output_regrid.ygrid,...
    plotmat);
xlim([-105.5 -102])
ylim([39.3 41])
set(h,'edgecolor','none');
colormap('jet')
caxis(Clim)

hc = colorbar('south');
set(hc,'position',[0.35 0.03 0.3 0.02])
set(get(hc,'xlabel'),'string','Ammonia column [log10 molec/cm2]')
hold on
for istate = 1:length(US_states)
    plot(US_states(istate).X,US_states(istate).Y,'color','w')
end
alpha(h,0.8)
plot_google_map('MapType','terrain')
pos = get(gca,'position');
% set(gca,'position',[pos(1) pos(2)+0.05 pos(3:4)])
end
%%
close all
h = surf(output_regrid.xgrid,output_regrid.ygrid,...
    double(C/1e16));
set(h,'edgecolor','none');
%%
Clim = [0 1e16];% min/max plotted nh3 column
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
plot_google_map('MapType','terrain')
xlim([inp_regrid.MinLon inp_regrid.MaxLon])
ylim([inp_regrid.MinLat inp_regrid.MaxLat])
pos = get(gca,'position');
set(gca,'position',[pos(1) pos(2)+0.05 pos(3:4)])
%%
Clim = [15 16.31];% min/max plotted nh3 column
% shape file for US states
US_states = shaperead('D:\GIS data\US_state\cb_2016_us_state_20m.shp');
opengl software
close all
figure('unit','inch','color','w','position',[0 1 8 5.5])
axes('position',[0.1300    0.1700    0.7750    0.8])
plotmat = double(log10(C));
plotmat(plotmat < Clim(1)) = nan;
h = pcolor(output_regrid.xgrid,output_regrid.ygrid,...
    plotmat);
xlim([-105.5 -102])
ylim([39.1 41])
set(h,'edgecolor','none');
colormap('jet')
caxis(Clim)
plot_google_map('MapType','terrain')
hc = colorbar('south');
set(hc,'position',[0.35 0.03 0.3 0.02])
Ytick = [1 2 5 10 20];
sbxlim = get(hc,'xlim');
sbxtick = interp1(Clim,sbxlim,log10(Ytick*1e15));
set(hc,'xtick',sbxtick,'xticklabel',Ytick)
set(get(hc,'xlabel'),'string','Ammonia column [10^{15} molec/cm2]')
hold on
for istate = 1:length(US_states)
    plot(US_states(istate).X,US_states(istate).Y,'color','w')
end
alpha(h,0.5)
% pos = get(gca,'position');
% set(gca,'position',[pos(1) pos(2)+0.06 pos(3:4)])
addpath('c:\users\Kang Sun\dropbox\matlab functions\export_fig\')
export_fig('East_Colorado_JJA_2008_2015.png','-r200')
