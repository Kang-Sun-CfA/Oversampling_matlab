clear
clc

%%% PLEASE MODIFY THIS SECTION, OTHERWISE IT WILL NOT WORK!!! %%%%

rootdir = '/data/tempo1/Shared/kangsun/IASI/';

% directory containing L2 data
L2dir = [rootdir,'IASI_2015_Global/'];

% file containing the pixel size lookup table
pixel_size_file = [rootdir,'pixel_size/daysss.mat'];

% where to save your plots
plotdir = [rootdir,'figures/'];

% directory containing figure saving function export_fig.m
% This is optional, you may save figure using other methods
export_fig_dir = '~/matlab functions/export_fig';

% directory containing necessary functions, and this script
codedir = '~/OMI/Oversampling_matlab/IASI_code';

%%% PLEASE MODIFY THIS SECTION, OTHERWISE IT WILL NOT WORK!!! %%%%

cd(codedir)
addpath(export_fig_dir)
load(pixel_size_file)
%% parameter inputs
% begin and start dates: from Startdate(1) to Enddate(1), and from 
% Startdate(2) to Enddate(2) ...
Startdate = [2005 3 1;
    2006 3 1;
    2007 3 1;
    2015 1 1];

Enddate = [2005 5 31;
    2006 5 31;
    2007 5 31;
    2015 12 31];

% lat lon box
% Front Range
MinLat = 39; MaxLat = 41; 
MinLon = -106; MaxLon = -101;

% % CONUS
% MinLat = 25; MaxLat = 50; 
% MinLon = -130; MaxLon = -63;

% grey area, L2 pixel center cannot be there, but pixel boundaries can
MarginLat = 0.5; 
MarginLon = 0.5;

% max cloud fraction, thermal constrast, etc
MaxCF = 25; % percent

% minimal positive thermal constrast, K
Min_pos_TC = 0.5;
% maximal negatie thermal constrast, K
Max_neg_TC = -0.5;
% minimal skin temperature
MinTskin = 263.15;

% sand index, sand should be filtered out
MaxSand = 0.5;

% daytime or nighttime measurement? choose 'day', or 'night'. otherwise
% both day and night. I guess day means fhour == 1. Otherwise we have a
% problem
day_or_night = 'day';

% Resolution of oversampled L3 data?
Res = 0.02; % in degree

% define x y grids
xgrid = (MinLon+0.5*Res):Res:MaxLon;
ygrid = (MinLat+0.5*Res):Res:MaxLat;
nrows = length(ygrid);
ncols = length(xgrid);

% define x y mesh
[xmesh, ymesh] = meshgrid(xgrid,ygrid);

% construct a rectangle envelopes the orginal pixel
xmargin = 3; % how many times to extend zonally
ymargin = 2; % how many times to extend meridonally

% how many points to define an IASI ellipse?
npoint_ellipse = 10;

filelist = dir(L2dir);
fileyear = nan(size(filelist));
filemonth = fileyear;
filedate = fileyear;

for i = 1:length(filelist)
    if length(filelist(i).name) > 10
        fileyear(i) = str2double(filelist(i).name(8:11));
        filemonth(i) = str2double(filelist(i).name(12:13));
        filedate(i) = str2double(filelist(i).name(14:15));
    end
end

int = ~isnan(fileyear);
filelist = filelist(int);
fileyear = fileyear(int);
filemonth = filemonth(int);
filedate = filedate(int);

fileday = datenum([fileyear filemonth filedate]);
useindex = false(size(filelist));
for iperiod = 1:size(Startdate,1)
    useindex = useindex | ...
        fileday >= datenum(Startdate(iperiod,:)) & fileday <= datenum(Enddate(iperiod,:));
end
% only use files on useful dates
subfilelist = filelist(useindex);
subfileday = fileday(useindex);
%%
disp('Subsetting and filtering the IASI data ...')
latall = [];
lonall = [];
ifovall = [];
colnh3all = [];
utcall = [];
totErrall = [];
fhourall = [];
for ifile = 1:length(subfilelist)
    disp(['Loading ',subfilelist(ifile).name,'...'])
    temp = load([L2dir,subfilelist(ifile).name]);
    TC = temp.tskin-temp.Tprof(1,:); % thermal contrast
    validmask = temp.lat >= MinLat+MarginLat & temp.lat <= MaxLat-MarginLat & ...
        temp.lon >= MinLon+MarginLon & temp.lon <= MaxLon-MarginLon & ...
        temp.CLcov <= MaxCF & ...
        temp.sand <= MaxSand & ...
        temp.tskin >= MinTskin & ...
        (TC >= Min_pos_TC | TC <= Max_neg_TC);
    switch day_or_night
        case 'day'
            validmask = validmask & temp.fhour;
        case 'night'
            validmask = validmask & ~temp.fhour;
    end

    if sum(validmask) > 0
        disp(['You have ',sprintf('%5d',sum(validmask(:))),' valid L2 pixels on ',datestr(subfileday(ifile))]);
        HH = double(floor(temp.hour(validmask)/10000));
        MM = double(floor((temp.hour(validmask)-10000*HH)/100));
        SS = double(mod(temp.hour(validmask),100));
        % I think it is a good practice to keep 1-D vectors vertical!
        UTC = datenum([HH(:), MM(:), SS(:)])+subfileday(ifile);
        utcall = cat(2,utcall,UTC');
        
        latall = cat(2,latall,temp.lat(validmask));
        lonall = cat(2,lonall,temp.lon(validmask));
        ifovall = cat(2,ifovall,temp.ifov(validmask));
        colnh3all = cat(2,colnh3all,temp.colnh3(validmask));
        totErrall = cat(2,totErrall,temp.totErr(validmask));
        fhourall = cat(2,fhourall,temp.fhour(validmask));
    end
    
end
% NH3 column error, absolute
colnh3errorall = abs(colnh3all.*totErrall/100);
% assign very large error if colnh3 or totErr are nan
colnh3errorall(isnan(colnh3all) | isnan(colnh3errorall)) = 1e25;
colnh3all(isnan(colnh3all)| isnan(colnh3errorall)) = 0;

%% get IASI ellipses' axes and rotation angles
[u, v, t] = F_define_IASI_pixel(latall,ifovall,uuu4,vvv4,ttt4);
%%
disp('Gridding L2 pixels into L3 ...')
Sum_Above = zeros(nrows,ncols);
Sum_Below = zeros(nrows,ncols);
count = 1;
for i = 1:length(lonall)
    % minlon_e is the minimum lon of the elliptical pixel, does not have to
    % be super accurate; minlat_e is the minmum lat; X is the polygon
    [~, minlon_e, minlat_e] =...
        F_construct_ellipse([lonall(i);latall(i)],v(i),u(i),t(i),npoint_ellipse,0);
    
    local_left = lonall(i)-xmargin*(lonall(i)-minlon_e);
    local_right = lonall(i)+xmargin*(lonall(i)-minlon_e);
    
    local_bottom = latall(i)-ymargin*(latall(i)-minlat_e);
    local_top = latall(i)+ymargin*(latall(i)-minlat_e);
    
    x_local_index = xgrid >= local_left & xgrid <= local_right;
    y_local_index = ygrid >= local_bottom & ygrid <= local_top;
    
    x_local_mesh = xmesh(y_local_index,x_local_index);
    y_local_mesh = ymesh(y_local_index,x_local_index);
    SG = F_2D_SG(x_local_mesh,y_local_mesh,lonall(i),latall(i),2*v(i),2*u(i),2,2,-t(i));
    
    Sum_Above(y_local_index,x_local_index) = Sum_Above(y_local_index,x_local_index)+...
        SG/(v(i)*u(i))/colnh3errorall(i)*colnh3all(i);
    Sum_Below(y_local_index,x_local_index) = Sum_Below(y_local_index,x_local_index)+...
        SG/(v(i)*u(i))/colnh3errorall(i);
    if i == count*round(length(lonall)/10)
        disp([num2str(count*10),' % finished'])
        count = count+1;
    end
end
Average = Sum_Above./Sum_Below;
%%
Clim = [0 10e16];% min/max plotted nh3 column
% shape file for US states
US_states = shaperead('/data/tempo1/Shared/kangsun/run_WRF/shapefiles/cb_2015_us_state_500k/cb_2015_us_state_500k.shp');

close all
figure('unit','inch','color','w','position',[-12 0 8 10])
ax1 = subplot(2,1,1);
h = pcolor(xgrid,ygrid,Average/1e16);set(h,'edgecolor','none');
caxis(Clim/1e16)
xlim([MinLon MaxLon])
ylim([MinLat MaxLat])
hc = colorbar('south');
set(hc,'position',[0.35 0.53 0.3 0.02])
set(get(hc,'xlabel'),'string','Ammonia column [10^{16} molec/cm2]')

hold on
for istate = 1:length(US_states)
    plot(US_states(istate).X,US_states(istate).Y,'color',[1 1 1 .5])
end

mcolorarray = colnh3all;
mxcolor = Clim(2);
mncolor = Clim(1);
mcolorarray(mcolorarray > mxcolor) = mxcolor;
mcolorarray(mcolorarray < mncolor) = mncolor;
mxlabel = 64;
mnlabel = 1;
CC = parula(mxlabel);
mcolorarray = round(interp1([mncolor mxcolor],[mnlabel mxlabel],mcolorarray));

ax2 = subplot(2,1,2);
for i = 1:length(lonall)
    [~,~,~,h] =...
        F_construct_ellipse([lonall(i);latall(i)],v(i),u(i),t(i),npoint_ellipse,1);
    set(h,'color',CC(mcolorarray(i),:),'linewidth',0.5)
end
hold on
for istate = 1:length(US_states)
    plot(US_states(istate).X,US_states(istate).Y,'color','y')
end
set(ax2,'xlim',get(ax1,'xlim'),'ylim',get(ax1,'ylim'))
%%
export_fig([plotdir,'Colorado_Res_',num2str(Res),'.png'],'-r150')
%%
% A = zeros(20,20);
% parfor i = 1:10
%     tmp = zeros(20,20);
%     tmp(i,i+1) = 1;
%     A = A+tmp;
% end