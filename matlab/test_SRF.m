% created by Kang Sun on 2017/06/03. Abandon the cakecut method, just use
% spatial response function (SRF)
clc;
if ~exist('savedata','var');clear;end
feature('numcores');
ncore = ans;
ncoreuse = input('Please decide how many cores to use ... ');
%% directory inputs
% directory of OMI L2 data
datadir = '/data/tempo1/Shared/OMHCHO/';

% directory for intermediate data, input to the oversampling algorthim
inputdir = '/data/tempo1/Shared/kangsun/Oversampling/RegridPixels/input/';

% directory for output from oversampling
outputdir = '/data/tempo1/Shared/kangsun/Oversampling/RegridPixels/output/';

% dirctory containing RegridPixels.x
rundir = '/data/tempo1/Shared/kangsun/Oversampling/RegridPixels/';

% directory for plot
plotdir = '/home/kangsun/OMI/Oversampling_matlab/plot/';

% matlab code dir
codedir = '/home/kangsun/OMI/Oversampling_matlab/';

addpath('/home/kangsun/matlab functions/export_fig/')
addpath('/home/kangsun/matlab functions/')
%% parameter inputs
% begin and start dates
Startdate = [2005 5 1;
    2006 5 1;
    2007 5 1;
    2008 5 1];

Enddate = [2005 8 31;
    2006 8 31;
    2007 8 31;
    2008 8 31];
% Startdate = [2005 5 1];

% Enddate = [2005 8 31];
if ~exist('savedata','var')
% Startdate = [2005 7 1];
% Enddate = [2005 7 31];
% step in dates, not very useful in this project
% Step = 1;
% 
% % do you wanna use destriped OMI data?
% if_destripe = true;

% lat lon box
MinLat = 28; MaxLat = 34;
MinLon = -99.5; MaxLon = -92.5;
%  MinLon = -inf; MaxLon = inf;

% max cloud fraction and SZA
MaxCF = 0.3;
MaxSZA = 60;

% xtrack position mask
usextrack = [11:50];

% Resolution of oversampled L3 data?
Res = 0.02;

% minimal number of L2 pixel per grid point?
minave = 5;
%%
disp('Reading the OMI L2 file name list ....')
fprintf('\n')
tic
if ~exist('filelist','var')
    filelist = dir(datadir);
end

if ~exist('fileday','var')
fileyear = nan(size(filelist));
filemonth = fileyear;
filedate = fileyear;
fileorbit = fileyear;
for i = 1:length(filelist)
    if length(filelist(i).name) > 10
        fileyear(i) = str2double(filelist(i).name(20:23));
        filemonth(i) = str2double(filelist(i).name(25:26));
        filedate(i) = str2double(filelist(i).name(27:28));
        fileorbit(i) = str2double(filelist(i).name(36:40));
    end
end
int = ~isnan(fileyear);
filelist = filelist(int);
fileyear = fileyear(int);
filemonth = filemonth(int);
filedate = filedate(int);
fileorbit = fileorbit(int);

fileday = datenum([fileyear filemonth filedate]);
end
tt = toc;
disp(['Took ',num2str(tt),' s'])
fprintf('\n')
%%
swathname = 'OMI Total Column Amount HCHO';

% variables to read from OMI L2 files
varname = {'AMFCloudFraction','ReferenceSectorCorrectedVerticalColumn',...
    'ColumnUncertainty','MainDataQualityFlag','PixelCornerLatitudes',...
    'PixelCornerLongitudes','AirMassFactor','AirMassFactorDiagnosticFlag'};
geovarname = {'Latitude','Longitude','TimeUTC','SolarZenithAngle',...
    'XtrackQualityFlags'};

useindex = false(size(filelist));
for iperiod = 1:size(Startdate,1)
useindex = useindex | ...
    fileday >= datenum(Startdate(iperiod,:)) & fileday <= datenum(Enddate(iperiod,:));
end

% useindex = useindex(1):Step:useindex(end);
subfilelist = filelist(useindex);
norbit = sum(useindex);
savedata = cell(norbit,1);
%%
p_obj = parpool(ncoreuse);
disp('Loading and subsetting OMI L2 data in parallel...')
fprintf('\n')
tic
parfor iorbit = 1:norbit
    fn = [datadir,subfilelist(iorbit).name];
    datavar = F_read_he5(fn,swathname,varname,geovarname);
    
    xtrackmask = false(size(datavar.Latitude));
    xtrackmask(usextrack,:) = true;
    
    validmask = datavar.Latitude >= MinLat & datavar.Latitude <= MaxLat & ...
        datavar.Longitude >= MinLon & datavar.Longitude <= MaxLon & ...
        datavar.MainDataQualityFlag.data == 0 & ...
        datavar.AirMassFactorDiagnosticFlag.data >= 0 & ...
        datavar.XtrackQualityFlags == 0 & ...
        datavar.SolarZenithAngle <= MaxSZA & ...
        datavar.AMFCloudFraction.data <= MaxCF & ...
        xtrackmask;
    if sum(validmask(:)) > 0
    disp(['You have ',sprintf('%5d',sum(validmask(:))),' valid L2 pixels in orbit ',num2str(iorbit),'.']);
    end
%     if if_destripe
%         tempVCD = datavar.ColumnAmountDestriped.data(validmask);
%     else
%         tempVCD = datavar.ColumnAmount.data(validmask);
%     end
    tempVCD = datavar.ReferenceSectorCorrectedVerticalColumn.data(validmask);
    tempVCD_unc = datavar.ColumnUncertainty.data(validmask);
    tempAMF = datavar.AirMassFactor.data(validmask);
    
    Lat_lowerleft = datavar.PixelCornerLatitudes.data(1:end-1,1:end-1);
    Lat_lowerright = datavar.PixelCornerLatitudes.data(2:end,1:end-1);
    Lat_upperleft = datavar.PixelCornerLatitudes.data(1:end-1,2:end);
    Lat_upperright = datavar.PixelCornerLatitudes.data(2:end,2:end);
    
    Lon_lowerleft = datavar.PixelCornerLongitudes.data(1:end-1,1:end-1);
    Lon_lowerright = datavar.PixelCornerLongitudes.data(2:end,1:end-1);
    Lon_upperleft = datavar.PixelCornerLongitudes.data(1:end-1,2:end);
    Lon_upperright = datavar.PixelCornerLongitudes.data(2:end,2:end);
    
    tempLatC = datavar.Latitude(validmask);
    tempLonC = datavar.Longitude(validmask);
    
    tempLat_lowerleft = Lat_lowerleft(validmask);
    tempLat_lowerright = Lat_lowerright(validmask);
    tempLat_upperleft = Lat_upperleft(validmask);
    tempLat_upperright = Lat_upperright(validmask);
    
    tempLon_lowerleft = Lon_lowerleft(validmask);
    tempLon_lowerright = Lon_lowerright(validmask);
    tempLon_upperleft = Lon_upperleft(validmask);
    tempLon_upperright = Lon_upperright(validmask);
    %     % plot the corners to see if it's alright
    %     plot(tempLonC,tempLatC,'.k',tempLon_lowerleft,tempLat_lowerleft,'.'...
    %         ,tempLon_lowerright,tempLat_lowerright,'o',tempLon_upperleft,tempLat_upperleft,'v'...
    %         ,tempLon_upperright,tempLat_upperright,'*')
    tempSZA = datavar.SolarZenithAngle(validmask);
    tempCF = datavar.AMFCloudFraction.data(validmask);
    
    tempdata = [tempLat_lowerleft(:),tempLat_upperleft(:),...
        tempLat_upperright(:),tempLat_lowerright(:),tempLatC(:),...
        tempLon_lowerleft(:),tempLon_upperleft(:),...
        tempLon_upperright(:),tempLon_lowerright(:),tempLonC(:),...
        tempSZA(:),tempCF(:),tempAMF(:),tempVCD(:),tempVCD_unc(:)];
    savedata{iorbit} = single(tempdata);
    
end
tt = toc;
disp(['Took ',num2str(tt),' s'])
fprintf('\n')
% %%
savedata = cell2mat(savedata);
savedata = cat(2,(1:size(savedata,1))',savedata);
input_fn = ['Lat_',num2str(MinLat),'_',num2str(MaxLat),'_Lon_',...
    num2str(MinLon),'_',num2str(MaxLon),'_Year_',num2str(Startdate(1,1)),...
    num2str(Startdate(1,2),'%02d'),'_',...
    num2str(Enddate(end,1)),num2str(Enddate(end,2),'%02d'),'.dat'];
cd(inputdir)
fid = fopen(input_fn,'w');
% print data for Lei Zhu' fortran program
fprintf(fid,['%8d',repmat('%15.6f',1,13),repmat('%15.6E',1,2),'\n'],savedata');
fclose(fid);
end
% %%
disp('Calculating SRF...')
fprintf('\n')
tic
m = 4;
n = 2;
xmargin = 2;
ymargin = 3;
cd(codedir)
np = size(savedata,1);
LAT = savedata(:,2:6);LON = savedata(:,7:11);
Lon_left = floor(min(LON(:)))-1;
Lon_right = ceil(max(LON(:)))+1;
Lat_low = floor(min(LAT(:)))-1;
Lat_up = ceil(max(LAT(:)))+1;
nrows = (Lat_up-Lat_low)/Res;
ncols = (Lon_right-Lon_left)/Res;

% define x y grids
ygrid = Lat_low+(1:nrows)*Res-0.5*Res;
xgrid = Lon_left+(1:ncols)*Res-0.5*Res;

% define x y mesh
[xmesh, ymesh] = meshgrid(xgrid,ygrid);

VCD = savedata(:,end-1);
VCD_Unc = savedata(:,end);

Sum_Above = zeros(nrows,ncols);
Sum_Below = zeros(nrows,ncols);
count = 1;
for ip = 1:np
    Lon_r = LON(ip,1:4);
    Lat_r = LAT(ip,1:4);
    Lon_c = LON(ip,5);
    Lat_c = LAT(ip,5);
    
vList = [Lon_r(:),Lat_r(:)];


leftpoint = mean(vList(1:2,:));
    rightpoint = mean(vList(3:4,:));
    
    uppoint = mean(vList(2:3,:));
    lowpoint = mean(vList([1 4],:));
    
    % calculate the FWHM of 2-D super gaussian SRF
    % x is the xtrack, different from the OMI pixel paper, which used y
    FWHMx = sqrt((leftpoint(1)-rightpoint(1))^2+(leftpoint(2)-rightpoint(2))^2);
    % y is the along track, different from the OMI pixel paper, which used x
    FWHMy = sqrt((uppoint(1)-lowpoint(1))^2+(uppoint(2)-lowpoint(2))^2);
    
    Angle = -atan((rightpoint(2)-leftpoint(2))/(rightpoint(1)-leftpoint(1)));
    rotation_matrix = [cos(Angle), -sin(Angle);
        sin(Angle),  cos(Angle)];
    
    local_left = Lon_c-xmargin*(Lon_c-Lon_r(2));
    local_right = Lon_c+xmargin*(Lon_r(4)-Lon_c);
    
    local_bottom = Lat_c-ymargin*(Lat_c-Lat_r(1));
    local_top = Lat_c+ymargin*(Lat_r(3)-Lat_c);
    
    x_local_index = xgrid >= local_left & xgrid <= local_right;
    y_local_index = ygrid >= local_bottom & ygrid <= local_top;
    
    x_local_mesh = xmesh(y_local_index,x_local_index);
    y_local_mesh = ymesh(y_local_index,x_local_index);
    SG = F_2D_SG(x_local_mesh,y_local_mesh,Lon_c,Lat_c,FWHMx,FWHMy,m,n,rotation_matrix);
    Sum_Above(y_local_index,x_local_index) = Sum_Above(y_local_index,x_local_index)+...
        SG/(FWHMx*FWHMy)/VCD_Unc(ip)*VCD(ip);
    Sum_Below(y_local_index,x_local_index) = Sum_Below(y_local_index,x_local_index)+...
        SG/(FWHMx*FWHMy)/VCD_Unc(ip);
    if ip == count*round(np/10)
        disp([num2str(count*10),' % finished'])
        count = count+1;
    end
end
Average = Sum_Above./Sum_Below;
tt = toc;
disp(['Took ',num2str(tt),' s'])
fprintf('\n')
%%
S         = shaperead('/data/tempo1/Shared/kangsun/run_WRF/shapefiles/cb_2015_us_state_500k/cb_2015_us_state_500k.shp');

close all
figure('unit','inch','position',[-12 0 10 8])
h = pcolor(xgrid,ygrid,Average);set(h,'edgecolor','none')
hold on
for istate = 1:length(S)
    plot(S(istate).X,S(istate).Y,'color','w')
end
% F_plot_polygon(vList);
%%
delete(p_obj);