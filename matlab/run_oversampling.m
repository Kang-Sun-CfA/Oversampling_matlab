clc;
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
plotdir = '/data/tempo1/Shared/kangsun/Oversampling/RegridPixels/plot/';

addpath('/home/kangsun/matlab functions/export_fig/')
addpath('/home/kangsun/matlab functions/')
%% parameter inputs
% begin and start dates
Startdate = [2005 3 1;
    2006 3 1;
    2007 3 1;
    2008 3 1];

Enddate = [2005 5 31;
    2006 5 31;
    2007 5 31;
    2008 5 31];

Startdate = [2011 12 1;
    2012 12 1;
    2013 12 1;
    2014 12 1;
    2015 12 1];

Enddate = [2012 3 1;
    2013 3 1;
    2014 3 1;
    2015 3 1;
    2016 3 1];
% step in dates, not very useful in this project
% Step = 1;

% do you wanna use destriped OMI data?
if_destripe = true;

% lat lon box
MinLat = 15; MaxLat = 50;
MinLon = 70; MaxLon = 135;
%  MinLon = -inf; MaxLon = inf;

% max cloud fraction and SZA
MaxCF = 0.3;
MaxSZA = 60;

% xtrack position mask
usextrack = [1:60];

% Resolution of oversampled L3 data?
Res = 0.5;

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
%%
savedata = cell2mat(savedata);
savedata = cat(2,(1:size(savedata,1))',savedata);
input_fn = ['Lat_',num2str(MinLat),'_',num2str(MaxLat),'_Lon_',...
    num2str(MinLon),'_',num2str(MaxLon),'_Year_',num2str(Startdate(1,1)),...
    num2str(Startdate(1,2),'%02d'),'_',...
    num2str(Enddate(end,1)),num2str(Enddate(end,2),'%02d'),'.dat'];
cd(inputdir)
fid = fopen(input_fn,'w');
% print data
fprintf(fid,['%8d',repmat('%15.6f',1,13),repmat('%15.6E',1,2),'\n'],savedata');
fclose(fid);
%%
% Res = 0.02;
disp('Running Lei Zhu''s pixel-regriding program ...')
fprintf('\n')
tic
output_fn = ['Res_',num2str(Res),'_Lat_',num2str(MinLat),'_',...
    num2str(MaxLat),'_Lon_',...
    num2str(MinLon),'_',num2str(MaxLon),'_Year_',num2str(Startdate(1,1)),...
    num2str(Startdate(1,2),'%02d'),'_',...
    num2str(Enddate(end,1)),num2str(Enddate(end,2),'%02d'),'.dat'];
cd(rundir)

% I know this is stupid, give me a better way

fid = fopen('run_KS.sh','w');
fprintf(fid,['set Input_Dir = "',inputdir(1:end),'"\n']);
fprintf(fid,['set Output_Dir = "',outputdir(1:end),'"\n']);

fprintf(fid,['set Res = ',num2str(Res),'\n']);

fprintf(fid,['set Input_Filename = "',input_fn,'"\n']);
fprintf(fid,['set Output_Filename = "',output_fn,'"\n']);

fprintf(fid,['./RegridPixels.x<<EOF\n',...
    '$Input_Dir\n',...
    '$Output_Dir \n',...
    '$Input_Filename\n',...
    '$Output_Filename\n',...
    '$Res\n',...
    'EOF\n',...
    'quit:\n',...
    'exit']);
fclose(fid);

unix('tcsh ./run_KS.sh');
tt = toc;
disp(['Took ',num2str(tt),' s'])
fprintf('\n')
%%
fid = fopen([outputdir,output_fn]);
C = cell2mat(textscan(fid,'%f%f%f%f%f%f','delimiter',' ','multipledelimsasone',1));
fclose(fid);
[nlat,Ilat] = max(C(:,1));
max_grid_lat = C(Ilat,3);
min_grid_lat = max_grid_lat-(nlat-1)*Res;
latgrid = (1:nlat)*Res+min_grid_lat-Res;

[nlon,Ilon] = max(C(:,2));
max_grid_lon = C(Ilon,4);
min_grid_lon = max_grid_lon-(nlon-1)*Res;
longrid = (1:nlon)*Res+min_grid_lon-Res;

value = nan(nlat,nlon);
nave = value;
for i = 1:size(C,1)
    value(C(i,1),C(i,2)) = C(i,5);
    nave(C(i,1),C(i,2)) = C(i,6);
end
%% plot
statelist = [8, 55, 32, 24 46];
S         = shaperead('/data/tempo1/Shared/kangsun/run_WRF/shapefiles/cb_2015_us_state_500k/cb_2015_us_state_500k.shp');

% S = shaperead('/data/tempo1/Shared/kangsun/run_WRF/shapefiles/asia/USDoS_LSIB4b_Eurasia_Sep2012.shp');% Slake     = shaperead('/data/tempo1/Shared/kangsun/run_WRF/shapefiles/ne_10m_lakes_north_america/ne_10m_lakes_north_america.shp');
if ~exist('lakelist','var')
Llake     = shaperead('/data/tempo1/Shared/kangsun/run_WRF/shapefiles/ne_10m_lakes/ne_10m_lakes.shp');
lakelist = [];
left = -82;right = -74;down = 39.5;up = 45;
for i = 1:length(Llake)
    tmp = Llake(i).BoundingBox;
    if ((tmp(1,1) > left && tmp(1,1) < right) || ...
            (tmp(2,1) > left && tmp(2,1) < right)) && ...
        ((tmp(2,1) > down && tmp(2,1) < up) || ...
            (tmp(2,2) > down && tmp(2,2) < up)) 
        lakelist = [lakelist i];
    end
end
end

close all
figure('color','w','unit','inch','position',[0 1 10 8])
set(0,'defaultaxesfontsize',13)
% h = scatter(C(:,4),C(:,3),[],C(:,5));
plotmat = value;
plotmat(nave < minave) = nan;
h = pcolor(longrid,latgrid,value);set(h,'edgecolor','none')
hc = colorbar;
set(get(hc,'ylabel'),'string','HCHO VCD [molecules cm^{-2}]')
caxis([0 1e16])
xlim([min(C(:,4)) max(C(:,4))])
ylim([min(C(:,3)) max(C(:,3))])
set(gca,'linewidth',1)
hold on
for istate = 1:length(S)
    plot(S(istate).X,S(istate).Y,'color','w')
end

% for ilake = lakelist
%     plot(Llake(ilake).X,Llake(ilake).Y,'color','w')
% end
title(['2012-2016 DJF, Resolution = ',num2str(Res)])
xlabel('Longitude');ylabel('Latitude')
%%
export_fig([plotdir,'Res_',num2str(Res),'_Lat_',num2str(MinLat),'_',...
    num2str(MaxLat),'_Lon_',num2str(MinLon),'_',num2str(MaxLon),...
    '_2012-2016 DJF.png'],'-r100')