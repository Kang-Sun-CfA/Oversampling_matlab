function output = F_subset_OMHCHO(inp)
% subset SAO OMHCHO, output consistent format like F_subset_IASI.m
% updated from F_subset_OMNO2.m by Kang Sun on 2017/07/16

olddir = pwd;

ntrack = 60; % OMI

Startdate = inp.Startdate;
Enddate = inp.Enddate;
MinLat = inp.MinLat;
MinLon = inp.MinLon;
MaxLat = inp.MaxLat;
MaxLon = inp.MaxLon;

% grey area, L2 pixel center cannot be there, but pixel boundaries can
MarginLat = 0.5;
MarginLon = 0.5;

% max cloud fraction and SZA
MaxCF = inp.MaxCF;
MaxSZA = inp.MaxSZA;

L2dir = inp.L2dir;

if ~isfield(inp,'usextrack')
    usextrack = 1:60;
else
usextrack = inp.usextrack;
end

swathname = 'OMI Total Column Amount HCHO';

% variables to read from OMI L2 files
varname = {'AMFCloudFraction','ReferenceSectorCorrectedVerticalColumn',...
    'ColumnUncertainty','MainDataQualityFlag','PixelCornerLatitudes',...
    'PixelCornerLongitudes','AirMassFactor','AirMassFactorDiagnosticFlag'};
geovarname = {'Latitude','Longitude','TimeUTC','SolarZenithAngle',...
    'XtrackQualityFlags'};

%TAI93ref = datenum('Jan-01-1993');

disp('Reading the OMI L2 file name list ....')
fprintf('\n')

if ~isfield(inp,'filelist')
filelist = dir(L2dir);
else
    filelist = inp.filelist;
end

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

useindex = false(size(filelist));
for iperiod = 1:size(Startdate,1)
useindex = useindex | ...
    fileday >= datenum(Startdate(iperiod,:)) & fileday <= datenum(Enddate(iperiod,:));
end

% useindex = useindex(1):Step:useindex(end);
subfilelist = filelist(useindex);
subfileorbit = fileorbit(useindex);
norbit = sum(useindex);

lonc = single([]);
latc = lonc;
lonr = lonc;latr = lonc;
sza = lonc;cloudfrac = lonc;ift = lonc;
utc = double([]);colhcho = lonc;colhchoerror = lonc;

for iorbit = 1:norbit
    fn = [L2dir,subfilelist(iorbit).name];
    datavar = F_read_he5(fn,swathname,varname,geovarname);
    
    xtrackmask = false(size(datavar.Latitude));
    xtrackmask(usextrack,:) = true;
    xtrack_N = repmat((1:ntrack)',[1,size(datavar.Latitude,2)]);
    
    validmask = datavar.Latitude >= MinLat-MarginLat & datavar.Latitude <= MaxLat+MarginLat & ...
        datavar.Longitude >= MinLon-MarginLon & datavar.Longitude <= MaxLon+MarginLon & ...
        datavar.MainDataQualityFlag.data == 0 & ...
        datavar.AirMassFactorDiagnosticFlag.data >= 0 & ...
        datavar.XtrackQualityFlags == 0 & ...
        datavar.SolarZenithAngle <= MaxSZA & ...
        datavar.AMFCloudFraction.data <= MaxCF & ...
        xtrackmask;
    if sum(validmask(:)) > 0
    disp(['You have ',sprintf('%5d',sum(validmask(:))),...
        ' valid L2 pixels in orbit ',num2str(subfileorbit(iorbit)),'.']);
    end
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
    tempSZA = datavar.SolarZenithAngle(validmask);
    tempCF = datavar.AMFCloudFraction.data(validmask);
    tempxtrack_N = xtrack_N(validmask);
    tempUTC = repmat(datenum(datavar.TimeUTC')',[ntrack,1]);
    tempUTC = tempUTC(validmask);
   % tempUTC = double(xTime(validmask))/86400+TAI93ref;

    lonc = cat(1,lonc,single(tempLonC(:)));
        latc = cat(1,latc,single(tempLatC(:)));
        templonr = [tempLon_lowerleft(:),tempLon_upperleft(:),...
                    tempLon_upperright(:),tempLon_lowerright(:)];
        lonr = cat(1,lonr,single(templonr));
        templatr = [tempLat_lowerleft(:),tempLat_upperleft(:),...
                    tempLat_upperright(:),tempLat_lowerright(:)];
        latr = cat(1,latr,single(templatr));
        sza = cat(1,sza,single(tempSZA(:)));
        cloudfrac = cat(1,cloudfrac,single(tempCF(:)));
        ift = cat(1,ift,single(tempxtrack_N(:)));
        utc = cat(1,utc,double(tempUTC(:)));
        colhcho = cat(1,colhcho,single(tempVCD(:)));
        colhchoerror = cat(1,colhchoerror,single(tempVCD_unc(:)));
end

output.lonc = lonc;
output.lonr = lonr;
output.latc = latc;
output.latr = latr;
output.sza = sza;
output.cloudfrac = cloudfrac;
output.ift = ift;
output.utc = utc;
output.colhcho = colhcho;
output.colhchoerror = colhchoerror;

cd(olddir)

function datavar = F_read_he5(filename,swathname,varname,geovarname)
% filename = '/home/kangsun/OMH2O/result/OMH2O_2005m0714t2324-o05311_test_TLCF.he5';
if isempty(swathname)
    swathname = 'OMI Total Column Amount H2O';
end
swn = ['/HDFEOS/SWATHS/',swathname];
% varname = {'FittingRMS','ColumnUncertainty','FitConvergenceFlag'};
% geovarname = {'Latitude','Longitude','TimeUTC','SolarZenithAngle','ViewingZenithAngle'};
file_id = H5F.open (filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');

for i = 1:length(geovarname)
    DATAFIELD_NAME = [swn,'/Geolocation Fields/',geovarname{i}];
    data_id=H5D.open(file_id, DATAFIELD_NAME);
    %     ATTRIBUTE = 'Title';
    %     attr_id = H5A.open_name (data_id, ATTRIBUTE);
    %     long_name=H5A.read (attr_id, 'H5ML_DEFAULT');
    datavar.(geovarname{i})=H5D.read(data_id,'H5T_NATIVE_DOUBLE', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT');
    
end

for i = 1:length(varname)
    % Open the dataset.
    DATAFIELD_NAME = [swn,'/Data Fields/',varname{i}];
    data_id = H5D.open (file_id, DATAFIELD_NAME);
    % Read attributes.
    try
        ATTRIBUTE = 'Offset';
        attr_id = H5A.open_name (data_id, ATTRIBUTE);
        datavar.(varname{i}).(ATTRIBUTE) = H5A.read (attr_id, 'H5ML_DEFAULT');
        
        ATTRIBUTE = 'ScaleFactor';
        attr_id = H5A.open_name (data_id, ATTRIBUTE);
        datavar.(varname{i}).(ATTRIBUTE) = H5A.read (attr_id, 'H5ML_DEFAULT');
    catch
        warning('No attributes to read!')
    end
    % Read the dataset.
    datavar.(varname{i}).data=H5D.read (data_id,'H5T_NATIVE_DOUBLE', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT');
    %     datavar.(varname{i}).name = long_name(:)';
end

% Close and release resources.
% H5A.close (attr_id)
H5D.close (data_id);
H5F.close (file_id);
