function output = F_subset_OMPIXCOR(inp)
% subset NASA OMNO2, output consistent format like F_subset_IASI.m
% updated from F_subset_OMNO2.m by Kang Sun on 2017/07/16

sfs = filesep;
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

L2dir = inp.L2dir;

if ~isfield(inp,'usextrack')
    usextrack = 1:60;
else
    usextrack = inp.usextrack;
end

if ~isfield(inp,'swathname')
    swathname = 'OMI Ground Pixel Corners VIS';
else
    swathname = inp.swathname;
end

% variables to read from OMI L2 files
varname = {'FoV75CornerLatitude','FoV75CornerLongitude','FoV75Area',...
    'TiledCornerLatitude','TiledCornerLongitude','TiledArea'};

geovarname = {'Latitude','Longitude','Time'};

TAI93ref = datenum('Jan-01-1993');

day_array = (datenum(Startdate):1:datenum(Enddate))';
datevec_array = datevec(day_array);
Jan01_array = datevec_array;
Jan01_array(:,2:end) = 0;
doy_array = day_array-datenum(Jan01_array);
year_array = datevec_array(:,1);

nday = length(day_array);

lonc = single([]);
latc = lonc;
lonr = lonc;latr = lonc;
sza = lonc;cloudfrac = lonc;ift = lonc;
utc = double([]);colno2 = lonc;colno2error = lonc;

for iday = 1:nday
    day_dir = [L2dir,num2str(year_array(iday)),sfs,sprintf('%03d',doy_array(iday)),sfs];
    %cd(day_dir);
    he5fn = dir([day_dir,'*.he5']);
    if isempty(he5fn);
        warning([day_dir,' contains no HE5 files!']);
        continue;
    end;
    for ifile = 1:length(he5fn)
        fn = [day_dir,he5fn(ifile).name];
        
        % open the he5 file, massaging the data
        datavar = F_read_he5(fn,swathname,varname,geovarname);
        xtrackmask = false(size(datavar.Latitude));
        xtrackmask(usextrack,:) = true;
        
        xtrack_N = repmat((1:ntrack)',[1,size(datavar.Latitude,2)]);
        
        validmask = datavar.Latitude >= MinLat-MarginLat & datavar.Latitude <= MaxLat+MarginLat & ...
            datavar.Longitude >= MinLon-MarginLon & datavar.Longitude <= MaxLon+MarginLon & ...
            xtrackmask;
        if sum(validmask(:)) > 0
            disp(['You have ',sprintf('%5d',sum(validmask(:))),' valid L2 pixels on orbit ',he5fn(ifile).name(3+(35:39)),'.']);
        end
        
        tempxtrack_N = xtrack_N(validmask);
        xTime = repmat(datavar.Time(:)',[ntrack,1]);
        tempUTC = double(xTime(validmask))/86400+TAI93ref;
        corner_data_size = size(datavar.FoV75CornerLatitude.data);
        if corner_data_size(3) == 4
            Lat_lowerleft = squeeze(datavar.FoV75CornerLatitude.data(:,:,1));
            Lat_lowerright = squeeze(datavar.FoV75CornerLatitude.data(:,:,2));
            Lat_upperleft = squeeze(datavar.FoV75CornerLatitude.data(:,:,4));
            Lat_upperright = squeeze(datavar.FoV75CornerLatitude.data(:,:,3));
            
            Lon_lowerleft = squeeze(datavar.FoV75CornerLongitude.data(:,:,1));
            Lon_lowerright = squeeze(datavar.FoV75CornerLongitude.data(:,:,2));
            Lon_upperleft = squeeze(datavar.FoV75CornerLongitude.data(:,:,4));
            Lon_upperright = squeeze(datavar.FoV75CornerLongitude.data(:,:,3));
        elseif corner_data_size(1) == 4
            Lat_lowerleft = squeeze(datavar.FoV75CornerLatitude.data(1,:,:));
            Lat_lowerright = squeeze(datavar.FoV75CornerLatitude.data(2,:,:));
            Lat_upperleft = squeeze(datavar.FoV75CornerLatitude.data(4,:,:));
            Lat_upperright = squeeze(datavar.FoV75CornerLatitude.data(3,:,:));
            
            Lon_lowerleft = squeeze(datavar.FoV75CornerLongitude.data(1,:,:));
            Lon_lowerright = squeeze(datavar.FoV75CornerLongitude.data(2,:,:));
            Lon_upperleft = squeeze(datavar.FoV75CornerLongitude.data(4,:,:));
            Lon_upperright = squeeze(datavar.FoV75CornerLongitude.data(3,:,:));
        end
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
        
        lonc = cat(1,lonc,single(tempLonC(:)));
        latc = cat(1,latc,single(tempLatC(:)));
        templonr = [tempLon_lowerleft(:),tempLon_upperleft(:),...
            tempLon_upperright(:),tempLon_lowerright(:)];
        lonr = cat(1,lonr,single(templonr));
        templatr = [tempLat_lowerleft(:),tempLat_upperleft(:),...
            tempLat_upperright(:),tempLat_lowerright(:)];
        latr = cat(1,latr,single(templatr));
        ift = cat(1,ift,single(tempxtrack_N(:)));
        utc = cat(1,utc,double(tempUTC(:)));
    end
end

output.lonc = lonc;
output.lonr = lonr;
output.latc = latc;
output.latr = latr;
output.sza = sza;
output.cloudfrac = cloudfrac;
output.ift = ift;
output.utc = utc;
output.colno2 = colno2;
output.colno2error = colno2error;

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
