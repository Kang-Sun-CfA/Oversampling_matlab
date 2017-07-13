function output = F_download_subset_OMNO2(inp)
% Download NASA OMNO2 data selectively, subset and filter the data, save
% filtered data into L2g folder for future conversion into L3 data
% The L2 data will be saved under L2dir/yyyy/doy. If the L2 files are
% already there, will simply subset/filter the L2 data.

% not work for windows OS.

% Updated by Kang Sun from save_OMNO2 on 2017/07/12 to be consistent with
% F_subset_IASI.m

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

% if download xml, meta data files
if_download_xml = inp.if_download_xml;

% if download the he5 data
if_download_he5 = inp.if_download_he5;

% if delete he5 files to save disk space
if_delete_he5 = inp.if_delete_he5;

L2dir = inp.L2dir;
cd(L2dir)

% location of a rough boundary of OMI swath. see /data/tempo1/Shared/kangsun/OMNO2/Important_constant/OMI_BDR.mat
% for an example
if ~isfield(inp,'swath_BDR_fn')
    if if_download_he5
        error('no bdr file!')
    else
        swath_BDR_fn = '';
    end
else
swath_BDR_fn = inp.swath_BDR_fn;
end

url0 = inp.url0;% = 'https://aura.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level2/OMNO2.003/';
%% define L2 variables
ntrack = 60;

usextrack = 1:ntrack; % just use all xtrack positions.

swathname = 'ColumnAmountNO2';

% variables to read from OMI L2 files
varname = {'CloudFraction','ColumnAmountNO2','ColumnAmountNO2Trop',...
    'ColumnAmountNO2Std','ColumnAmountNO2TropStd','MeasurementQualityFlags',...
    'XTrackQualityFlags','VcdQualityFlags','AmfTrop'};

geovarname = {'Latitude','Longitude','Time','SolarZenithAngle',...
    'FoV75CornerLatitude','FoV75CornerLongitude'};

% extent of orbit given equator crossing lon == 0
if length(swath_BDR_fn) > 1
bdrstruct = load(swath_BDR_fn);
left_bdr = bdrstruct.left_bdr;
right_bdr = bdrstruct.right_bdr;
else
    left_bdr = [];
    right_bdr = [];
end

TAI93ref = datenum('Jan-01-1993');

day_array = (datenum(Startdate):1:datenum(Enddate))';
datevec_array = datevec(day_array);
Jan01_array = datevec_array;
Jan01_array(:,2:end) = 0;
doy_array = day_array-datenum(Jan01_array);
year_array = datevec_array(:,1);

nday = length(day_array);

savedata = cell(nday,1);

for iday = 1:nday
    day_dir = [num2str(year_array(iday)),'/',sprintf('%03d',doy_array(iday)),'/'];
    url1 = [url0,day_dir];
    if ~exist(day_dir,'dir')
        mkdir(day_dir)
    end
    cd(day_dir)
    % wget commend to download all xml meta data
    if if_download_xml
        str = ['wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A xml "',...
            url1,'"'];
        unix(str);
    end
    flist = dir;
    tmp_savedata = [];
    for ixml = 1:length(flist)
        % all xml files, not too small (. and ..), not too large (he5)
        if flist(ixml).bytes > 10000 && flist(ixml).bytes < 1000000
            % open xml and decide if need to download he5 data
            fid = fopen(flist(ixml).name);
            while 1
                s = fgetl(fid);
                if ~ischar(s), break, end;
                if strfind(s,'<EquatorCrossingLongitude>')
                    EqLon = cell2mat(textscan(s,'<EquatorCrossingLongitude>%f</EquatorCrossingLongitude>'));
                    break;
                end
            end
            fclose(fid);
            this_left_bdr = left_bdr;
            this_right_bdr = right_bdr;
            this_left_bdr(1,:) = this_left_bdr(1,:)+EqLon;
            this_right_bdr(1,:) = this_right_bdr(1,:)+EqLon;
            %             plot(this_left_bdr(1,:),this_left_bdr(2,:),this_right_bdr(1,:),this_right_bdr(2,:))
            orbit_ul = interp1(this_left_bdr(2,:),this_left_bdr(1,:),MaxLat);
            orbit_ur = interp1(this_right_bdr(2,:),this_right_bdr(1,:),MaxLat);
            orbit_ll = interp1(this_left_bdr(2,:),this_left_bdr(1,:),MinLat);
            orbit_lr = interp1(this_right_bdr(2,:),this_right_bdr(1,:),MinLat);
            % decide if he5 file worth download
            if (MinLon >= orbit_ul && MinLon <= orbit_ur) ||...
                    (MaxLon >= orbit_ul && MaxLon <= orbit_ur) ||...
                    (MinLon >= orbit_ll && MinLon <= orbit_lr) ||...
                    (MaxLon >= orbit_ll && MaxLon <= orbit_lr) ||...
                    (MinLon <= orbit_ul && MaxLon >= orbit_ur) ||...
                    (MinLon <= orbit_ll && MaxLon >= orbit_lr)
                fn = flist(ixml).name(1:end-4);
                if ~exist(fn,'file')
                    if if_download_he5
                        % download the he5 file
                        str = ['wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies ',...
                            url1,fn];
                        unix(str);
                    end
                end
                % open the he5 file, massaging the data
                datavar = F_read_he5(fn,swathname,varname,geovarname);
                xtrackmask = false(size(datavar.Latitude));
                xtrackmask(usextrack,:) = true;
                
                xtrack_N = repmat((1:60)',[1,size(datavar.Latitude,2)]);

                validmask = datavar.Latitude >= MinLat-MarginLat & datavar.Latitude <= MaxLat+MarginLat & ...
                    datavar.Longitude >= MinLon-MarginLon & datavar.Longitude <= MaxLon+MarginLon & ...
                    datavar.VcdQualityFlags.data == 0 & ...
                    (datavar.XTrackQualityFlags.data == 0 | datavar.XTrackQualityFlags.data == 255) & ...
                    datavar.SolarZenithAngle <= MaxSZA & ...
                    datavar.CloudFraction.data.*datavar.CloudFraction.ScaleFactor <= MaxCF & ...
                    xtrackmask;
                if sum(validmask(:)) > 0
                    disp(['You have ',sprintf('%5d',sum(validmask(:))),' valid L2 pixels on orbit ',fn(35:39),'.']);
                end
                
                tempVCD = datavar.ColumnAmountNO2Trop.data(validmask);
                tempVCD_unc = datavar.ColumnAmountNO2TropStd.data(validmask);
                tempAMF = datavar.AmfTrop.data(validmask);
                tempxtrack_N = xtrack_N(validmask);
                tempUTC = datavar.Time(validmask)/86400+TAI93ref;
                corner_data_size = size(datavar.FoV75CornerLatitude);
                if corner_data_size(3) == 4
                    Lat_lowerleft = squeeze(datavar.FoV75CornerLatitude(:,:,1));
                    Lat_lowerright = squeeze(datavar.FoV75CornerLatitude(:,:,2));
                    Lat_upperleft = squeeze(datavar.FoV75CornerLatitude(:,:,4));
                    Lat_upperright = squeeze(datavar.FoV75CornerLatitude(:,:,3));
                    
                    Lon_lowerleft = squeeze(datavar.FoV75CornerLongitude(:,:,1));
                    Lon_lowerright = squeeze(datavar.FoV75CornerLongitude(:,:,2));
                    Lon_upperleft = squeeze(datavar.FoV75CornerLongitude(:,:,4));
                    Lon_upperright = squeeze(datavar.FoV75CornerLongitude(:,:,3));
                elseif corner_data_size(1) == 4
                    Lat_lowerleft = squeeze(datavar.FoV75CornerLatitude(1,:,:));
                    Lat_lowerright = squeeze(datavar.FoV75CornerLatitude(2,:,:));
                    Lat_upperleft = squeeze(datavar.FoV75CornerLatitude(4,:,:));
                    Lat_upperright = squeeze(datavar.FoV75CornerLatitude(3,:,:));
                    
                    Lon_lowerleft = squeeze(datavar.FoV75CornerLongitude(1,:,:));
                    Lon_lowerright = squeeze(datavar.FoV75CornerLongitude(2,:,:));
                    Lon_upperleft = squeeze(datavar.FoV75CornerLongitude(4,:,:));
                    Lon_upperright = squeeze(datavar.FoV75CornerLongitude(3,:,:));
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
                % plot the corners to see if it's alright
                %                 plot(tempLonC,tempLatC,'.k',tempLon_lowerleft,tempLat_lowerleft,'.'...
                %                     ,tempLon_lowerright,tempLat_lowerright,'o',tempLon_upperleft,tempLat_upperleft,'v'...
                %                     ,tempLon_upperright,tempLat_upperright,'*')
                tempSZA = datavar.SolarZenithAngle(validmask);
                tempCF = datavar.CloudFraction.data(validmask).*datavar.CloudFraction.ScaleFactor;
                
                tempdata = [tempLat_lowerleft(:),tempLat_upperleft(:),...
                    tempLat_upperright(:),tempLat_lowerright(:),tempLatC(:),...
                    tempLon_lowerleft(:),tempLon_upperleft(:),...
                    tempLon_upperright(:),tempLon_lowerright(:),tempLonC(:),...
                    tempSZA(:),tempCF(:),tempAMF(:),tempVCD(:),tempVCD_unc(:),...
                    tempxtrack_N(:),tempUTC(:)];
                
                tmp_savedata = cat(1,tmp_savedata,double(tempdata));
                
                if if_delete_he5
                    delete(fn);
                end
            end
        end
    end
    savedata{iday} = tmp_savedata;
    cd(L2dir)
end
savedata = cell2mat(savedata);
output.lat = single(savedata(:,5));
output.lon = single(savedata(:,10));
output.lat_r = single(savedata(:,1:4));
output.lon_r = single(savedata(:,6:9));
output.sza = single(savedata(:,11));
output.cloudfrac = single(savedata(:,12));
output.amf = single(savedata(:,13));
output.colno2 = single(savedata(:,14));
output.colno2error = single(savedata(:,15));
output.xtrackN = single(savedata(:,16));
output.utc = double(savedata(:,17));

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
