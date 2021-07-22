function output = F_subset_BEHR(inp)
% subset NASA BEHR, output consistent format like F_subset_OMNO2.m
% updated from F_subset_OMNO2.m by Kang Sun on 2018/09/06

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

% max cloud fraction and SZA
if isfield(inp,'MaxCF')
MaxCF = inp.MaxCF;
else
    MaxCF = 0.2;
end
if MaxCF > 0.2
    warning('Only cloud fraction <= 20% is kept. Customize your filter to allow higher cloud fraction')
end
MaxSZA = inp.MaxSZA;

L2dir = inp.L2dir;

if ~isfield(inp,'usextrack')
    usextrack = 1:60;
else
usextrack = inp.usextrack;
end

% variables to read from BEHR L2 files
varname = {'BEHRColumnAmountNO2Trop','BEHRQualityFlags','CloudFraction',...
    'FoV75CornerLatitude','FoV75CornerLongitude','Latitude','Longitude',...
    'SolarZenithAngle','Time','ColumnAmountNO2TropStd','XTrackQualityFlags','VcdQualityFlags'};

TAI93ref = datenum('Jan-01-1993');

day_array = (datenum(Startdate):1:datenum(Enddate))';
datevec_array = datevec(day_array);
year_array = datevec_array(:,1);
month_array = datevec_array(:,2);
date_array = datevec_array(:,3);

nday = length(day_array);

lonc = single([]);
latc = lonc;
lonr = lonc;latr = lonc;
sza = lonc;cloudfrac = lonc;ift = lonc;
utc = double([]);colno2 = lonc;colno2error = lonc;

for iday = 1:nday
    day_dir = [L2dir,'OMI_BEHR-DAILY_US_v3-0B_',num2str(year_array(iday)),sprintf('%02d',month_array(iday)),sfs];
    fn = [day_dir,'OMI_BEHR-DAILY_US_v3-0B_',num2str(year_array(iday)),...
        num2str(month_array(iday),'%02d'),num2str(date_array(iday),'%02d'),'.hdf'];
    if ~exist(fn,'file')
        warning([fn,' does not exist!']);
        continue
    end
    inp_behr = [];
    inp_behr.fn = fn;
    inp_behr.varname = varname;
    behr = F_read_BEHR_h5(inp_behr);
    behr_fieldnames = fieldnames(behr);
    nswath = length(behr_fieldnames);
    for iswath = 1:nswath
        % open the he5 file, massaging the data
        datavar = behr.(behr_fieldnames{iswath});
        xtrackmask = false(size(datavar.Latitude));
        % come on! why swith column and row!
%         xtrackmask(usextrack,:) = true;
        xtrackmask(:,usextrack) = true;
        xtrack_N = repmat((1:ntrack),[1,size(datavar.Latitude,1)]);
        xtrack_N = xtrack_N';
        validmask = datavar.Latitude >= MinLat-MarginLat & datavar.Latitude <= MaxLat+MarginLat & ...
            datavar.Longitude >= MinLon-MarginLon & datavar.Longitude <= MaxLon+MarginLon & ...
            mod(datavar.BEHRQualityFlags,2) == 0 & ...
            datavar.SolarZenithAngle <= MaxSZA & ...
            datavar.CloudFraction <= MaxCF & ...
            xtrackmask;
        if sum(validmask(:)) > 0
            disp(['You have ',sprintf('%5d',sum(validmask(:))),...
                ' valid L2 pixels on orbit ',num2str(behr_fieldnames{iswath}(6:end)),'.']);
        end
        
        tempVCD = datavar.BEHRColumnAmountNO2Trop(validmask);
        % use uncertainty from SP
        tempVCD_unc = datavar.ColumnAmountNO2TropStd(validmask);
        tempxtrack_N = xtrack_N(validmask);
        xTime = repmat(datavar.Time(:)',[ntrack,1]);
        tempUTC = double(xTime(validmask))/86400+TAI93ref;
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
        tempSZA = datavar.SolarZenithAngle(validmask);
        tempCF = datavar.CloudFraction(validmask);
        
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
        colno2 = cat(1,colno2,single(tempVCD(:)));
        colno2error = cat(1,colno2error,single(tempVCD_unc(:)));
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