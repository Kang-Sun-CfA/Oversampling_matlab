% Download NASA OMNO2 data selectively, subset and filter the data, save
% filtered data into L2g folder for future conversion into L3 data

% Written by Kang Sun on 2017/06/03-05
%% define directories
addpath('~/matlab functions/')
L2_dir = '/data/tempo1/Shared/kangsun/OMNO2/L2_data/';
% intermediate data dir
L2g_dir = '/data/tempo1/Shared/kangsun/OMNO2/L2g_data/';
cd(L2_dir)

%% define cores
feature('numcores');
ncore = ans;
ncoreuse = input('Please decide how many cores to use ... ');

%% define L2 variables
swathname = 'ColumnAmountNO2';

% variables to read from OMI L2 files
varname = {'CloudFraction','ColumnAmountNO2','ColumnAmountNO2Trop',...
    'ColumnAmountNO2Std','ColumnAmountNO2TropStd','MeasurementQualityFlags',...
    'XTrackQualityFlags','VcdQualityFlags','AmfTrop'};

geovarname = {'Latitude','Longitude','Time','SolarZenithAngle',...
    'FoV75CornerLatitude','FoV75CornerLongitude'};
%% load critical data
% extent of orbit given equator crossing lon == 0
load('/data/tempo1/Shared/kangsun/OMNO2/Important_constant/OMI_BDR.mat')
% load('/data/tempo1/Shared/kangsun/OMNO2/Important_constant/FoV75Area.mat')
%% knobs to turn
% high latitude beyond +/- 50 may not work. better download all data
MinLat = 25; MaxLat = 50;
MinLon = -130; MaxLon = -63;

% max cloud fraction and SZA
MaxCF = 0.2;
MaxSZA = 75;

% xtrack position mask
usextrack = [1:60];

% if download xml, meta data files
if_download_xml = true;

% if download the he5 data
if_download_he5 = true;

% if delete he5 files to save disk space
if_delete_he5 = true;

% if save ZL, KY cake cut input file
if_save_cakecut_input = false;

% start and end date to download data
Startdate = '2004-10-01';
Enddate = '2004-12-31';
%%

day_array = (datenum(Startdate):1:datenum(Enddate))';
datevec_array = datevec(day_array);
Jan01_array = datevec_array;
Jan01_array(:,2:end) = 0;
doy_array = day_array-datenum(Jan01_array);
year_array = datevec_array(:,1);

nday = length(day_array);

p_obj = parpool(ncoreuse);

savedata = cell(nday,1);

parfor iday = 1:nday
    url0 = 'https://aura.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level2/OMNO2.003/';
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
    tmp_savedata = single([]);
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

                validmask = datavar.Latitude >= MinLat & datavar.Latitude <= MaxLat & ...
                    datavar.Longitude >= MinLon & datavar.Longitude <= MaxLon & ...
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
                    tempxtrack_N(:),ones(size(tempLatC(:)))*day_array(iday)];
                
                tmp_savedata = cat(1,tmp_savedata,single(tempdata));
                
                if if_delete_he5
                    delete(fn);
                end
            end
        end
    end
    savedata{iday} = tmp_savedata;
    cd(L2_dir)
end
% FoV75Area = [(1:60)',datavar.FoV75Area];
% plot(FoV75Area(:,1),FoV75Area(:,2))
% save('/data/tempo1/Shared/kangsun/OMNO2/Important_constant/FoV75Area.mat','FoV75Area');
savedata = cell2mat(savedata);
tmp = datevec(double(savedata(:,end)));
tmp = tmp(:,1);
savedata = cat(2,(1:size(savedata,1))',savedata);
unique_year = unique(year_array);
cd(L2g_dir)
for iyear = 1:length(unique_year)
    if ~exist(num2str(unique_year(iyear)),'dir')
        mkdir(num2str(unique_year(iyear)))
    end
    year_savedata = savedata(tmp == unique_year(iyear),:);
    save([L2g_dir,num2str(unique_year(iyear)),'/Lat_',num2str(MinLat),'_',...
        num2str(MaxLat),'_Lon_',...
        num2str(MinLon),'_',num2str(MaxLon),'.mat'],...
        'year_savedata');
    
    if if_save_cakecut_input
        fid = fopen([L2g_dir,'Lat_',num2str(MinLat),'_',num2str(MaxLat),'_Lon_',...
            num2str(MinLon),'_',num2str(MaxLon),'.dat'],'w');
        % print data for Lei Zhu' fortran program
        fprintf(fid,['%8d',repmat('%15.6f',1,13),repmat('%15.6E',1,2),'\n'],year_savedata(:,1:16)');
        fclose(fid);
    end
end
delete(p_obj);
% %%
% close all
% for iday = 100:100%nday
%     day_dir = [num2str(year_array(iday)),'/',sprintf('%03d',doy_array(iday)),'/'];
%     cd([L2_dir,day_dir])
%     flist = dir;
%     for ihdf = 1:length(flist)
%         if flist(ihdf).bytes > 1000000
%             datavar = F_read_he5(flist(ihdf).name,swathname,varname,geovarname);
%             xtrackmask = false(size(datavar.Latitude));
%                 xtrackmask(usextrack,:) = true;
%             validmask = datavar.Latitude >= MinLat & datavar.Latitude <= MaxLat & ...
%                     datavar.Longitude >= MinLon & datavar.Longitude <= MaxLon & ...
%                     datavar.XTrackQualityFlags.data == 0 & ...
%                     datavar.VcdQualityFlags.data == 0 & ...
%                     datavar.SolarZenithAngle <= MaxSZA & ...
%                     datavar.CloudFraction.data.*datavar.CloudFraction.ScaleFactor <= MaxCF & ...
%                     xtrackmask;
%             lat = datavar.Latitude;lon = datavar.Longitude;
%             data = datavar.ColumnAmountNO2Trop.data;
%             data(~validmask) = nan;
%
%             figure('unit','inch','position',[-17 0 15 12])
%             axesm('MapProjection','eqdcylin','MapLatLimit', ...
%                 [-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','on', ...
%                 'MeridianLabel','on','ParallelLabel','on','MLabelParallel','south')
%             coast = load('coast.mat');
%             caxis([0 1e17])
%             surfacem(lat, lon, data);
%             plotm(coast.lat,coast.long,'k');
%         end
%     end
% end
% %%
% fn = '/data/tempo1/Shared/kangsun/OMNO2/L2_data/OMI-Aura_L2-OMNO2_2004m1001t0956-o01138_v003-2016m0819t183533.he5';
% datavar = F_read_he5(fn,swathname,varname,geovarname);
% %%
% ix = 30;
% il = 800;
% % lat_r = squeeze(datavar.FoV75CornerLatitude(ix,il,:));
% % lon_r = squeeze(datavar.FoV75CornerLongitude(ix,il,:));
% lat_r = squeeze(datavar.FoV75CornerLatitude(:,ix,il));
% lon_r = squeeze(datavar.FoV75CornerLongitude(:,ix,il));
% lat_c = datavar.Latitude(ix,il);
% lon_c = datavar.Longitude(ix,il);
% plot(lon_r(1),lat_r(1),'.',lon_r(2),lat_r(2),'o',lon_r(3),lat_r(3),'*',lon_r(4),lat_r(4),'v',...
%     lon_c,lat_c,'.k')
% % OMNO2 stupid corner definition:
% % ur, ul, ll, lr
% %%
% lat = datavar.Latitude;lon = datavar.Longitude;
% data = datavar.ColumnAmountNO2Trop.data;
% Filter = datavar.VcdQualityFlags.data;
% data(Filter~=0) = nan;
% close all
% figure('unit','inch','position',[-12 0 10 8])
% axesm('MapProjection','eqdcylin','MapLatLimit', ...
%     [-90 90],'MapLonLimit',[-180 180],'Frame','on','Grid','on', ...
%     'MeridianLabel','on','ParallelLabel','on','MLabelParallel','south')
% coast = load('coast.mat');
%
% surfacem(lat, lon, data);
% plotm(coast.lat,coast.long,'k');
% %%
% close all
% lat = datavar.Latitude;lon = datavar.Longitude;
% plot(lon(1,:)-(44.1),lat(1,:),lon(end,:)-(44.1),lat(end,:))
% left_bdr = [lon(1,:)-(44.1);lat(1,:)];
% right_bdr = [lon(end,:)-(44.1);lat(end,:)];
% save('/data/tempo1/Shared/kangsun/OMNO2/Important_constant/OMI_BDR.mat','left_bdr','right_bdr')
% %%
% str = 'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2TMNXSLV.5.12.4/1981/MERRA2_100.tavgM_2d_slv_Nx.198101.nc4';
% unix(str)
% %%
% str = 'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A he5 "https://aura.gesdisc.eosdis.nasa.gov/data/Aura_OMI_Level2/OMNO2.003/2004/275/"';
% unix(str)