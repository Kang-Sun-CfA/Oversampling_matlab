function datavar = F_read_OMPS_h5(fn,varname,geovarname)

% fn = '/data/tempo1/Shared/kangsun/OMPS_NO2/L2_data/OMPS-NPP_NMNO2-L2_2016m0101t180538_o21656_2017m0531t045531.h5';
% varname = {'CloudFraction','ColumnAmountNO2','ColumnAmountNO2tropo',...
%     'PixelQualityFlags','SceneRefletivity','GroundRefletivity'};
% 
% geovarname = {'Latitude','Longitude','SolarZenithAngle',...
%     'LatitudeCorner','LongitudeCorner','GroundPixelQualityFlags'};
% clc
datavar = [];
file_id = H5F.open (fn, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');

for i = 1:length(geovarname)
    DATAFIELD_NAME = ['/GeolocationData/',geovarname{i}];
    data_id=H5D.open(file_id, DATAFIELD_NAME);
    datavar.(geovarname{i})=H5D.read(data_id,'H5T_NATIVE_DOUBLE', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT');   
end

for i = 1:length(varname)
    % Open the dataset.
    DATAFIELD_NAME = ['/ScienceData/',varname{i}];
    data_id = H5D.open (file_id, DATAFIELD_NAME);
    datavar.(varname{i}).data=H5D.read (data_id,'H5T_NATIVE_DOUBLE', 'H5S_ALL', 'H5S_ALL', 'H5P_DEFAULT');
end

H5D.close (data_id);
H5F.close (file_id);