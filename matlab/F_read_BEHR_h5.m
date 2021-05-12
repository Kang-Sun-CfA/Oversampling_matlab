function behr = F_read_BEHR_h5(inp)
% matlab function to read hdf5 files of BEHR L2 data. Written by Kang Sun
% on 2018/09/06

% clc;clear
% fn = '/mnt/Data2/BEHR/OMI_BEHR-DAILY_US_v3-0B_200505/OMI_BEHR-DAILY_US_v3-0B_20050501.hdf';
% varname = {'BEHRColumnAmountNO2Trop','BEHRQualityFlags','CloudFraction',...
%     'FoV75CornerLatitude','FoV75CornerLongitude','Latitude','Longitude',...
%     'SolarZenithAngle','Time'};

fn = inp.fn;
varname = inp.varname;

behr = [];

nvar = length(varname);
fninfo = h5info(fn);
swathnames = {fninfo.Groups.Groups.Name};
nswath = length(swathnames);

fid = H5F.open(fn);

for iswath = 1:nswath
    for ivar = 1:nvar
        dset_id = H5D.open(fid,[swathnames{iswath},'/',varname{ivar}]);
        behr.(swathnames{iswath}(7:end)).(varname{ivar}) = H5D.read(dset_id);
    end
end