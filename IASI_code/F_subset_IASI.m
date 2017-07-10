function output = F_subset_IASI(inp)
% extract useful IASI data from L2 files in L2dir. The L2 files should be
% named as "nh3fix_yyyymmdd.mat"

% written by Kang Sun on 2017/07/06

% I don't know why there are so many NaNs in critical variables such as
% CLcov and tskin.
Startdate = inp.Startdate;
Enddate = inp.Enddate;
MinLat = inp.MinLat;
MinLon = inp.MinLon;
MaxLat = inp.MaxLat;
MaxLon = inp.MaxLon;

% grey area, L2 pixel center cannot be there, but pixel boundaries can
MarginLat = 0.5; 
MarginLon = 0.5;

% MaxCF = inp.MaxCF;
Min_pos_TC = inp.Min_pos_TC;
Max_neg_TC = inp.Max_neg_TC;
MinTskin = inp.MinTskin;
MaxSand = inp.MaxSand;

day_or_night = inp.day_or_night;

L2dir = inp.L2dir;

if isfield(inp,'showfilter')
    showfilter = inp.showfilter;
else
    showfilter = false;
end

%% find out useful filenames
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
latall = single([]);
lonall = latall;
ifovall = latall;
colnh3all = latall;
utcall = []; % time needs to be double
totErrall = latall;
cfall = latall;
% fhourall = latall;
TCall = latall;
for ifile = 1:length(subfilelist)
    disp(['Loading ',subfilelist(ifile).name,'...'])
    try
        temp = load([L2dir,subfilelist(ifile).name]);
    catch
        warning(['Error reading ',L2dir,subfilelist(ifile).name,'!!!'])
        continue
    end
    TC = temp.tskin-temp.Tprof(1,:); % thermal contrast
    
    f1 = temp.lat >= MinLat+MarginLat & temp.lat <= MaxLat-MarginLat & ...
        temp.lon >= MinLon+MarginLon & temp.lon <= MaxLon-MarginLon;
    switch day_or_night
        case 'day'
            f1 = f1 & temp.fhour;
        case 'night'
            f1 = f1 & ~temp.fhour;
    end    
    f2 = temp.fcloud;
    CLcov = temp.CLcov;
    if max(CLcov) > 2;CLcov = CLcov/100;end
    f2a = ~(CLcov > 0.25);
    f3 = ~(temp.sand > MaxSand);
    f4 = ~(temp.tskin < MinTskin);
    f5 = ~(TC < Min_pos_TC & TC > Max_neg_TC);
    validmask = f1 & f2 & f3 & f4 & f5;

    if sum(validmask) > 0
        disp(['You have ',sprintf('%5d',sum(validmask(:))),' valid L2 pixels on ',datestr(subfileday(ifile))]);
        if showfilter
            disp(['In box: ',num2str(sum(f1)),'; after cloud: ',num2str(sum(f1 & f2)),...
                '; after fc < 0.25: ',num2str(sum(f1 & f2a)),'; after sand: ',...
                num2str(sum(f1 & f2 & f3)),'; after tskin: ',num2str(sum(f1 & f2 & f3 & f4)),...
                '; after TC: ',num2str(sum(f1 & f2 & f3 & f4 & f5))])
        end
        HH = double(floor(temp.hour(validmask)/10000));
        MM = double(floor((temp.hour(validmask)-10000*HH)/100));
        SS = double(mod(temp.hour(validmask),100));
        UTC = HH(:)/24+MM(:)/60/24+SS(:)/3600/24+subfileday(ifile);
        utcall = cat(2,utcall,UTC');
        
        latall = cat(2,latall,single(temp.lat(validmask)));
        lonall = cat(2,lonall,single(temp.lon(validmask)));
        ifovall = cat(2,ifovall,single(temp.ifov(validmask)));
        colnh3all = cat(2,colnh3all,single(temp.colnh3(validmask)));
        totErrall = cat(2,totErrall,single(temp.totErr(validmask)));
%         fhourall = cat(2,fhourall,single(temp.fhour(validmask)));
        TCall = cat(2,TCall,single(TC(validmask)));
        cfall = cat(2,cfall,single(CLcov(validmask)));
    end
end
% NH3 column error, absolute
colnh3errorall = abs(colnh3all.*totErrall/100);
% % assign very large error if colnh3 or totErr are nan
% colnh3errorall(isnan(colnh3all) | isnan(colnh3errorall)) = 1e25;
% colnh3all(isnan(colnh3all)| isnan(colnh3errorall)) = 0;

output.colnh3 = colnh3all;
output.colnh3error = colnh3errorall;
output.lat = latall;
output.lon = lonall;
output.ifov = ifovall;
output.utc = utcall;
output.tc = TCall;
output.cloudfrac = cfall;