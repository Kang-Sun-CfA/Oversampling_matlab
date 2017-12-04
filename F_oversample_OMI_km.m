function output_oversample = F_oversample_OMI_km(inp,output_subset)
% function to take in the output from F_subset_OM*.m and regrid these L2
% data to a L3 grid, centered at clon and clat and bounded by max_x and 
% max_y with resolution res in km.

% Modified from F_regrid_OMI.m by Kang Sun on 2017/10/02

output_oversample = [];

% Area of OMI pixel for each x-track position. Independent of location 
A_array = [4596.47998046875;3296.36010742188;2531.02001953125;2022.28002929688;1664.44995117188;1402.01000976563;1202.15002441406;1046.56994628906;922.939025878906;823.557983398438;742.541015625000;676.026977539063;620.728027343750;574.632019042969;535.745971679688;502.919006347656;474.923004150391;451.101989746094;430.649993896484;413.221008300781;398.187011718750;385.433013916016;374.503997802734;365.376007080078;357.684997558594;351.493011474609;346.483001708984;342.816009521484;340.184997558594;338.792999267578;338.364990234375;339.161010742188;340.954010009766;344.029998779297;348.161010742188;353.752990722656;360.536010742188;369.022003173828;378.957000732422;390.996002197266;404.904998779297;421.479003906250;440.553985595703;463.213012695313;489.364013671875;520.383972167969;556.742004394531;599.838012695313;651.012023925781;712.447998046875;786.658996582031;877.330993652344;989.809020996094;1131.55004882813;1313.92004394531;1554.43994140625;1881.34997558594;2340.41992187500;3012.80004882813;4130.99023437500];

Startdate = inp.Startdate;
Enddate = inp.Enddate;

res = inp.res;
max_x = inp.max_x;
max_y = inp.max_y;
clon = inp.clon;
clat = inp.clat;
R = inp.R;

if ~isfield(inp,'do_weight')
    do_weight = false;
else
    do_weight = inp.do_weight;
end

max_lon = clon+max_x*1.2/110/cos((abs(clat)+max_y/111)/180*pi);
min_lon = clon-max_x*1.2/110/cos((abs(clat)+max_y/111)/180*pi);
max_lat = clat+max_y*1.2/110;
min_lat = clat-max_y*1.2/110;

% define x y grids
xgrid = (-max_x+0.5*res):res:max_x;
ygrid = (-max_y+0.5*res):res:max_y;
nrows = length(ygrid);
ncols = length(xgrid);

% define x y mesh
[xmesh, ymesh] = meshgrid(single(xgrid),single(ygrid));

% max cloud fraction and SZA
MaxCF = inp.MaxCF;
MaxSZA = inp.MaxSZA;

% xtrack to use
if ~isfield(inp,'usextrack')
    usextrack = 1:60;
else
    usextrack = inp.usextrack;
end

vcdname = inp.vcdname;
vcderrorname = inp.vcderrorname;

% if ~isfield(inp,'lon_offset_array')
%     x_offset_array = zeros(1,60);
%     y_offset_array = zeros(1,60);
% else
%     x_offset_array = inp.x_offset_array;
%     y_offset_array = inp.y_offset_array;
% end

if isfield(inp,'useweekday')
    useweekday = inp.useweekday;
end

if ~isfield(inp,'if_parallel')
    if_parallel = false;
else
    if_parallel = inp.if_parallel;
end

f1 = output_subset.utc >= datenum([Startdate 0 0 0]) ...
    & output_subset.utc <= datenum([Enddate 23 59 59]);
% pixel corners are all 0 in OMNO2 orbit 04420. W. T. F.
f2 = output_subset.latc >= min_lat-0.5 & output_subset.latc <= max_lat+0.5...
    & output_subset.lonc >= min_lon-0.5 & output_subset.lonc <= max_lon+0.5 ...
    & output_subset.latr(:,1) >= min_lat-1 & output_subset.latr(:,1) <= max_lat+1 ...
    & output_subset.lonr(:,1) >= min_lon-1 & output_subset.lonr(:,1) <= max_lon+1;
f3 = output_subset.sza <= MaxSZA;
f4 = output_subset.cloudfrac <= MaxCF;
f5 = ismember(output_subset.ift,usextrack);

validmask = f1&f2&f3&f4&f5;

if exist('useweekday','var')
    wkdy = weekday(output_subset.utc);
    f6 = ismember(wkdy,useweekday);
    validmask = validmask & f6;
end

nL2 = sum(validmask);
if nL2 <= 0;return;end
disp(['Regriding pixels from ',datestr([Startdate 0 0 0]),' to ',...
    datestr([Enddate 23 59 59])])
disp([num2str(nL2),' pixels to be regridded...'])

Lat_c = output_subset.latc(validmask);
Lon_c = output_subset.lonc(validmask);
Xtrack = output_subset.ift(validmask);
VCD = output_subset.(vcdname)(validmask);
VCDe = output_subset.(vcderrorname)(validmask);

disp('Converting pixel lat lon to x y in km...')
% call function F_latlon2xy to convert pixel lat lon to x y.
inp_xy = [];
inp_xy.clon = clon;
inp_xy.clat = clat;
inp_xy.lon = Lon_c(:);
inp_xy.lat = Lat_c(:);
outp_xy = F_latlon2xy(inp_xy);

disp('Calculating spatial response functions pixel by pixel...')
x = outp_xy.x;
y = outp_xy.y;

D = zeros(nrows,ncols,'single');
O = D;

count = 1;
for irow = 1:nrows
    for icol = 1:ncols
        	xcenter = xgrid(icol);ycenter = ygrid(irow);
            tmp_ind = find(abs(x-xcenter) <= R & abs(y-ycenter) <= R);
            Distance = sqrt((x(tmp_ind)-xcenter).^2+(y(tmp_ind)-ycenter).^2);
            ind = tmp_ind(Distance <= R);
            if do_weight
            weight = 1./VCDe(ind)./(A_array(Xtrack(ind)));
            weight = weight/nansum(weight);
            O(irow,icol) = nansum(VCD(ind).*weight);
            else
                O(irow,icol) = nanmean(VCD(ind));
            end
            D(irow,icol) = length(ind);
    end
    if irow == count*round(nrows/10)
        disp([num2str(count*10),' % finished'])
        count = count+1;
    end
end

output_oversample.O = O;
output_oversample.D = D;

output_oversample.xgrid = xgrid;
output_oversample.ygrid = ygrid;

output_oversample.xmesh = single(xmesh);
output_oversample.ymesh = single(ymesh);

disp('Transfer x y coordinates back to lat lon...')
[tmplat,tmplon] = minvtran(outp_xy.mstruct,xmesh(:),ymesh(:));
output_oversample.latmesh = single(reshape(tmplat,size(xmesh)));
output_oversample.lonmesh = single(reshape(tmplon,size(xmesh)));

output_oversample.mstruct = outp_xy.mstruct;
output_oversample.max_lon = max_lon;
output_oversample.min_lon = min_lon;

output_oversample.max_lat = max_lat;
output_oversample.min_lat = min_lat;
return

function outp = F_latlon2xy(inp)
% Function to transfer lat lon to x y in km using Lambert standard
% projection. Need to define center lat lon.

% Written by Kang Sun on 2017/09/23

% major bug-fix on 2017/11/20 to use mercator projection near equator
% instead of lambert

clon = inp.clon;
clat = inp.clat;

if clon < 0
    clon = 360+clon;
end
if abs(clat) < 15
    mstruct = defaultm('mercator');
else
    mstruct = defaultm('lambertstd');
end
mstruct.origin = [clat clon 0];
mstruct.mapparallels = clat;
mstruct.nparallels = 1;
mstruct.scalefactor = 6371229/1e3;
mstruct.falseeasting = 0;
mstruct.falsenorthing = 0;
mstruct = defaultm(mstruct);

nloop = size(inp.lat,2);
x = inp.lat;y = inp.lat;
for i = 1:nloop
    [x(:,i), y(:,i)] = mfwdtran(mstruct,inp.lat(:,i),inp.lon(:,i));
end

outp.x = x;
outp.y = y;
outp.mstruct = mstruct;
return