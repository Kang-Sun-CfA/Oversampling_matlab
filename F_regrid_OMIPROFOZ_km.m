function output_regrid = F_regrid_OMIPROFOZ_km(inp,output_subset)
% function to take in the output from Subset_PROFOZ.py
% and regrid these L2 data to a L3 grid, centered at clon and clat and bounded by max_x and 
% max_y with resolution res in km.

% rewritten by Kang Sun from F_regrid_OMIPROFOZ.m on 2017/12/31
sfn = fieldnames(output_subset);
nsounding = length(output_subset.cloudfrac);
for i = 1:length(sfn)
    if size(output_subset.(sfn{i}),2) == nsounding
        output_subset.(sfn{i}) = output_subset.(sfn{i})';
    end
end
if ~isfield(output_subset,'ift')
    output_subset.ift = output_subset.pix;
end
output_regrid = [];

% Area of OMI pixel for each x-track position. Independent of location 
A_array = [15222.3149986107;10498.3518814635;7945.47902847710;6377.76649260743;5322.23556240396;4583.96549482750;4037.72737197957;3635.41864587816;3323.01075984179;3091.41186996674;2907.83739603056;2777.87425809979;2677.58128508113;2618.21803638261;2580.24402152602;2578.63829899980;2596.44102963661;2651.45608589670;2729.58631631376;2851.81699830249;3008.09880886615;3225.10671159599;3501.02437564678;3874.42711591147;4362.79686081269;5040.28588879711;5985.27983361525;7393.59412907937;9629.38874875802;13653.7417563649];

Startdate = inp.Startdate;
Enddate = inp.Enddate;

res = inp.res;
max_x = inp.max_x;
max_y = inp.max_y;
clon = inp.clon;
clat = inp.clat;

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

% construct a rectangle envelopes the orginal pixel
xmargin = 3; % how many times to extend zonally
ymargin = 2; % how many times to extend meridonally

% max cloud fraction and SZA
MaxCF = inp.MaxCF;
MaxSZA = inp.MaxSZA;

% xtrack to use
if ~isfield(inp,'usextrack')
    usextrack = 1:30;
else
    usextrack = inp.usextrack;
end

vcdname = inp.vcdname;
vcderrorname = inp.vcderrorname;

% parameters to define pixel SRF
if ~isfield(inp,'inflatex_array')
    inflatex_array = ones(1,30);
    inflatey_array = ones(1,30);
else
    inflatex_array = inp.inflatex_array;
    inflatey_array = inp.inflatey_array;
end

if ~isfield(inp,'lon_offset_array')
    x_offset_array = zeros(1,30);
    y_offset_array = zeros(1,30);
else
    x_offset_array = inp.x_offset_array;
    y_offset_array = inp.y_offset_array;
end

if ~isfield(inp,'m_array')
    m_array = 4*ones(1,30);
    n_array = 4*ones(1,30);
else
    m_array = inp.m_array;
    n_array = inp.n_array;
end

if isfield(inp,'useweekday')
    useweekday = inp.useweekday;
end

if ~isfield(inp,'if_parallel')
    if_parallel = false;
else
    if_parallel = inp.if_parallel;
end
ozdatemat = [double(output_subset.year(:)) double(output_subset.month(:))...
    double(output_subset.day(:))];
output_subset.utc = datenum(ozdatemat)+double(output_subset.hour(:))/24;
f1 = output_subset.utc >= datenum([Startdate 0 0 0]) ...
    & output_subset.utc <= datenum([Enddate 23 59 59]);
% pixel corners are all 0 in OMNO2 orbit 04420. W. T. F.
f2 = output_subset.lat_c >= min_lat-0.5 & output_subset.lat_c <= max_lat+0.5...
    & output_subset.lon_c >= min_lon-0.5 & output_subset.lon_c <= max_lon+0.5 ...
    & output_subset.lat_r(:,1) >= min_lat-1 & output_subset.lat_r(:,1) <= max_lat+1 ...
    & output_subset.lon_r(:,1) >= min_lon-1 & output_subset.lon_r(:,1) <= max_lon+1;
% f3 = output_subset.sza <= MaxSZA;
f4 = output_subset.cloudfrac <= MaxCF;
f5 = ismember(output_subset.ift,usextrack);

validmask = f1&f2&f4&f5;

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

Sum_Above = zeros(nrows,ncols,'single');
Sum_Below = zeros(nrows,ncols,'single');
D = zeros(nrows,ncols,'single');

Lat_r = output_subset.lat_r(validmask,:);
Lon_r = output_subset.lon_r(validmask,:);
Lat_c = output_subset.lat_c(validmask);
Lon_c = output_subset.lon_c(validmask);
Xtrack = output_subset.ift(validmask);
VCD = output_subset.(vcdname)(validmask);
VCDe = output_subset.(vcderrorname)(validmask);

disp('Converting pixel lat lon to x y in km...')
% call function F_latlon2xy to convert pixel lat lon to x y.
inp_xy = [];
inp_xy.clon = double(clon);
inp_xy.clat = double(clat);
inp_xy.lon = [Lon_r,Lon_c(:)];
inp_xy.lat = [Lat_r,Lat_c(:)];
outp_xy = F_latlon2xy(inp_xy);

disp('Calculating spatial response functions pixel by pixel...')
if if_parallel
parfor iL2 = 1:nL2
    y_r = outp_xy.y(iL2,1:4);
    x_r = outp_xy.x(iL2,1:4);
    y_c = outp_xy.y(iL2,5);
    x_c = outp_xy.x(iL2,5);
    vcd = VCD(iL2);
    vcd_unc = VCDe(iL2);
    xtrack = Xtrack(iL2);
    
    inflatex = inflatex_array(xtrack);
    inflatey = inflatey_array(xtrack);
    x_offset = x_offset_array(xtrack);
    y_offset = y_offset_array(xtrack);
    m = m_array(xtrack);
    n = n_array(xtrack);
    % area is more elegant if using km rather than lat lon
%     A = polyarea([x_r(:);x_r(1)],[y_r(:);y_r(1)]);
    A = A_array(xtrack);
    y_min_local = min(y_r);
    x_min_local = min(x_r);
    local_left = x_c-xmargin*(x_c-x_min_local);
    local_right = x_c+xmargin*(x_c-x_min_local);
    
    local_bottom = y_c-ymargin*(y_c-y_min_local);
    local_top = y_c+ymargin*(y_c-y_min_local);
    
    x_index = xgrid >= local_left & xgrid <= local_right;
    y_index = ygrid >= local_bottom & ygrid <= local_top;
    
    x_local_mesh = xmesh(y_index,x_index);
    y_local_mesh = ymesh(y_index,x_index);
    
    SG = F_2D_SG_affine(x_local_mesh,y_local_mesh,x_r,y_r,x_c,y_c,...
        m,n,inflatex,inflatey,x_offset,y_offset);
    
    sum_above_local = zeros(nrows,ncols,'single');
    sum_below_local = zeros(nrows,ncols,'single');
    D_local = zeros(nrows,ncols,'single');
    
    D_local(y_index,x_index) = SG;
    sum_above_local(y_index,x_index) = SG/A/vcd_unc*vcd;
    sum_below_local(y_index,x_index) = SG/A/vcd_unc;
    Sum_Above = Sum_Above + sum_above_local;
    Sum_Below = Sum_Below + sum_below_local;
    D = D+D_local;
end
else
count = 1;
for iL2 = 1:nL2
    y_r = outp_xy.y(iL2,1:4);
    x_r = outp_xy.x(iL2,1:4);
    y_c = outp_xy.y(iL2,5);
    x_c = outp_xy.x(iL2,5);
    vcd = VCD(iL2);
    vcd_unc = VCDe(iL2);
    xtrack = Xtrack(iL2);
    
    inflatex = inflatex_array(xtrack);
    inflatey = inflatey_array(xtrack);
    x_offset = x_offset_array(xtrack);
    y_offset = y_offset_array(xtrack);
    m = m_array(xtrack);
    n = n_array(xtrack);
    % area is more elegant if using km rather than lat lon
%     A = polyarea([x_r(:);x_r(1)],[y_r(:);y_r(1)]);
    A = A_array(xtrack);
    y_min_local = min(y_r);
    x_min_local = min(x_r);
    local_left = x_c-xmargin*(x_c-x_min_local);
    local_right = x_c+xmargin*(x_c-x_min_local);
    
    local_bottom = y_c-ymargin*(y_c-y_min_local);
    local_top = y_c+ymargin*(y_c-y_min_local);
    
    x_index = xgrid >= local_left & xgrid <= local_right;
    y_index = ygrid >= local_bottom & ygrid <= local_top;
    
    x_local_mesh = xmesh(y_index,x_index);
    y_local_mesh = ymesh(y_index,x_index);
    
    SG = F_2D_SG_affine(x_local_mesh,y_local_mesh,x_r,y_r,x_c,y_c,...
        m,n,inflatex,inflatey,x_offset,y_offset);
    
    sum_above_local = zeros(nrows,ncols,'single');
    sum_below_local = zeros(nrows,ncols,'single');
    D_local = zeros(nrows,ncols,'single');
    
    D_local(y_index,x_index) = SG;
    sum_above_local(y_index,x_index) = SG/A/vcd_unc*vcd;
    sum_below_local(y_index,x_index) = SG/A/vcd_unc;
    Sum_Above = Sum_Above + sum_above_local;
    Sum_Below = Sum_Below + sum_below_local;
    D = D+D_local;
    
    if iL2 == count*round(nL2/10)
        disp([num2str(count*10),' % finished'])
        count = count+1;
    end
end
end

output_regrid.A = Sum_Above;
output_regrid.B = Sum_Below;
output_regrid.C = Sum_Above./Sum_Below;
output_regrid.D = D;

output_regrid.xgrid = xgrid;
output_regrid.ygrid = ygrid;
output_regrid.xmesh = single(xmesh);
output_regrid.ymesh = single(ymesh);

disp('Transfer x y coordinates back to lat lon...')
[tmplat,tmplon] = minvtran(outp_xy.mstruct,xmesh(:),ymesh(:));
output_regrid.latmesh = single(reshape(tmplat,size(xmesh)));
output_regrid.lonmesh = single(reshape(tmplon,size(xmesh)));

output_regrid.mstruct = outp_xy.mstruct;
output_regrid.max_lon = max_lon;
output_regrid.min_lon = min_lon;

output_regrid.max_lat = max_lat;
output_regrid.min_lat = min_lat;
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

function SG = F_2D_SG_affine(lon_mesh,lat_mesh,lon_r,lat_r,lon_c,lat_c,...
    m,n,inflatex,inflatey,lon_offset,lat_offset)
% This function is updated from F_2D_SG.m to include affine transform, so
% that the 2-D super Gaussian spatial response function (SRF) can be easily
% sheared and rotated, adjusting to the parallelogram OMI/TEMPO pixel shape

% Written by Kang Sun on 2017/06/07

% Updated by Kang Sun on 2017/06/08. The previous one only works for
% perfectly zonally aligned pixels

vList = [lon_r(:)-lon_c,lat_r(:)-lat_c];

leftpoint = mean(vList(1:2,:));
rightpoint = mean(vList(3:4,:));

uppoint = mean(vList(2:3,:));
lowpoint = mean(vList([1 4],:));

xvector = rightpoint-leftpoint;
yvector = uppoint-lowpoint;

FWHMx = norm(xvector);
FWHMy = norm(yvector);

fixedPoints = [-FWHMx,-FWHMy;
    -FWHMx, FWHMy;
    FWHMx, FWHMy;
    FWHMx, -FWHMy]/2;

tform = fitgeotrans(fixedPoints,vList,'affine');

xym1 = [lon_mesh(:)-lon_c-lon_offset, lat_mesh(:)-lat_c-lat_offset];
xym2 = transformPointsInverse(tform,xym1);

FWHMy = FWHMy*inflatey;
FWHMx = FWHMx*inflatex;

wx = FWHMx/2/(log(2))^(1/m);
wy = FWHMy/2/(log(2))^(1/n);

SG = exp(-(abs(xym2(:,1))/wx).^m-(abs(xym2(:,2))/wy).^n);
SG = reshape(SG(:),size(lon_mesh,1),size(lon_mesh,2));
return
