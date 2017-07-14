function output_regrid = F_regrid_OMI(inp,output_subset)
% function to take in the output from F_subset_OMNO2.m or F_subset_OMHCHO.m
% and regrid these L2 data to a L3 grid, defined by a lat lon box and
% resolution (Res).

% the output contains four matrices, A, B, C, and D. A and B are cumulative
% terms. C = A./B is the regridded concentration over this period. D is the
% sum of unnormalized spatial response function. Large D means L3 grid is
% covered by more L2 pixels.

% written by Kang Sun on 2017/07/14

output_regrid = [];
Res = inp.Res;
MinLon = inp.MinLon;
MaxLon = inp.MaxLon;
MinLat = inp.MinLat;
MaxLat = inp.MaxLat;
if isfield(inp,'MarginLon')
    MarginLon = inp.MarginLon;
else
    MarginLon = 0.5;
end
if isfield(inp,'MarginLat')
    MarginLat = inp.MarginLat;
else
    MarginLat = 0.5;
end
Startdate = inp.Startdate;
Enddate = inp.Enddate;

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

% parameters to define pixel SRF
if ~isfield(inp,'inflatex_array')
    inflatex_array = ones(1,60);
    inflatey_array = ones(1,60);
else
    inflatex_array = inp.inflatex_array;
    inflatey_array = inp.inflatey_array;
end

if ~isfield(inp,'lon_offset_array')
    lon_offset_array = zeros(1,60);
    lat_offset_array = zeros(1,60);
else
    lon_offset_array = inp.lon_offset_array;
    lat_offset_array = inp.lat_offset_array;
end

if ~isfield(inp,'m_array')
    m_array = 4*ones(1,60);
    n_array = 2*ones(1,60);
else
    m_array = inp.m_array;
    n_array = inp.n_array;
end

% define x y grids
xgrid = (MinLon+0.5*Res):Res:MaxLon;
ygrid = (MinLat+0.5*Res):Res:MaxLat;
nrows = length(ygrid);
ncols = length(xgrid);

% define x y mesh
[Lon_mesh, Lat_mesh] = meshgrid(single(xgrid),single(ygrid));

% construct a rectangle envelopes the orginal pixel
xmargin = 3; % how many times to extend zonally
ymargin = 2; % how many times to extend meridonally


f1 = output_subset.utc >= datenum([Startdate 0 0 0]) ...
    & output_subset.utc <= datenum([Enddate 0 0 0]);
f2 = output_subset.lat >= MinLat-MarginLat & output_subset.lat <= MaxLat+MarginLat...
    & output_subset.lon >= MinLon-MarginLon & output_subset.lon <= MaxLon+MarginLon;
f3 = output_subset.sza <= MaxSZA;
f4 = output_subset.cloudfrac <= MaxCF;
f5 = ismember(output_subset.ift,usextrack);

validmask = f1&f2&f3&f4&f5;

nL2 = sum(validmask);

disp(['Regriding pixels from ',datestr([Startdate 0 0 0]),' to ',...
    datestr([Enddate 0 0 0])])
disp([num2str(nL2),' pixels to be regridded...'])

Sum_Above = zeros(nrows,ncols,'single');
Sum_Below = zeros(nrows,ncols,'single');
D = zeros(nrows,ncols,'single');

Lat_r = output_subset.latr(validmask,:);
Lon_r = output_subset.lonr(validmask,:);
Lat_c = output_subset.latc(validmask);
Lon_c = output_subset.lonc(validmask);
Xtrack = output_subset.ift(validmask);
VCD = output_subset.(vcdname)(validmask);
VCDe = output_subset.(vcderrorname)(validmask);

parfor iL2 = 1:nL2
    lat_r = Lat_r(iL2,:);
    lon_r = Lon_r(iL2,:);
    lat_c = Lat_c(iL2);
    lon_c = Lon_c(iL2);
    vcd = VCD(iL2);
    vcd_unc = VCDe(iL2);
    xtrack = Xtrack(iL2);
    
    inflatex = inflatex_array(xtrack);
    inflatey = inflatey_array(xtrack);
    lon_offset = lon_offset_array(xtrack);
    lat_offset = lat_offset_array(xtrack);
    m = m_array(xtrack);
    n = n_array(xtrack);
    A = polyarea([lon_r(:);lon_r(1)],[lat_r(:);lat_r(1)]);
    
    lat_min = min(lat_r);
    lon_min = min(lon_r);
    local_left = lon_c-xmargin*(lon_c-lon_min);
    local_right = lon_c+xmargin*(lon_c-lon_min);
    
    local_bottom = lat_c-ymargin*(lat_c-lat_min);
    local_top = lat_c+ymargin*(lat_c-lat_min);
    
    lon_index = Lon_grid >= local_left & Lon_grid <= local_right;
    lat_index = Lat_grid >= local_bottom & Lat_grid <= local_top;
    
    lon_mesh = Lon_mesh(lat_index,lon_index);
    lat_mesh = Lat_mesh(lat_index,lon_index);
    
    SG = F_2D_SG_affine(lon_mesh,lat_mesh,lon_r,lat_r,lon_c,lat_c,...
        m,n,inflatex,inflatey,lon_offset,lat_offset);
    
    sum_above_local = zeros(nrows,ncols,'single');
    sum_below_local = zeros(nrows,ncols,'single');
    D_local = zeros(nrows,ncols,'single');
    
    D_local(lat_index,lon_index) = SG;
    sum_above_local(lat_index,lon_index) = SG/A/vcd_unc*vcd;
    sum_below_local(lat_index,lon_index) = SG/A/vcd_unc;
    Sum_Above = Sum_Above + sum_above_local;
    Sum_Below = Sum_Below + sum_below_local;
    D = D+D_local;
end
output_regrid.A = Sum_Above;
output_regrid.B = Sum_Below;
output_regrid.C = Sum_Above./Sum_Below;
output_regrid.D = D;

output_regrid.xgrid = xgrid;
output_regrid.ygrid = ygrid;

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