function output_regrid = F_regrid_TROPOMI_met(inp,output_subset)
% major undertake to combine tropomi, projection 
% coordinate system, wind rotation, and level 2 data slicing. 
% updated from F_regrid_TROPOMI.m on 2019/06/20

output_regrid = [];

wd_bin = inp.wd_bin;

if ~isfield(inp,'ws_bin')
    ws_bin = [0 inf];
else
    ws_bin = inp.ws_bin;
end

if ~isfield(inp,'T_bin')% T was meant to be temperature but can be any scalar (albedo for example)
    T_bin = [-inf inf];
else
    T_bin = inp.T_bin;% in K
end

nwd = length(wd_bin);
nws = length(ws_bin)-1;
nT = length(T_bin)-1;

if isfield(inp,'which_wind')
    which_wind = inp.which_wind;
else
    which_wind = 'U10M';
end

if isfield(inp,'which_T')
    which_T = inp.which_wind;
else
    which_T = 'T2M';
end

if ~isfield(output_subset,which_wind)
    error(['Wind source ',which_wind,' does not exist in your L2g data!']);
end

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
if isfield(inp,'errorpower')
    errorpower = inp.errorpower;
else
    errorpower = 1;
end

Startdate = inp.Startdate;
Enddate = inp.Enddate;

% max cloud fraction and SZA, VZA, QA and min and max values for NO2
if isfield(inp,'MinQA')
    MinQA = inp.MinQA;
else
    MinQA = 0.5;
end
if isfield(inp,'MaxCF')
    MaxCF = inp.MaxCF;
else
    MaxCF = inf;
end
if isfield(inp,'MaxSZA')
    MaxSZA = inp.MaxSZA;
else
    MaxSZA = inf;
end
if isfield(inp,'MaxVZA')
    MaxVZA = inp.MaxVZA;
else
    MaxVZA = inf;
end
if isfield(inp,'MaxNO2')
    inp.MaxCol = inp.MaxNO2;
    inp.MinCol = inp.MinNO2;
end
if isfield(inp,'MaxCol')
    MaxCol = inp.MaxCol;
    MinCol = inp.MinCol;
else
    MaxCol = inf;
    MinCol = -inf;
end

% xtrack to use
if ~isfield(inp,'usextrack')
    usextrack = 1:1000;
else
    usextrack = inp.usextrack;
end

vcdname = inp.vcdname;
vcderrorname = inp.vcderrorname;

% % parameters to define pixel SRF
% if ~isfield(inp,'inflatex_array')
%     inflatex_array = ones(1,450);
%     inflatey_array = ones(1,450);
% else
%     inflatex_array = inp.inflatex_array;
%     inflatey_array = inp.inflatey_array;
% end
%
% if ~isfield(inp,'lon_offset_array')
%     lon_offset_array = zeros(1,450);
%     lat_offset_array = zeros(1,450);
% else
%     lon_offset_array = inp.lon_offset_array;
%     lat_offset_array = inp.lat_offset_array;
% end
%
% if ~isfield(inp,'m_array')
%     m_array = 4*ones(1,450);
%     n_array = 2*ones(1,450);
% else
%     m_array = inp.m_array;
%     n_array = inp.n_array;
% end

% if ~isfield(inp,'if_parallel')
%     if_parallel = false;
% else
%     if_parallel = inp.if_parallel;
% end

if isfield(inp,'useweekday')
    useweekday = inp.useweekday;
end

% define x y grids
% be careful, these are linear grid, in km
res = inp.res;
max_x = inp.max_x;
max_y = inp.max_y;
clon = inp.clon;
clat = inp.clat;

max_lon = clon+max_x*1.2/110/cos((abs(clat)+max_y/111)/180*pi);
min_lon = clon-max_x*1.2/110/cos((abs(clat)+max_y/111)/180*pi);
max_lat = clat+max_y*1.2/110;
min_lat = clat-max_y*1.2/110;

xgrid = (-max_x+0.5*res):res:max_x;
ygrid = (-max_y+0.5*res):res:max_y;

nrows = length(ygrid);
ncols = length(xgrid);

% define x y mesh
[xmesh, ymesh] = meshgrid(xgrid,ygrid);

% construct a rectangle envelopes the orginal pixel
xmargin = 2; % how many times to extend zonally
ymargin = 2; % how many times to extend meridonally

f1 = output_subset.utc >= datenum([Startdate 0 0 0]) & output_subset.utc <= datenum([Enddate 23 59 59]);
f2 = output_subset.latc >= min_lat-0.5 & output_subset.latc <= max_lat+0.5...
    & output_subset.lonc >= min_lon-0.5 & output_subset.lonc <= max_lon+0.5 ...
    & output_subset.latr(:,1) >= min_lat-1 & output_subset.latr(:,1) <= max_lat+1 ...
    & output_subset.lonr(:,1) >= min_lon-1 & output_subset.lonr(:,1) <= max_lon+1;
if isfield(output_subset,'cloudfrac')
    f4 = output_subset.cloudfrac <= MaxCF;
else
    f4 = true(size(output_subset.utc));
end
% f9, f10, f11 and f12 were added to take care of QA for TROPOMI data (added by Lorena Castro)
f9 = output_subset.qa_value > MinQA;
if isfield(output_subset,'vza')
    f10 = output_subset.vza <= MaxVZA;
else
    f10 = true(size(output_subset.utc));
end
if isfield(output_subset,'sza')
    f3 = output_subset.sza <= MaxSZA;
else
    f3 = true(size(output_subset.utc));
end
f11 = output_subset.(vcdname) >= MinCol;
f12 = output_subset.(vcdname) <= MaxCol;

f5 = ismember(output_subset.ift,usextrack);

% add on 2018/03/28 for OMCHOCHO
f7 = output_subset.(vcdname) > -1e26;

% add on 2018/09/10 to make sure uncertainties are all positive
f8 = output_subset.(vcderrorname) > 0;

validmask = f1&f2&f3&f4&f5&f7&f8&f9&f10&f11&f12;

if exist('useweekday','var')
    wkdy = weekday(output_subset.utc);
    f6 = ismember(wkdy,useweekday);
    validmask = validmask & f6;
end

nL2 = sum(validmask);
if nL2 <= 0;return;end
disp(['Regriding pixels from ',datestr([Startdate 0 0 0]),' to ', datestr([Enddate 0 0 0])])
disp([num2str(nL2),' pixels to be regridded...'])

Lat_r = output_subset.latr(validmask,:);
Lon_r = output_subset.lonr(validmask,:);
Lat_c = output_subset.latc(validmask);
Lon_c = output_subset.lonc(validmask);
Xtrack = output_subset.ift(validmask);
VCD = output_subset.(vcdname)(validmask);
VCDe = (output_subset.(vcderrorname)(validmask)).^errorpower;
% pixel_west = Lon_c-range(Lon_r,2)/2*xmargin;
% pixel_east = Lon_c+range(Lon_r,2)/2*xmargin;
% pixel_south = Lat_c-range(Lat_r,2)/2*ymargin;
% pixel_north = Lat_c+range(Lat_r,2)/2*ymargin;
u_vec_rot = output_subset.(which_wind)(validmask);
v_vec_rot = output_subset.(['V',which_wind(2:end)])(validmask);
if isfield(output_subset,which_T)
    T_vec = output_subset.(which_T)(validmask);
else
    warning(['Met data ',which_T,' does not exist in your L2g data.']);
    T_vec = ones(size(u_vec_rot));
end
totalindex = 1:nL2;
disp('Converting pixel lat lon to x y in km...')
% call function F_latlon2xy to convert pixel lat lon to x y. rotation
% center is already [0, 0]
inp_xy = [];
inp_xy.clon = clon;
inp_xy.clat = clat;
inp_xy.lon = [Lon_r,Lon_c(:)];
inp_xy.lat = [Lat_r,Lat_c(:)];
outp_xy = F_latlon2xy(inp_xy);
% categorization
disp('Categorize data based on WS, WD...')

ws_vec_rot = sqrt(u_vec_rot.^2+v_vec_rot.^2);
wd_vec_rot = u_vec_rot;
wd_vec_rot(v_vec_rot >= 0) = acos(u_vec_rot(v_vec_rot >= 0)./ws_vec_rot(v_vec_rot >= 0));
wd_vec_rot(v_vec_rot < 0) = 2*pi-acos(u_vec_rot(v_vec_rot < 0)./ws_vec_rot(v_vec_rot < 0));

[~,~,ws_idx] = histcounts(ws_vec_rot,ws_bin);
[~,~,wd_idx] = histcounts(wd_vec_rot,[0 wd_bin 2*pi]);
wd_idx(wd_idx == nwd+1) = 1;
[~,~,T_idx] = histcounts(T_vec,T_bin);

A = zeros(nwd,nws,nT,nrows,ncols,'single');
B = zeros(nwd,nws,nT,nrows,ncols,'single');
D = zeros(nwd,nws,nT,nrows,ncols,'single');

A_r = zeros(nwd,nws,nT,nrows,ncols,'single');
B_r = zeros(nwd,nws,nT,nrows,ncols,'single');
D_r = zeros(nwd,nws,nT,nrows,ncols,'single');

m = 4;n = 2;

for iwd = 1:nwd
    for iws = 1:nws
        for iT = 1:nT
            use_idx = wd_idx == iwd & ws_idx == iws & T_idx == iT;
            nl2 = sum(use_idx);
            disp([num2str([iwd iws iT]),' has ',num2str(nl2),' pixels'])
            if nl2 > 0
               x_r_inp = outp_xy.x(use_idx,1:4);
               y_r_inp = outp_xy.y(use_idx,1:4);
               x_c_inp = outp_xy.x(use_idx,5);
               y_c_inp = outp_xy.y(use_idx,5);

               ws_inp = ws_vec_rot(use_idx);
               wd_inp = wd_vec_rot(use_idx);

               vcd_inp = VCD(use_idx);
               vcde_inp = VCDe(use_idx);
               xtrack_inp = Xtrack(use_idx);

               Sum_Above = zeros(nrows,ncols,'single');
               Sum_Below = zeros(nrows,ncols,'single');
               Sum_SG = zeros(nrows,ncols,'single');

               Sum_Abover = zeros(nrows,ncols,'single');
               Sum_Belowr = zeros(nrows,ncols,'single');
               Sum_SGr = zeros(nrows,ncols,'single');
               for il2 = 1:nl2
                   x_r = x_r_inp(il2,:);
                   y_r = y_r_inp(il2,:);
                   x_c = x_c_inp(il2);
                   y_c = y_c_inp(il2);

%                    ws = ws_inp(il2);
                   wd = wd_inp(il2);

                   xy_rot = [cos(wd) sin(wd);-sin(wd) cos(wd)]*[x_r x_c;y_r y_c];
                   x_rr = xy_rot(1,1:4);
                   y_rr = xy_rot(2,1:4);
                   x_cr = xy_rot(1,5);
                   y_cr = xy_rot(2,5);

%                    xtrack = xtrack_inp(il2);
                   vcd = vcd_inp(il2);
                   vcd_unc = vcde_inp(il2);

%                    inflatex = inflatex_array(xtrack);
%                    inflatey = inflatey_array(xtrack);
%                    x_offset = x_offset_array(xtrack);
%                    y_offset = y_offset_array(xtrack);
%                    m = m_array(xtrack);
%                    n = n_array(xtrack);
                   Area = polyarea([x_r(:);x_r(1)],[y_r(:);y_r(1)]);

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
                       m,n,1,1,0,0);
                   
                   Sum_Above(y_index,x_index) = Sum_Above(y_index,x_index) + SG/Area/vcd_unc*vcd;
                   Sum_Below(y_index,x_index) = Sum_Below(y_index,x_index) + SG/Area/vcd_unc;
                   Sum_SG(y_index,x_index) = Sum_SG(y_index,x_index)+SG;
                   
                   % repeat it with rotated x y
                   y_min_local = min(y_rr);
                   x_min_local = min(x_rr);
                   local_left = x_cr-xmargin*(x_cr-x_min_local);
                   local_right = x_cr+xmargin*(x_cr-x_min_local);

                   local_bottom = y_cr-ymargin*(y_cr-y_min_local);
                   local_top = y_cr+ymargin*(y_cr-y_min_local);

                   x_index = xgrid >= local_left & xgrid <= local_right;
                   y_index = ygrid >= local_bottom & ygrid <= local_top;

                   x_local_mesh = xmesh(y_index,x_index);
                   y_local_mesh = ymesh(y_index,x_index);
                   
                   SG = F_2D_SG_affine(x_local_mesh,y_local_mesh,x_rr,y_rr,x_cr,y_cr,...
                       m,n,1,1,0,0);
                   
                   Sum_Abover(y_index,x_index) = Sum_Abover(y_index,x_index) + SG/Area/vcd_unc*vcd;
                   Sum_Belowr(y_index,x_index) = Sum_Belowr(y_index,x_index) + SG/Area/vcd_unc;
                   Sum_SGr(y_index,x_index) = Sum_SGr(y_index,x_index)+SG;
               end
               A(iwd,iws,iT,:,:) = Sum_Above;
               B(iwd,iws,iT,:,:) = Sum_Below;
               D(iwd,iws,iT,:,:) = Sum_SG;

               A_r(iwd,iws,iT,:,:) = Sum_Abover;
               B_r(iwd,iws,iT,:,:) = Sum_Belowr;
               D_r(iwd,iws,iT,:,:) = Sum_SGr;
            end
        end
    end
end

output_regrid.A = A;
output_regrid.B = B;
output_regrid.D = D;

output_regrid.A_r = A_r;
output_regrid.B_r = B_r;
output_regrid.D_r = D_r;

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

tform = fitgeotrans(fixedPoints,vList,'projective');

xym1 = [lon_mesh(:)-lon_c-lon_offset, lat_mesh(:)-lat_c-lat_offset];
xym2 = transformPointsInverse(tform,xym1);

FWHMy = FWHMy*inflatey;
FWHMx = FWHMx*inflatex;

wx = FWHMx/2/(log(2))^(1/m);
wy = FWHMy/2/(log(2))^(1/n);

SG = exp(-(abs(xym2(:,1))/wx).^m-(abs(xym2(:,2))/wy).^n);
SG = reshape(SG(:),size(lon_mesh,1),size(lon_mesh,2));

