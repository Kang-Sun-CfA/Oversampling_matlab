function output_regrid = F_regrid_OMIPROFOZ_NARR(inp,output_subset)
% rewritten by Kang Sun from F_regrid_OMI_NARR.m on 2018/01/01
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

if ~isfield(inp,'NARR_download_dir')
    error('You have to know where NARR data are!')
else
    NARR_download_dir = inp.NARR_download_dir;
end

wd_bin = inp.wd_bin;

if ~isfield(inp,'ws_bin')
    ws_bin = [0 inf];
else
    ws_bin = inp.ws_bin;
end

if ~isfield(inp,'T_bin')
    T_bin = [0 inf];
else
    T_bin = inp.T_bin;% in K
end

nwd = length(wd_bin);
nws = length(ws_bin)-1;
nT = length(T_bin)-1;

which_wind = inp.which_wind;

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
if ~isfield(inp,'vcderrorname')
    vcderrorname = 'none';
else
    vcderrorname = inp.vcderrorname;
end

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

% just pure parallel!!!
% if ~isfield(inp,'if_parallel')
%     if_parallel = false;
% else
%     if_parallel = inp.if_parallel;
% end
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

Lat_r = output_subset.lat_r(validmask,:);
Lon_r = output_subset.lon_r(validmask,:);
Lat_c = output_subset.lat_c(validmask);
Lon_c = output_subset.lon_c(validmask);
Xtrack = output_subset.ift(validmask);
VCD = output_subset.(vcdname)(validmask);
if strcmpi(vcderrorname,'none')
    VCDe = ones(size(VCD),'single');
else
    VCDe = output_subset.(vcderrorname)(validmask);
end
UTC = output_subset.utc(validmask);

[UTC, I] = sort(UTC);
Lat_r = Lat_r(I,:);
Lon_r = Lon_r(I,:);
Lat_c = Lat_c(I);
Lon_c = Lon_c(I);
Xtrack = Xtrack(I);
VCD = VCD(I);
VCDe = VCDe(I);

% allocate space for wind, associated with each satellite pixel
U_sat_pbl = nan(size(Lat_c),'single');
V_sat_pbl = U_sat_pbl;
T_sat_pbl = U_sat_pbl;

% allocate space for wind, associated with each rotation center
U_c_pbl = nan(size(Lat_c),'single');
V_c_pbl = U_sat_pbl;
T_c_pbl = U_sat_pbl;

day_1 = datevec(UTC(1));
day_end = datevec(UTC(end));

% saved NARR data in which hours
narr_hour_vec = 12:3:24;
% work out NARR's map projection
mstruct_narr = defaultm('lambertstd');
mstruct_narr.origin = [50 360-107 0];
mstruct_narr.mapparallels = 50;
mstruct_narr.nparallels = 1;
mstruct_narr.scalefactor = 6367470/1e3;
mstruct_narr.falseeasting = 0;
mstruct_narr.falsenorthing = 0;
mstruct_narr = defaultm(mstruct_narr);
% map center lat lon to narr projection x y
[x_c, y_c] = mfwdtran(mstruct_narr,clat,clon);

totalindex = 1:nL2;

% work out the wind interpolation day-by-day
disp('Interpolating NARR wind to satellite pixel locations...')

for narr_date = datenum(day_1(1:3)):1:datenum(day_end(1:3));
    int = UTC > datenum(narr_date) & UTC <= datenum(narr_date)+1;
    if sum(int) > 0
        [N,~,bin] = histcounts(UTC(int),narr_date+[narr_hour_vec-1.5,narr_hour_vec(end)+1.5]/24);
        lon_interp_day = Lon_c(int);
        lat_interp_day = Lat_c(int);
        idx_interp_day = totalindex(int);
        for ibin = 1:length(N)
            if N(ibin) > 0
                tmp = datevec(narr_date+narr_hour_vec(ibin)/24);
                narr_year = num2str(tmp(1));
                narr_month = num2str(tmp(2),'%02d');
                narr_day = num2str(tmp(3),'%02d');
                narr_hour = num2str(tmp(4),'%02d');
                disp(['Interpolating NARR for ',datestr(tmp)])
                inp_interp_narr = [];
                inp_interp_narr.narr_data_3d = load([NARR_download_dir,'/',narr_year,'/',narr_month,'/',...
                    'subset_3D_',narr_day,'_',narr_hour,'.mat']);
                inp_interp_narr.narr_data_sfc = load([NARR_download_dir,'/',narr_year,'/',narr_month,'/',...
                    'subset_sfc_',narr_day,'_',narr_hour,'.mat']);
                inp_interp_narr.lon_interp_hour = lon_interp_day(bin == ibin);
                inp_interp_narr.lat_interp_hour = lat_interp_day(bin == ibin);
                inp_interp_narr.x_c = x_c;
                inp_interp_narr.y_c = y_c;
                inp_interp_narr.mstruct_narr = mstruct_narr;
                inp_interp_narr.max_x = max_x;
                inp_interp_narr.max_y = max_y;
                inp_interp_narr.P_pblmax = 200;% maximum pbl thickness in hPa
                outp_interp_narr = F_interp_narr(inp_interp_narr);
                
                idx_interp_hour = idx_interp_day(bin == ibin);
                U_sat_pbl(idx_interp_hour) = outp_interp_narr.u_sat_pbl;
                V_sat_pbl(idx_interp_hour) = outp_interp_narr.v_sat_pbl;
                T_sat_pbl(idx_interp_hour) = outp_interp_narr.T_sat_pbl;
                U_c_pbl(idx_interp_hour) = outp_interp_narr.u_c_pbl;
                V_c_pbl(idx_interp_hour) = outp_interp_narr.v_c_pbl;
                T_c_pbl(idx_interp_hour) = outp_interp_narr.T_c_pbl;
            end
        end
    end
end

disp('Converting pixel lat lon to x y in km...')
% call function F_latlon2xy to convert pixel lat lon to x y. rotation
% center is already [0, 0]
inp_xy = [];
inp_xy.clon = clon;
inp_xy.clat = clat;
inp_xy.lon = [Lon_r,Lon_c(:)];
inp_xy.lat = [Lat_r,Lat_c(:)];
outp_xy = F_latlon2xy(inp_xy);
%% categorization
disp('Categorize data based on WS/WD/T...')
switch which_wind
    case 'pbl_pixel'
        u_vec_rot = U_sat_pbl;
        v_vec_rot = V_sat_pbl;
    case 'pbl_center'
        u_vec_rot = U_c_pbl;
        v_vec_rot = V_c_pbl;
    case 'pbl_average'
        u_vec_rot = 0.5*(U_sat_pbl+U_c_pbl);
        v_vec_rot = 0.5*(V_sat_pbl+V_c_pbl);
end

ws_vec_rot = sqrt(u_vec_rot.^2+v_vec_rot.^2);
wd_vec_rot = u_vec_rot;
wd_vec_rot(v_vec_rot >= 0) = acos(u_vec_rot(v_vec_rot >= 0)./ws_vec_rot(v_vec_rot >= 0));
wd_vec_rot(v_vec_rot < 0) = 2*pi-acos(u_vec_rot(v_vec_rot < 0)./ws_vec_rot(v_vec_rot < 0));

[~,~,ws_idx] = histcounts(ws_vec_rot,ws_bin);
[~,~,wd_idx] = histcounts(wd_vec_rot,[0 wd_bin 2*pi]);
wd_idx(wd_idx == nwd+1) = 1;
[~,~,T_idx] = histcounts(T_sat_pbl,T_bin);

A = zeros(nwd,nws,nT,nrows,ncols,'single');
B = zeros(nwd,nws,nT,nrows,ncols,'single');
D = zeros(nwd,nws,nT,nrows,ncols,'single');

A_r = zeros(nwd,nws,nT,nrows,ncols,'single');
B_r = zeros(nwd,nws,nT,nrows,ncols,'single');
D_r = zeros(nwd,nws,nT,nrows,ncols,'single');

disp([num2str(nL2),' pixels to be regridded...'])

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
               
               parfor il2 = 1:nl2
                   x_r = x_r_inp(il2,:);
                   y_r = y_r_inp(il2,:);
                   x_c = x_c_inp(il2);
                   y_c = y_c_inp(il2);
                   
                   ws = ws_inp(il2);
                   wd = wd_inp(il2);
                   
                   xy_rot = [cos(wd) sin(wd);-sin(wd) cos(wd)]*[x_r x_c;y_r y_c];
                   x_rr = xy_rot(1,1:4);
                   y_rr = xy_rot(2,1:4);
                   x_cr = xy_rot(1,5);
                   y_cr = xy_rot(2,5);
                   
                   xtrack = xtrack_inp(il2);
                   vcd = vcd_inp(il2);
                   vcd_unc = vcde_inp(il2);
                   
                   inflatex = inflatex_array(xtrack);
                   inflatey = inflatey_array(xtrack);
                   x_offset = x_offset_array(xtrack);
                   y_offset = y_offset_array(xtrack);
                   m = m_array(xtrack);
                   n = n_array(xtrack);
                   Area = A_array(xtrack);
                   
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
                   sum_above_local(y_index,x_index) = SG/Area/vcd_unc*vcd;
                   sum_below_local(y_index,x_index) = SG/Area/vcd_unc;
                   Sum_Above = Sum_Above + sum_above_local;
                   Sum_Below = Sum_Below + sum_below_local;
                   Sum_SG = Sum_SG+D_local;
                   
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
                       m,n,inflatex,inflatey,x_offset,y_offset);
                   
                   sum_above_local = zeros(nrows,ncols,'single');
                   sum_below_local = zeros(nrows,ncols,'single');
                   D_local = zeros(nrows,ncols,'single');
                   
                   D_local(y_index,x_index) = SG;
                   sum_above_local(y_index,x_index) = SG/Area/vcd_unc*vcd;
                   sum_below_local(y_index,x_index) = SG/Area/vcd_unc;
                   Sum_Abover = Sum_Abover + sum_above_local;
                   Sum_Belowr = Sum_Belowr + sum_below_local;
                   Sum_SGr = Sum_SGr+D_local;
                   
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
