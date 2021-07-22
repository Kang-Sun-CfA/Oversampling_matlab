function output_oversample = F_oversample_OMI_NARR(inp,output_subset)
% very similar to F_regrid_OMI_NARR.m, but instead of using 2-D Gaussian
% SRF, it uses the neighbors-within-a-circle oversampling

% Modified from F_regrid_OMI_NARR.m by Kang Sun on 2017/11/29

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

% % parameters to define pixel SRF
% if ~isfield(inp,'inflatex_array')
%     inflatex_array = ones(1,60);
%     inflatey_array = ones(1,60);
% else
%     inflatex_array = inp.inflatex_array;
%     inflatey_array = inp.inflatey_array;
% end
% 
% if ~isfield(inp,'lon_offset_array')
%     x_offset_array = zeros(1,60);
%     y_offset_array = zeros(1,60);
% else
%     x_offset_array = inp.x_offset_array;
%     y_offset_array = inp.y_offset_array;
% end

% if ~isfield(inp,'m_array')
%     m_array = 4*ones(1,60);
%     n_array = 2*ones(1,60);
% else
%     m_array = inp.m_array;
%     n_array = inp.n_array;
% end

if isfield(inp,'useweekday')
    useweekday = inp.useweekday;
end

% just pure parallel!!!
% if ~isfield(inp,'if_parallel')
%     if_parallel = false;
% else
%     if_parallel = inp.if_parallel;
% end

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
UTC = output_subset.utc(validmask);

[UTC, I] = sort(UTC);
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
                inp_interp_narr.P_pblmax = 100;% maximum pbl thickness in hPa
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
inp_xy.lon = Lon_c(:);
inp_xy.lat = Lat_c(:);
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

O = zeros(nwd,nws,nT,nrows,ncols,'single');
D = zeros(nwd,nws,nT,nrows,ncols,'single');

O_r = zeros(nwd,nws,nT,nrows,ncols,'single');
D_r = zeros(nwd,nws,nT,nrows,ncols,'single');

disp([num2str(nL2),' pixels to be regridded...'])

for iwd = 1:nwd
    for iws = 1:nws
        for iT = 1:nT
            use_idx = wd_idx == iwd & ws_idx == iws & T_idx == iT;
            nl2 = sum(use_idx);
            disp([num2str([iwd iws iT]),' has ',num2str(nl2),' pixels'])
            if nl2 > 0
               x_c_inp = outp_xy.x(use_idx);
               y_c_inp = outp_xy.y(use_idx);
               
%                ws_inp = ws_vec_rot(use_idx);
               wd_inp = wd_vec_rot(use_idx);
               
               vcd_inp = VCD(use_idx);
               vcde_inp = VCDe(use_idx);
               xtrack_inp = Xtrack(use_idx);
               
               x_cr_inp = x_c_inp;
               y_cr_inp = y_c_inp;
                
                % rotate pixel according to wind direction
                for il2 = 1:nl2
                   x_c = x_c_inp(il2);
                   y_c = y_c_inp(il2);
                   
%                    ws = ws_inp(il2);
                   wd = wd_inp(il2);
                   
                   xy_rot = [cos(wd) sin(wd);-sin(wd) cos(wd)]*[x_c;y_c];
%                    x_rr = xy_rot(1,1:4);
%                    y_rr = xy_rot(2,1:4);
                   x_cr_inp(il2) = xy_rot(1,:);
                   y_cr_inp(il2) = xy_rot(2,:);
                end
                
                tmpO = zeros(nrows,ncols,'single');
                tmpD = zeros(nrows,ncols,'single');
                
                tmpOr = zeros(nrows,ncols,'single');
                tmpDr = zeros(nrows,ncols,'single');
                % sorry for many nested loops
                for irow = 1:nrows
                    for icol = 1:ncols
                        xcenter = xgrid(icol);ycenter = ygrid(irow);
                        
                        % for each grid center, find the closeby unrotated
                        % pixels
                        x = x_c_inp; y = y_c_inp;
                        tmp_ind = find(abs(x-xcenter) <= R & abs(y-ycenter) <= R);
                        Distance = sqrt((x(tmp_ind)-xcenter).^2+(y(tmp_ind)-ycenter).^2);
                        ind = tmp_ind(Distance <= R);
                        if do_weight
                            weight = 1./vcde_inp(ind)./(A_array(xtrack_inp(ind)));
                            weight = weight/nansum(weight);
                            tmpO(irow,icol) = nansum(vcd_inp(ind).*weight);
                        else
                            tmpO(irow,icol) = nanmean(vcd_inp(ind));
                        end
                        tmpD(irow,icol) = length(ind);
                        
                        % for each grid center, find the closeby rotated
                        % pixels
                        x = x_cr_inp; y = y_cr_inp;
                        tmp_ind = find(abs(x-xcenter) <= R & abs(y-ycenter) <= R);
                        Distance = sqrt((x(tmp_ind)-xcenter).^2+(y(tmp_ind)-ycenter).^2);
                        ind = tmp_ind(Distance <= R);
                        if do_weight
                            weight = 1./vcde_inp(ind)./(A_array(xtrack_inp(ind)));
                            weight = weight/nansum(weight);
                            tmpOr(irow,icol) = nansum(vcd_inp(ind).*weight);
                        else
                            tmpOr(irow,icol) = nanmean(vcd_inp(ind));
                        end
                        tmpDr(irow,icol) = length(ind);
                    end
                    
                end
                          
                O(iwd,iws,iT,:,:) = tmpO;
                D(iwd,iws,iT,:,:) = tmpD;
                
                O_r(iwd,iws,iT,:,:) = tmpOr;
                D_r(iwd,iws,iT,:,:) = tmpDr;
            end
        end
    end
end

output_oversample.O = O;
output_oversample.D = D;

output_oversample.O_r = O_r;
output_oversample.D_r = D_r;

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