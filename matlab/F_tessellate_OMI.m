function output_regrid = F_tessellate_OMI(inp,output_subset)
% function to take in the output from F_subset_OM*.m and regrid these L2
% data to a L3 grid, centered at clon and clat and bounded by max_x and 
% max_y with resolution res in km.

% Use tessellation method, originated from Lei Zhu and Kai Yang

% Modified by Kang Sun from F_tessellate_OMI_km.m on 2018/06/28

output_regrid = [];

Startdate = inp.Startdate;
Enddate = inp.Enddate;

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
if isfield(inp,'errorpower')
    errorpower = inp.errorpower;
else
    errorpower = 1;
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
    if_inflate = false;
else
    inflatex_array = inp.inflatex_array;
    inflatey_array = inp.inflatey_array;
    if_inflate = true;
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
f2 = output_subset.latc >= MinLat-MarginLat & output_subset.latc <= MaxLat+MarginLat...
    & output_subset.lonc >= MinLon-MarginLon & output_subset.lonc <= MaxLon+MarginLon ...
    & output_subset.latr(:,1) >= MinLat-MarginLat*2 & output_subset.latr(:,1) <= MaxLat+2*MarginLat...
    & output_subset.lonr(:,1) >= MinLon-MarginLon*2 & output_subset.lonr(:,1) <= MaxLon+2*MarginLon;
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


disp('Tessellating pixel by pixel...')
if if_parallel
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
    
    pixel = [];
    pixel.nv = length(lon_r);
    pixel.vList = [(lon_r(:)-MinLon)/Res,(lat_r(:)-MinLat)/Res];
    pixel.center = [(lon_c-MinLon)/Res,(lat_c-MinLat)/Res];
    % Used for the next step
%     id_all = 0;
    pixel_area = 0;
    
    if if_inflate
    % make pixel inflation possible, start
    leftpoint = mean(pixel.vList(1:2,:));
    rightpoint = mean(pixel.vList(3:4,:));

    uppoint = mean(pixel.vList(2:3,:));
    lowpoint = mean(pixel.vList([1 4],:));

    % calculate the FWHM of 2-D super gaussian SRF
    % x is the xtrack, different from the OMI pixel paper, which used y
    FWHMx = sqrt((leftpoint(1)-rightpoint(1))^2+(leftpoint(2)-rightpoint(2))^2);
    % y is the along track, different from the OMI pixel paper, which used x
    FWHMy = sqrt((uppoint(1)-lowpoint(1))^2+(uppoint(2)-lowpoint(2))^2);

    Angle = -atan((rightpoint(2)-leftpoint(2))/(rightpoint(1)-leftpoint(1)));
    rotation_matrix = [cos(Angle), -sin(Angle);
        sin(Angle),  cos(Angle)];
    diamond_orth = (rotation_matrix*...
        [[leftpoint(1) uppoint(1) rightpoint(1) lowpoint(1)]-pixel.center(1);...
        [leftpoint(2) uppoint(2) rightpoint(2) lowpoint(2)]-pixel.center(2)]);

    xleft = diamond_orth(1,1)*inflatex;
    xright = diamond_orth(1,3)*inflatex;
    ytop = diamond_orth(2,2)*inflatey;
    ybottom = diamond_orth(2,4)*inflatey;

    vList_orth = [xleft xleft xright xright;
        ybottom ytop ytop ybottom];

    vList_inflate = [cos(-Angle), -sin(-Angle);
        sin(-Angle),  cos(-Angle)]*vList_orth;

    vList_inflate = [vList_inflate(1,:)'+pixel.center(1),vList_inflate(2,:)'+pixel.center(2)];
    pixel.vList = vList_inflate;
    % make pixel inflation possible, end
    end
    
    % Perform Horizontal cut first at the integer grid lines
    [sub_pixels,n_sub_pixels] = F_HcakeCut( pixel );

    Sub_Area = zeros(nrows,ncols,'single');
    Pixels_count = Sub_Area;
    % Then perform Vertical cut for each sub pixel obtainted
    % from the Horizontal cut at the integer grid lines
    for id_sub = 1: n_sub_pixels
        [final_pixels, n_final_pixels] = F_VcakeCut( sub_pixels(id_sub) );
        for id_final = 1: n_final_pixels
            row = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,2))) + 1;
            col = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,1))) + 1;
            
            if row >= 1 && row <= nrows && col >= 1 && col <= ncols
%             id_all = id_all + 1;
            % temp_area is the area of each sub polygon, at this stage
            [ifsquare,edges] = F_if_square(final_pixels(id_final));
            if ifsquare
                temp_area = edges(1)*edges(2)*Res^2;
            else
                temp_area = F_polyarea(final_pixels(id_final))*Res^2;
            end
            
            pixel_area = pixel_area + temp_area;
            
            % Get the overlaped area between the pixel and each cell
            Sub_Area(row,col) = temp_area;
            
            Pixels_count(row,col) = Pixels_count(row,col) + 1;
            end
        end
    end
%     if abs((pixel_area-A_IASI(Ifov(iL2))*pi)/(A_IASI(Ifov(iL2))*pi)) > 0.1
%         error('Pixel area is more inaccurate than 10%. Increase your npoints!')
%     end
    % Sum weighted value and weights
    % Here use temp_area/A/VCD_Unc(p) as averaging weight, meaning that
    % averaging weight is assumed to be proportional to the ratio of the overlap area (temp_area) to the
    % pixel size (A) and inversely proportional to the error standard deviation (VCD_Unc(p)).
    % If you just want fraction of overlap area as averaging weight, use: temp_area/A
    % If you just want area weighted average, use: temp_area
    if ~isnan(vcd) && ~isnan(vcd_unc)
    Sum_Above = Sum_Above+Sub_Area/(A)/vcd_unc*vcd;
    Sum_Below = Sum_Below+Sub_Area/(A)/vcd_unc;
    D = D+Pixels_count;
    end
end
else
count = 1;
for iL2 = 1:nL2
    lat_r = Lat_r(iL2,:);
    lon_r = Lon_r(iL2,:);
    lat_c = Lat_c(iL2);
    lon_c = Lon_c(iL2);

    vcd = VCD(iL2);
    vcd_unc = VCDe(iL2);
    xtrack = Xtrack(iL2);
    
    inflatex = inflatex_array(xtrack);
    inflatey = inflatey_array(xtrack);
%     x_offset = x_offset_array(xtrack);
%     y_offset = y_offset_array(xtrack);
    m = m_array(xtrack);
    n = n_array(xtrack);
    % area is more elegant if using km rather than lat lon
%     A = polyarea([x_r(:);x_r(1)],[y_r(:);y_r(1)]);
    A = polyarea([lon_r(:);lon_r(1)],[lat_r(:);lat_r(1)]);
    
    pixel = [];
    pixel.nv = length(lon_r);
    pixel.vList = [(lon_r(:)-MinLon)/Res,(lat_r(:)-MinLat)/Res];
    pixel.center = [(lon_c-MinLon)/Res,(lat_c-MinLat)/Res];
    % Used for the next step
%     id_all = 0;
    pixel_area = 0;
    
    if if_inflate
    % make pixel inflation possible, start
    leftpoint = mean(pixel.vList(1:2,:));
    rightpoint = mean(pixel.vList(3:4,:));

    uppoint = mean(pixel.vList(2:3,:));
    lowpoint = mean(pixel.vList([1 4],:));

    % calculate the FWHM of 2-D super gaussian SRF
    % x is the xtrack, different from the OMI pixel paper, which used y
    FWHMx = sqrt((leftpoint(1)-rightpoint(1))^2+(leftpoint(2)-rightpoint(2))^2);
    % y is the along track, different from the OMI pixel paper, which used x
    FWHMy = sqrt((uppoint(1)-lowpoint(1))^2+(uppoint(2)-lowpoint(2))^2);

    Angle = -atan((rightpoint(2)-leftpoint(2))/(rightpoint(1)-leftpoint(1)));
    rotation_matrix = [cos(Angle), -sin(Angle);
        sin(Angle),  cos(Angle)];
    diamond_orth = (rotation_matrix*...
        [[leftpoint(1) uppoint(1) rightpoint(1) lowpoint(1)]-pixel.center(1);...
        [leftpoint(2) uppoint(2) rightpoint(2) lowpoint(2)]-pixel.center(2)]);

    xleft = diamond_orth(1,1)*inflatex;
    xright = diamond_orth(1,3)*inflatex;
    ytop = diamond_orth(2,2)*inflatey;
    ybottom = diamond_orth(2,4)*inflatey;

    vList_orth = [xleft xleft xright xright;
        ybottom ytop ytop ybottom];

    vList_inflate = [cos(-Angle), -sin(-Angle);
        sin(-Angle),  cos(-Angle)]*vList_orth;

    vList_inflate = [vList_inflate(1,:)'+pixel.center(1),vList_inflate(2,:)'+pixel.center(2)];
    pixel.vList = vList_inflate;
    % make pixel inflation possible, end
    end
    
    % Perform Horizontal cut first at the integer grid lines
    [sub_pixels,n_sub_pixels] = F_HcakeCut( pixel );

    Sub_Area = zeros(nrows,ncols,'single');
    Pixels_count = Sub_Area;
    % Then perform Vertical cut for each sub pixel obtainted
    % from the Horizontal cut at the integer grid lines
    for id_sub = 1: n_sub_pixels
        [final_pixels, n_final_pixels] = F_VcakeCut( sub_pixels(id_sub) );
        for id_final = 1: n_final_pixels
            row = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,2))) + 1;
            col = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,1))) + 1;
            
            if row >= 1 && row <= nrows && col >= 1 && col <= ncols
%             id_all = id_all + 1;
            % temp_area is the area of each sub polygon, at this stage
            [ifsquare,edges] = F_if_square(final_pixels(id_final));
            if ifsquare
                temp_area = edges(1)*edges(2)*Res^2;
            else
                temp_area = F_polyarea(final_pixels(id_final))*Res^2;
            end
            
            pixel_area = pixel_area + temp_area;
            
            % Get the overlaped area between the pixel and each cell
            Sub_Area(row,col) = temp_area;
            
            Pixels_count(row,col) = Pixels_count(row,col) + 1;
            end
        end
    end
%     if abs((pixel_area-A_IASI(Ifov(iL2))*pi)/(A_IASI(Ifov(iL2))*pi)) > 0.1
%         error('Pixel area is more inaccurate than 10%. Increase your npoints!')
%     end
    % Sum weighted value and weights
    % Here use temp_area/A/VCD_Unc(p) as averaging weight, meaning that
    % averaging weight is assumed to be proportional to the ratio of the overlap area (temp_area) to the
    % pixel size (A) and inversely proportional to the error standard deviation (VCD_Unc(p)).
    % If you just want fraction of overlap area as averaging weight, use: temp_area/A
    % If you just want area weighted average, use: temp_area
    if ~isnan(vcd) && ~isnan(vcd_unc)
    Sum_Above = Sum_Above+Sub_Area/(A)/vcd_unc*vcd;
    Sum_Below = Sum_Below+Sub_Area/(A)/vcd_unc;
    D = D+Pixels_count;
    end
    
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
output_regrid.xmesh = single(Lon_mesh);
output_regrid.ymesh = single(Lat_mesh);


return

