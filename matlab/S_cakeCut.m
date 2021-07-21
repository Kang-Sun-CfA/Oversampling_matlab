warning off
parfor ip = 1:np
    Lon_r = LON(ip,1:4);
    Lat_r = LAT(ip,1:4);
    Lon_c = LON(ip,5);
    Lat_c = LAT(ip,5);
    
if ~isfield(options,'inflate_pixel')
    inflate_pixel = false;
else
    inflate_pixel = options.inflate_pixel;
end

if ~isfield(options,'use_SRF')
    use_SRF = false;
else
    use_SRF = options.use_SRF;
end

if ~isfield(options,'use_simplified_area')
    use_simplified_area = false;
else
    use_simplified_area = options.use_simplified_area;
end
pixel = [];
pixel.nv = length(Lon_r);
pixel.vList = [(Lon_r(:)-Lon_left)/Res,(Lat_r(:)-Lat_low)/Res];
pixel.center = [(Lon_c-Lon_left)/Res,(Lat_c-Lat_low)/Res];
vList = pixel.vList;

% inflate_pixel and use_SRF should be used together, but don't have to
if ~use_SRF % if not use SRF, there is no point of inflating!
    inflate_pixel = false;
end

if inflate_pixel || use_SRF
    leftpoint = mean(vList(1:2,:));
    rightpoint = mean(vList(3:4,:));
    
    uppoint = mean(vList(2:3,:));
    lowpoint = mean(vList([1 4],:));
    
    % calculate the FWHM of 2-D super gaussian SRF
    % x is the xtrack, different from the OMI pixel paper, which used y
    FWHMx = sqrt((leftpoint(1)-rightpoint(1))^2+(leftpoint(2)-rightpoint(2))^2);
    % y is the along track, different from the OMI pixel paper, which used x
    FWHMy = sqrt((uppoint(1)-lowpoint(1))^2+(uppoint(2)-lowpoint(2))^2);
    
    Angle = -atan((rightpoint(2)-leftpoint(2))/(rightpoint(1)-leftpoint(1)));
    rotation_matrix = [cos(Angle), -sin(Angle);
        sin(Angle),  cos(Angle)];
end

% m is the super gaussian exponient in x (xtrack), n is in y (along track)
if use_SRF
    if ~isfield(options,'m')
        m = 4;
    else
        m = options.m;
    end
    if ~isfield(options,'n')
        n = 2;
    else
        n = options.n;
    end
end

if inflate_pixel % if inflate the pixel, update the pixel object
    % inflation factor xtrack
    if ~isfield(options,'inflationx')
        inflationx = 1.5;
    else
        inflationx = options.inflationx;
    end
    % inflation factor along track
    if ~isfield(options,'inflationy')
        inflationy = 2;
    else
        inflationy = options.inflationy;
    end
    
    diamond_orth = (rotation_matrix*...
        [[leftpoint(1) uppoint(1) rightpoint(1) lowpoint(1)]-pixel.center(1);...
        [leftpoint(2) uppoint(2) rightpoint(2) lowpoint(2)]-pixel.center(2)]);
    
    xleft = diamond_orth(1,1)*inflationx;
    xright = diamond_orth(1,3)*inflationx;
    ytop = diamond_orth(2,2)*inflationy;
    ybottom = diamond_orth(2,4)*inflationy;
    
    vList_orth = [xleft xleft xright xright;
        ybottom ytop ytop ybottom];
    
    vList_inflate = [cos(-Angle), -sin(-Angle);
        sin(-Angle),  cos(-Angle)]*vList_orth;
    
    vList_inflate = [vList_inflate(1,:)'+pixel.center(1),vList_inflate(2,:)'+pixel.center(2)];
    pixel.vList = vList_inflate;
end
% Used for the next step
id_all = 0;
pixel_area = 0;

% Perform Horizontal cut first at the integer grid lines
[sub_pixels,n_sub_pixels] = F_HcakeCut( pixel );

% Sub_Area = zeros(size(Sum_Above));
Sub_Area = zeros(nrows,ncols);
% Then perform Vertical cut for each sub pixel obtainted
% from the Horizontal cut at the integer grid lines
for id_sub = 1: n_sub_pixels
    [final_pixels, n_final_pixels] = F_VcakeCut( sub_pixels(id_sub) );
    for id_final = 1: n_final_pixels
        id_all = id_all + 1;
        % temp_area is the area of each sub polygon, at this stage
        if use_simplified_area
            temp_area = Res^2;
        else
            [ifsquare,edges] = F_if_square(final_pixels(id_final));
            if ifsquare
                temp_area = edges(1)*edges(2)*Res^2;
            else
                temp_area = F_polyarea(final_pixels(id_final))*Res^2;
            end
        end
        row = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,2))) + 1;
        col = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,1))) + 1;
        
        % if necessary, update temp_area to include the weighting from SRF
        if use_SRF
            temp_area = temp_area * ...
                F_2D_SG(col,row,pixel.center(1),pixel.center(2),FWHMx,FWHMy,m,n,rotation_matrix);
        end
        pixel_area = pixel_area + temp_area;
        
        % Get the overlaped area between the pixel and each cell
        Sub_Area(row,col) = temp_area;
        
%         Pixels_count(row,col) = Pixels_count(row,col) + 1;
    end
end
% Sum weighted value and weights
% Here use temp_area/A/VCD_Unc(p) as averaging weight, meaning that
% averaging weight is assumed to be proportional to the ratio of the overlap area (temp_area) to the
% pixel size (A) and inversely proportional to the error standard deviation (VCD_Unc(p)).
% If you just want fraction of overlap area as averaging weight, use: temp_area/A
% If you just want area weighted average, use: temp_area
Sum_Above = Sum_Above+Sub_Area/pixel_area/VCD_Unc(ip)*VCD(ip);
Sum_Below = Sum_Below+Sub_Area/pixel_area/VCD_Unc(ip);

end
warning on
Sum_Below(Sum_Below == 0) = nan;
Average = Sum_Above./Sum_Below;
