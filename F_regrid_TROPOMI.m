function output_regrid = F_regrid_TROPOMI(inp,output_subset)
% written by Lorena Castro from Iowa University to oversample TROPOMI data
% updated by Kang Sun on 2019/05/08 for more flexibility
% updated on 2019/05/11 for block-parallel
% updated on 2019/06/30 for multi-variable oversampling

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
if ~iscell(vcdname)
    if_old_vcd = true;
    vcdname = {vcdname};
else
    if_old_vcd = false;
end
nv = length(vcdname);
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

if ~isfield(inp,'if_parallel')
    if_parallel = false;
else
    if_parallel = inp.if_parallel;
end

if isfield(inp,'useweekday')
    useweekday = inp.useweekday;
end

% define x y grids
xgrid = (MinLon+0.5*Res):Res:MaxLon;
ygrid = (MinLat+0.5*Res):Res:MaxLat;
nrows = length(ygrid);
ncols = length(xgrid);

% define x y mesh
[Lon_mesh, Lat_mesh] = meshgrid(double(xgrid),double(ygrid));

% construct a rectangle envelopes the orginal pixel
xmargin = 1.5; % how many times to extend zonally
ymargin = 2; % how many times to extend meridonally

f1 = output_subset.utc >= datenum([Startdate 0 0 0]) & output_subset.utc <= datenum([Enddate 23 59 59]);

f2 = output_subset.latc >= MinLat-MarginLat & output_subset.latc <= MaxLat+MarginLat...
    & output_subset.lonc >= MinLon-MarginLon & output_subset.lonc <= MaxLon+MarginLon ...
    & output_subset.latr(:,1) >= MinLat-MarginLat*2 & output_subset.latr(:,1) <= MaxLat+2*MarginLat...
    & output_subset.lonr(:,1) >= MinLon-MarginLon*2 & output_subset.lonr(:,1) <= MaxLon+2*MarginLon;
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

f11 = output_subset.(vcdname{1}) >= MinCol;
f12 = output_subset.(vcdname{1}) <= MaxCol;

f5 = ismember(output_subset.ift,usextrack);

f7 = output_subset.(vcdname{1}) > -1e26;

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
% VCD is a structure if do_multi_var is true, a vector if otherwise
VCD = nan(length(Lat_c),nv);
for iv = 1:nv
    VCD(:,iv) = output_subset.(vcdname{iv})(validmask);
end
VCDe = (output_subset.(vcderrorname)(validmask)).^errorpower;
pixel_west = Lon_c-range(Lon_r,2)/2*xmargin;
pixel_east = Lon_c+range(Lon_r,2)/2*xmargin;
pixel_south = Lat_c-range(Lat_r,2)/2*ymargin;
pixel_north = Lat_c+range(Lat_r,2)/2*ymargin;

m = 4;n = 2;
if if_parallel
    
    if isfield(inp,'block_length')
        block_length = inp.block_length;
    else
        block_length = 200;
    end
    nblock_row = ceil(nrows/block_length);
    nblock_col = ceil(ncols/block_length);
    row_block_length = ones(1,nblock_row)*block_length;
    row_block_length(end) = row_block_length(end)-(sum(row_block_length)-nrows);
    if sum(row_block_length) ~= nrows
        error('row block number is wrong')
    end
    col_block_length = ones(1,nblock_col)*block_length;
    col_block_length(end) = col_block_length(end)-(sum(col_block_length)-ncols);
    if sum(row_block_length) ~= nrows
        error('row block number is wrong')
    end
    c_xmesh = mat2cell(Lon_mesh,row_block_length,col_block_length);
    c_ymesh = mat2cell(Lat_mesh,row_block_length,col_block_length);
    nblock = numel(c_xmesh);
    if nblock ~= length(row_block_length)*length(col_block_length)
        error('number of blocks are wrong')
    end
    c_xmesh = c_xmesh(:);
    c_ymesh = c_ymesh(:);
    c_A = cell(nblock,1);
    c_B = c_xmesh;
    c_D = c_xmesh;
    c_xgrid = cell(nblock,1);
    c_ygrid = cell(nblock,1);
    c_Lat_r = cell(nblock,1);
    c_Lon_r = cell(nblock,1);
    c_Lat_c = cell(nblock,1);
    c_Lon_c = cell(nblock,1);
    c_VCD = cell(nblock,nv);
    c_VCDe = cell(nblock,1);
    c_pixel_west = cell(nblock,1);
    c_pixel_east = cell(nblock,1);
    c_pixel_south = cell(nblock,1);
    c_pixel_north = cell(nblock,1);
    % il2 = 100;plot(Lon_c(il2),Lat_c(il2),'o',...
    %     pixel_west(il2),pixel_south(il2),'s',...
    %     pixel_west(il2),pixel_north(il2),'^',...
    %     pixel_east(il2),pixel_south(il2),'*',...
    %     pixel_east(il2),pixel_north(il2),'v',...
    %     Lon_r(il2,[1:4 1]),Lat_r(il2,[1:4 1]))
    nl2_block = zeros(nblock,1);
    for iblock = 1:nblock
        c_B{iblock} = single(c_B{iblock})*0;
        c_D{iblock} = c_D{iblock}*0;
        c_xgrid{iblock} = c_xmesh{iblock}(1,:);
        c_ygrid{iblock} = c_ymesh{iblock}(:,1);
        in = pixel_east >= min(c_xgrid{iblock}) & ...
            pixel_west <= max(c_xgrid{iblock}) & ...
            pixel_north >= min(c_ygrid{iblock}) & ...
            pixel_south <= max(c_ygrid{iblock});
        c_Lat_r{iblock} = Lat_r(in,:);
        c_Lon_r{iblock} = Lon_r(in,:);
        c_Lat_c{iblock} = Lat_c(in);
        c_Lon_c{iblock} = Lon_c(in);
        for iv = 1:nv
            c_VCD{iblock,iv} = VCD(in,iv);
        end
        c_VCDe{iblock} = VCDe(in);
        c_pixel_west{iblock} = pixel_west(in);
        c_pixel_east{iblock} = pixel_east(in);
        c_pixel_south{iblock} = pixel_south(in);
        c_pixel_north{iblock} = pixel_north(in);
        nl2_block(iblock) = sum(in);
        disp([num2str(nl2_block(iblock)),' pixels to be regridded in block ',num2str(iblock)])
        %     clf
        %     plot(c_xmesh{iblock}(:),c_ymesh{iblock}(:),'.k')
        %     hold on
        %     patch(c_Lon_r{iblock}',c_Lat_r{iblock}',c_VCD{iblock})
    end
    %
    parfor iblock = 1:nblock
        if nl2_block(iblock) == 0
            disp(['block ',num2str(iblock),' has no data'])
            c_A{iblock} = zeros(size(c_B{iblock},1),size(c_B{iblock},2),nv);
            continue
        end
        b_Lat_r = c_Lat_r{iblock};
        b_Lon_r = c_Lon_r{iblock};
        b_Lat_c = c_Lat_c{iblock};
        b_Lon_c = c_Lon_c{iblock};
        %         b_VCD = c_VCD(iblock,:);
        b_VCDe = c_VCDe{iblock};
        %         b_A = c_A(iblock,:);
        b_B = c_B{iblock};
        b_A = zeros(size(b_B,1),size(b_B,2),nv);
        b_D = c_D{iblock};
        b_nl2 = nl2_block(iblock);
        b_xgrid = c_xgrid{iblock};
        b_ygrid = c_ygrid{iblock};
        b_xmesh = c_xmesh{iblock};
        b_ymesh = c_ymesh{iblock};
        b_pixel_west = c_pixel_west{iblock};
        b_pixel_east = c_pixel_east{iblock};
        b_pixel_south = c_pixel_south{iblock};
        b_pixel_north = c_pixel_north{iblock};
        for iL2 = 1:b_nl2
            lat_r = b_Lat_r(iL2,:);
            lon_r = b_Lon_r(iL2,:);
            lat_c = b_Lat_c(iL2);
            lon_c = b_Lon_c(iL2);
            vcd_unc = b_VCDe(iL2);
            A = polyarea([lon_r(:);lon_r(1)],[lat_r(:);lat_r(1)]);
            
            lon_index = b_xgrid >= b_pixel_west(iL2) & b_xgrid <= b_pixel_east(iL2);
            lat_index = b_ygrid >= b_pixel_south(iL2) & b_ygrid <= b_pixel_north(iL2);
            
            lon_mesh = b_xmesh(lat_index,lon_index);
            lat_mesh = b_ymesh(lat_index,lon_index);
            
            SG = F_2D_SG_affine(lon_mesh,lat_mesh,lon_r,lat_r,lon_c,lat_c,...
                m,n,1,1,0,0);
            b_B(lat_index,lon_index) = b_B(lat_index,lon_index) + SG/A/vcd_unc;
            b_D(lat_index,lon_index) = b_D(lat_index,lon_index)+SG;
            for iv = 1:nv
                b_VCD = c_VCD{iblock,iv};
                vcd = b_VCD(iL2);
                
                b_A(lat_index,lon_index,iv) = b_A(lat_index,lon_index,iv) + SG/A/vcd_unc*vcd;
            end
            
        end
        disp(['block ',num2str(iblock),' has ',num2str(b_nl2),' pixels, finished on ',datestr(now)])
        c_A{iblock} = b_A;
        c_B{iblock} = b_B;
        c_D{iblock} = b_D;
    end
    c_a = cell(nblock,nv);
    for iv = 1:nv
        for iblock = 1:nblock
            c_a{iblock,iv} = squeeze(c_A{iblock}(:,:,iv));
        end
    end
    if if_old_vcd
        Sum_Above = single(cell2mat(reshape(c_A,[nblock_row,nblock_col])));
    else
        for iv = 1:nv
            Sum_Above.(vcdname{iv}) = single(cell2mat(reshape(c_a(:,iv),[nblock_row,nblock_col])));
        end
    end
    Sum_Below = cell2mat(reshape(c_B,[nblock_row,nblock_col]));
    D = cell2mat(reshape(c_D,[nblock_row,nblock_col]));
    output_regrid.nblock = nblock;
else
    %error('serial is not supported yet!')
    Sum_Above = zeros(nrows,ncols,nv,'single');
    Sum_Below = zeros(nrows,ncols,'single');
    D = zeros(nrows,ncols,'single');
    
    count = 1;
    for iL2 = 1:nL2
        lat_r = Lat_r(iL2,:);
        lon_r = Lon_r(iL2,:);
        lat_c = Lat_c(iL2);
        lon_c = Lon_c(iL2);
        
        vcd_unc = VCDe(iL2);
        A = polyarea([lon_r(:);lon_r(1)],[lat_r(:);lat_r(1)]);
        
        lon_index = xgrid >= pixel_west(iL2) & xgrid <= pixel_east(iL2);
        lat_index = ygrid >= pixel_south(iL2) & ygrid <= pixel_north(iL2);
        
        lon_mesh = Lon_mesh(lat_index,lon_index);
        lat_mesh = Lat_mesh(lat_index,lon_index);
        
        SG = F_2D_SG_affine(lon_mesh,lat_mesh,lon_r,lat_r,lon_c,lat_c,...
            m,n,1,1,0,0);
        for iv = 1:nv
            vcd = VCD(iL2,iv);
            Sum_Above(lat_index,lon_index,iv) = Sum_Above(lat_index,lon_index,iv) + SG/A/vcd_unc*vcd;
        end
        Sum_Below(lat_index,lon_index) = Sum_Below(lat_index,lon_index) + SG/A/vcd_unc;
        D(lat_index,lon_index) = D(lat_index,lon_index)+SG;
        
        if iL2 == count*round(nL2/10)
            disp([num2str(count*10),' % finished'])
            count = count+1;
        end
    end
    if if_old_vcd
        Sum_Above = squeeze(Sum_Above);
    else
        tmp = Sum_Above;
        Sum_Above = [];
        for iv = 1:nv
            Sum_Above.(vcdname{iv}) = squeeze(tmp(:,:,iv));
        end
    end
end
output_regrid.A = Sum_Above;
output_regrid.B = Sum_Below;
if if_old_vcd
    output_regrid.C = Sum_Above./Sum_Below;
end
output_regrid.D = single(D);

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

tform = fitgeotrans(fixedPoints,vList,'projective');

xym1 = [lon_mesh(:)-lon_c-lon_offset, lat_mesh(:)-lat_c-lat_offset];
xym2 = transformPointsInverse(tform,xym1);

FWHMy = FWHMy*inflatey;
FWHMx = FWHMx*inflatex;

wx = FWHMx/2/(log(2))^(1/m);
wy = FWHMy/2/(log(2))^(1/n);

SG = exp(-(abs(xym2(:,1))/wx).^m-(abs(xym2(:,2))/wy).^n);
SG = reshape(SG(:),size(lon_mesh,1),size(lon_mesh,2));

