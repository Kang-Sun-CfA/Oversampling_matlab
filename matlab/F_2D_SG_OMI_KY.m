function SG = F_2D_SG_OMI_KY(lon_meshr,lat_meshr,lon_mesh,lat_mesh,lon_r,lat_r,lon_c,lat_c,...
    m,n,inflatex,inflatey,lon_offset,lat_offset)
% super gaussian for quadrilateral fov. use Kai Yang's discretization.
% updated from F_2D_SG_affine.m by Kang Sun on 2018/07/17

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

xym1 = [[lon_mesh(:)-lon_c-lon_offset;lon_meshr(:)-lon_c-lon_offset],...
    [lat_mesh(:)-lat_c-lat_offset;lat_meshr(:)-lat_c-lat_offset]];
xym2 = transformPointsInverse(tform,xym1);

FWHMy = FWHMy*inflatey;
FWHMx = FWHMx*inflatex;

wx = FWHMx/2/(log(2))^(1/m);
wy = FWHMy/2/(log(2))^(1/n);

meshcn = length(lat_mesh(:));
SGc = exp(-(abs(xym2(1:meshcn,1))/wx).^m-(abs(xym2(1:meshcn,2))/wy).^n);
SGc = reshape(SGc(:),size(lon_mesh,1),size(lon_mesh,2));

SGr = exp(-(abs(xym2(meshcn+1:end,1))/wx).^m-(abs(xym2(meshcn+1:end,2))/wy).^n);
SGr = reshape(SGr(:),size(lon_meshr,1),size(lon_meshr,2));

SG = (2*SGc+conv2(SGr,ones(2,2),'valid'))/6;
% Shear_cos = dot(xvector,yvector)/(FWHMx*FWHMy);
% Shear_sin = sqrt(1-Shear_cos^2);
% Shear = -Shear_cos/Shear_sin;
% 
% FWHMy = FWHMy*Shear_sin*inflatey;
% FWHMx = FWHMx*inflatex;
% 
% Angle = -atan((rightpoint(2)-leftpoint(2))/(rightpoint(1)-leftpoint(1)));
% rotation_matrix = [cos(Angle), -sin(Angle);
%     sin(Angle),  cos(Angle)];
% Shear_matrix = [1 Shear;0 1];
% 
% wx = FWHMx/2/(log(2))^(1/m);
% wy = FWHMy/2/(log(2))^(1/n);
% 
% xym1 = [lon_mesh(:)'-lon_c-lon_offset; lat_mesh(:)'-lat_c-lat_offset];
% xym2 = rotation_matrix*Shear_matrix*xym1;
% SG = exp(-(abs(xym2(1,:))/wx).^m-(abs(xym2(2,:))/wy).^n);
% SG = reshape(SG(:),size(lon_mesh,1),size(lon_mesh,2));