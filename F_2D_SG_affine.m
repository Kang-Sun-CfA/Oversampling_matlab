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