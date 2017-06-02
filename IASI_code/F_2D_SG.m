function SG = F_2D_SG(xmesh,ymesh,x0,y0,FWHMx,FWHMy,m,n,Angle)
% two dimensional super gaussian, note x with m, y with n, different from
% the OMI pixel shape paper

% written by Kang Sun on 2017/05/06

% angle in rad

wx = FWHMx/2/(log(2))^(1/m);
wy = FWHMy/2/(log(2))^(1/n);
if ~exist('Angle','var')
SG = exp(-((xmesh-x0)/wx).^m-((ymesh-y0)/wy).^n);
return;
end
if length(Angle) > 1
    xym1 = [xmesh(:)'-x0; ymesh(:)'-y0];
xym2 = Angle*xym1;
SG = exp(-((xym2(1,:))/wx).^m-((xym2(2,:))/wy).^n);
SG = reshape(SG(:),size(xmesh,1),size(xmesh,2));
else
   rotation_matrix = [cos(Angle), -sin(Angle);
    sin(Angle),  cos(Angle)]; 
xym1 = [xmesh(:)'-x0; ymesh(:)'-y0];
xym2 = rotation_matrix*xym1;
SG = exp(-((xym2(1,:))/wx).^m-((xym2(2,:))/wy).^n);
SG = reshape(SG(:),size(xmesh,1),size(xmesh,2));
end