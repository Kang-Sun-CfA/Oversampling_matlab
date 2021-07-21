function outp_interp_narr = F_interp_narr(inp_interp_narr)
% matlab function to interpolate narr met field to given satellite pixel
% locations (lat_interp_hour, lon_interp_hour) and central locations (x_c,
% y_c, which should already been transformed to km using narr projection)

% outputs are surface and pbl-integrated meteorology at satellite locations 
% and central location

% written by Kang Sun on 2017/11/09

narr_data_3d = inp_interp_narr.narr_data_3d;
narr_data_sfc = inp_interp_narr.narr_data_sfc;
lon_interp_hour = inp_interp_narr.lon_interp_hour;
lat_interp_hour = inp_interp_narr.lat_interp_hour;
x_c = inp_interp_narr.x_c;
y_c = inp_interp_narr.y_c;
mstruct_narr = inp_interp_narr.mstruct_narr;
max_x = inp_interp_narr.max_x;
max_y = inp_interp_narr.max_y;

intx = narr_data_3d.x > x_c - max_x-500 & narr_data_3d.x < x_c + max_x+500;
inty = narr_data_3d.y > y_c - max_y-500 & narr_data_3d.y < y_c + max_y+500;

x = narr_data_3d.x(intx);
y = narr_data_3d.y(inty);

u = narr_data_3d.u(:,inty,intx);
v = narr_data_3d.v(:,inty,intx);
T = narr_data_3d.T(:,inty,intx);

P_surf = narr_data_sfc.P_surf(inty,intx);
T_surf = narr_data_sfc.T_surf(inty,intx);
PBLH = narr_data_sfc.PBLH(inty,intx);

[x_sat, y_sat] = mfwdtran(mstruct_narr,lat_interp_hour,lon_interp_hour);

% pressure grid of narr data, in Pa
P = [550 600 650 700 725 750 775 800 825 850 875 900 925 950 975 1000]'*100;
% maximum pbl thickness in Pa
if ~isfield(inp_interp_narr,'P_pblmax')
P_pblmax = 200*100;
else
    P_pblmax = inp_interp_narr.P_pblmax*100;% from hPa to Pa
end
%% work out the pbl pressure weighted wind
% % find the surface layer pressure index, not worth it
% sfc_idx = zeros(size(narr_data_sfc.P_surf));
% for i = length(P):-1:1
%     mask = narr_data_sfc.P_surf-P(i) > 0 & sfc_idx == 0;
%     sfc_idx(mask) = i;
%     if sum(mask(:)) == 0;
%         break;
%     end
% end
nx = sum(intx);
ny = sum(inty);
u_pbl = nan(ny,nx,'single');
u_sfc = nan(ny,nx,'single');
v_pbl = nan(ny,nx,'single');
v_sfc = nan(ny,nx,'single');
T_pbl = nan(ny,nx,'single');

% pbl top pressure
P_pbl = nan(ny,nx,'single');
for ix = 1:nx
    for iy = 1:ny
        scaleH = 287/9.8*(T_surf(iy,ix)-10);
        P_pbl(iy,ix) = max([P_surf(iy,ix)*exp(-PBLH(iy,ix)/scaleH) P_surf(iy,ix)-P_pblmax]);
        
        int = P >= P_pbl(iy,ix) & P <= P_surf(iy,ix) & P >= P_surf(iy,ix)-P_pblmax;
        % sometimes the pbl is so thin...
        if sum(int) == 0
            int = find(P <= P_surf(iy,ix),1,'last');
        end
%         sum(int)
        localu = squeeze(u(int,iy,ix));
        localv = squeeze(v(int,iy,ix));
        u_pbl(iy,ix) = sum(localu.*P(int)/sum(P(int)));
        v_pbl(iy,ix) = sum(localv.*P(int)/sum(P(int)));
        
        u_sfc(iy,ix) = localu(end);
        v_sfc(iy,ix) = localv(end);
        
        T_pbl(iy,ix) = sum(squeeze(T(int,iy,ix)).*P(int)/sum(P(int)));
    end
end
%%
outp_interp_narr = [];
outp_interp_narr.u_sat_pbl = interp2(x,y,u_pbl,x_sat,y_sat);
outp_interp_narr.v_sat_pbl = interp2(x,y,v_pbl,x_sat,y_sat);
outp_interp_narr.T_sat_pbl = interp2(x,y,T_pbl,x_sat,y_sat);

nz = length(outp_interp_narr.u_sat_pbl);

outp_interp_narr.u_c_pbl = ones(nz,1,'single')*interp2(x,y,u_pbl,x_c,y_c);
outp_interp_narr.v_c_pbl = ones(nz,1,'single')*interp2(x,y,v_pbl,x_c,y_c);
outp_interp_narr.T_c_pbl = ones(nz,1,'single')*interp2(x,y,T_pbl,x_c,y_c);

outp_interp_narr.u_sat_sfc = interp2(x,y,u_sfc,x_sat,y_sat);
outp_interp_narr.v_sat_sfc = interp2(x,y,v_sfc,x_sat,y_sat);
outp_interp_narr.T_sat_sfc = interp2(x,y,T_surf,x_sat,y_sat);

outp_interp_narr.u_c_sfc = ones(nz,1,'single')*interp2(x,y,u_sfc,x_c,y_c);
outp_interp_narr.v_c_sfc = ones(nz,1,'single')*interp2(x,y,v_sfc,x_c,y_c);
outp_interp_narr.T_c_sfc = ones(nz,1,'single')*interp2(x,y,T_surf,x_c,y_c);
% %%
% close all
% hold on
% h = pcolor(x,y,double(P_pbl));
% % h = pcolor(narr_data_sfc.x,narr_data_sfc.y,double(narr_data_sfc.P_surf));
% % h = pcolor(narr_data_sfc.x,narr_data_sfc.y,squeeze(double(narr_data_3d.q(16,:,:))));
% % h = pcolor(narr_data_sfc.x,narr_data_sfc.y,double(narr_data_sfc.PBLH));
% colormap('gray')
% colorbar
% set(h,'edgecolor','none')
% plot(x_c,y_c,'r*',x_sat,y_sat,'.')
% quiver(x_sat,y_sat,u_sat_pbl,v_sat_pbl,0)
% quiver(x,y,u_pbl,v_pbl,.0)
% quiver(x_c,y_c,u_c_pbl,v_c_pbl,0)
% %%
% quiver(x_sat,y_sat,u_sat_pbl-u_c_pbl,v_sat_pbl-v_c_pbl)
% hold on
% quiver(x_sat,y_sat,u_sat_pbl,v_sat_pbl)
