function outp_interp_era = F_interp_era(inp_interp_era)
% interpolate ERA data to lat lon and time
% written on 2018/01/15

ERA_download_dir = inp_interp_era.ERA_download_dir;
u_all = [];
v_all = [];
T2m_all = [];
nn = length(inp_interp_era.hour);

tt = nan(nn,1);
for ihour = 1:nn
    era_datenum = inp_interp_era.datenum+inp_interp_era.hour(ihour)/24;
    tt(ihour) = era_datenum;
    tmp = datevec(era_datenum);
    era_data = load([ERA_download_dir,'/',num2str(tmp(1)),'/',num2str(tmp(2),'%02d'),'/',...
        'sfc_',num2str(tmp(3),'%02d'),'_',num2str(tmp(4),'%02d'),'.mat']);
    u_all = cat(3,u_all,era_data.u);
    v_all = cat(3,v_all,era_data.v);
    T2m_all = cat(3,T2m_all,era_data.T2m);
end

llo = double(era_data.lon);
lla = double(era_data.lat);
lonv = double(inp_interp_era.lon);
lonv(lonv < 0) = 360+lonv(lonv < 0);
outp_interp_era.u = interpn(lla,llo,tt,double(u_all),double(inp_interp_era.lat),lonv,inp_interp_era.utc);
outp_interp_era.v = interpn(lla,llo,tt,double(v_all),double(inp_interp_era.lat),lonv,inp_interp_era.utc);
outp_interp_era.T2m = interpn(lla,llo,tt,double(T2m_all),double(inp_interp_era.lat),lonv,inp_interp_era.utc);