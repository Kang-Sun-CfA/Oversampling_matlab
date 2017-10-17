clc;
clear
clon = -1.182436849000000e+02;
clat = 34.052234200000000;
if ispc
%%
function_dir = 'c:\users\Kang Sun\Documents\gitHub\PU_KS_share\';
% file containing the pixel size lookup table
pixel_size_file = '.\daysss.mat';
cd(function_dir)
addpath('C:\Users\Kang Sun\Documents\GitHub\Oversampling_matlab\')

%% find out IASI pixel size, in km
pixel_shape = load(pixel_size_file);
u_km = nan(120,1);
v_km = nan(120,1);
t_km = nan(120,1);
pixel_left = nan(120,1);
pixel_down = nan(120,1);
for i = 1:120
    [u,v,t] = F_define_IASI_pixel(0,i,...
        pixel_shape.uuu4,pixel_shape.vvv4,pixel_shape.ttt4);
    u_km(i) = 111*u;
    v_km(i) = 111*v;
    t_km(i) = t;
    [X, pixel_left(i), pixel_down(i)] =...
        F_construct_ellipse([0; 0],v_km(i),u_km(i),t_km(i),50,0);   
end

%%
inp = [];
inp.res = 1;
inp.max_x = 133;
inp.max_y = 100;
inp.clon = clon;
inp.clat = clat;

inp.u_km = u_km;
inp.v_km = v_km;
inp.t_km = t_km;
inp.pixel_left = pixel_left;
inp.pixel_down = pixel_down;
run_year = 2013:2015;
outp_b = cell(length(run_year),12);
for iyear = 1:length(run_year)
L2g_fn = ['CONUS_',num2str(run_year(iyear)),'.mat'];
load(['D:\Research_CfA\IASIb\L2g\',L2g_fn],'inp_subset','output_subset')

end_date_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    if iyear == 2004 || iyear == 2008 || iyear == 2016
        end_date_month = [31 29 31 30 31 30 31 31 30 31 30 31];
    end
    for imonth = 1:12
        inp.Startdate = [run_year(iyear) imonth 1];
        inp.Enddate = [run_year(iyear) imonth end_date_month(imonth)];
outp_b{iyear,imonth} = F_regrid_IASI_km(inp,output_subset);
    end
end
%%
inp = [];
inp.res = 1;
inp.max_x = 133;
inp.max_y = 100;
inp.clon = clon;
inp.clat = clat;

inp.u_km = u_km;
inp.v_km = v_km;
inp.t_km = t_km;
inp.pixel_left = pixel_left;
inp.pixel_down = pixel_down;
run_year = 2008:2015;
outp = cell(length(run_year),12);
for iyear = 1:length(run_year)
L2g_fn = ['CONUS_',num2str(run_year(iyear)),'.mat'];
load(['D:\Research_CfA\IASI\L2g\',L2g_fn],'inp_subset','output_subset')

end_date_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    if iyear == 2004 || iyear == 2008 || iyear == 2016
        end_date_month = [31 29 31 30 31 30 31 31 30 31 30 31];
    end
    for imonth = 1:12
        inp.Startdate = [run_year(iyear) imonth 1];
        inp.Enddate = [run_year(iyear) imonth end_date_month(imonth)];
outp{iyear,imonth} = F_regrid_IASI_km(inp,output_subset);
    end
end
%%
load('D:\Research_CfA\IASI\car_counties.mat')
load('D:\Research_CfA\IASI\cow_counties.mat')
%% select sub counties
casubcounty = shaperead('D:\GIS data\California County Shape Files\Sub County\CaSubCounty.shp');
plot_county = struct;
count = 0;
for i = 1:length(casubcounty)
    if casubcounty(i).BoundingBox(1,2) <= outp{1,1}.max_lat
        count = count+1;
        [x, y] = mfwdtran(outp{1,1}.mstruct,casubcounty(i).Y,casubcounty(i).X);
        plot_county(count).x = x;
        plot_county(count).y = y;
        plot_county(count).name = casubcounty(i).NAME;
        [xc, yc] = mfwdtran(outp{1,1}.mstruct,...
            str2double(casubcounty(i).INTPTLAT),...
            str2double(casubcounty(i).INTPTLON));
        plot_county(count).yc = yc;
        plot_county(count).xc = xc;

    end
end
plot_subcounty = plot_county;
%%
cacounty = shaperead('D:\GIS data\California County Shape Files\County\CaliforniaCounty.shp');
plot_county = struct;
count = 0;
for i = 1:length(cacounty)
    if cacounty(i).BoundingBox(1,2) <= outp{1,1}.max_lat
        count = count+1;
        [x, y] = mfwdtran(outp{1,1}.mstruct,cacounty(i).Y,cacounty(i).X);
        plot_county(count).x = x;
        plot_county(count).y = y;
        plot_county(count).name = cacounty(i).NAME;
        [xc, yc] = mfwdtran(outp{1,1}.mstruct,...
            str2double(cacounty(i).INTPTLAT),...
            str2double(cacounty(i).INTPTLON));
        plot_county(count).yc = yc;
        plot_county(count).xc = xc;

    end
end
%%
tmpa = zeros(size(outp{1,1}));
tmpb = tmpa;
for iyear = 1:length(run_year)
    for imonth = 1:12
        tmpa = tmpa+outp{iyear,imonth}.A;
        tmpb = tmpb+outp{iyear,imonth}.B;
    end
end
%
Clim = [15.2 16.2];% min/max plotted nh3 column
Ytick = [2 3 5 10 15];
figfn = 'C:\Users\Kang Sun\Dropbox\Code\SAO\NH3_East_Colorado_JJA_2008_2015.png';
clabel = 'NH_3 column [10^{15} molec cm^{-2}]';
clc
opengl software
plotmat = log10(double(tmpa./tmpb));
plotmat(plotmat < Clim(1)) = Clim(1);
close all
figure('color','w','unit','inch','position',[15 0 10 10/inp.max_x*inp.max_y])
axes('position',[0.1 0.1 0.8 0.8])
h = pcolor(outp{1,1}.xgrid,outp{1,1}.ygrid,...
    plotmat);
caxis(Clim)

hc = colorbar;
set(hc,'position',[0.92 0.2 0.02 0.6])
sbxlim = get(hc,'ylim');
sbxtick = interp1(Clim,sbxlim,log10(Ytick*1e15));
set(hc,'ytick',sbxtick,'yticklabel',Ytick)
set(get(hc,'ylabel'),'string',clabel)

set(h,'edgecolor','none');
hold on
for i = 1:length(plot_subcounty)
    
    plot(plot_subcounty(i).x,plot_subcounty(i).y,'--k','linewidth',0.2)
    if plot_subcounty(i).xc > -inp.max_x && plot_subcounty(i).xc < inp.max_x ...
            && plot_subcounty(i).yc > -inp.max_y && plot_subcounty(i).yc < inp.max_y
%     text(plot_county(i).xc,plot_county(i).yc,plot_county(i).name,...
%         'fontsize',7,'HorizontalAlignment','center')
    end
end
for i = 1:length(plot_county)
    
    plot(plot_county(i).x,plot_county(i).y,'k','linewidth',1.)
    if plot_county(i).xc > -inp.max_x && plot_county(i).xc < inp.max_x ...
            && plot_county(i).yc > -inp.max_y && plot_county(i).yc < inp.max_y
    text(plot_county(i).xc,plot_county(i).yc,plot_county(i).name,...
        'fontsize',14,'HorizontalAlignment','center')
    end
end

% plot(carx,cary,'--y',cowx,cowy,'--w','linewidth',2)
hold off

else
addpath('/home/kangsun/OMI/Oversampling_matlab/')
inp = [];

inp.res = 1;
inp.max_x = 133;
inp.max_y = 100;
inp.clon = clon;
inp.clat = clat;
inp.MaxCF = 0.3;
inp.MaxSZA = 60;
inp.vcdname = 'colno2';
inp.vcderrorname = 'colno2error';    
run_year = 2008:2015;
outp_no2 = cell(length(run_year),12);
for iyear = 1:length(run_year)
L2g_fn = ['CONUS_',num2str(run_year(iyear)),'.mat'];
load(['/data/tempo1/Shared/kangsun/OMNO2/L2g/',L2g_fn],'inp_subset','output_subset')

end_date_month = [31 28 31 30 31 30 31 31 30 31 30 31];
    if iyear == 2004 || iyear == 2008 || iyear == 2016
        end_date_month = [31 29 31 30 31 30 31 31 30 31 30 31];
    end
    for imonth = 1:12
        inp.Startdate = [run_year(iyear) imonth 1];
        inp.Enddate = [run_year(iyear) imonth end_date_month(imonth)];
outp_no2{iyear,imonth} = F_regrid_OMI_km(inp,output_subset);
    end
end
save('/data/tempo1/Shared/kangsun/OMNO2/outp_no2.mat','outp_no2')
% load('/data/tempo1/Shared/kangsun/IASI/car_counties.mat')
% load('/data/tempo1/Shared/kangsun/IASI/cow_counties.mat')
end
%%
if ispc
    load('d:\Research_CfA\OMNO2\L3\outp_no2.mat')
    car_mean_no2 = nan(length(run_year),12);
cow_mean_no2 = nan(length(run_year),12);
in_car = inpolygon(outp{1,1}.xmesh,outp{1,1}.ymesh,carx,cary);
in_cow = inpolygon(outp{1,1}.xmesh,outp{1,1}.ymesh,cowx,cowy);

for iyear = 1:length(run_year)
    for imonth = 1:12
        car_mean_no2(iyear,imonth) = nanmean(outp_no2{iyear,imonth}.C(in_car));
        cow_mean_no2(iyear,imonth) = nanmean(outp_no2{iyear,imonth}.C(in_cow));
    end
end
end
%%

tmpa = zeros(size(outp_no2{1,1}));
tmpb = tmpa;
for iyear = 1:length(run_year)
    for imonth = 1:12
        tmpa = tmpa+outp_no2{iyear,imonth}.A;
        tmpb = tmpb+outp_no2{iyear,imonth}.B;
    end
end

Clim = [15 16];% min/max plotted nh3 column
Ytick = [1 2 3 5 10 15];
figfn = 'C:\Users\Kang Sun\Dropbox\Code\SAO\NH3_East_Colorado_JJA_2008_2015.png';
clabel = 'NO_2 column [10^{15} molec cm^{-2}]';
clc
opengl software
plotmat = log10(double(tmpa./tmpb));
plotmat(plotmat < Clim(1)) = Clim(1);
close all
figure('color','w','unit','inch','position',[15 0 10 10/inp.max_x*inp.max_y])
axes('position',[0.1 0.1 0.8 0.8])
h = pcolor(outp_no2{1,1}.xgrid,outp_no2{1,1}.ygrid,...
    plotmat);
caxis(Clim)

hc = colorbar;
set(hc,'position',[0.92 0.2 0.02 0.6])
sbxlim = get(hc,'ylim');
sbxtick = interp1(Clim,sbxlim,log10(Ytick*1e15));
set(hc,'ytick',sbxtick,'yticklabel',Ytick)
set(get(hc,'ylabel'),'string',clabel)

set(h,'edgecolor','none');
hold on
for i = 1:length(plot_subcounty)
    
    plot(plot_subcounty(i).x,plot_subcounty(i).y,'-.k','linewidth',0.2)
    if plot_subcounty(i).xc > -inp.max_x && plot_subcounty(i).xc < inp.max_x ...
            && plot_subcounty(i).yc > -inp.max_y && plot_subcounty(i).yc < inp.max_y
%     text(plot_county(i).xc,plot_county(i).yc,plot_county(i).name,...
%         'fontsize',7,'HorizontalAlignment','center')
    end
end
for i = 1:length(plot_county)
    
    plot(plot_county(i).x,plot_county(i).y,'k','linewidth',1.)
    if plot_county(i).xc > -inp.max_x && plot_county(i).xc < inp.max_x ...
            && plot_county(i).yc > -inp.max_y && plot_county(i).yc < inp.max_y
    text(plot_county(i).xc,plot_county(i).yc,plot_county(i).name,...
        'fontsize',14,'HorizontalAlignment','center')
    end
end
% for i = 1:length(plot_county)
%     
%     plot(plot_county(i).x,plot_county(i).y,'k')
%     if plot_county(i).xc > -inp.max_x && plot_county(i).xc < inp.max_x ...
%             && plot_county(i).yc > -inp.max_y && plot_county(i).yc < inp.max_y
%     text(plot_county(i).xc,plot_county(i).yc,plot_county(i).name,...
%         'fontsize',7,'HorizontalAlignment','center')
%     end
% end
% plot(carx,cary,'--y',cowx,cowy,'--w','linewidth',2)
hold off

%%
car_mean = nan(length(run_year),12);
cow_mean = nan(length(run_year),12);
in_car = inpolygon(outp{1,1}.xmesh,outp{1,1}.ymesh,carx,cary);
in_cow = inpolygon(outp{1,1}.xmesh,outp{1,1}.ymesh,cowx,cowy);

for iyear = 1:length(run_year)
    for imonth = 1:12
        car_mean(iyear,imonth) = nanmean(outp_b{iyear,imonth}.C(in_car));
        cow_mean(iyear,imonth) = nanmean(outp_b{iyear,imonth}.C(in_cow));
    end
end
%%
car_mean = nan(length(run_year),12);
cow_mean = nan(length(run_year),12);
in_car = inpolygon(outp{1,1}.xmesh,outp{1,1}.ymesh,carx,cary);
in_cow = inpolygon(outp{1,1}.xmesh,outp{1,1}.ymesh,cowx,cowy);

for iyear = 1:length(run_year)
    for imonth = 1:12
        car_mean(iyear,imonth) = nanmean(outp{iyear,imonth}.C(in_car));
        cow_mean(iyear,imonth) = nanmean(outp{iyear,imonth}.C(in_cow));
    end
end
%%
close;figure
subplot(2,1,1)
plot(1:12*length(run_year),reshape(car_mean',[12*length(run_year),1]),...
    1:12*length(run_year),reshape(car_mean_no2',[12*length(run_year),1]))
ylim([0 2e16])
subplot(2,1,2)
plot(1:12*length(run_year),reshape(cow_mean',[12*length(run_year),1]),...
    1:12*length(run_year),reshape(cow_mean_no2',[12*length(run_year),1]))
ylim([0 4e16])