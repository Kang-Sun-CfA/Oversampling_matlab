testdata = [1,34.928440,35.046413,35.157444,35.039295,35.045200,-70.678810,-70.721886,-70.306282,-70.263763,-70.486397,43.510700,0.16052000,1.6698500,1.4008900e+15,3.4824599e+15];

Lon_r = testdata(7:10);
Lat_r = testdata(2:5);
Lon_c = testdata(11);
Lat_c = testdata(6);

% % ellipse footprint
% [h,xx,yy]=ellipse(0.2,0.08,pi/6,Lon_c,Lat_c,'b',15);
%
% Lon_r = xx(end-1:-1:1);Lat_r = yy(end-1:-1:1);

VCD = testdata(15);VCD_Unc = testdata(16);

Lon_left = floor(min(Lon_r))-1;
Lon_right = ceil(max(Lon_r))+1;
Lat_low = floor(min(Lat_r))-1;
Lat_up = ceil(max(Lat_r))+1;

Lon_left = floor(min(Lon_r));
Lon_right = ceil(max(Lon_r));
Lat_low = floor(min(Lat_r));
Lat_up = ceil(max(Lat_r));

nrows = (Lat_up-Lat_low)/Res;
ncols = (Lon_right-Lon_left)/Res;



pixel.nv = length(Lon_r);
pixel.vList = [(Lon_r(:)-Lon_left)/Res,(Lat_r(:)-Lat_low)/Res];

[pout,npout] = F_HcakeCut(pixel);
close all;
subplot(1,2,1)
hold on
for i = 1:npout
    F_plot_polygon(pout(i));
end
[pout,npout] = F_VcakeCut(pixel);
axis equal
subplot(1,2,2)
hold on
for i = 1:npout
    F_plot_polygon(pout(i));
end
axis equal
% test clockwise
figure
hold on
plot(Lon_c,Lat_c,'ok')
for i = 1:length(Lon_r)
    plot(Lon_r(i),Lat_r(i),'*');
end
hold off
%%
Sub_Area = zeros(nrows,ncols);
Sum_Above = zeros(nrows,ncols);
Sum_Below = zeros(nrows,ncols);
Pixels_count = zeros(nrows,ncols);
Average = zeros(nrows,ncols);
close all
figure('unit','inch','color','w','position',[0 1 5 8])
ax1 = subplot(3,1,1);

% Get the area of the pixel
A = F_polyarea([Lon_r(:),Lat_r(:)]);

% Used for the next step
id_all = 0;
pixel_area = 0;

% Perform Horizontal cut first at the integer grid lines
[sub_pixels,n_sub_pixels] = F_HcakeCut( pixel );
CC = parula(n_sub_pixels);
hold on
for i = 1:n_sub_pixels
    h = F_plot_polygon(sub_pixels(i));
    set(h,'color',CC(i,:))
end
axis equal
axis off
title('Horizontal cut')
ax2 = subplot(3,1,2);
hold on
% Then perform Vertical cut for each sub pixel obtainted
% from the Horizontal cut at the integer grid lines
tic
for id_sub = 1: n_sub_pixels
    [final_pixels, n_final_pixels] = F_VcakeCut( sub_pixels(id_sub) );
    for id_final = 1: n_final_pixels
        h = F_plot_polygon(final_pixels(id_final));
        set(h,'color',CC(id_sub,:))
        id_all = id_all + 1;
        temp_area = F_polyarea(final_pixels(id_final))*Res^2;
        pixel_area = pixel_area + temp_area;
        %DO i = 1, final_pixels(id_final)%nv
        %  WRITE(*,'(A,F10.3,A,F10.3,A)') '{', final_pixels(id_final)%vList(i,1),',', &
        %                                      final_pixels(id_final)%vList(i,2),'},'
        %ENDDO
        row = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,2))) + 1;
        col = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,1))) + 1;
        %WRITE(*,*) 'id =', id_all, ', aera=',temp_area, ', row=', row, ',col=',col
        
        % Get the overlaped area between the pixel and each cell
        Sub_Area(row,col) = temp_area;
        
        % Sum weighted value and weights
        % Here use temp_area/A/VCD_Unc(p) as averaging weight, meaning that
        % averaging weight is assumed to be proportional to the ratio of the overlap area (temp_area) to the
        % pixel size (A) and inversely proportional to the error standard deviation (VCD_Unc(p)).
        % If you just want fraction of overlap area as averaging weight, use: temp_area/A
        % If you just want area weighted average, use: temp_area
        Sum_Above(row,col) = Sum_Above(row,col) + temp_area/A/VCD_Unc(1)*VCD(1);
        Sum_Below(row,col) = Sum_Below(row,col) + temp_area/A/VCD_Unc(1);
        Pixels_count(row,col) = Pixels_count(row,col) + 1;
    end
end
toc
axis equal
axis off
title('Vertical cut')
ax3 = subplot(3,1,3);
h = pcolor((1:ncols)-1,(1:nrows)-1,Sub_Area);
set(h,'edgecolor','w')
set(ax3,'xlim',get(ax2,'xlim'),'ylim',get(ax2,'ylim'),'box','off')
title('Calculate each sub polygon area')
axis off
% Check area consvertiveness
if( abs(A-pixel_area)/A >=0.05  )
    disp('------------------------------------------------------')
    disp('  -Area not conservative at pixel: ')%, p, A, pixel_area
    disp([num2str( Lon_o),' ', num2str(Lat_o)])
end
%%
% export_fig([plotdir,'ellipse_pixel.pdf'])
export_fig([plotdir,'omi_pixel.pdf'])
%% test 2D SG
testdata = [1,34.928440,35.046413,35.157444,35.039295,35.045200,-70.678810,-70.721886,-70.306282,-70.263763,-70.486397,43.510700,0.16052000,1.6698500,1.4008900e+15,3.4824599e+15];

Lon_r = testdata(7:10);
Lat_r = testdata(2:5);
Lon_c = testdata(11);
Lat_c = testdata(6);
Lon_left = floor(min(Lon_r));
Lon_right = ceil(max(Lon_r));
Lat_low = floor(min(Lat_r));
Lat_up = ceil(max(Lat_r));

nrows = (Lat_up-Lat_low)/Res;
ncols = (Lon_right-Lon_left)/Res;

pixel.nv = length(Lon_r);
pixel.vList = [(Lon_r(:)-Lon_left)/Res,(Lat_r(:)-Lat_low)/Res];
pixel.center = [(Lon_c-Lon_left)/Res,(Lat_c-Lat_low)/Res];
vList = pixel.vList;

leftpoint = mean(vList(1:2,:));
rightpoint = mean(vList(3:4,:));

uppoint = mean(vList(2:3,:));
lowpoint = mean(vList([1 4],:));

% x is the xtrack, different from the OMI pixel paper, which used y
FWHMx = sqrt((leftpoint(1)-rightpoint(1))^2+(leftpoint(2)-rightpoint(2))^2);
% y is the along track, different from the OMI pixel paper, which used x
FWHMy = sqrt((uppoint(1)-lowpoint(1))^2+(uppoint(2)-lowpoint(2))^2);

Angle = -atan((rightpoint(2)-leftpoint(2))/(rightpoint(1)-leftpoint(1)));
rotation_matrix = [cos(Angle), -sin(Angle);
    sin(Angle),  cos(Angle)];
xgrid = 1:ncols;ygrid = 1:nrows;
[xmesh,ymesh] = meshgrid(xgrid,ygrid);
%%
Xlim = [0 50];
Ylim = [40 65];
close all
figure('unit','inch','color','w','position',[0 1 5 8])
ax1 = subplot(3,1,1);
m = 500;n = 500;
% Angle = -pi/6;
SG = F_2D_SG(xmesh,ymesh,pixel.center(1),pixel.center(2),FWHMx,FWHMy,m,n,Angle);
% figure
h = pcolor(xgrid-.5,ygrid-.5,SG);
hold on
h = F_plot_polygon(pixel);set(h,'color','r','linewidth',2)
set(gca,'box','off','xlim',Xlim,'ylim',Ylim)
axis equal
axis off
colorbar
title(['m = ',num2str(m),', n = ',num2str(n)])

ax2 = subplot(3,1,2);
m = 2;n = 2;
% Angle = -pi/6;
SG = F_2D_SG(xmesh,ymesh,pixel.center(1),pixel.center(2),FWHMx,FWHMy,m,n,Angle);
% figure
h = pcolor(xgrid-.5,ygrid-.5,SG);
hold on
h = F_plot_polygon(pixel);set(h,'color','r','linewidth',2)
set(gca,'box','off','xlim',Xlim,'ylim',Ylim)
axis equal
axis off
caxis([0 1])
hc = colorbar;set(get(hc,'ylabel'),'string','Spatial response function');
title(['m = ',num2str(m),', n = ',num2str(n)])

ax3 = subplot(3,1,3);
m = 4;n = 2;
% Angle = -pi/6;
SG = F_2D_SG(xmesh,ymesh,pixel.center(1),pixel.center(2),FWHMx,FWHMy,m,n,Angle);
% figure
h = pcolor(xgrid-.5,ygrid-.5,SG);
hold on
h = F_plot_polygon(pixel);set(h,'color','r','linewidth',2)
set(gca,'box','off','xlim',Xlim,'ylim',Ylim)
axis equal
axis off
caxis([0 1])
colorbar
title(['m = ',num2str(m),', n = ',num2str(n)])
export_fig([plotdir,'spatial_response.pdf'])
%% test pixel inflation
Res = 0.05;
testdata = [1,34.928440,35.046413,35.157444,35.039295,35.045200,-70.678810,-70.721886,-70.306282,-70.263763,-70.486397,43.510700,0.16052000,1.6698500,1.4008900e+15,3.4824599e+15];

Lon_r = testdata(7:10);
Lat_r = testdata(2:5);
Lon_c = testdata(11);
Lat_c = testdata(6);
% inflation factor xtrack
inflationx = 1.5;
% inflation factor along track
inflationy = 2;

Lon_left = floor(min(Lon_r));
Lon_right = ceil(max(Lon_r));
Lat_low = floor(min(Lat_r));
Lat_up = ceil(max(Lat_r));

nrows = ceil((Lat_up-Lat_low)/Res);
ncols = ceil((Lon_right-Lon_left)/Res);

pixel.nv = length(Lon_r);
pixel.vList = [(Lon_r(:)-Lon_left)/Res,(Lat_r(:)-Lat_low)/Res];
pixel.center = [(Lon_c-Lon_left)/Res,(Lat_c-Lat_low)/Res];
vList = pixel.vList;

leftpoint = mean(vList(1:2,:));
rightpoint = mean(vList(3:4,:));

uppoint = mean(vList(2:3,:));
lowpoint = mean(vList([1 4],:));

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

% diamond_orth = [diamond_orth(1,:)+pixel.center(1);diamond_orth(2,:)+pixel.center(2)];
xleft = diamond_orth(1,1)*inflationx;
xright = diamond_orth(1,3)*inflationx;
ytop = diamond_orth(2,2)*inflationy;
ybottom = diamond_orth(2,4)*inflationy;

vList_orth = [xleft xleft xright xright;
    ybottom ytop ytop ybottom];

vList_inflate = [cos(-Angle), -sin(-Angle);
    sin(-Angle),  cos(-Angle)]*vList_orth;

vList_inflate = [vList_inflate(1,:)'+pixel.center(1),vList_inflate(2,:)'+pixel.center(2)];
plot(vList_inflate(:,1),vList_inflate(:,2),'o',vList(:,1),vList(:,2),'*')
axis equal
xgrid = 1:ncols;ygrid = 1:nrows;
[xmesh,ymesh] = meshgrid(xgrid,ygrid);
pixel.vList = vList_inflate;

Sub_Area = zeros(nrows,ncols);
Sub_SRF = zeros(nrows,ncols);
Sum_Above = zeros(nrows,ncols);
Sum_Below = zeros(nrows,ncols);
Pixels_count = zeros(nrows,ncols);
Average = zeros(nrows,ncols);


% Get the area of the pixel
A = F_polyarea([Lon_r(:),Lat_r(:)]);

% Used for the next step
id_all = 0;
pixel_area = 0;

% Perform Horizontal cut first at the integer grid lines
[sub_pixels,n_sub_pixels] = F_HcakeCut( pixel );
CC = parula(n_sub_pixels);

use_simplied_area = false;
% Then perform Vertical cut for each sub pixel obtainted
% from the Horizontal cut at the integer grid lines
tic
for id_sub = 1: n_sub_pixels
    [final_pixels, n_final_pixels] = F_VcakeCut( sub_pixels(id_sub) );
    for id_final = 1: n_final_pixels
        
        id_all = id_all + 1;
        if use_simplied_area
            temp_area = Res^2;
        else
            [ifsquare,edges] = F_if_square(final_pixels(id_final));
            if ifsquare
                temp_area = edges(1)*edges(2)*Res^2;
            else
                temp_area = F_polyarea(final_pixels(id_final))*Res^2;
            end
        end
        pixel_area = pixel_area + temp_area;
        %DO i = 1, final_pixels(id_final)%nv
        %  WRITE(*,'(A,F10.3,A,F10.3,A)') '{', final_pixels(id_final)%vList(i,1),',', &
        %                                      final_pixels(id_final)%vList(i,2),'},'
        %ENDDO
        row = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,2))) + 1;
        col = floor(min(final_pixels(id_final).vList(1:final_pixels(id_final).nv,1))) + 1;
        %WRITE(*,*) 'id =', id_all, ', aera=',temp_area, ', row=', row, ',col=',col
        
        % Get the overlaped area between the pixel and each cell
        Sub_Area(row,col) = temp_area;
        Sub_SRF(row,col) = ...
            F_2D_SG(col,row,pixel.center(1),pixel.center(2),FWHMx,FWHMy,m,n,rotation_matrix);
        % Sum weighted value and weights
        % Here use temp_area/A/VCD_Unc(p) as averaging weight, meaning that
        % averaging weight is assumed to be proportional to the ratio of the overlap area (temp_area) to the
        % pixel size (A) and inversely proportional to the error standard deviation (VCD_Unc(p)).
        % If you just want fraction of overlap area as averaging weight, use: temp_area/A
        % If you just want area weighted average, use: temp_area
        Sum_Above(row,col) = Sum_Above(row,col) + temp_area/A/VCD_Unc(1)*VCD(1);
        Sum_Below(row,col) = Sum_Below(row,col) + temp_area/A/VCD_Unc(1);
        Pixels_count(row,col) = Pixels_count(row,col) + 1;
    end
end
toc
xext = 5*0.02/Res;yext = 2*0.02/Res;
Xlim = [min(pixel.vList(:,1))-xext max(pixel.vList(:,1))+xext];
Ylim = [min(pixel.vList(:,2))-yext max(pixel.vList(:,2))+yext];
close all
figure('unit','inch','color','w','position',[0 1 5 8])
ax1 = subplot(3,1,1);
h = pcolor((1:ncols)-1,(1:nrows)-1,Sub_Area/Res^2);

hold on
h1 = F_plot_polygon([(Lon_r(:)-Lon_left)/Res,(Lat_r(:)-Lat_low)/Res]);
h2 = F_plot_polygon(pixel);
set(h,'edgecolor','w')
% title('Calculate each sub polygon area')
axis equal
set(gca,'xlim',Xlim,'ylim',Ylim,'box','off')
axis off
% caxis([0 1])
colorbar
title(['m = ',num2str(m),', n = ',num2str(n),', Res = ',num2str(Res),...
    char(10),'inflate OMI pixel by ',num2str(inflationx),'x',num2str(inflationy)])
text(Xlim(1),Ylim(1)-2*0.02/Res,'(a) sub polygon areas','fontsize',12)

ax2 = subplot(3,1,2);
h = pcolor((1:ncols)-1,(1:nrows)-1,Sub_SRF);
hold on
h1 = F_plot_polygon([(Lon_r(:)-Lon_left)/Res,(Lat_r(:)-Lat_low)/Res]);
h2 = F_plot_polygon(pixel);
set(h,'edgecolor','w')
axis equal
set(gca,'xlim',Xlim,'ylim',Ylim,'box','off')
axis off
caxis([0 1])
colorbar
text(Xlim(1),Ylim(1)-2*0.02/Res,'(b) spatial response function','fontsize',12)

ax3 = subplot(3,1,3);
h = pcolor((1:ncols)-1,(1:nrows)-1,Sub_SRF.*Sub_Area/Res^2);
hold on
h1 = F_plot_polygon([(Lon_r(:)-Lon_left)/Res,(Lat_r(:)-Lat_low)/Res]);
h2 = F_plot_polygon(pixel);
set(h,'edgecolor','w')
axis equal
set(gca,'xlim',Xlim,'ylim',Ylim,'box','off')
axis off
caxis([0 1])
colorbar
text(Xlim(1),Ylim(1)-2*0.02/Res,'(c) total weight','fontsize',12)
%%
export_fig([plotdir,'SRF_weight_inflate_',num2str(inflationx),'_',num2str(inflationy),...
    '_Res_',num2str(Res),'_simplearea_',num2str(use_simplied_area),'.pdf'])
%% test cakecut function
testdata = [1,34.928440,35.046413,35.157444,35.039295,35.045200,-70.678810,-70.721886,-70.306282,-70.263763,-70.486397,43.510700,0.16052000,1.6698500,1.4008900e+15,3.4824599e+15];

Res = 0.02;

Lon_r = testdata(7:10);
Lat_r = testdata(2:5);
Lon_c = testdata(11);
Lat_c = testdata(6);

VCD = testdata(end-1);
VCD_Unc = testdata(end);
Lon_left = floor(min(Lon_r));
Lon_right = ceil(max(Lon_r));
Lat_low = floor(min(Lat_r));
Lat_up = ceil(max(Lat_r));

nrows = (Lat_up-Lat_low)/Res;
ncols = (Lon_right-Lon_left)/Res;

Pixels_count = zeros(nrows,ncols);
Sum_Above = zeros(nrows,ncols);
Sum_Below = zeros(nrows,ncols);

options = [];

options.use_SRF = true;
options.m = 2;
options.n = 2;
options.use_simplified_area = true;
options.inflate_pixel = true;
options.inflationx = 1.2;
options.inflationy = 1.5;

tic
[Sum_Above,Sum_Below,Pixels_count] = ...
    F_cakeCut(Sum_Above,Sum_Below,Pixels_count,...
    Lon_r,Lat_r,Lon_c,Lat_c,Lon_left,Lat_low,VCD,VCD_Unc,Res,options);
toc
Sum_Below(Sum_Below == 0) = nan;
Average = Sum_Above./Sum_Below;

xext = 5*0.02/Res;yext = 2*0.02/Res;
Xlim = [min(pixel.vList(:,1))-xext max(pixel.vList(:,1))+xext];
Ylim = [min(pixel.vList(:,2))-yext max(pixel.vList(:,2))+yext];

close all
figure('unit','inch','color','w','position',[0 1 5 8])
ax1 = subplot(3,1,1);
h = pcolor((1:ncols)-1,(1:nrows)-1,Average);
set(h,'edgecolor','w')
colorbar
% set(gca,'xlim',Xlim,'ylim',Ylim,'box','off')

ax1 = subplot(3,1,2);
h = pcolor((1:ncols)-1,(1:nrows)-1,Sum_Above);
set(h,'edgecolor','w')
colorbar
% set(gca,'xlim',Xlim,'ylim',Ylim,'box','off')

ax1 = subplot(3,1,3);
h = pcolor((1:ncols)-1,(1:nrows)-1,Sum_Below);
set(h,'edgecolor','w')
colorbar
% set(gca,'xlim',Xlim,'ylim',Ylim,'box','off')
%% test cakecut function
testdata = [1,34.928440,35.046413,35.157444,35.039295,35.045200,-70.678810,-70.721886,-70.306282,-70.263763,-70.486397,43.510700,0.16052000,1.6698500,1.4008900e+15,3.4824599e+15];

Res = 0.02;

Lon_r = testdata(7:10);
Lat_r = testdata(2:5);
Lon_c = testdata(11);
Lat_c = testdata(6);

VCD = testdata(end-1);
VCD_Unc = testdata(end);
Lon_left = floor(min(Lon_r));
Lon_right = ceil(max(Lon_r));
Lat_low = floor(min(Lat_r));
Lat_up = ceil(max(Lat_r));

nrows = (Lat_up-Lat_low)/Res;
ncols = (Lon_right-Lon_left)/Res;

Pixels_count = zeros(nrows,ncols);
Sum_Above = zeros(nrows,ncols);
Sum_Below = zeros(nrows,ncols);

options = [];

options.use_SRF = true;
options.m = 2;
options.n = 2;
options.use_simplified_area = false;
options.inflate_pixel = true;
options.inflationx = 2;
options.inflationy = 2;

tic
[Sum_Above,Sum_Below,Pixels_count] = ...
    F_cakeCut(Sum_Above,Sum_Below,Pixels_count,...
    Lon_r,Lat_r,Lon_c,Lat_c,Lon_left,Lat_low,VCD,VCD_Unc,Res,options);
toc
Sum_Below(Sum_Below == 0) = nan;
Average = Sum_Above./Sum_Below;
close
h = pcolor((1:ncols)-1,(1:nrows)-1,Sum_Above);set(h,'edgecolor','w')
hold on
h1 = F_plot_polygon([(Lon_r(:)-Lon_left)/Res,(Lat_r(:)-Lat_low)/Res]);
set(h1,'linewidth',2,'color','r')
axis equal
axis off
%%
% export_fig(['/data/wdocs/kangsun/www-docs/files/tessellation_unify.pdf'])
export_fig(['/data/wdocs/kangsun/www-docs/files/srf_unify.pdf'])