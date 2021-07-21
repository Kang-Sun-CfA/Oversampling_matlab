function h = F_plot_polygon(p)
if isstruct(p)
h = plot([p.vList(1:p.nv,1);p.vList(1,1)],[p.vList(1:p.nv,2);p.vList(1,2)]);
else
    h = plot([p(:,1);p(1,1)],[p(:,2);p(1,2)]);
end