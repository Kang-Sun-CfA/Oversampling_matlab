function [ifornot, edgelength] = F_if_square(p)
% determin if a polygon is a rectangle
% written by Kang Sun on 2017/05/07
if p.nv ~= 4
    ifornot = false;edgelength = nan;return;
else
vList = p.vList(1:p.nv,:);
xsort = sort(vList(:,1));
xedge = xsort(3)-xsort(2);
if xsort(2)-xsort(1) > 0.01*xedge || xsort(4)-xsort(3) > 0.01*xedge
    ifornot = false;edgelength = nan;return;
end

ysort = sort(vList(:,2));
yedge = ysort(3)-ysort(2);
if ysort(2)-ysort(1) > 0.01*yedge || ysort(4)-ysort(3) > 0.01*yedge
    ifornot = false;edgelength = nan;return;
end
ifornot = true;edgelength = nan(2,1);
edgelength(1) = xedge;edgelength(2) = yedge;
end
