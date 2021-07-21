function A = F_polyarea(p)

if isstruct(p)
A = polyarea([p.vList(1:p.nv,1);p.vList(1,1)],[p.vList(1:p.nv,2);p.vList(1,2)]);
else
    A = polyarea([p(:,1);p(1,1)],[p(:,2);p(1,2)]);
end