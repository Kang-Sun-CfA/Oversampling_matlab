function p = F_append1(p,x,y)

iv = p.nv+1;
p.vList(iv,1) = x;
p.vList(iv,2) = y;
p.nv = iv;
