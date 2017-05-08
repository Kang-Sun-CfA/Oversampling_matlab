function p = F_append2(p,v)

iv = p.nv+1;
p.vList(iv,1:2) = v(1:2);
p.nv = iv;