function [Pouts,npouts] = F_HcakeCut(Pin)
vList = nan(Pin.nv,2,'single');
nv = Pin.nv;
vList(1:nv,1:2) = Pin.vList(1:nv,1:2);
if( nv < 3 )
    npouts = 0;
    return
end

% find ymin, ymax and their corresponding indices
[ymin,jmin] = min(vList(:,2));
[ymax,jmax] = max(vList(:,2));


irowMax = ceil( ymax ); irowMin = floor( ymin );
npouts = irowMax - irowMin;

if( npouts <= 0 );return;end
Pouts(npouts).nv = 0;
if( npouts == 1 )
    Pouts(1).nv = nv;
    Pouts(1).vList(1:nv,1:2) = vList(1:nv,1:2);
    return
end

for i = 1: npouts
    Pouts(i).nv = 0;
    Pouts(i).vList = zeros(nv,2,'single');
end

i  = 0; ip = 1; sign = 1;
j  = jmin;
j1 = mod( j, nv) + 1;
y  = floor( ymin ) + 1;

while( i < nv )
    if( sign*vList(j1, 2) > sign* y  )  %true, for intersecting y
        x = vList( j,1 ) + (y-vList(j,2)) ...
            *(vList(j1,1 ) -  vList(j,1))/(vList(j1,2) - vList(j,2));
        Pouts(ip) = F_append1( Pouts(ip), x, y );
        ip = ip + sign;
        Pouts(ip) = F_append1( Pouts(ip), x, y );
        y  =  y + sign;
    else
        Pouts(ip) = F_append2( Pouts(ip), vList(j1,1:2) );
        if( sign*vList(j1, 2) >= sign* y  ) % > not possible, for ==
            if( 1 <= ip + sign && ip + sign <= npouts )
                ip = ip + sign;
                y  =  y + sign;
                Pouts(ip) = F_append2( Pouts(ip), vList(j1,1:2) );
            end
        end
        j = j1; j1 = mod(j, nv) + 1;
        if( j == jmax )
            sign = -1; y = y + (sign);
        end
        i = i + 1;
    end
end
