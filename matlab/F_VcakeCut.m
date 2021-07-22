function [Pouts,npouts] = F_VcakeCut(Pin)
vList = nan(Pin.nv,2,'single');
nv = Pin.nv;
vList(1:nv,1:2) = Pin.vList(1:nv,1:2);
if( nv < 3 )
    npouts = 0;
    return
end

% find xmin, xmax and their corresponding indices
[xmin,jmin] = min(vList(:,1));
[xmax,jmax] = max(vList(:,1));


icolMax = ceil( xmax ); icolMin = floor( xmin );
npouts = icolMax - icolMin;

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
x  = floor( xmin ) + 1;

while( i < nv )
    if( sign*vList(j1, 1) > sign* x  )  %true, for intersecting x
        y = vList( j,2 ) + (x-vList(j,1)) ...
            *(vList(j1,2 ) -  vList(j,2))/(vList(j1,1) - vList(j,1));
        Pouts(ip) = F_append1( Pouts(ip), x, y );
        ip = ip + sign;
        Pouts(ip) = F_append1( Pouts(ip), x, y );
        x  =  x + sign;
    else
        Pouts(ip) = F_append2( Pouts(ip), vList(j1,1:2) );
        if( sign*vList(j1, 1) >= sign* x  ) % > not possible, for ==
            if( 1 <= ip + sign && ip + sign <= npouts )
                ip = ip + sign;
                x  =  x + sign;
                Pouts(ip) = F_append2( Pouts(ip), vList(j1,1:2) );
            end
        end
        j = j1; j1 = mod(j, nv) + 1;
        if( j == jmax )
            sign = -1; x = x + (sign);
        end
        i = i + 1;
    end
end
