function [X, minlon_e, minlat_e, h] = F_construct_ellipse(z,a,b,alpha,npoint,if_plot)
%PLOTELLIPSE   Plot parametrically specified ellipse
%
%   PLOTELLIPSE(Z, A, B, ALPHA) Plots the ellipse specified by Z, A, B,
%       ALPHA (as returned by FITELLIPSE)
%
%       A, B are positive scalars, Z a 2x1 column vector, and
%       ALPHA a rotation angle, such that the equation of the ellipse is:
%           X = Z + Q(ALPHA) * [A * cos(phi); B * sin(phi)]
%       where Q(ALPHA) is the rotation matrix
%           Q(ALPHA) = [cos(ALPHA) -sin(ALPHA);
%                       sin(AlPHA) cos(ALPHA)]
%
%   H = PLOTELLIPSE(...) returns a handle to the created lineseries object
%       created by the plot command
%   
%   Example:
%       % Ellipse centred at 10,10, with semiaxes 5 and 3, rotated by pi/4
%       a = 5;
%       b = 3;
%       z = [10; 10]
%       alpha = pi/4;
%       plotellipse(z, a, b, alpha)
%
%   See also FITELLIPSE

% Copyright Richard Brown. This code can be freely reused and modified so
% long as it retains this copyright clause
%
% Just kept the essentials for simplicity and speed; introduced the fill option
% (LC 2011)

t = fliplr(linspace(0, 2*pi, npoint));

Q = [cos(alpha), -sin(alpha); sin(alpha) cos(alpha)];% Rotation matrix
X = Q * [a * cos(t); b * sin(t)] + repmat(z, 1, npoint);% Ellipse points

% The actual plotting one-liner
%h = plot(hAx, X(1,:), X(2,:), linespec);
    minlon_e=min(X(1,:));
%     maxlon_e=max(X(1,:));
        
    minlat_e=min(X(2,:));
%     maxlat_e=max(X(2,:));

if if_plot
    h = plot(X(1,:), X(2,:)); hold on;
else
    h = [];
end;