function out = vl_nnavr(x,dzdy,varargin)

y = abs(x);

if nargin <= 1 || isempty(dzdy)
  out = y ;
else
  out = dzdy .* sign(x);
end














