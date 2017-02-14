function out = vl_nnfla(x,dzdy,varargin)

if nargin <= 1 || isempty(dzdy)
  % forward pass
  out = reshape(x, [1, 1, 4 * 4 * size(x,3), size(x,4)]);
else
  % backwards pass
  out = reshape(x, [4, 4, size(x,3) / (4 * 4), size(x,4)]);
end














