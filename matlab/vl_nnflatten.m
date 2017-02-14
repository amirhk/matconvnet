function out = vl_nnfla(x,dzdy,x_layer_before,varargin)

if nargin <= 1 || isempty(dzdy)
  % forward pass
  % out = reshape(x, [1, 1, 4 * 4 * size(x,3), size(x,4)]);
  out = reshape(x, [1, 1, size(x,1) * size(x,2) * size(x,3), size(x,4)]);
else
  % backwards pass
  assert(size(x_layer_before.x, 1) * size(x_layer_before.x, 2) * size(x_layer_before.x, 3) == size(x, 3));
  original_pre_flatten_dimensions = size(x_layer_before.x);
  out = reshape(x, original_pre_flatten_dimensions);
end














