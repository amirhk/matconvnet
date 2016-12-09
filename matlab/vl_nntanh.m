function y = vl_nntanh(x,dzdy,varargin)
  % AMIR's implementation that copies almost everything from vl_nnrelu
  % only did s/(x > single(0))/tanh(x)
opts.leak = 0 ;
opts = vl_argparse(opts, varargin) ;

if opts.leak == 0
  if nargin <= 1 || isempty(dzdy)
    y = max(x, single(0)) ;
  else
    y = dzdy .* tanh(x) ;
  end
  % testing absolute value non-linearity
  % disp(dzdy);
  % disp(x);
  % if nargin <= 1 || isempty(dzdy)
  %   y = abs(x);
  % else
  %   y = dzdy .* abs(x);
  % end
else
  if nargin <= 1 || isempty(dzdy)
    y = x .* (opts.leak + (1 - opts.leak) * tanh(x)) ;
  else
    y = dzdy .* (opts.leak + (1 - opts.leak) * tanh(x)) ;
  end
end
