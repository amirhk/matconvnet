function out = vl_nnavr(x,dzdy,varargin)
%VL_NNRELU CNN rectified linear unit.
%   Y = VL_NNRELU(X) applies the rectified linear unit to the data
%   X. X can have arbitrary size.
%
%   DZDX = VL_NNRELU(X, DZDY) computes the derivative of the block
%   projected onto DZDY. DZDX and DZDY have the same dimensions as
%   X and Y respectively.
%
%   VL_NNRELU(...,'OPT',VALUE,...) takes the following options:
%
%   `Leak`:: 0
%      Set the leak factor, a non-negative number. Y is equal to X if
%      X is not smaller than zero; otherwise, Y is equal to X
%      multipied by the leak factor. By default, the leak factor is
%      zero; for values greater than that one obtains the leaky ReLU
%      unit.
%
%   ADVANCED USAGE
%
%   As a further optimization, in the backward computation it is
%   possible to replace X with Y, namely, if Y = VL_NNRELU(X), then
%   VL_NNRELU(X,DZDY) gives the same result as VL_NNRELU(Y,DZDY).
%   This is useful because it means that the buffer X does not need to
%   be remembered in the backward pass.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% opts.leak = 0 ;
% opts = vl_argparse(opts, varargin) ;

% y = max(x, -x);

% if opts.leak == 0
%   if nargin <= 1 || isempty(dzdy)
%     out = max(x, single(0)) ;
%   else
%     out = dzdy .* (x > single(0)) ;
%   end
%   % testing absolute value non-linearity
%   % disp(dzdy);
%   % disp(x);
%   % if nargin <= 1 || isempty(dzdy)
%   %   out = abs(x);
%   % else
%   %   out = dzdy .* abs(x);
%   % end
% else
%   if nargin <= 1 || isempty(dzdy)
%     out = x .* (opts.leak + (1 - opts.leak) * single(x > 0)) ;
%   else
%     out = dzdy .* (opts.leak + (1 - opts.leak) * single(x > 0)) ;
%   end
% end

y = abs(x);

if nargin <= 1 || isempty(dzdy)
  out = y ;
else
  out = dzdy .* sign(x);
end














