% -------------------------------------------------------------------------
function full_backprop_depth = getFullBackPropDepthForConvArchitecture(conv_architecture)
% -------------------------------------------------------------------------
% Copyright (c) 2017, Amir-Hossein Karimi
% All rights reserved.

% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are
% met:
%     * Redistributions of source code must retain the above copyright
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright
%       notice, this list of conditions and the following disclaimer in
%       the documentation and/or other materials provided with the distribution

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.

  switch conv_architecture

    case 'convV0P0RL0+fcV1-RF32CH3'
      full_backprop_depth = 4;
    case 'convV0P0RL0+fcV1-RF32CH64'
      full_backprop_depth = 4;
    case 'convV0P0RL0+fcV1-RF16CH64'
      full_backprop_depth = 4;
    case 'convV0P0RL0+fcV1-RF4CH64'
      full_backprop_depth = 4;

    case 'convV1P0RL0-RF32CH3+fcV1-RF32CH64'
      full_backprop_depth = 5;
    case 'convV1P0RL1-RF32CH3+fcV1-RF32CH64'
      full_backprop_depth = 6;
    case 'convV1P1RL1-RF32CH3+fcV1-RF16CH64'
      full_backprop_depth = 7;
    case 'convV3P0RL0-RF32CH3+fcV1-RF32CH64'
      full_backprop_depth = 7;
    case 'convV3P0RL3-RF32CH3+fcV1-RF32CH64'
      full_backprop_depth = 10;
    case 'convV3P1RL3-RF32CH3+fcV1-RF16CH64'
      full_backprop_depth = 11;
    case 'convV3P3RL0-RF32CH3+fcV1-RF4CH64'
      full_backprop_depth = 10;
    case 'convV3P3RL3-RF32CH3+fcV1-RF4CH64'
      full_backprop_depth = 13;

    case 'convV5P1RL5-RF32CH3+fcV1-RF16CH64'
      full_backprop_depth = 15;

    case 'convV3P3RL3-RF32CH3+fcV1-RF4CH64-input64x64x3'
      full_backprop_depth = 13;
    case 'convV3P3RL3-RF32CH3+fcV1-RF4CH64-input64x64x3-with-dropout'
      full_backprop_depth = 17;
    case 'convV5P3RL5-input64x64x3'
      full_backprop_depth = 17;


  end
