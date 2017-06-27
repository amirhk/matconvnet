% -------------------------------------------------------------------------
function matching_conv_architecture = getMatchingConvArchitectureForLarpArchitecture(larp_network_arch, non_larp_network_arch, mlp_version)
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

  matching_conv_architecture = '';
  % index_of_smooth_boolean = findstr(larp_network_arch, 'S');
  % larp_network_arch = larp_network_arch(1:index_of_smooth_boolean - 1);

  if ~strcmp(non_larp_network_arch, 'convV0P0RL0+fcV1')
    assert(strcmp(larp_network_arch, 'larpV0P0RL0'));
  end

  if strcmp(mlp_version, 'v1')
    switch larp_network_arch

      case 'larpV0P0RL0'
        switch non_larp_network_arch
          case 'convV0P0RL0+fcV1'
            matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH3';
          case 'convV1P0RL0+fcV1'
            matching_conv_architecture = 'convV1P0RL0-RF32CH3+fcV1-RF32CH64';
          case 'convV1P0RL1+fcV1'
            matching_conv_architecture = 'convV1P0RL1-RF32CH3+fcV1-RF32CH64';
          case 'convV1P1RL1+fcV1'
            matching_conv_architecture = 'convV1P1RL1-RF32CH3+fcV1-RF16CH64';
          case 'convV3P0RL0+fcV1'
            matching_conv_architecture = 'convV3P0RL0-RF32CH3+fcV1-RF32CH64';
          case 'convV3P0RL3+fcV1'
            matching_conv_architecture = 'convV3P0RL3-RF32CH3+fcV1-RF32CH64';
          case 'convV3P1RL3+fcV1'
            matching_conv_architecture = 'convV3P1RL3-RF32CH3+fcV1-RF16CH64';
          case 'convV3P3RL0+fcV1'
            matching_conv_architecture = 'convV3P3RL0-RF32CH3+fcV1-RF4CH64';
          case 'convV3P3RL3+fcV1'
            matching_conv_architecture = 'convV3P3RL3-RF32CH3+fcV1-RF4CH64';
        end

      case 'larpV0P0RL0-single-dense-rp'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH3';
      case 'larpV1P0RL0-single-sparse-rp'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH3';
      case 'larpV1P0RL0-ensemble-sparse-rp' % = 'larpV1P0RL0'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH64';
      case 'larpV1P0RL1-ensemble-sparse-rp' % = 'larpV1P0RL1'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH64';

      case 'larpV1P0RL0' % = 'larpV1P0-ensemble-sparse-rp'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH64';
      case 'larpV1P0RL1' % = 'larpV1P0-ensemble-sparse-rp'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH64';

      case 'larpV1P1RL1-non-decimated-pooling'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH64';
      case 'larpV1P1RL1'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF16CH64';
      case 'larpV3P0RL0'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH64';
      case 'larpV3P0RL3'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF32CH64';
      case 'larpV3P1RL3'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF16CH64';
      case 'larpV3P3RL0'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF4CH64';
      case 'larpV3P3RL3'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF4CH64';
      case 'larpV5P3RL5'
        matching_conv_architecture = 'convV0P0RL0+fcV1-RF4CH64';
    end

  % elseif strcmp(mlp_version, 'v2')
  %   switch larp_network_arch
  %     case 'larpV0P0'
  %       matching_conv_architecture = 'convV0P0+fcV2RF32CH3';
  %     case 'larpV1P0'
  %       matching_conv_architecture = 'convV0P0+fcV2RF32CH64';
  %     case 'larpV1P1'
  %       matching_conv_architecture = 'convV0P0+fcV2RF16CH64';
  %     case 'larpV3P0'
  %       matching_conv_architecture = 'convV0P0+fcV2RF32CH64';
  %     case 'larpV3P1'
  %       matching_conv_architecture = 'convV0P0+fcV2RF16CH64';
  %     case 'larpV3P3'
  %       matching_conv_architecture = 'convV0P0+fcV2RF4CH64';
  %     case 'larpV5P3'
  %       matching_conv_architecture = 'convV0P0+fcV2RF4CH64';
  %   end
  else
    throwException('[ERROR] `mlp_version` not recongized.');
  end

