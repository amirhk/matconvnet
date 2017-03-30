% -------------------------------------------------------------------------
function weight_init_sequence = getWeightInitSequenceForWeightInitTypeAndNetworkArch(larp_weight_init_type, network_arch)
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

  % Note, only larp layers may be `generated` with the weight_init_type.
  % Conv andmlp layers are still `initialized` using randn().

  switch network_arch

    case 'larpV0P0+convV0P0+fcV1'
      weight_init_sequence = {'gaussian', 'gaussian'};


    case 'larpV0P0+convV1P1+fcV1'
      weight_init_sequence = {larp_weight_init_type, 'gaussian', 'gaussian'};


    case 'larpV1aP1+convV0P0+fcV1-TEST'
      weight_init_sequence = {larp_weight_init_type, 'gaussian', 'gaussian'};
    case 'larpV3aP1+convV0P0+fcV1-TEST'
      weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, 'gaussian', 'gaussian', 'gaussian'};
    case 'larpV1P1+convV0P0+fcV1-TEST'
      weight_init_sequence = {larp_weight_init_type, 'gaussian', 'gaussian'};
    case 'larpV3P1+convV0P0+fcV1-TEST'
      weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, 'gaussian', 'gaussian', 'gaussian'};


    case 'larpV1P1+convV0P0+fcV1'
      weight_init_sequence = {larp_weight_init_type, 'gaussian', 'gaussian'};
    case 'larpV3P1+convV0P0+fcV1'
      weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, 'gaussian', 'gaussian'};
    case 'convV0P0+fcV1RF16CH64'
      weight_init_sequence = {'gaussian', 'gaussian'};
    case 'convV0P0+fcV1RF4CH64'
      weight_init_sequence = {'gaussian', 'gaussian'};

    case 'larpV0P0+convV0P0+fcV2'
      weight_init_sequence = {'gaussian', 'gaussian', 'gaussian'};
    case 'larpV3P1+convV0P0+fcV2'
      weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, 'gaussian', 'gaussian', 'gaussian'};

  end
