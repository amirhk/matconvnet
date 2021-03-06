% -------------------------------------------------------------------------
function weight_init_sequence = getLarpWeightInitSequence(larp_weight_init_type, larp_network_arch)
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

  if strfind(larp_network_arch, 'custom-')
    number_of_blocks = str2num(getStringParameterStartingAtIndex(larp_network_arch, 8));
  else
    assert(strcmp(larp_network_arch(1:5), 'larpV'));
    number_of_blocks = str2num(larp_network_arch(6));
  end

  if number_of_blocks == 0
    weight_init_sequence = {};
  else
    weight_init_sequence = repmat({larp_weight_init_type}, 1, number_of_blocks);
  end

  % switch larp_network_arch

  %   case 'larpV0P0RL0'
  %     weight_init_sequence = {};

  %   case 'larpV0P0RL0-single-dense-rp'
  %     weight_init_sequence = {};
  %   case 'larpV1P0RL0-single-sparse-rp'
  %     weight_init_sequence = {larp_weight_init_type};
  %   case 'larpV1P0RL0-ensemble-sparse-rp' % = 'larpV1P0'
  %     weight_init_sequence = {larp_weight_init_type};
  %   case 'larpV1P0RL1-ensemble-sparse-rp' % = 'larpV1P0'
  %     weight_init_sequence = {larp_weight_init_type};

  %   case 'larpV1P0RL0' % = 'larpV1P0-ensemble-sparse-rp'
  %     weight_init_sequence = {larp_weight_init_type};
  %   case 'larpV1P0RL1' % = 'larpV1P0-ensemble-sparse-rp'
  %     weight_init_sequence = {larp_weight_init_type};
  %   case 'larpV1P1RL1-non-decimated-pooling'
  %     weight_init_sequence = {larp_weight_init_type};
  %   case 'larpV1P1RL1'
  %     weight_init_sequence = {larp_weight_init_type};






  %   case 'larpV1P1RL1-special-1'
  %     weight_init_sequence = {larp_weight_init_type};
  %   case 'larpV1P1RL1-special-2'
  %     weight_init_sequence = {larp_weight_init_type};






  %   case 'larpV3P0RL0'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV3P0RL3'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV3P1RL3-non-decimated-pooling'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV3P1RL3'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV3P3RL0'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV3P2RL3'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV3P3RL3'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV3P3RL3-final-conv-16-kernels'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};






  %   case 'larpV5P1RL5-non-decimated-pooling'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV5P1RL5'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV5P2RL5'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV5P3RL5'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV5P3RL5-final-conv-16-kernels'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};



  %   case 'larpV5P0RL0'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV5P0RL5'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV5P3RL0'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV5P3RL5'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV5P5RL0'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV5P5RL5'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  %   case 'larpV8P0RL8'
  %     weight_init_sequence = {larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type, larp_weight_init_type};
  % end


