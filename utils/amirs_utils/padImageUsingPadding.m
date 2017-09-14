% -------------------------------------------------------------------------
function padded_multi_channel_image = padImageUsingPadding(I, padding)
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

  number_of_channels = size(I, 3);
  tmp_padded_single_channel = padSingleChannelImageUsingPadding(I(:,:,1), padding);

  padded_multi_channel_image = zeros(size(tmp_padded_single_channel, 1), size(tmp_padded_single_channel, 2), number_of_channels);

  for i = 1 : number_of_channels
    padded_multi_channel_image(:,:,i) = padSingleChannelImageUsingPadding(I(:,:,i), padding);
  end


% -------------------------------------------------------------------------
function padded_image = padSingleChannelImageUsingPadding(I, padding)
% -------------------------------------------------------------------------
  padded_image = I;
  padded_image = padarray(padded_image, [padding(1), 0], 'pre');
  padded_image = padarray(padded_image, [padding(2), 0], 'post');
  padded_image = padarray(padded_image, [0, padding(3)], 'pre');
  padded_image = padarray(padded_image, [0, padding(4)], 'post');























