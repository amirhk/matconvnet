function network_opts = cnnInit(input_opts)
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

  % -------------------------------------------------------------------------
  %                                                              Parse inputs
  % -------------------------------------------------------------------------

  dataset              = input_opts.general.dataset; % Used in 2 places: 1) convLayer loading weights 2) based on the dataset, networks decide how many outputs nodes in FC
  network_arch         = input_opts.general.network_arch;
  learning_rate        = input_opts.train.learning_rate;
  weight_init_source   = input_opts.net.weight_init_source; % Always 'gen' I think...
  weight_init_sequence = input_opts.net.weight_init_sequence;

  % -------------------------------------------------------------------------
  %                                                         Set learning rate
  % -------------------------------------------------------------------------

  s = rng;
  rng(0);
  fh = networkInitializationUtils;
  net.layers = {};
  if strcmp(learning_rate, 'default_keyword')
    network_opts.train.learning_rate = getLearningRate(dataset, network_arch);
  else
    network_opts.train.learning_rate = learning_rate;
  end
  network_opts.train.num_epochs = numel(network_opts.train.learning_rate);

  architecture_type = network_arch(1:4);
  switch architecture_type
    case 'larp'
      net = getLarpArchitecture(dataset, network_arch, weight_init_sequence);
    case 'cust' % custom-
      net = getLarpArchitecture(dataset, network_arch, weight_init_sequence);
    case 'conv'
      net = getConvArchitecture(dataset, network_arch);
    otherwise
      throwException('[ERROR] architecture type can only be `larp` or `conv`.')
  end

  % -------------------------------------------------------------------------
  %    VERY IMPORTANT: reset this afterwards so other modules are true random
  % -------------------------------------------------------------------------

  rng(s);
  network_opts.net = net;
