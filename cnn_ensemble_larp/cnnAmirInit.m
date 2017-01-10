function output_opts = cnn_amir_init(input_opts)
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

  opts.dataset = input_opts.general.dataset;
  opts.network_arch = input_opts.general.network_arch;
  opts.weight_init_source = input_opts.net.weight_init_source;
  opts.weight_init_sequence = input_opts.net.weight_init_sequence;

  if strcmp(opts.dataset, 'prostate-v2-20-patients') || ...
    strcmp(opts.dataset, 'mnist-two-class-9-4') || ...
    strcmp(opts.dataset, 'svhn-two-class-9-4') || ...
    strcmp(opts.dataset, 'cifar-two-deer-horse') || ...
    strcmp(opts.dataset, 'cifar-two-deer-truck')
    assert(strcmp(opts.network_arch, 'two-class-lenet'));
  end

  tic;
  s = rng;
  rng(0);
  net.layers = {};
  % Meta parameters
  switch opts.network_arch
    case 'mnistnet'
      switch opts.dataset
        case 'mnist'
          output_opts.train.learning_rate = [0.001*ones(1,50)]; % matconvnet default
      end
    case 'two-class-lenet'
      switch opts.dataset
        case 'cifar-two-class-deer-horse'
          output_opts.train.learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
        case 'cifar-two-class-deer-truck'
          output_opts.train.learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
        case 'mnist-two-class-9-4'
          output_opts.train.learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
          output_opts.train.learning_rate = [0.05*ones(1,10)];
        case 'svhn-two-class-9-4'
          % output_opts.train.learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
          output_opts.train.learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)]; % matconvnet default
        case 'prostate-v2-20-patients'
          output_opts.train.learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)];
          % output_opts.train.learning_rate = [0.05*ones(1,5)];
      end
    case 'lenet'
      switch opts.dataset
        case 'cifar'
          output_opts.train.learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
        case 'coil-100'
          output_opts.train.learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
        case 'mnist'
          output_opts.train.learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
        case 'stl-10'
          output_opts.train.learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)]; % javad-LR
        case 'svhn'
          % output_opts.train.learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)]; % matconvnet default
          output_opts.train.learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)]; % matconvnet default
      end
    case 'alexnet'
      switch opts.dataset
        case 'cifar'
          output_opts.train.learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
        case 'stl-10'
          % still testing
          % output_opts.train.learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)]; % testing
          % output_opts.train.learning_rate = [2*ones(1,20)]; % testing
          output_opts.train.learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      end
    case 'alexnet-bnorm'
      output_opts.train.learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)];
    case 'alexnet-bottleneck'
      output_opts.train.learning_rate = [0.005*ones(1,50)];
  end

  output_opts.train.num_epochs = numel(output_opts.train.learning_rate);

  switch opts.network_arch
    case 'mnistnet'
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- ---                    MNISTNET                         --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = 1;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 1, 20, 1/100, 0, char(opts.weight_init_sequence{1}), opts.weight_init_source);
      net.layers{end+1} = poolingLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 20, 50, 1/100, 0, char(opts.weight_init_sequence{1}), opts.weight_init_source);
      net.layers{end+1} = poolingLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 4, 50, 500, 1/100, 0, char(opts.weight_init_sequence{1}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 500, 10, 1/100, 0, 'compRand', 'gen');

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % Loss layer
      net.layers{end+1} = struct('type', 'softmaxloss');
    case 'two-class-lenet'
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- ---                TWO-CLASS-LENET                      --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = 1;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), opts.weight_init_source);
      net.layers{end+1} = poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerLeNetAvg(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerLeNetAvg(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % FULLY CONNECTED
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 2, 5/100, 0, 'compRand', 'gen');

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % Loss layer
      net.layers{end+1} = struct('type', 'softmaxloss');
    case 'lenet'
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- ---                     LENET                           --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = 1;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 32, 1/100, 2, char(opts.weight_init_sequence{1}), opts.weight_init_source);
      net.layers{end+1} = poolingLayerLeNetMax(layer_number);
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 32, 5/100, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerLeNetAvg(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 32, 64, 5/100, 2, char(opts.weight_init_sequence{3}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerLeNetAvg(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % FULLY CONNECTED
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 4, 64, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      if strcmp(opts.dataset, 'coil-100')
        net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 100, 5/100, 0, 'compRand', 'gen');
      else
        net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
      end

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % Loss layer
      net.layers{end+1} = struct('type', 'softmaxloss');
    case 'alexnet'
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- ---                   ALEXNET                           --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = 1;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 96, 5/1000, 2, char(opts.weight_init_sequence{1}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96, 256, 5/1000, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerAlexNet(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256, 384, 5/1000, 1, char(opts.weight_init_sequence{3}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerAlexNet(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384, 5/1000, 1, char(opts.weight_init_sequence{4}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 256, 5/1000, 1, char(opts.weight_init_sequence{5}), opts.weight_init_source);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerAlexNet(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % FULLY CONNECTED
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 4, 256, 128, 5/1000, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 128, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % Loss layer
      net.layers{end+1} = struct('type', 'softmaxloss');
    case 'alexnet-bnorm'
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- ---                ALEXNET-BNORM                        --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = 1;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 96, 5/1000, 2, char(opts.weight_init_sequence{1}), opts.weight_init_source);
      % net.layers{end+1} = bnormLayer(layer_number, 96);
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96, 256, 5/1000, 2, char(opts.weight_init_sequence{2}), opts.weight_init_source);
      net.layers{end+1} = bnormLayer(layer_number, 256);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerAlexNet(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256, 384, 5/1000, 1, char(opts.weight_init_sequence{3}), opts.weight_init_source);
      % net.layers{end+1} = bnormLayer(layer_number, 384);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerAlexNet(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384, 5/1000, 1, char(opts.weight_init_sequence{4}), opts.weight_init_source);
      % net.layers{end+1} = bnormLayer(layer_number, 384);
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 256, 5/1000, 1, char(opts.weight_init_sequence{5}), opts.weight_init_source);
      net.layers{end+1} = bnormLayer(layer_number, 256);
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerAlexNet(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % FULLY CONNECTED
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 4, 256, 128, 5/1000, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 128, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % Loss layer
      net.layers{end+1} = struct('type', 'softmaxloss');
    case 'alexnet-bottleneck'
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- ---               ALEXNET-BOTTLENECK                    --- --- --
      % --- --- ---                                                     --- --- --
      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % k = [1,4,8,16,32]
      k = opts.bottleneckDivideBy;
      layer_number = 1;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 3, 96, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      % net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96, 256, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96, 96/k, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 5, 96/k, 256, 5/1000, 2, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerAlexNet(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256, 384, 5/1000, 1, 'compRand', 'gen');
      % net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256, 256/k, 5/1000, 1, 'compRand', 'gen');
      % net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 256/k, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerAlexNet(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384, 5/1000, 1, 'compRand', 'gen');
      % net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384/k, 5/1000, 1, 'compRand', 'gen');
      % net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384/k, 384, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 256, 5/1000, 1, 'compRand', 'gen');
      % net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384, 384/k, 5/1000, 1, 'compRand', 'gen');
      % net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 3, 384/k, 256, 5/1000, 1, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);
      net.layers{end+1} = poolingLayerAlexNet(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % FULLY CONNECTED
      layer_number = layer_number + 3;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 4, 256, 128, 5/1000, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 128, 64, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      layer_number = layer_number + 2;
      net.layers{end+1} = convLayer(opts.dataset, opts.network_arch, layer_number, 1, 64, 10, 5/100, 0, 'compRand', 'gen');
      net.layers{end+1} = reluLayer(layer_number);

      % --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
      % Loss layer
      net.layers{end+1} = struct('type', 'softmaxloss');
  end

  % VERY IMPORTANT to reset this afterwards so other modules are true random
  rng(s);
  output_opts.net.net = net;


% --------------------------------------------------------------------
function structuredLayer = convLayer(dataset, network_arch, layer_number, k, m, n, init_multiplier, pad, weight_init_type, weight_init_source);
% --------------------------------------------------------------------
  switch weight_init_source
    case 'load'
      layerWeights = loadWeights(dataset, network_arch, layer_number, weight_init_type);
    case 'gen'
      if ~strcmp(weight_init_type, 'compRand')
        utils = networkExtractionUtils;
        baselineWeights = loadWeights(dataset, network_arch, layer_number, 'baseline'); % used for its size
      end
      switch weight_init_type
        case 'compRand'
          layerWeights{1} = init_multiplier * randn(k, k, m, n, 'single');
          layerWeights{2} = zeros(1, n, 'single');
        otherwise
          throwException('[ERROR] Generating non-compRand weights not supported from this code.');
      end
  end
  structuredLayer = constructConvLayer(network_arch, layer_number, layerWeights, pad, weight_init_type, weight_init_source);

% --------------------------------------------------------------------
function weights = loadWeights(dataset, network_arch, layer_number, weight_init_type)
% --------------------------------------------------------------------
  fprintf( ...
    '[INFO] Loading %s weights (layer %d) from saved directory...\t', ...
    weight_init_type, ...
    layer_number);
  dev_path = getDevPath();

  % sub_dir_path = fullfile('data', 'cifar-alexnet', sprintf('w_%s', weight_init_type));
  % TODO: search subtstring... if network_arch starts with 'alexnet' use the 'alexnet' folder
  sub_dir_path = fullfile( ...
    'data', ...
    'generated_weights', ...
    sprintf('%s', network_arch), ...
    sprintf('w-%s', weight_init_type));
  file_name_suffix = sprintf('-layer-%d.mat', layer_number);
  tmp = load(fullfile(dev_path, sub_dir_path, sprintf('W1%s', file_name_suffix)));
  weights{1} = tmp.W1;
  tmp = load(fullfile(dev_path, sub_dir_path, sprintf('W2%s', file_name_suffix)));
  weights{2} = tmp.W2;
  fprintf('Done!\n');

% --------------------------------------------------------------------
function structuredLayer = constructConvLayer(network_arch, layer_number, weights, pad, weight_init_type, weight_init_source)
% --------------------------------------------------------------------
  lr = [.1 2];
  if strcmp(network_arch, 'alexnet') && layer_number == 18
    lr = lr * .1;
  elseif strcmp(network_arch, 'lenet') && layer_number == 12
  end
  structuredLayer = struct( ...
    'type', 'conv', ...
    'name', sprintf('conv%s-%s-%s', layer_number, weight_init_type, weight_init_source), ...
    'weights', {weights}, ...
    'learning_rate', lr, ...
    'stride', 1, ...
    'pad', pad);

% --------------------------------------------------------------------
function structuredLayer = reluLayer(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'relu', ...
    'name', sprintf('relu%s', layer_number));

% --------------------------------------------------------------------
function structuredLayer = tanhLayer(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'tanh', ...
    'name', sprintf('tanh%s', layer_number));

% --------------------------------------------------------------------
function structuredLayer = poolingLayer(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layer_number), ...
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 0); % Emulate caffe_n

% --------------------------------------------------------------------
function structuredLayer = poolingLayerAlexNet(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layer_number), ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe_n

% --------------------------------------------------------------------
function structuredLayer = poolingLayerLeNetAvg(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layer_number), ...
    'method', 'avg', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe_n

% --------------------------------------------------------------------
function structuredLayer = poolingLayerLeNetMax(layer_number)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'pool', ...
    'name', sprintf('pool%s', layer_number), ...
    'method', 'max', ...
    'pool', [3 3], ...
    'stride', 2, ...
    'pad', [0 1 0 1]); % Emulate caffe_n

% --------------------------------------------------------------------
function structuredLayer = dropoutLayer(layer_number, dropout_ratio)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'dropout', ...
    'name', sprintf('dropout%s', layer_number), ...
    'rate', dropout_ratio);

% --------------------------------------------------------------------
function structuredLayer = bnormLayer(layer_number, ndim)
% --------------------------------------------------------------------
  structuredLayer = struct( ...
    'type', 'bnorm', ...
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
    'learningRate', [1 1], ...
    'weightDecay', [0 0]);

% --------------------------------------------------------------------
function throwException(msg)
% --------------------------------------------------------------------
  msgID = 'MYFUN:BadIndex';
  throw(MException(msgID,msg));
