function learning_rate = getLearningRate(dataset, network_arch)
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

switch network_arch
  case 'lenet'
    switch dataset
      % multi-class
      case 'cifar'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'coil-100'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'mnist'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'stl-10'
        learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];
      case 'svhn'
        learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];
      case 'cifar-two-class-deer-horse'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
      case 'cifar-two-class-deer-truck'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
      case 'cifar-no-white-two-class-deer-truck'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];

      % multi-class subsampled
      case 'mnist-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'svhn-multi-class-subsampled'
        learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];
      case 'cifar-multi-class-subsampled'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,25)];
      case 'stl-10-multi-class-subsampled'
        learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];

      % two-class
      case 'mnist-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
      case 'stl-10-two-class-airplane-bird'
        learning_rate = [0.05*ones(1,20) 0.005*ones(1,20) 0.001*ones(1,110)] * 10;
      case 'stl-10-two-class-airplane-cat'
        learning_rate = [0.05*ones(1,20) 0.005*ones(1,20) 0.001*ones(1,110)] * 10;
        % learning_rate = [0.05*ones(1,20) 0.005*ones(1,20) 0.001*ones(1,100) 0.0001*ones(1,310)] * 10;
      case 'svhn-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,120)];
      case 'prostate-v2-20-patients'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)];
      case 'prostate-v3-104-patients'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20)];
    end



  case 'fc_lenet_with_conv'
    switch dataset
      % multi-class
      case 'mnist'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      case 'cifar'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      case 'stl-10'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      case 'svhn'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;

      % multi-class subsampled
      case 'mnist-multi-class-subsampled'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      case 'cifar-multi-class-subsampled'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      case 'stl-10-multi-class-subsampled'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
      case 'svhn-multi-class-subsampled'
        learning_rate = [0.1*ones(1,5) 0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,45)] / 20;
    end
  case 'lenet_with_conv'
    switch dataset
      % multi-class
      case 'mnist'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'cifar'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'stl-10'
        % learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];             % 15.05 / 14.69
        % learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)]; % 27.15 / 27.48
        % learning_rate = [0.5*ones(1,50) 0.05*ones(1,100)];                                % 36.10 / 36.60
        % learning_rate = [0.5*ones(1,150)];                                                % 45.32 / 44.85
        learning_rate = [1 * ones(1,20) 0.5*ones(1,130)];                                   %
      case 'svhn'
        % learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];             % 72.80 / 70.39
        % learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];             %
        % learning_rate = [0.05*ones(1,150)];                                               %
        learning_rate = [0.05*ones(1,50) 0.01*ones(1,50) 0.005*ones(1,50)];                 %

      % multi-class subsampled
      case 'mnist-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'cifar-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'stl-10-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
      case 'svhn-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,25)];
    end




  case 'fc_lenet_with_larger_fc_conv'
    switch dataset
      % multi-class
      case 'mnist'
        learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % testing!!!
      case 'mnist-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % testing!!!
    end
  case 'lenet_with_larger_fc_conv'
    switch dataset
      % multi-class
      case 'mnist'
        learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % best
      case 'mnist-multi-class-subsampled'
        learning_rate = [0.05*ones(1,15) 0.01*ones(1,20) 0.005*ones(1,10) 0.0005*ones(1,25)]; % best
    end





  case 'lenet+1'
    switch dataset
      case 'cifar-two-class-deer-truck'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'mnist-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'svhn-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'stl-10-two-class-airplane-bird'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
      case 'stl-10-two-class-airplane-cat'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
    end
  case 'lenet++1'
    switch dataset
      case 'cifar-two-class-deer-truck'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'mnist-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'svhn-two-class-9-4'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)];
      case 'stl-10-two-class-airplane-bird'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
      case 'stl-10-two-class-airplane-cat'
        learning_rate = [0.05*ones(1,10) 0.005*ones(1,20) 0.001*ones(1,20) 0.0005*ones(1,50) 0.0001*ones(1,100)] * 10;
    end
  case 'alexnet'
    switch dataset
      case 'cifar'
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      case 'coil-100'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      case 'mnist'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      case 'stl-10'
        learning_rate = [0.5*ones(1,20) 0.05*ones(1,15)  0.1:-0.01:0.06 0.05*ones(1,10)];
      case 'svhn'
        learning_rate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,10)];
      case 'cifar-two-class-deer-horse'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
      case 'cifar-two-class-deer-truck'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
      case 'mnist-two-class-9-4'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
      case 'svhn-two-class-9-4'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,55)]; % amir-LR
      case 'prostate-v2-20-patients'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
      case 'prostate-v3-104-patients'
        % still testing ...
        learning_rate = [0.01*ones(1,5) 0.005*ones(1,25) 0.001*ones(1,10) 0.0005*ones(1,5) 0.0001*ones(1,5)]; % amir-LR
    end
end




































