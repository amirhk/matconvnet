% -------------------------------------------------------------------------
function trained_net = loadTrainedNet(entire_architecture, dataset, posneg_balance)
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

  if ispc
    datapath = 'H:\Amir\';
  else
    datapath = '/Volumes/Amir/';
  end
  folder_path = fullfile(datapath, 'some trained networks', entire_architecture, dataset, posneg_balance);
  trained_nets = dir(fullfile(folder_path, 'net-epoch-*.mat'));
  assert(numel(trained_nets) == 1);
  tmp = load(fullfile(folder_path, trained_nets(1).name));
  trained_net = tmp.net;


  % tmp = load(path_convV1P0RL0);
  % projection_net = tmp.net;

  % if ispc
  %   datapath = 'H:\Amir\';
  %   path_1 = fullfile(datapath, 'some trained networks/balanced-38/v3p3/cifar/k=3-fold-cifar-multi-class-subsampled-31-May-2017-07-07-19-single-cnn/cnn-31-May-2017-07-07-20-cifar-multi-class-subsampled-convV3P3-RF32CH3+fcV1-RF4CH64-batch-size-100-weight-decay-0.0100-GPU-2-bpd-13/net-epoch-100.mat');
  %   path_2 = fullfile(datapath, 'Parent11.mat');
  % else
  %   path_1 = '/Volumes/Amir/some trained networks/balanced-38/v3p3/cifar/k=3-fold-cifar-multi-class-subsampled-31-May-2017-07-07-19-single-cnn/cnn-31-May-2017-07-07-20-cifar-multi-class-subsampled-convV3P3-RF32CH3+fcV1-RF4CH64-batch-size-100-weight-decay-0.0100-GPU-2-bpd-13/net-epoch-100.mat';
  %   path_2 = '/Volumes/Amir/Parent11.mat';
  %   path_convV1P0RL0 = '/Volumes/Amir/some trained networks/larpV0P0RL0+convV1P0RL0/cifar/balance-38/k=3-fold-cifar-multi-class-subsampled-7-Jun-2017-17-14-44-single-cnn/cnn-7-Jun-2017-17-14-45-cifar-multi-class-subsampled-convV1P0RL0-RF32CH3+fcV1-RF32CH64-batch-size-100-weight-decay-0.0001-GPU-1-bpd-05/net-epoch-100.mat';
  %   path_convV1P0RL1 = '/Volumes/Amir/some trained networks/larpV0P0RL0+convV1P0RL1/cifar/balance-38/k=3-fold-cifar-multi-class-subsampled-7-Jun-2017-23-56-51-single-cnn/cnn-7-Jun-2017-23-56-52-cifar-multi-class-subsampled-convV1P0RL1-RF32CH3+fcV1-RF32CH64-batch-size-50-weight-decay-0.0010-GPU-1-bpd-06/net-epoch-100.mat';
  %   path_convV3P0RL0 = '/Volumes/Amir/some trained networks/larpV0P0RL0+convV3P0RL0/cifar/balance-38/k=3-fold-cifar-multi-class-subsampled-9-Jun-2017-02-46-18-single-cnn/cnn-9-Jun-2017-02-46-19-cifar-multi-class-subsampled-convV3P0RL0-RF32CH3+fcV1-RF32CH64-batch-size-100-weight-decay-0.0001-GPU-1-bpd-07/net-epoch-100.mat';
  %   path_convV3P0RL3 = '/Volumes/Amir/some trained networks/larpV0P0RL0+convV3P0RL3/cifar/balance-38/k=3-fold-cifar-multi-class-subsampled-8-Jun-2017-21-13-14-single-cnn/cnn-8-Jun-2017-21-13-15-cifar-multi-class-subsampled-convV3P0RL3-RF32CH3+fcV1-RF32CH64-batch-size-100-weight-decay-0.0010-GPU-2-bpd-10/net-epoch-100.mat';
  %   path_convV3P3RL0 = '/Volumes/Amir/some trained networks/larpV0P0RL0+convV3P3RL0/cifar/balance-38/k=3-fold-cifar-multi-class-subsampled-7-Jun-2017-05-36-41-single-cnn/cnn-7-Jun-2017-05-36-42-cifar-multi-class-subsampled-convV3P3RL0-RF32CH3+fcV1-RF4CH64-batch-size-50-weight-decay-0.0010-GPU-3-bpd-10/net-epoch-100.mat';
  %   path_convV3P3RL3 = '/Volumes/Amir/some trained networks/larpV0P0RL0+convV3P3RL3/cifar/balance-38/k=3-fold-cifar-multi-class-subsampled-7-Jun-2017-13-35-35-single-cnn/cnn-7-Jun-2017-13-35-36-cifar-multi-class-subsampled-convV3P3RL3-RF32CH3+fcV1-RF4CH64-batch-size-50-weight-decay-0.0100-GPU-3-bpd-13/net-epoch-100.mat';
  % end
