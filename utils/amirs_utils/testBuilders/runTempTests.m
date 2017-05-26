% -------------------------------------------------------------------------
function runTempTests()
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
% % POSSIBILITY OF SUCH DAMAGE.


  % -------------------------------------------------------------------------
  %                                                              opts.general
  % -------------------------------------------------------------------------
  opts.general.dataset = 'cifar';

  % -------------------------------------------------------------------------
  %                                                                opts.train
  % -------------------------------------------------------------------------
  opts.train.gpus = 2;


  % -------------------------------------------------------------------------
  %                                                                opts.paths
  % -------------------------------------------------------------------------
  opts.paths.time_string = sprintf('%s',datetime('now', 'Format', 'd-MMM-y-HH-mm-ss'));
  opts.paths.experiment_parent_dir = getValueFromFieldOrDefault( ...
    {}, ... % no input_opts here! :)
    'experiment_parent_dir', ...
    fullfile(vl_rootnn, 'experiment_results'));
  opts.paths.experiment_dir = fullfile(opts.paths.experiment_parent_dir, sprintf( ...
    'temp-test-%s-GPU-%d', ...
    opts.paths.time_string, ...
    opts.train.gpus));
  if ~exist(opts.paths.experiment_dir)
    mkdir(opts.paths.experiment_dir);
  end
  opts.paths.options_file_path = fullfile(opts.paths.experiment_dir, 'options.txt');
  % opts.paths.results_file_path = fullfile(opts.paths.experiment_dir, 'results.txt');

  % -------------------------------------------------------------------------
  %                                                    save experiment setup!
  % -------------------------------------------------------------------------
  saveStruct2File(opts, opts.paths.options_file_path, 0);



  % larp_network_arch = 'larpV0P0-single-dense-rp-no-nl';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % larp_network_arch = 'larpV1P0-single-sparse-rp-no-nl';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-no-nl';
  % non_larp_network_arch = 'convV0P0+fcV1';



  % larp_network_arch = 'larpV1P0';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % larp_network_arch = 'larpV1P1';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % larp_network_arch = 'larpV1P1-non-decimated-pooling';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % larp_network_arch = 'larpV3P1';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % larp_network_arch = 'larpV3P3';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % larp_network_arch = 'larpV5P3';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % larp_network_arch = 'larpV5P3';
  % non_larp_network_arch = 'convV0P0+fcV2';



  % dataset_name = sprintf('%s-multi-class-subsampled', opts.general.dataset);
  % balance_name = 'balanced-38';

  % dataset_name = sprintf('%s-multi-class-subsampled', opts.general.dataset);
  % balance_name = 'balanced-707';

  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1',                          non_larp_network_arch, opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, 'gaussian-SmoothedCovariance-3-MuDivide-1-SigmaDivide-1',                        non_larp_network_arch, opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, 'gaussian-CentreSurroundCovariance-randomDivide-10-MuDivide-1-SigmaDivide-1',    non_larp_network_arch, opts.train.gpus);








  % larp_network_arch = 'larpV0P0-single-dense-rp-no-nl';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1',                          non_larp_network_arch, opts.train.gpus);

  % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-no-nl';
  % non_larp_network_arch = 'convV0P0+fcV1';

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1',                          non_larp_network_arch, opts.train.gpus);











  % opts.general.dataset = 'cifar';
  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';
  % % larp_network_arch = 'larpV0P0';
  % % larp_network_arch = 'larpV0P0-single-dense-rp-no-nl';
  % larp_network_arch = 'larpV1P0-single-sparse-rp-no-nl';
  % % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-no-nl';
  % non_larp_network_arch = 'convV0P0+fcV1';
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1',                          non_larp_network_arch, opts.train.gpus);

  % opts.general.dataset = 'mnist';
  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';
  % % larp_network_arch = 'larpV0P0';
  % % larp_network_arch = 'larpV0P0-single-dense-rp-no-nl';
  % larp_network_arch = 'larpV1P0-single-sparse-rp-no-nl';
  % % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-no-nl';
  % non_larp_network_arch = 'convV0P0+fcV1';
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1',                          non_larp_network_arch, opts.train.gpus);

  % opts.general.dataset = 'svhn';
  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';
  % % larp_network_arch = 'larpV0P0';
  % % larp_network_arch = 'larpV0P0-single-dense-rp-no-nl';
  % larp_network_arch = 'larpV1P0-single-sparse-rp-no-nl';
  % % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-no-nl';
  % non_larp_network_arch = 'convV0P0+fcV1';
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1',                          non_larp_network_arch, opts.train.gpus);

  % opts.general.dataset = 'stl-10';
  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';
  % % larp_network_arch = 'larpV0P0';
  % % larp_network_arch = 'larpV0P0-single-dense-rp-no-nl';
  % larp_network_arch = 'larpV1P0-single-sparse-rp-no-nl';
  % % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-no-nl';
  % non_larp_network_arch = 'convV0P0+fcV1';
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1',                          non_larp_network_arch, opts.train.gpus);















  % balance_name = 'whatever';
  balance_name = 'balanced-38';



  % larp_network_arch = 'larpV0P0';
  % larp_network_arch = 'larpV0P0-single-dense-rp-no-nl';
  % larp_network_arch = 'larpV1P0-single-sparse-rp-no-nl';
  % larp_network_arch = 'larpV1P0-ensemble-sparse-rp-no-nl';
  % larp_network_arch = 'larpV1P1';
  larp_network_arch = 'larpV3P1';
  % larp_network_arch = 'larpV3P3';



  non_larp_network_arch = 'convV0P0+fcV1';



  larp_weight_init_type = 'gaussian-IdentityCovariance-MuDivide-1-SigmaDivide-1';
  % larp_weight_init_type = 'gaussian-SmoothedCovariance-3-MuDivide-1-SigmaDivide-1';
  % larp_weight_init_type = 'gaussian-CentreSurroundCovariance-randomDivide-10-MuDivide-1-SigmaDivide-1';
  % larp_weight_init_type = 'logNormal-layer5-ratVisualCortex';


  % dataset_name = 'cifar';    runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, non_larp_network_arch, larp_weight_init_type, opts.train.gpus);
  % dataset_name = 'mnist';    runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, non_larp_network_arch, larp_weight_init_type, opts.train.gpus);
  % dataset_name = 'svhn';   runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, non_larp_network_arch, larp_weight_init_type, opts.train.gpus);
  % dataset_name = 'stl-10'; runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, non_larp_network_arch, larp_weight_init_type, opts.train.gpus);


  dataset_name = 'cifar-multi-class-subsampled';    runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, non_larp_network_arch, larp_weight_init_type, opts.train.gpus);
  dataset_name = 'mnist-multi-class-subsampled';    runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, non_larp_network_arch, larp_weight_init_type, opts.train.gpus);
  dataset_name = 'svhn-multi-class-subsampled';   runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, non_larp_network_arch, larp_weight_init_type, opts.train.gpus);
  % dataset_name = 'stl-10-multi-class-subsampled'; runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, larp_network_arch, non_larp_network_arch, larp_weight_init_type, opts.train.gpus);
























