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
  opts.train.gpus = 4;


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


  % dataset_name = sprintf('%s-multi-class-subsampled', opts.general.dataset);
  % balance_name = 'balanced-38';
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'no-projection',                opts.train.gpus);

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P0SF', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P1SF', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P0SF', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P1SF', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P3SF', opts.train.gpus);

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P0ST', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P1ST', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P0ST', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P1ST', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P3ST', opts.train.gpus);


  % -------------------------------------------------------------------------s


  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'no-projection',                opts.train.gpus);

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P0SF', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P1SF', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P0SF', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P1SF', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P3SF', opts.train.gpus);

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P0ST', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P1ST', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P0ST', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P1ST', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P3ST', opts.train.gpus);







  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'no-projection', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'no-projection', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'no-projection', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'no-projection', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'no-projection', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'no-projection', opts.train.gpus);

  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P0SF', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV1P1SF', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P0SF', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P1SF', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'projected-through-larpV3P3SF', opts.train.gpus);











  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';
  dataset_name = sprintf('%s-multi-class-subsampled', opts.general.dataset);
  balance_name = 'balanced-707';
  % balance_name = 'balanced-1880';
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV0P0+convV0P0+fcV1', 'NA',                                               'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussian',                                         'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3',                               'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-Cov-Sampling-ScaleUp-3',        'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-Cov-Sampling',                  'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-Cov-Sampling-ScaleDown-3',      'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-Cov-Sampling-ScaleDown-10',     'no-projection', opts.train.gpus);

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-Cov-Sampling-ScaleDown-100',    'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-Cov-Sampling-ScaleDown-1000',   'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-Cov-Sampling-ScaleDown-10000',  'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-Cov-Sampling-ScaleDown-100000', 'no-projection', opts.train.gpus);

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'mixedKernelsWithGaussianIdentityCovariance-MuDivide-1-SigmaDivide-1',    'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'mixedKernelsWithGaussianIdentityCovariance-MuDivide-1-SigmaDivide-10',   'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'mixedKernelsWithGaussianIdentityCovariance-MuDivide-1-SigmaDivide-100',  'no-projection', opts.train.gpus);

  runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'mixedKernelsWithGaussianIdentityCovariance-MuDivide-10-SigmaDivide-1',   'no-projection', opts.train.gpus);
  runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'mixedKernelsWithGaussianIdentityCovariance-MuDivide-10-SigmaDivide-10',  'no-projection', opts.train.gpus);
  runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'mixedKernelsWithGaussianIdentityCovariance-MuDivide-10-SigmaDivide-100', 'no-projection', opts.train.gpus);







  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-Cov-Sampling-TEST',             'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-4',                               'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-4-Cov-Sampling',                  'no-projection', opts.train.gpus);

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianAnisoDiffed-2',                            'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianAnisoDiffed-4',                            'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianAnisoDiffed-6',                            'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianAnisoDiffed-8',                            'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussian-mult2DGaussian',                          'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianSmoothed-3-mult2DGaussian',                'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussianAnisoDiffed-2-mult2DGaussian',             'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'bernoulli',                                        'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'bernoulliSmoothed-3',                              'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'bernoulliAnisoDiffed-2',                           'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussian2D',                                       'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussian2DMeanSubtracted',                         'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV1P1+convV0P0+fcV1', 'gaussian2DMeanSubtractedRandomlyFlipped',          'no-projection', opts.train.gpus);





  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV0P0+convV0P0+fcV2', 'NA',                                      'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussian',                                'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianSmoothed-3',                      'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianSmoothed-4',                      'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianSmoothed-3-Cov-Sampling',         'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianSmoothed-3-Cov-Sampling-TEST',    'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianSmoothed-4-Cov-Sampling',         'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianAnisoDiffed-2',                   'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianAnisoDiffed-4',                   'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianAnisoDiffed-6',                   'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianAnisoDiffed-8',                   'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussian-mult2DGaussian',                 'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianSmoothed-3-mult2DGaussian',       'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussianAnisoDiffed-2-mult2DGaussian',    'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'bernoulli',                               'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'bernoulliSmoothed-3',                     'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'bernoulliAnisoDiffed-2',                  'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussian2D',                              'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussian2DMeanSubtracted',                'no-projection', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV2', 'gaussian2DMeanSubtractedRandomlyFlipped', 'no-projection', opts.train.gpus);



  % % dataset_name = sprintf('%s-multi-class-subsampled', opts.general.dataset);
  % % balance_name = 'balanced-38';
  % dataset_name = sprintf('%s', opts.general.dataset);
  % balance_name = 'whatever';
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'larpV3P1+convV0P0+fcV1', 'gaussian', 'no-projection',                opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'convV0P0+fcV1RF16CH64',  'gaussian', 'projected-through-larpV3P1SF', opts.train.gpus);

  % % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'convV0P0+fcV1RF16CH64',  'gaussian', 'projected-through-larpV3P1SF-noflag', opts.train.gpus);
  % % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'convV0P0+fcV1RF16CH64',  'gaussian', 'projected-through-larpV3P1SF-73flag', opts.train.gpus);

  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'convV0P0+fcV1RF4CH64',  'gaussian', 'projected-through-larpV3P3SF-noflag', opts.train.gpus);
  % % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'convV0P0+fcV1RF4CH64',  'gaussian', 'projected-through-larpV3P3SF-73flag', opts.train.gpus);

  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'convV0P0+fcV1RF16CH64',  'gaussian', 'projected-through-larpV1P1-gaussian', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'convV0P0+fcV1RF16CH64',  'gaussian', 'projected-through-larpV3P1-gaussian', opts.train.gpus);
  % runLarpTests(opts.paths.experiment_dir, dataset_name, balance_name, 'convV0P0+fcV1RF4CH64',   'gaussian', 'projected-through-larpV3P3-gaussian', opts.train.gpus);











