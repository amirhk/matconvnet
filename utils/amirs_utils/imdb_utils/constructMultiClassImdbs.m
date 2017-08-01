% --------------------------------------------------------------------
function constructMultiClassImdbs(dataset)
% --------------------------------------------------------------------
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

  afprintf(sprintf('[INFO] Constructing multi-class `%s` imdbs...\n', dataset));
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', dataset);
  switch dataset
  case 'pathology'
      all_class_imdb = constructPathologyImdb(opts);
    case 'imagenet-tiny'
      all_class_imdb = constructImageNetTinyImdb(opts);
    case 'mnist'
      opts.general.network_arch = 'lenet';
      all_class_imdb = constructMnistImdb(opts);
    case 'mnist-784'
      opts.general.network_arch = 'mnistnet';
      all_class_imdb = constructMnistImdb(opts);
    case 'cifar'
      opts.imdb.imdb_portion = 1.0;
      opts.imdb.contrast_normalization = true;
      opts.imdb.whiten_data = true;
      all_class_imdb = constructCifarImdb(opts);
    case 'cifar-no-white'
      opts.imdb.imdb_portion = 1.0;
      opts.imdb.contrast_normalization = true;
      opts.imdb.whiten_data = false;
      all_class_imdb = constructCifarImdb(opts);
    case 'stl-10'
      all_class_imdb = constructStl10Imdb(opts);
    case 'svhn'
      opts.imdb.contrast_normalization = true;
      opts.imdb.whiten_data = false;
      all_class_imdb = constructSvhnImdb(opts);
    case 'svhn-yes-white'
      opts.imdb.contrast_normalization = true;
      opts.imdb.whiten_data = true;
      all_class_imdb = constructSvhnImdb(opts);
    case 'uci-ion'
      all_class_imdb = constructUCIIonImdb(opts);
    case 'uci-spam'
      all_class_imdb = constructUCISpamImdb(opts);


    case 'gaussian-2D-mean-1-var-0'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 2, 1, 0);
    case 'gaussian-2D-mean-1-var-1'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 2, 1, 1);
    case 'gaussian-2D-mean-1-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 2, 1, 10);

    case 'gaussian-2D-mean-3-var-3'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 2, 3, 3);

    case 'gaussian-3D-mean-3-var-3'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 3, 3, 3);

    case 'gaussian-3D-mean-1-var-1-diag'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 3, 1, 1);
    case 'gaussian-3D-mean-3-var-1-diag'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 3, 3, 1);

    % case 'gaussian-5D-mean-1-var-.1'
    %   all_class_imdb = constructSyntheticGaussianImdb(1100, 5, 1, .1);
    case 'gaussian-5D-mean-1-var-0'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 5, 1, 0);
    case 'gaussian-5D-mean-1-var-1'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 5, 1, 1);
    case 'gaussian-5D-mean-1-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 5, 1, 10);

    case 'gaussian-5D-mean-9-var-0'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 5, 1, 0);
    case 'gaussian-5D-mean-9-var-1'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 5, 1, 1);
    case 'gaussian-5D-mean-9-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 5, 1, 10);

    case 'gaussian-50D-mean-1-var-0'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 1, 0);
    case 'gaussian-50D-mean-1-var-1'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 1, 1);
    case 'gaussian-50D-mean-1-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 1, 10);

    case 'gaussian-50D-mean-1-var-1-diag'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 1, 1);
    case 'gaussian-50D-mean-3-var-1-diag'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 3, 1);
    case 'gaussian-50D-mean-3-var-1-diag2'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 3, 1);

    case 'gaussian-50D-mean-9-var-0'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 1, 0);
    case 'gaussian-50D-mean-9-var-1'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 1, 1);
    case 'gaussian-50D-mean-9-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 1, 10);

    case 'gaussian-50D-mean-1-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(5500, 50, 1, 10);




    case 'gaussian-3D-mean-1-var-1'
      % all_class_imdb = constructSyntheticGaussianImdb(1100, 3, 1, 1);
      all_class_imdb = constructSyntheticGaussianImdb(11000, 3, 1, 1);
      % all_class_imdb = constructSyntheticGaussianImdb(110000, 3, 1, 1);
    case 'gaussian-50D-mean-1-var-1'
      % all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 1, 1);
      all_class_imdb = constructSyntheticGaussianImdb(11000, 50, 1, 1);
      % all_class_imdb = constructSyntheticGaussianImdb(110000, 50, 1, 1);
    case 'gaussian-250D-mean-1-var-1'
      % all_class_imdb = constructSyntheticGaussianImdb(1100, 250, 1, 1);
      all_class_imdb = constructSyntheticGaussianImdb(11000, 250, 1, 1);
      % all_class_imdb = constructSyntheticGaussianImdb(110000, 250, 1, 1);
    case 'gaussian-1000D-mean-1-var-1'
      % all_class_imdb = constructSyntheticGaussianImdb(1100, 1000, 1, 1);
      all_class_imdb = constructSyntheticGaussianImdb(11000, 1000, 1, 1);
      % all_class_imdb = constructSyntheticGaussianImdb(110000, 1000, 1, 1);
    case 'gaussian-2500D-mean-1-var-1'
      % all_class_imdb = constructSyntheticGaussianImdb(1100, 2500, 1, 1);
      all_class_imdb = constructSyntheticGaussianImdb(11000, 2500, 1, 1);
      % all_class_imdb = constructSyntheticGaussianImdb(110000, 2500, 1, 1);




    case 'gaussian-3D-mean-1-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 3, 1, 10);
    case 'gaussian-50D-mean-1-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 50, 1, 10);
    case 'gaussian-250D-mean-1-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 250, 1, 10);
    case 'gaussian-1000D-mean-1-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 1000, 1, 10);
    case 'gaussian-2500D-mean-1-var-10'
      all_class_imdb = constructSyntheticGaussianImdb(1100, 2500, 1, 10);




    case 'circles-3D-mean-0-var-1-radius-1'
      all_class_imdb = constructSyntheticCirclesImdb(5000, 3, 0, 1);
    case 'circles-50D-mean-0-var-1-radius-1'
      all_class_imdb = constructSyntheticCirclesImdb(5000, 50, 0, 1);
    case 'circles-250D-mean-0-var-1-radius-1'
      all_class_imdb = constructSyntheticCirclesImdb(5000, 250, 0, 1);
    case 'circles-1000D-mean-0-var-1-radius-1'
      all_class_imdb = constructSyntheticCirclesImdb(5000, 1000, 0, 1);
    case 'circles-2500D-mean-0-var-1-radius-1'
      all_class_imdb = constructSyntheticCirclesImdb(5000, 2500, 0, 1);


    case 'spirals-2D'
      all_class_imdb = constructSyntheticSpiralsImdb(2000);
  end
  keyboard

  % OLD: logspace(1 + log10(3.76), 3 + log10(5), 6): [38, 100, 266, 1880, 5000]
  % NEW:                                             [10, 50, 100, 250, 500, 1000, 2500]
  % -------------------------------------------------------------------------
  % createImdbWithBalance(dataset, all_class_imdb, 10, 'default');
  % createImdbWithBalance(dataset, all_class_imdb, 50, 'default');
  % createImdbWithBalance(dataset, all_class_imdb, 100, 'default');
  % createImdbWithBalance(dataset, all_class_imdb, 250, 'default');
  % createImdbWithBalance(dataset, all_class_imdb, 500, 'default');
  % createImdbWithBalance(dataset, all_class_imdb, 1000, 'default');
  % createImdbWithBalance(dataset, all_class_imdb, 2500, 'default');

  % createImdbWithBalance(dataset, all_class_imdb, 500, 500);
  % createImdbWithBalance(dataset, all_class_imdb, 5000, 5000);
  % createImdbWithBalance(dataset, all_class_imdb, 50000, 50000);


function createImdbWithBalance(dataset, imdb, train_balance_count, test_balance_count)
  fh_imdb_utils = imdbMultiClassUtils;
  % posneg_balance = sprintf('balanced-%d-%d', balance_count, balance_count);
  % afprintf(sprintf('[INFO] Constructing `%s`...\n', posneg_balance));
  printConsoleOutputSeparator();
  afprintf(sprintf('[INFO] INITIAL IMDB INFO...\n'));
  fh_imdb_utils.getImdbInfo(imdb, 1);
  imdb = fh_imdb_utils.balanceAllClassesInImdb(imdb, 'train', train_balance_count);
  imdb = fh_imdb_utils.balanceAllClassesInImdb(imdb, 'test', test_balance_count);
  afprintf(sprintf('[INFO] FINAL IMDB INFO...\n'));
  fh_imdb_utils.getImdbInfo(imdb, 1);
  % Save
  fh_imdb_utils.saveImdb(dataset, imdb, train_balance_count, test_balance_count)
  afprintf(sprintf('done!\n\n'));

















