% --------------------------------------------------------------------
function constructTwoClassImdbs(dataset, network_arch, positive_class_number, negative_class_number)
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

  afprintf(sprintf('[INFO] Constructing two-class `%s` imdbs...\n', dataset));
  opts.imdb.data_dir = fullfile(getDevPath(), 'data', 'source', dataset);
  switch dataset
    case 'mnist'
      opts.general.network_arch = network_arch;
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
      all_class_imdb = constructSvhnImdb(opts);
  end



  % % -------------------------------------------------------------------------
  % %                                                              balanced-low
  % % -------------------------------------------------------------------------
  % posneg_balance = 'balanced-low';
  % afprintf(sprintf('[INFO] `%s`...\n', posneg_balance));
  % imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 200);
  % imdb = fh_imdb_utils.balanceImdb(imdb, 'train', 'downsample');
  % imdb = fh_imdb_utils.balanceImdb(imdb, 'test', 'downsample'); % all test sets should be balanced so acc = avg(sens, spec)
  % fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  % afprintf(sprintf('done!\n\n'));

  % % -------------------------------------------------------------------------
  % %                                                                unbalanced
  % % -------------------------------------------------------------------------
  % posneg_balance = 'unbalanced';
  % afprintf(sprintf('[INFO] `%s`...\n', posneg_balance));
  % imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 200);
  % imdb = fh_imdb_utils.balanceImdb(imdb, 'test', 'downsample'); % all test sets should be balanced so acc = avg(sens, spec)
  % fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  % afprintf(sprintf('done!\n\n'));

  % % -------------------------------------------------------------------------
  % %                                                             balanced-high
  % % -------------------------------------------------------------------------
  % posneg_balance = 'balanced-high';
  % afprintf(sprintf('[INFO] `%s`...\n', posneg_balance));
  % imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 1);
  % imdb = fh_imdb_utils.balanceImdb(imdb, 'test', 'downsample'); % all test sets should be balanced so acc = avg(sens, spec)
  % fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  % afprintf(sprintf('done!\n\n'));


  % logspace(1 + log10(3.76), 3 + log10(5), 6)
  % -------------------------------------------------------------------------
  % balance_count = 38; createImdbWithBalance(balance_count, dataset, all_class_imdb, positive_class_number, negative_class_number);
  % balance_count = 100; createImdbWithBalance(balance_count, dataset, all_class_imdb, positive_class_number, negative_class_number);
  % balance_count = 266; createImdbWithBalance(balance_count, dataset, all_class_imdb, positive_class_number, negative_class_number);
  % balance_count = 707; createImdbWithBalance(balance_count, dataset, all_class_imdb, positive_class_number, negative_class_number);
  % balance_count = 1880; createImdbWithBalance(balance_count, dataset, all_class_imdb, positive_class_number, negative_class_number);
  balance_count = 5000; createImdbWithBalance(balance_count, dataset, all_class_imdb, positive_class_number, negative_class_number);

  % balance_count = 500; createImdbWithBalance(balance_count, dataset, all_class_imdb, positive_class_number, negative_class_number);


function createImdbWithBalance(balance_count, dataset, all_class_imdb, positive_class_number, negative_class_number)
  fh_imdb_utils = imdbTwoClassUtils;
  posneg_balance = sprintf('balanced-%d-%d', balance_count, balance_count);
  afprintf(sprintf('[INFO] Constructing `%s`...\n', posneg_balance));
  imdb = fh_imdb_utils.constructTwoClassImdbFromMultiClassImdb(all_class_imdb, positive_class_number, negative_class_number);
  imdb = fh_imdb_utils.subsampleImdb(imdb, 'train', 'positive', balance_count);
  imdb = fh_imdb_utils.subsampleImdb(imdb, 'train', 'negative', balance_count);
  imdb = fh_imdb_utils.balanceImdb(imdb, 'test', 'downsample'); % all test sets should be balanced so acc = avg(sens, spec)
  fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  afprintf(sprintf('done!\n\n'));

















