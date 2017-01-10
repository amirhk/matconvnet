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

  afprintf(sprintf('[INFO] Constructing unbalanced `%s` imdbs...\n', dataset));
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
    case 'svhn'
      opts.imdb.contrast_normalization = true;
      all_class_imdb = constructSvhnImdb(opts);
  end

  fh_imdb_utils = imdbTwoClassUtils;

  % -------------------------------------------------------------------------
  %                                                              balanced-low
  % -------------------------------------------------------------------------
  posneg_balance = 'balanced-low';
  afprintf(sprintf('[INFO] `%s`...\n', posneg_balance));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 200);
  imdb = fh_imdb_utils.balanceImdb(imdb, 'train', 'downsample');
  fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  afprintf(sprintf('done!\n\n'));

  % -------------------------------------------------------------------------
  %                                                                unbalanced
  % -------------------------------------------------------------------------
  posneg_balance = 'unbalanced';
  afprintf(sprintf('[INFO] `%s`...\n', posneg_balance));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 200);
  fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  afprintf(sprintf('done!\n\n'));

  % -------------------------------------------------------------------------
  %                                                             balanced-high
  % -------------------------------------------------------------------------
  posneg_balance = 'balanced-high';
  afprintf(sprintf('[INFO] `%s`...\n', posneg_balance));
  imdb = fh_imdb_utils.constructTwoClassUnbalancedImdb(all_class_imdb, positive_class_number, negative_class_number, 1);
  fh_imdb_utils.saveImdb(imdb, dataset, posneg_balance, positive_class_number, negative_class_number)
  afprintf(sprintf('done!\n\n'));
