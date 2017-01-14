%-------------------------------------------------------------------------
function constructProstateImdbs(input_opts)
  % TODO: this can be extended to support both v2 and v3, with different fold counts....
%-------------------------------------------------------------------------
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

  posneg_balance                   = getValueFromFieldOrDefault(input_opts, 'posneg_balance', 'unbalanced');
  imdb_opts.dataset_version        = getValueFromFieldOrDefault(input_opts, 'dataset_version', 'v2-20-patients');
  switch imdb_opts.dataset_version
    case 'v2-20-patients'
      number_of_patients = 20;
      imdb_opts.dataDir            = getValueFromFieldOrDefault(input_opts, 'dataDir', fullfile(getDevPath(), 'data/source/prostate/v2-20-patients'));
    case 'v3-104-patients'
      number_of_patients = 104;
      imdb_opts.dataDir            = getValueFromFieldOrDefault(input_opts, 'dataDir', fullfile(getDevPath(), 'data/source/prostate/v3-104-patients'));
  end

  imdb_opts.leave_out_type         = getValueFromFieldOrDefault(input_opts, 'leave_out_type', 'special');

  switch posneg_balance
    case 'balanced-low'
      % no augment, just balance
      imdb_opts.train_augment_healthy  = getValueFromFieldOrDefault(input_opts, 'train_augment_healthy', 'none');
      imdb_opts.train_augment_cancer   = getValueFromFieldOrDefault(input_opts, 'train_augment_cancer', 'none');
      imdb_opts.train_balance          = getValueFromFieldOrDefault(input_opts, 'train_balance', true);
    case 'unbalanced'
      imdb_opts.train_augment_healthy  = getValueFromFieldOrDefault(input_opts, 'train_augment_healthy', 'none');
      imdb_opts.train_augment_cancer   = getValueFromFieldOrDefault(input_opts, 'train_augment_cancer', 'none');
      imdb_opts.train_balance          = getValueFromFieldOrDefault(input_opts, 'train_balance', false);
    case 'balanced-high'
      % first augment then balance
      imdb_opts.train_augment_healthy  = getValueFromFieldOrDefault(input_opts, 'train_augment_healthy', 'none');
      imdb_opts.train_augment_cancer   = getValueFromFieldOrDefault(input_opts, 'train_augment_cancer', 'rotate');
      imdb_opts.train_balance          = getValueFromFieldOrDefault(input_opts, 'train_balance', true);
    otherwise
      fprintf('TODO: implement!');
  end
  imdb_opts.test_balance           = getValueFromFieldOrDefault(input_opts, 'test_balance', false);
  imdb_opts.test_augment_healthy   = getValueFromFieldOrDefault(input_opts, 'test_augment_healthy', 'none');
  imdb_opts.test_augment_cancer    = getValueFromFieldOrDefault(input_opts, 'test_augment_cancer', 'none');
  imdb_opts.contrast_normalization = getValueFromFieldOrDefault(input_opts, 'contrast_normalization', true);

  leave_one_out = true;
  if leave_one_out
    number_of_folds = number_of_patients;
  else
    number_of_folds = 10;
  end
  patients_per_fold = ceil(number_of_patients / number_of_folds);
  fh_imdb_utils = imdbTwoClassUtils;
  % WARNING: this is a new order every time, very unlikely to replicate numbers from prev. run.
  random_patient_indices = randperm(number_of_patients);
  for i = 1:number_of_folds
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Randomly dividing for fold #%d / %d...\n', i, number_of_folds));
    start_index = 1 + (i - 1) * patients_per_fold;
    end_index = min(number_of_patients, i * patients_per_fold);
    afprintf(sprintf('[INFO] done!\n'));
    afprintf(sprintf('[INFO] Constructing imdb for fold #%d / %d...\n', i, number_of_folds));
    imdb_opts.leave_out_indices = random_patient_indices(start_index : end_index);
    imdb = constructProstateImdb(imdb_opts);
    if leave_one_out
      single_patient_index = start_index;
      fh_imdb_utils.saveImdb(imdb, sprintf('prostate-%s', imdb_opts.dataset_version), sprintf('patient-%02d-%s', single_patient_index, posneg_balance), 2, 1);
    else
      fh_imdb_utils.saveImdb(imdb, sprintf('prostate-%s', imdb_opts.dataset_version), posneg_balance, 2, 1);
    end
  end
