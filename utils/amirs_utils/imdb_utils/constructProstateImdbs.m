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

  posneg_balance                   = getValueFromFieldOrDefault(input_opts, 'posneg_balance', 'unbalaced');
  imdb_opts.dataset                = getValueFromFieldOrDefault(input_opts, 'dataset', 'prostate-v2-20-patients');
  imdb_opts.dataDir                = getValueFromFieldOrDefault(input_opts, 'dataDir', '/Users/a6karimi/dev/data/source/prostate/v2 - 20 patients');
  imdb_opts.leave_out_type         = getValueFromFieldOrDefault(input_opts, 'leave_out_type', 'special');
  imdb_opts.train_balance          = getValueFromFieldOrDefault(input_opts, 'train_balance', false);
  imdb_opts.train_augment_healthy  = getValueFromFieldOrDefault(input_opts, 'train_augment_healthy', 'none');
  switch posneg_balance
    case 'unbalaced'
      imdb_opts.train_augment_cancer   = getValueFromFieldOrDefault(input_opts, 'train_augment_cancer', 'none');
    case 'balanced-high'
      imdb_opts.train_augment_cancer   = getValueFromFieldOrDefault(input_opts, 'train_augment_cancer', 'rotate');
    otherwise
      fprintf('TODO: implement!');
  end
  imdb_opts.test_balance           = getValueFromFieldOrDefault(input_opts, 'test_balance', false);
  imdb_opts.test_augment_healthy   = getValueFromFieldOrDefault(input_opts, 'test_augment_healthy', 'none');
  imdb_opts.test_augment_cancer    = getValueFromFieldOrDefault(input_opts, 'test_augment_cancer', 'none');
  imdb_opts.contrast_normalization = getValueFromFieldOrDefault(input_opts, 'contrast_normalization', true);

  switch imdb_opts.dataset
    case 'prostate-v2-20-patients'
      number_of_patients = 20;
    case 'prostate-v3-104-patients'
      number_of_patients = 104;
  end

  leave_one_out = false;
  if leave_one_out
    number_of_folds = 20;
  else
    number_of_folds = 5;
  end
  patients_per_fold = ceil(number_of_patients / number_of_folds);
  fh_imdb_utils = imdbTwoClassUtils;
  % WARNING: this is a new order every time, very unlikely to replicate numbers from prev. run.
  random_patient_indices = randperm(number_of_patients);
  for i = 1:number_of_folds
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Randomly dividing for fold #%d...\n', i));
    start_index = 1 + (i - 1) * patients_per_fold;
    end_index = min(number_of_patients, i * patients_per_fold);
    afprintf(sprintf('[INFO] done!\n'));
    afprintf(sprintf('[INFO] Constructing imdb for fold #%d...\n', i));
    imdb_opts.leave_out_indices = random_patient_indices(start_index : end_index);
    imdb = constructProstateImdb(imdb_opts);
    if leave_one_out
      single_patient_index = start_index;
      fh_imdb_utils.saveImdb(imdb, imdb_opts.dataset, sprintf('patient-%d-%s', single_patient_index, posneg_balance), 2, 1);
    else
      fh_imdb_utils.saveImdb(imdb, imdb_opts.dataset, posneg_balance, 2, 1);
    end
  end
