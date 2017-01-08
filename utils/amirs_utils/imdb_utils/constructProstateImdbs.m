%-------------------------------------------------------------------------
function constructProstateImdbs(input_opts)
  % TODO: this can be extended to support both v2 and v3, with different fold counts....
%-------------------------------------------------------------------------
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

  number_of_folds = 5;
  switch imdb_opts.dataset
    case 'prostate-v2-20-patients'
      number_of_patients = 20;
    case 'prostate-v3-104-patients'
      number_of_patients = 104;
  end

  fh_imdb_utils = imdbTwoClassUtils;
  patients_per_fold = ceil(number_of_patients / number_of_folds);
  % WARNING: this is a new order every time, very unlikely to replicate numbers from prev. run.
  random_patient_indices = randperm(number_of_patients);
  for i = 1:number_of_folds
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Randomly dividing for fold #%d...\n', i));
    start_index = 1 + (i - 1) * patients_per_fold;
    end_index = min(104, i * patients_per_fold);
    folds.(sprintf('fold_%d', i)).patient_indices = random_patient_indices(start_index : end_index);
    afprintf(sprintf('[INFO] done!\n'));
    afprintf(sprintf('[INFO] Constructing imdb for fold #%d...\n', i));
    imdb_opts.leave_out_indices = folds.(sprintf('fold_%d', i)).patient_indices;
    imdb = constructProstateImdb(imdb_opts);
    fh_imdb_utils.saveImdb(imdb, imdb_opts.dataset, posneg_balance, 2, 1)
  end
