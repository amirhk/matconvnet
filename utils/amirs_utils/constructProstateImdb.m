% -------------------------------------------------------------------------
function imdb = constructProstateImdb(opts)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing / loading Prostate imdb.\n'));
  all_patient_indices = 1:1:104;
  switch opts.leaveOutType
    case 'special'
      train_patient_indices = setdiff(all_patient_indices, opts.leaveOutIndices);
      train_balance = false;
      train_augment_healthy = 'none';
      train_augment_cancer = 'none';
      test_patient_indices = opts.leaveOutIndices;
      test_balance = false;
      test_augment_healthy = 'none';
      test_augment_cancer = 'none';
      % just use the helper with the indices above.
      imdb = constructProstateImdbHelper( ...
        opts, ...
        train_patient_indices, ...
        train_balance, ...
        train_augment_healthy, ...
        train_augment_cancer, ...
        test_patient_indices, ...
        test_balance, ...
        test_augment_healthy, ...
        test_augment_cancer);
    case 'patient'
      train_patient_indices = all_patient_indices(all_patient_indices ~= opts.leaveOutIndex);
      train_balance = true;
      train_augment_healthy = 'none';
      train_augment_cancer = 'none';
      test_patient_indices = opts.leaveOutIndex;
      test_balance = true;
      test_augment_healthy = 'rotate-flip';
      test_augment_cancer = 'rotate-flip';
      % just use the helper with the indices above.
      imdb = constructProstateImdbHelper( ...
        opts, ...
        train_patient_indices, ...
        train_balance, ...
        train_augment_healthy, ...
        train_augment_cancer, ...
        test_patient_indices, ...
        test_balance, ...
        test_augment_healthy, ...
        test_augment_cancer);
    case 'sample'
      % 1. check for the existence of a consistent / saved (!) balanaced,
      %    non-augmented imdb (both train and test)
      if ~exist(opts.imdbBalancedDir)
        mkdir(opts.imdbBalancedDir);
      end
      if exist(opts.imdbBalancedPath, 'file')
        afprintf(sprintf('[INFO] Loading Prostate imdb...\n'));
        imdb = load(opts.imdbBalancedPath);
      else
        afprintf(sprintf('[INFO] No saved Prostate imdb found, creating new one...\n'));
        % 1.5. if not saved, create it
        train_patient_indices = all_patient_indices;
        train_balance = true;
        train_augment_healthy = 'none';
        train_augment_cancer = 'none';
        test_patient_indices = [];
        test_balance = true;
        test_augment_healthy = 'none';
        test_augment_cancer = 'none';
        imdb = constructProstateImdbHelper( ...
          opts, ...
          train_patient_indices, ...
          train_balance, ...
          train_augment_healthy, ...
          train_augment_cancer, ...
          test_patient_indices, ...
          test_balance, ...
          test_augment_healthy, ...
          test_augment_cancer);
        save(opts.imdbBalancedPath, '-struct', 'imdb');
      end;
      % 2. then from the imdb, choose indices from there for train and test set.
      imdb.images.set(opts.leaveOutIndex) = 3;
      % A) find all tumors / non-tumors (other class of index i)
        % B) choose one at random
        % C) assign it as part of the test set as well
      switch imdb.images.labels(opts.leaveOutIndex)
        case 1
          other_class_index = 2;
        case 2
          other_class_index = 1;
      end
      tmp = find(imdb.images.labels == other_class_index);
      % choose one at random
      random_shuffle = randperm(length(tmp));
      first_random_shuffle = random_shuffle(1);
      % make that index also part of test set
      imdb.images.set(tmp(first_random_shuffle)) = 3;
  end

  afprintf(sprintf('[INFO] Finished constructing / loading Prostate imdb.\n'));

% -------------------------------------------------------------------------
function imdb = constructProstateImdbHelper( ...
  opts, ...
  train_patient_indices, ...
  train_balance, ...
  train_augment_healthy, ...
  train_augment_cancer, ...
  test_patient_indices, ...
  test_balance, ...
  test_augment_healthy, ...
  test_augment_cancer)
% -------------------------------------------------------------------------

  label_class = 'Gleason'; % 'Gleason' | 'PIRAD'
  modalitites_in_use = { ...
    'ADC_crop', ...
    'CDI_crop', ...
    'HBV_crop', ...
    % 'T2_crop', ...
    % 'I_b0_crop', ...
    % 'I_b1_crop', ...
    % 'I_b2_crop', ...
    % 'I_b3_crop', ...
  };

  % TRAIN
  afprintf(sprintf('== == == == == == == == == == == == ==  TRAIN  == == == == == == == == == == == == == == == == == == == == ==\n\n'));
  [data_train, labels_train] = loadSamples(opts, train_patient_indices, modalitites_in_use, label_class);
  [data_train, labels_train] = balanceData(data_train, labels_train, train_balance);
  [data_train, labels_train] = augmentData(data_train, labels_train, train_augment_healthy, train_augment_cancer);

  % TEST
  afprintf(sprintf('== == == == == == == == == == == == ==  TEST  == == == == == == == == == == == == == == == == == == == == ==\n\n'));
  [data_test, labels_test] = loadSamples(opts, test_patient_indices, modalitites_in_use, label_class);
  [data_test, labels_test] = balanceData(data_test, labels_test, test_balance);
  [data_test, labels_test] = augmentData(data_test, labels_test, test_augment_healthy, test_augment_cancer);

  set_train = [1 * ones(1, length(labels_train))];
  set_test = [3 * ones(1, length(labels_test))];

  if numel(data_test)
    data = cat(4, data_train, data_test);
    labels = cat(2, labels_train, labels_test);
    set = cat(2, set_train, set_test);
  else
    data = data_train;
    labels = labels_train;
    set = set_train;
  end

  totalNumberOfSamples = size(data,4);
  assert(totalNumberOfSamples == length(labels));
  assert(totalNumberOfSamples == length(set));

  % remove mean in any case
  data = single(data);
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  afprintf(sprintf('== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==\n\n'));

  if opts.contrastNormalization
    afprintf(sprintf('[INFO] contrast-normalizing data... '));
    z = reshape(data,[],totalNumberOfSamples);
    z = bsxfun(@minus, z, mean(z,1));
    n = std(z,0,1);
    z = bsxfun(@times, z, mean(n) ./ max(n, 40));
    data = reshape(z, 32, 32, numel(modalitites_in_use), []);
    afprintf(sprintf('done.\n'));
  end

  % if opts.whitenData
  %   afprintf(sprintf('[INFO] whitening data... '));
  %   z = reshape(data,[],totalNumberOfSamples);
  %   W = z(:,set == 1)*z(:,set == 1)'/totalNumberOfSamples;
  %   [V,D] = eig(W);
  %   % the scale is selected to approximately preserve the norm of W
  %   d2 = diag(D);
  %   en = sqrt(mean(d2));
  %   z = V*diag(en./max(sqrt(d2), 10))*V'*z;
  %   data = reshape(z, 32, 32, numberOfModalities, []);
  %   afprintf(sprintf('done.\n'));
  % end

  afprintf(sprintf('== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==\n\n'));

  afprintf(sprintf('[INFO] total number of samples: %d\n', totalNumberOfSamples));
  afprintf(sprintf('[INFO] number of `train` data - healthy: %d\n', size(data_train(:,:,:,labels_train == 1),4)));
  afprintf(sprintf('[INFO] number of `train` data - cancer: %d\n', size(data_train(:,:,:,labels_train == 2),4)));
  afprintf(sprintf('[INFO] number of `test` data - healthy: %d\n', size(data_test(:,:,:,labels_test == 1),4)));
  afprintf(sprintf('[INFO] number of `test` data - cancer: %d\n', size(data_test(:,:,:,labels_test == 2),4)));

  imdb.images.data = data;
  imdb.images.labels = single(labels);
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  % imdb.meta.classes = ...
  afprintf(sprintf('done!\n\n'));

% --------------------------------------------------------------------
function [data, labels] = loadSamples(opts, patient_indices_in_set, modalitites_in_use, label_class)
% --------------------------------------------------------------------
  data = [];
  labels = [];
  set = [];

  % if ~numel(patient_indices_in_set)
  %   return ([data, labels, set]);
  % end
  if ~numel(patient_indices_in_set)
    return;
  end

  modality_count = numel(modalitites_in_use);

  total_suspicious_tissue_count_in_set = 0;
  all_patients_list = dir(fullfile(opts.dataDir, 'P0*'));
  for i = patient_indices_in_set
    single_patient_directory = char(all_patients_list(i).name);
    suspicious_tissues_for_patient = dir(fullfile(opts.dataDir, char(single_patient_directory), '*_Candidate*'));
    for j = 1:length(suspicious_tissues_for_patient)
      total_suspicious_tissue_count_in_set = total_suspicious_tissue_count_in_set + 1;
    end
  end
  afprintf(sprintf('[INFO] Total suspicious tissue count (healthy or cancer) for %d patients: %d\n', ...
    numel(patient_indices_in_set), ...
    total_suspicious_tissue_count_in_set));

  % alloc memory for the matrices
  data = zeros(32, 32, modality_count, total_suspicious_tissue_count_in_set);
  labels_gleason = zeros(1, total_suspicious_tissue_count_in_set);
  labels_pirad = zeros(1, total_suspicious_tissue_count_in_set);
  labels = zeros(1, total_suspicious_tissue_count_in_set);
  set = zeros(1, total_suspicious_tissue_count_in_set);

  total_patient_count_in_set = length(patient_indices_in_set);
  sample_count = 0;
  patient_count = 0;
  afprintf(sprintf('[INFO] Loading patients... #'));
  for i = patient_indices_in_set
    patient_count = patient_count + 1;
    for j = 0:log10(patient_count - 1) + 5 % + 5 because of ` / ###`
      fprintf('\b'); % delete previous counter display
    end
    fprintf('%d / %d', patient_count, total_patient_count_in_set);

    % afprintf(sprintf('[INFO] Loading up suspicious tissues from patient #%03d (%03d of %03d)... ', i, patient_count, total_patient_count_in_set));
    single_patient_directory = char(all_patients_list(i).name);
    suspicious_tissues_for_patient = dir(fullfile(opts.dataDir, single_patient_directory, '*_Candidate*'));
    for j = 1:length(suspicious_tissues_for_patient)
      sample_count = sample_count + 1;
      suspicious_tissue_file = char(suspicious_tissues_for_patient(j).name);
      suspicious_tissue = load(fullfile(opts.dataDir, single_patient_directory, suspicious_tissue_file));
      tmp = zeros(32, 32, modality_count);
      for k = 1:modality_count
        tmp(:,:,k) = suspicious_tissue.(modalitites_in_use{k});
      end
      data(:,:,:,sample_count) = tmp;
      labels_gleason(1, sample_count) = suspicious_tissue.Gleason;
      labels_pirad(1, sample_count) = suspicious_tissue.PIRAD;
    end
    % afprintf(sprintf('done.\n'));
  end
  switch label_class
    case 'Gleason'
      labels = labels_gleason >= 6;
    case 'PIRAD'
      labels = labels_pirad >= 4;
  end
  % labels start from 1
  labels = labels + 1;
  afprintf(sprintf('done.\n'));

% --------------------------------------------------------------------
function [new_data, new_labels] = balanceData(data, labels, should_balance)
% --------------------------------------------------------------------
  if ~should_balance
    new_data = data;
    new_labels = labels;
  else
    afprintf(sprintf('[INFO] Balancing healthy and cancer tissues...\n'));
    afprintf(sprintf('[INFO] Identified %d total tissues\n', size(data, 4)));
    healthy_data = data(:,:,:,labels == 1);
    cancer_data = data(:,:,:,labels == 2);
    healthy_count = size(healthy_data, 4);
    cancer_count = size(cancer_data, 4);
    afprintf(sprintf('healthy:  %d \n', healthy_count));
    afprintf(sprintf('cancer: %d \n', cancer_count));

    % choose N random indices from healthy, where N = number of cancer tumors
    afprintf(sprintf('[INFO] Choosing %d out of %d healthy tissues... ', cancer_count, healthy_count));
    ix = randperm(healthy_count);
    ix = ix(1:cancer_count);
    subsampled_healthy_data = healthy_data(:,:,:,ix);
    new_data = cat(4, subsampled_healthy_data, cancer_data);
    new_labels = [1*ones(1,size(subsampled_healthy_data, 4)) 2*ones(1,cancer_count)]; % same number of healthy and cancer now.
    afprintf(sprintf('done.\n'));

    % shuffle them so we have intermixed subsampled_healthy_data and cancer_data
    total_new_count = size(new_data, 4);
    ix = randperm(total_new_count);
    new_data = new_data(:,:,:,ix);
    new_labels = new_labels(ix);

    afprintf(sprintf('[INFO] New data count: %d...\n', total_new_count));
    afprintf(sprintf('healthy:  %d \n', size(new_data(:,:,:,new_labels == 2), 4)));
    afprintf(sprintf('cancer: %d \n', size(new_data(:,:,:,new_labels == 1), 4)));

    afprintf(sprintf('done.\n'));
  end

% --------------------------------------------------------------------
function [new_data, new_labels] = augmentData(data, labels, augment_healthy, augment_cancer)
% --------------------------------------------------------------------
  healthy_data = data(:,:,:,labels == 1);
  cancer_data = data(:,:,:,labels == 2);

  augmented_healthy_data = augmentDataHelper('healthy', healthy_data, augment_healthy);
  augmented_cancer_data = augmentDataHelper('cancer', cancer_data, augment_cancer);
  augmented_healthy_labels = 1 * ones(1, size(augmented_healthy_data, 4));
  augmented_cancer_labels = 2 * ones(1, size(augmented_cancer_data, 4));

  new_data = cat(4, augmented_healthy_data, augmented_cancer_data);
  new_labels = cat(2, augmented_healthy_labels, augmented_cancer_labels);

  % shuffle them so we have intermixed healthy and cancer data
  total_new_count = size(new_data, 4);
  ix = randperm(total_new_count);
  new_data = new_data(:,:,:,ix);
  new_labels = new_labels(ix);

% --------------------------------------------------------------------
function [new_data] = augmentDataHelper(data_class, data, augment_type)
% --------------------------------------------------------------------
  if ~strcmp(augment_type, 'none')
    afprintf(sprintf('[INFO] Augmenting `%s` data (type: %s)...\n', data_class, augment_type));
    afprintf(sprintf('[INFO] Initial data count: %d.\n', size(data, 4)));
    rotation_angle = 45;
    degrees = 0:rotation_angle:360 - rotation_angle;

    if strcmp(augment_type, 'special')
      % randomly choose 120% of the inital size
      percent = 150;
      % so create 16x samples
      afprintf(sprintf('num_degrees: %d.\n', length(degrees)));
      afprintf(sprintf('num_flips: %d.\n', 2));
      new_data = zeros(size(data, 1), size(data, 2), size(data, 3), size(data, 4) * length(degrees) * 2);
      for i = 1:size(data, 4)
        for degree = degrees
          new_index = (i - 1) * length(degrees) * 2 + (degree / rotation_angle) * 2 + 1;
          new_index_left = new_index + 0;
          new_index_right = new_index + 1;
          rotated_image = imrotate(data(:,:,:,i), degree, 'crop');
          new_data(:,:,:,new_index_left) = rotated_image;
          new_data(:,:,:,new_index_right) = fliplr(rotated_image);
        end
      end
      % shuffle them, and choose the first 120%
      original_count = size(data, 4);
      total_new_count = size(new_data, 4);
      ix = randperm(total_new_count);
      new_data = new_data(:,:,:,ix);
      new_data = new_data(:,:,:,1:floor(original_count * percent / 100));
    elseif strcmp(augment_type, 'rotate-flip')
      afprintf(sprintf('num_degrees: %d.\n', length(degrees)));
      afprintf(sprintf('num_flips: %d.\n', 2));
      new_data = zeros(size(data, 1), size(data, 2), size(data, 3), size(data, 4) * length(degrees) * 2);
      for i = 1:size(data, 4)
        for degree = degrees
          new_index = (i - 1) * length(degrees) * 2 + (degree / rotation_angle) * 2 + 1;
          new_index_left = new_index + 0;
          new_index_right = new_index + 1;
          rotated_image = imrotate(data(:,:,:,i), degree, 'crop');
          new_data(:,:,:,new_index_left) = rotated_image;
          new_data(:,:,:,new_index_right) = fliplr(rotated_image);
        end
      end
    elseif strcmp(augment_type, 'rotate')
      afprintf(sprintf('num_degrees: %d.\n', length(degrees)));
      new_data = zeros(size(data, 1), size(data, 2), size(data, 3), size(data, 4) * length(degrees));
      for i = 1:size(data, 4)
        for degree = degrees
          new_index = (i - 1) * length(degrees) + (degree / rotation_angle) + 1;
          rotated_image = imrotate(data(:,:,:,i), degree, 'crop');
          new_data(:,:,:,new_index) = rotated_image;
        end
      end
    elseif strcmp(augment_type, 'flip')
      afprintf(sprintf('num_flips: %d.\n', 2));
      new_data = zeros(size(data, 1), size(data, 2), size(data, 3), size(data, 4) * 2);
      for i = 1:size(data, 4)
        new_index = (i - 1) * 2 + 1;
        new_index_left = new_index + 0;
        new_index_right = new_index + 1;
        new_data(:,:,:,new_index_left) = data(:,:,:,i);
        new_data(:,:,:,new_index_right) = fliplr(data(:,:,:,i));
      end
    elseif strcmp(augment_type, 'none')
      new_data = data;
    else
      afprintf(sprintf('\nWRONG!!!!!!!!\n\n'));
    end


    % shuffle them so we have intermixed rotations and flippings of different images
    total_new_count = size(new_data, 4);
    ix = randperm(total_new_count);
    new_data = new_data(:,:,:,ix);
    % new_labels = new_labels(ix);

    afprintf(sprintf('[INFO] Final data count: %d.\n', size(new_data, 4)));
    afprintf(sprintf('done.\n'));
  else
    new_data = data;
  end

% --------------------------------------------------------------------
function imdb = testProstateImdbConstructor()
% --------------------------------------------------------------------
  opts.dataDir = '/Users/a6karimi/dev/matconvnet/data_1/_prostate';
  opts.imdbBalancedDir = '/Users/a6karimi/dev/matconvnet/data_1/balanced-prostate-prostatenet';
  opts.imdbBalancedPath = '/Users/a6karimi/dev/matconvnet/data_1/balanced-prostate-prostatenet/imdb.mat';
  opts.leaveOutType = 'none';
  opts.leaveOutIndex = 1;
  opts.contrastNormalization = true;
  opts.whitenData = true;
  imdb = constructProstateImdb(opts);
