% -------------------------------------------------------------------------
function imdb = constructProstateImdb3(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing / loading Prostate imdb.\n');

  all_patient_indices = 1:1:104;
  switch opts.leaveOutType
    case 'patient'
      train_patient_indices = all_patient_indices(all_patient_indices ~= opts.leaveOutIndex);
      train_balance = true;
      train_augment = true;
      test_patient_indices = opts.leaveOutIndex;
      test_balance = true;
      test_augment = true;
      % just use the helper with the indices above.
      imdb = constructProstateImdbHelper( ...
        opts, ...
        train_patient_indices, ...
        train_balance, ...
        train_augment, ...
        test_patient_indices, ...
        test_balance, ...
        test_augment);
    case 'sample'
      % 1. check for the existence of a consistent / saved (!) balanaced,
      %    non-augmented imdb (both train and test)
      if ~exist(opts.imdbBalancedDir)
        mkdir(opts.imdbBalancedDir);
      end
      if exist(opts.imdbBalancedPath, 'file')
        fprintf('[INFO] Loading Prostate imdb...\n');
        imdb = load(opts.imdbBalancedPath);
      else
        fprintf('[INFO] No saved Prostate imdb found, creating new one...\n');
        % 1.5. if not saved, create it
        train_patient_indices = all_patient_indices;
        train_balance = true;
        train_augment = false;
        test_patient_indices = [];
        test_balance = true;
        test_augment = false;
        imdb = constructProstateImdbHelper( ...
          opts, ...
          train_patient_indices, ...
          train_balance, ...
          train_augment, ...
          test_patient_indices, ...
          test_balance, ...
          test_augment);
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

  fprintf('[INFO] Finished constructing / loading Prostate imdb.\n');

% -------------------------------------------------------------------------
function imdb = constructProstateImdbHelper( ...
  opts, ...
  train_patient_indices, ...
  train_balance, ...
  train_augment, ...
  test_patient_indices, ...
  test_balance, ...
  test_augment)
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
  [data_train, labels_train, set_train] = loadSamples(opts, 'train', train_patient_indices, modalitites_in_use, label_class);
  if train_balance; [data_train, labels_train, set_train] = balanceData('train', data_train, labels_train); end;
  if train_augment; [data_train, labels_train, set_train] = augmentData('train', data_train, labels_train); end;

  % TEST
  [data_test, labels_test, set_test] = loadSamples(opts, 'test', test_patient_indices, modalitites_in_use, label_class);
  if test_balance; [data_test, labels_test, set_test] = balanceData('test', data_test, labels_test); end;
  if test_augment; [data_test, labels_test, set_test] = augmentData('test', data_test, labels_test); end;

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

  fprintf('[INFO] total number of samples: %d\n', totalNumberOfSamples);
  fprintf('[INFO] number of `train` data - cancer: %d\n', size(data_train(:,:,:,labels_train == 2),4));
  fprintf('[INFO] number of `train` data - healthy: %d\n', size(data_train(:,:,:,labels_train == 1),4));
  fprintf('[INFO] number of `test` data - cancer: %d\n', size(data_test(:,:,:,labels_test == 2),4));
  fprintf('[INFO] number of `test` data - healthy: %d\n', size(data_test(:,:,:,labels_test == 1),4));

  % remove mean in any case
  data = single(data);
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  if opts.contrastNormalization
    fprintf('[INFO] contrast-normalizing data... ');
    z = reshape(data,[],totalNumberOfSamples);
    z = bsxfun(@minus, z, mean(z,1));
    n = std(z,0,1);
    z = bsxfun(@times, z, mean(n) ./ max(n, 40));
    data = reshape(z, 32, 32, numel(modalitites_in_use), []);
    fprintf('done.\n');
  end

  % if opts.whitenData
  %   fprintf('[INFO] whitening data... ');
  %   z = reshape(data,[],totalNumberOfSamples);
  %   W = z(:,set == 1)*z(:,set == 1)'/totalNumberOfSamples;
  %   [V,D] = eig(W);
  %   % the scale is selected to approximately preserve the norm of W
  %   d2 = diag(D);
  %   en = sqrt(mean(d2));
  %   z = V*diag(en./max(sqrt(d2), 10))*V'*z;
  %   data = reshape(z, 32, 32, numberOfModalities, []);
  %   fprintf('done.\n');
  % end

  imdb.images.data = data;
  imdb.images.labels = single(labels);
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  % imdb.meta.classes = ...
  fprintf('done!\n\n');



% --------------------------------------------------------------------
function [data, labels, set] = loadSamples(opts, set_type, patient_indices_in_set, modalitites_in_use, label_class)
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
  fprintf('\t[INFO] Total suspicious tissue count for %d `%s` patients: ', length(patient_indices_in_set), set_type);
  for i = patient_indices_in_set
    single_patient_directory = char(all_patients_list(i).name);
    suspicious_tissues_for_patient = dir(fullfile(opts.dataDir, char(single_patient_directory), '*_Candidate*'));
    for j = 1:length(suspicious_tissues_for_patient)
      total_suspicious_tissue_count_in_set = total_suspicious_tissue_count_in_set + 1;
    end
  end
  fprintf('%d\n', total_suspicious_tissue_count_in_set);

  % alloc memory for the matrices
  data = zeros(32, 32, modality_count, total_suspicious_tissue_count_in_set);
  labels_gleason = zeros(1, total_suspicious_tissue_count_in_set);
  labels_pirad = zeros(1, total_suspicious_tissue_count_in_set);
  labels = zeros(1, total_suspicious_tissue_count_in_set);
  set = zeros(1, total_suspicious_tissue_count_in_set);

  total_patient_count_in_set = length(patient_indices_in_set);
  sample_count = 0;
  patient_count = 0;
  fprintf('\t[INFO] Loading `%s` patients...\n', set_type);
  for i = patient_indices_in_set
    patient_count = patient_count + 1;
    fprintf('\t\t[INFO] Loading up suspicious tissues from patient #%03d (%03d of %03d)... ', i, patient_count, total_patient_count_in_set);
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
    fprintf('done.\n');
  end
  switch label_class
    case 'Gleason'
      labels = labels_gleason >= 6;
    case 'PIRAD'
      labels = labels_pirad >= 4;
  end
  % labels start from 1
  labels = labels + 1;
  if strcmp(set_type, 'train')
    set = [1 * ones(1, length(labels))];
  else
    set = [3 * ones(1, length(labels))];
  end
  fprintf('\tdone.\n');

% --------------------------------------------------------------------
function [new_data, new_labels, new_set] = balanceData(set_type, data, labels)
% --------------------------------------------------------------------
  fprintf('\t[INFO] Balancing malignant and benign tissues in `%s` set...\n', set_type);
  fprintf('\t\t[INFO] Identified %d total tissues\n', size(data, 4));
  benign_data = data(:,:,:,labels == 1);
  malignant_data = data(:,:,:,labels == 2);
  benign_count = size(benign_data, 4);
  malignant_count = size(malignant_data, 4);
  fprintf('\t\t\tbenign:  %d \n', benign_count);
  fprintf('\t\t\tmalignant: %d \n', malignant_count);

  % if ~malignant_count
  %   malignant_count = 1;
  % end

  % choose N random indices from benign, where N = number of malignant tumors
  fprintf('\t\t[INFO] Choosing %d out of %d benign tissues... ', malignant_count, benign_count);
  ix = randperm(benign_count);
  ix = ix(1:malignant_count);
  subsampled_benign_data = benign_data(:,:,:,ix);
  new_data = cat(4, subsampled_benign_data, malignant_data);
  new_labels = [1*ones(1,size(subsampled_benign_data, 4)) 2*ones(1,malignant_count)]; % same number of benign and malignant now.
  fprintf('done.\n');

  % shuffle them so we have intermixed subsampled_benign_data and malignant_data
  total_new_count = size(new_data, 4);
  ix = randperm(total_new_count);
  new_data = new_data(:,:,:,ix);
  new_labels = new_labels(ix);
  if strcmp(set_type, 'train')
    new_set = [1 * ones(1, length(new_labels))];
  else
    new_set = [3 * ones(1, length(new_labels))];
  end

  fprintf('\t\t[INFO] New `%s` data count: %d...\n', set_type, total_new_count);
  fprintf('\t\t\tbenign:  %d \n', size(new_data(:,:,:,new_labels == 2), 4));
  fprintf('\t\t\tmalignant: %d \n', size(new_data(:,:,:,new_labels == 1), 4));

  fprintf('\tdone.\n');

% --------------------------------------------------------------------
function [new_data, new_labels, new_set] = augmentData(set_type, data, labels)
% --------------------------------------------------------------------
  benign_data = data(:,:,:,labels == 1);
  malignant_data = data(:,:,:,labels == 2);

  if strcmp(set_type, 'train')
    augmented_benign_data = augmentDataHelper(set_type, benign_data, 'none');
    augmented_malignant_data = augmentDataHelper(set_type, malignant_data, 'special');
    augmented_benign_labels = 1 * ones(1, size(augmented_benign_data, 4));
    augmented_malignant_labels = 2 * ones(1, size(augmented_malignant_data, 4));
  else
    augmented_benign_data = augmentDataHelper(set_type, benign_data, 'rotate-flip');
    augmented_malignant_data = augmentDataHelper(set_type, malignant_data, 'rotate-flip');
    augmented_benign_labels = 1 * ones(1, size(augmented_benign_data, 4));
    augmented_malignant_labels = 2 * ones(1, size(augmented_malignant_data, 4));
  end


  new_data = cat(4, augmented_benign_data, augmented_malignant_data);
  new_labels = cat(2, augmented_benign_labels, augmented_malignant_labels);

  % shuffle them so we have intermixed benign and malignant data
  total_new_count = size(new_data, 4);
  ix = randperm(total_new_count);
  new_data = new_data(:,:,:,ix);
  new_labels = new_labels(ix);
  if strcmp(set_type, 'train')
    new_set = [1 * ones(1, length(new_labels))];
  else
    new_set = [3 * ones(1, length(new_labels))];
  end

% --------------------------------------------------------------------
function [new_data] = augmentDataHelper(set_type, data, augment_type)
% --------------------------------------------------------------------
  fprintf('\t[INFO] Augmenting `%s` data...\n', set_type);
  fprintf('\t\t[INFO] Initial `%s` data count: %d.\n', set_type, size(data, 4));
  rotation_angle = 45;
  degrees = 0:rotation_angle:360 - rotation_angle;

  if strcmp(augment_type, 'special')
    % randomly choose 120% of the inital size
    percent = 150;
    % so create 16x samples
    fprintf('\t\t\t[INFO] Number of degrees: %d.\n', length(degrees));
    fprintf('\t\t\t[INFO] Number of flips: %d.\n', 2);
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
    fprintf('\t\t\t[INFO] Number of degrees: %d.\n', length(degrees));
    fprintf('\t\t\t[INFO] Number of flips: %d.\n', 2);
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
    fprintf('\t\t\t[INFO] Number of degrees: %d.\n', length(degrees));
    new_data = zeros(size(data, 1), size(data, 2), size(data, 3), size(data, 4) * length(degrees));
    for i = 1:size(data, 4)
      for degree = degrees
        new_index = (i - 1) * length(degrees) + (degree / rotation_angle) + 1;
        rotated_image = imrotate(data(:,:,:,i), degree, 'crop');
        new_data(:,:,:,new_index) = rotated_image;
      end
    end
  elseif strcmp(augment_type, 'flip')
    fprintf('\t\t\t[INFO] Number of flips: %d.\n', 2);
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
    fprintf('\n\nWRONG!!!!!!!!\n\n');
  end


  % shuffle them so we have intermixed rotations and flippings of different images
  total_new_count = size(new_data, 4);
  ix = randperm(total_new_count);
  new_data = new_data(:,:,:,ix);
  % new_labels = new_labels(ix);

  fprintf('\t\t[INFO] Final `%s` data count: %d.\n', set_type, size(new_data, 4));
  fprintf('\tdone.\n');

