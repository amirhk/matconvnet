% -------------------------------------------------------------------------
function imdb = constructProstateImdb(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing Prostate imdb...\n');

  useLabels = 'Gleason'; % 'Gleason' | 'PIRAD'
  % percentageTrain = .90;
  numberOfTestPatients = 5;
  modalititesInUse = { ...
    'ADC_crop', ...
    'CDI_crop', ...
    'HBV_crop', ...
    'T2_crop', ...
    % 'I_b0_crop', ...
    % 'I_b1_crop', ...
    % 'I_b2_crop', ...
    % 'I_b3_crop', ...
  };
  numberOfModalities = numel(modalititesInUse);

  totalSuspiciousTissueCount = 0;
  allPatientsList = dir(fullfile(opts.dataDir, 'P0*'));
  totalNumberOfPatients = length(allPatientsList);
  % for every patient
  for i = 1:totalNumberOfPatients
    % for every suspicious tissue
    singlePatientDirectory = char(allPatientsList(i).name);
    suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, char(singlePatientDirectory), '*_Candidate*'));
    for j = 1:length(suspiciousTissuesForPatient)
      totalSuspiciousTissueCount = totalSuspiciousTissueCount + 1;
    end
  end

  data = zeros(32, 32, numberOfModalities, totalSuspiciousTissueCount);
  labelsGleason = zeros(1, totalSuspiciousTissueCount);
  labelsPIRAD = zeros(1, totalSuspiciousTissueCount);
  labels = zeros(1, totalSuspiciousTissueCount);
  set = zeros(1, totalSuspiciousTissueCount);

  % randomly select some to be training and others to be test
  ix = randperm(totalNumberOfPatients);

  % ---- ---- ---- ---- TRAIN ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

  train_count = 0;
  patient_count = 0;
  fprintf('\t[INFO] Loading TRAIN Patients...\n');
  for i = ix(1 : totalNumberOfPatients - numberOfTestPatients)
    patient_count = patient_count + 1;
    fprintf('\t\t[INFO] Loading up suspicious tissues from patient #%02d (%02d of %d)... ', i, patient_count, totalNumberOfPatients);
    % for every suspicious tissue
    singlePatientDirectory = char(allPatientsList(i).name);
    suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, singlePatientDirectory, '*_Candidate*'));
    for j = 1:length(suspiciousTissuesForPatient)
      train_count = train_count + 1;
      suspiciousTissueFile = char(suspiciousTissuesForPatient(j).name);
      suspiciousTissue = load(fullfile(opts.dataDir, singlePatientDirectory, suspiciousTissueFile));
      tmp = zeros(32, 32, numberOfModalities);
      for k = 1:numberOfModalities
        tmp(:,:,k) = suspiciousTissue.(modalititesInUse{k});
      end
      index = train_count;
      data(:,:,:,index) = tmp;
      labelsGleason(1, index) = suspiciousTissue.Gleason;
      labelsPIRAD(1, index) = suspiciousTissue.PIRAD;
      set(1, index) = 1; % training data
    end
    fprintf('done.\n');
  end
  fprintf('\tdone.\n');

  % ---- ---- ---- ---- TEST ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

  test_count = 0;
  fprintf('\t[INFO] Loading TEST Patients...\n');
  for i = ix(totalNumberOfPatients - numberOfTestPatients + 1 : end)
    patient_count = patient_count + 1;
    fprintf('\t\t[INFO] Loading up suspicious tissues from patient #%02d (%02d of %d)... ', i, patient_count, totalNumberOfPatients);
    % for every suspicious tissue
    singlePatientDirectory = char(allPatientsList(i).name);
    suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, singlePatientDirectory, '*_Candidate*'));
    for j = 1:length(suspiciousTissuesForPatient)
      test_count = test_count + 1;
      suspiciousTissueFile = char(suspiciousTissuesForPatient(j).name);
      suspiciousTissue = load(fullfile(opts.dataDir, singlePatientDirectory, suspiciousTissueFile));
      tmp = zeros(32, 32, numberOfModalities);
      for k = 1:numberOfModalities
        tmp(:,:,k) = suspiciousTissue.(modalititesInUse{k});
      end
      index = train_count + test_count;
      data(:,:,:,index) = tmp;
      labelsGleason(1, index) = suspiciousTissue.Gleason;
      labelsPIRAD(1, index) = suspiciousTissue.PIRAD;
      set(1, index) = 3; % testing data
    end
    fprintf('done.\n');
  end
  fprintf('\tdone.\n');

  switch useLabels
    case 'Gleason'
      labels = labelsGleason >= 6;
    case 'PIRAD'
      labels = labelsPIRAD >= 4;
  end
  % labels start from 1
  labels = labels + 1;

  [data_train, labels_train] = balanceMalignantAndBenignTissues('train', data(:,:,:,1:train_count), labels(1:train_count));
  [data_test, labels_test] = balanceMalignantAndBenignTissues('test', data(:,:,:,train_count+1:end), labels(train_count+1:end));
  data = cat(4, data_train, data_test);
  labels = cat(2, labels_train, labels_test);
  set = [1*ones(1,length(labels_train)) 3*ones(1,length(labels_test))];

  totalNumberOfSamples = size(data,4);

  assert(totalNumberOfSamples == length(labels));
  assert(totalNumberOfSamples == length(set));

  % % shuffle data and labels the same way
  % ix = randperm(totalSuspiciousTissueCount);
  % data = single(data(:,:,:,ix));
  % labels = labels(ix);

  % % take the first 90% to be training data and last 10% to be testing data
  % totalNumberOfSamples = totalSuspiciousTissueCount;
  % numberOfTrainSamples = floor(totalNumberOfSamples * percentageTrain);
  % numberOfTestSamples = totalNumberOfSamples - numberOfTrainSamples;
  % set = [ones(1, numberOfTrainSamples) 3 * ones(1, numberOfTestSamples)];

  data = single(data);
  % remove mean in any case
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  if opts.contrastNormalization
    fprintf('[INFO] contrast-normalizing data... ');
    z = reshape(data,[],totalNumberOfSamples);
    z = bsxfun(@minus, z, mean(z,1));
    n = std(z,0,1);
    z = bsxfun(@times, z, mean(n) ./ max(n, 40));
    data = reshape(z, 32, 32, numberOfModalities, []);
    fprintf('done.\n');
  end

  if opts.whitenData
    fprintf('[INFO] whitening data... ');
    z = reshape(data,[],totalNumberOfSamples);
    W = z(:,set == 1)*z(:,set == 1)'/totalNumberOfSamples;
    [V,D] = eig(W);
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D);
    en = sqrt(mean(d2));
    z = V*diag(en./max(sqrt(d2), 10))*V'*z;
    data = reshape(z, 32, 32, numberOfModalities, []);
    fprintf('done.\n');
  end

  imdb.images.data = data;
  imdb.images.labels = single(labels);
  imdb.images.labelsGleason = single(labelsGleason);
  imdb.images.labelsPIRAD = single(labelsPIRAD);
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  % imdb.meta.classes = train_file.class_names; % = test_file.class_names
  fprintf('done!\n\n');


% --------------------------------------------------------------------
function [new_data, new_labels] = balanceMalignantAndBenignTissues(data_type, data, labels)
% --------------------------------------------------------------------
  fprintf('\t[INFO] Balancing malignant and benign tissues in `%s` set...\n', data_type);
  fprintf('\t\t[INFO] Identified %d total tissues\n', size(data, 4));
  benign_data = data(:,:,:,labels == 1);
  malignant_data = data(:,:,:,labels == 2);
  benign_count = size(benign_data, 4);
  malignant_count = size(malignant_data, 4);
  fprintf('\t\t\tbenign:  %d \n', benign_count);
  fprintf('\t\t\tmalignant: %d \n', malignant_count);

  if ~benign_count
    error('[ERROR]. Was not able to identify any benign tissues\n');
  end

  if ~malignant_count
    error('[ERROR]. Was not able to identify any malignant tissues\n');
  end

  % choose N random indices from benign, where N = number of malignant tumors
  fprintf('\t\t[INFO] Choosing %d out of %d benign tissues... ', malignant_count, benign_count);
  ix = randperm(benign_count);
  ix = ix(1:malignant_count);
  subsampled_benign_data = benign_data(:,:,:,ix);
  new_data = cat(4, subsampled_benign_data, malignant_data);
  new_labels = [1*ones(1,malignant_count) 2*ones(1,malignant_count)]; % same number of benign and malignant now.
  fprintf('done.\n');

  % shuffle them so we have intermixed subsampled_benign_data and malignant_data
  total_new_count = size(new_data, 4);
  ix = randperm(total_new_count);
  new_data = new_data(:,:,:,ix);
  new_labels = new_labels(ix);

  fprintf('\t\t[INFO] New `%s` data count: %d...\n', data_type, total_new_count);
  fprintf('\t\t\tbenign:  %d \n', size(new_data(:,:,:,new_labels == 2), 4));
  fprintf('\t\t\tmalignant: %d \n', size(new_data(:,:,:,new_labels == 1), 4));

  fprintf('\tdone.\n');
