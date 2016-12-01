% % -------------------------------------------------------------------------
% function imdb = constructProstateImdb(opts)
% % -------------------------------------------------------------------------
%   fprintf('[INFO] Constructing Prostate imdb...\n');

%   useLabels = 'Gleason'; % 'Gleason' | 'PIRAD'
%   % percentageTrain = .90;
%   numberOfTestPatients = 1;
%   modalititesInUse = { ...
%     'ADC_crop', ...
%     'CDI_crop', ...
%     'HBV_crop', ...
%     'T2_crop', ...
%     % 'I_b0_crop', ...
%     % 'I_b1_crop', ...
%     % 'I_b2_crop', ...
%     % 'I_b3_crop', ...
%   };
%   numberOfModalities = numel(modalititesInUse);

%   totalSuspiciousTissueCount = 0;
%   allPatientsList = dir(fullfile(opts.dataDir, 'P0*'));
%   totalNumberOfPatients = length(allPatientsList);
%   % for every patient
%   for i = 1:totalNumberOfPatients
%     % for every suspicious tissue
%     singlePatientDirectory = char(allPatientsList(i).name);
%     suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, char(singlePatientDirectory), '*_Candidate*'));
%     for j = 1:length(suspiciousTissuesForPatient)
%       totalSuspiciousTissueCount = totalSuspiciousTissueCount + 1;
%     end
%   end

%   % randomly select some to be training and others to be test.
%   % make sure the test index contains cancerous tissues!!!!
%   found_test_index_with_cancerous_tissue = false;
%   fprintf('[INFO] Trying to find test patient index with cancerous tissue...\n');
%   while ~found_test_index_with_cancerous_tissue
%     ix = randperm(totalNumberOfPatients);
%     test_patient_index = ix(end);
%     singlePatientDirectory = char(allPatientsList(test_patient_index).name);
%     suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, singlePatientDirectory, '*_Candidate*'));
%     for j = 1:length(suspiciousTissuesForPatient)
%       % test_count = test_count + 1;
%       suspiciousTissueFile = char(suspiciousTissuesForPatient(j).name);
%       suspiciousTissue = load(fullfile(opts.dataDir, singlePatientDirectory, suspiciousTissueFile));
%       gleason = suspiciousTissue.Gleason;
%       pirad = suspiciousTissue.PIRAD;
%       switch useLabels
%         case 'Gleason'
%           if gleason >= 6
%             found_test_index_with_cancerous_tissue = true;
%           end
%         case 'PIRAD'
%           if pirad >= 4
%             found_test_index_with_cancerous_tissue = true;
%           end
%       end
%     end
%     fprintf('\tNot found. Retrying...\n');
%   end
%   fprintf('[INFO] Found (index %d)!!!\n', test_patient_index);

%   data = zeros(32, 32, numberOfModalities, totalSuspiciousTissueCount);
%   labelsGleason = zeros(1, totalSuspiciousTissueCount);
%   labelsPIRAD = zeros(1, totalSuspiciousTissueCount);
%   labels = zeros(1, totalSuspiciousTissueCount);  set = zeros(1, totalSuspiciousTissueCount);


%   % ---- ---- ---- ---- TRAIN ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

%   train_count = 0;
%   patient_count = 0;
%   fprintf('\t[INFO] Loading TRAIN Patients...\n');
%   for i = ix(1 : totalNumberOfPatients - numberOfTestPatients)
%     patient_count = patient_count + 1;
%     fprintf('\t\t[INFO] Loading up suspicious tissues from patient #%03d (%03d of %d)... ', i, patient_count, totalNumberOfPatients);
%     % for every suspicious tissue
%     singlePatientDirectory = char(allPatientsList(i).name);
%     suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, singlePatientDirectory, '*_Candidate*'));
%     for j = 1:length(suspiciousTissuesForPatient)
%       train_count = train_count + 1;
%       suspiciousTissueFile = char(suspiciousTissuesForPatient(j).name);
%       suspiciousTissue = load(fullfile(opts.dataDir, singlePatientDirectory, suspiciousTissueFile));
%       tmp = zeros(32, 32, numberOfModalities);
%       for k = 1:numberOfModalities
%         tmp(:,:,k) = suspiciousTissue.(modalititesInUse{k});
%       end
%       index = train_count;
%       data(:,:,:,index) = tmp;
%       labelsGleason(1, index) = suspiciousTissue.Gleason;
%       labelsPIRAD(1, index) = suspiciousTissue.PIRAD;
%       set(1, index) = 1; % training data
%     end
%     fprintf('done.\n');
%   end
%   fprintf('\tdone.\n');

%   % ---- ---- ---- ---- TEST ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

%   test_count = 0;
%   fprintf('\t[INFO] Loading TEST Patients...\n');
%   for i = ix(totalNumberOfPatients - numberOfTestPatients + 1 : end)
%     patient_count = patient_count + 1;
%     fprintf('\t\t[INFO] Loading up suspicious tissues from patient #%03d (%03d of %d)... ', i, patient_count, totalNumberOfPatients);
%     % for every suspicious tissue
%     singlePatientDirectory = char(allPatientsList(i).name);
%     suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, singlePatientDirectory, '*_Candidate*'));
%     for j = 1:length(suspiciousTissuesForPatient)
%       test_count = test_count + 1;
%       suspiciousTissueFile = char(suspiciousTissuesForPatient(j).name);
%       suspiciousTissue = load(fullfile(opts.dataDir, singlePatientDirectory, suspiciousTissueFile));
%       tmp = zeros(32, 32, numberOfModalities);
%       for k = 1:numberOfModalities
%         tmp(:,:,k) = suspiciousTissue.(modalititesInUse{k});
%       end
%       index = train_count + test_count;
%       data(:,:,:,index) = tmp;
%       labelsGleason(1, index) = suspiciousTissue.Gleason;
%       labelsPIRAD(1, index) = suspiciousTissue.PIRAD;
%       set(1, index) = 3; % testing data
%     end
%     fprintf('done.\n');
%   end
%   fprintf('\tdone.\n');

%   switch useLabels
%     case 'Gleason'
%       labels = labelsGleason >= 6;
%     case 'PIRAD'
%       labels = labelsPIRAD >= 4;
%   end
%   % labels start from 1
%   labels = labels + 1;

%   [data_train, labels_train] = balanceMalignantAndBenignTissues('train', data(:,:,:,1:train_count), labels(1:train_count));
%   [data_test, labels_test] = balanceMalignantAndBenignTissues('test', data(:,:,:,train_count+1:end), labels(train_count+1:end));
%   [data_train, labels_train] = augmentData('train', data_train, labels_train);
%   % [data_test, labels_test] = augmentData('test', data_test, labels_test);

%   data = cat(4, data_train, data_test);
%   labels = cat(2, labels_train, labels_test);
%   set = [1*ones(1,length(labels_train)) 3*ones(1,length(labels_test))];

%   totalNumberOfSamples = size(data,4);

%   assert(totalNumberOfSamples == length(labels));
%   assert(totalNumberOfSamples == length(set));

%   fprintf('[INFO] total number of samples: %d\n', totalNumberOfSamples);
%   fprintf('[INFO] number of `train` data - cancer: %d\n', size(data_train(:,:,:,labels_train == 2),4));
%   fprintf('[INFO] number of `train` data - healthy: %d\n', size(data_train(:,:,:,labels_train == 1),4));
%   fprintf('[INFO] number of `test` data - cancer: %d\n', size(data_test(:,:,:,labels_test == 2),4));
%   fprintf('[INFO] number of `test` data - healthy: %d\n', size(data_test(:,:,:,labels_test == 1),4));

%   % % shuffle data and labels the same way
%   % ix = randperm(totalSuspiciousTissueCount);
%   % data = single(data(:,:,:,ix));
%   % labels = labels(ix);

%   % % take the first 90% to be training data and last 10% to be testing data
%   % totalNumberOfSamples = totalSuspiciousTissueCount;
%   % numberOfTrainSamples = floor(totalNumberOfSamples * percentageTrain);
%   % numberOfTestSamples = totalNumberOfSamples - numberOfTrainSamples;
%   % set = [ones(1, numberOfTrainSamples) 3 * ones(1, numberOfTestSamples)];


%   data = single(data);
%   % remove mean in any case
%   dataMean = mean(data(:,:,:,set == 1), 4);
%   data = bsxfun(@minus, data, dataMean);

%   if opts.contrastNormalization
%     fprintf('[INFO] contrast-normalizing data... ');
%     z = reshape(data,[],totalNumberOfSamples);
%     z = bsxfun(@minus, z, mean(z,1));
%     n = std(z,0,1);
%     z = bsxfun(@times, z, mean(n) ./ max(n, 40));
%     data = reshape(z, 32, 32, numberOfModalities, []);
%     fprintf('done.\n');
%   end

%   % if opts.whitenData
%   %   fprintf('[INFO] whitening data... ');
%   %   z = reshape(data,[],totalNumberOfSamples);
%   %   W = z(:,set == 1)*z(:,set == 1)'/totalNumberOfSamples;
%   %   [V,D] = eig(W);
%   %   % the scale is selected to approximately preserve the norm of W
%   %   d2 = diag(D);
%   %   en = sqrt(mean(d2));
%   %   z = V*diag(en./max(sqrt(d2), 10))*V'*z;
%   %   data = reshape(z, 32, 32, numberOfModalities, []);
%   %   fprintf('done.\n');
%   % end

%   imdb.images.data = data;
%   imdb.images.labels = single(labels);
%   imdb.images.labelsGleason = single(labelsGleason);
%   imdb.images.labelsPIRAD = single(labelsPIRAD);
%   imdb.images.set = set;
%   imdb.meta.sets = {'train', 'val', 'test'};
%   % imdb.meta.classes = train_file.class_names; % = test_file.class_names
%   fprintf('done!\n\n');


% % --------------------------------------------------------------------
% function [new_data, new_labels] = balanceMalignantAndBenignTissues(data_type, data, labels)
% % --------------------------------------------------------------------
%   fprintf('\t[INFO] Balancing malignant and benign tissues in `%s` set...\n', data_type);
%   fprintf('\t\t[INFO] Identified %d total tissues\n', size(data, 4));
%   benign_data = data(:,:,:,labels == 1);
%   malignant_data = data(:,:,:,labels == 2);
%   benign_count = size(benign_data, 4);
%   malignant_count = size(malignant_data, 4);
%   fprintf('\t\t\tbenign:  %d \n', benign_count);
%   fprintf('\t\t\tmalignant: %d \n', malignant_count);

%   if ~benign_count
%     error('[ERROR]. Was not able to identify any benign tissues\n');
%   end

%   if ~malignant_count
%     error('[ERROR]. Was not able to identify any malignant tissues\n');
%   end

%   % choose N random indices from benign, where N = number of malignant tumors
%   fprintf('\t\t[INFO] Choosing %d out of %d benign tissues... ', malignant_count, benign_count);
%   ix = randperm(benign_count);
%   ix = ix(1:malignant_count);
%   subsampled_benign_data = benign_data(:,:,:,ix);
%   new_data = cat(4, subsampled_benign_data, malignant_data);
%   new_labels = [1*ones(1,malignant_count) 2*ones(1,malignant_count)]; % same number of benign and malignant now.
%   fprintf('done.\n');

%   % shuffle them so we have intermixed subsampled_benign_data and malignant_data
%   total_new_count = size(new_data, 4);
%   ix = randperm(total_new_count);
%   new_data = new_data(:,:,:,ix);
%   new_labels = new_labels(ix);

%   fprintf('\t\t[INFO] New `%s` data count: %d...\n', data_type, total_new_count);
%   fprintf('\t\t\tbenign:  %d \n', size(new_data(:,:,:,new_labels == 2), 4));
%   fprintf('\t\t\tmalignant: %d \n', size(new_data(:,:,:,new_labels == 1), 4));

%   fprintf('\tdone.\n');



% % --------------------------------------------------------------------
% function [new_data, new_labels] = augmentData(data_type, data, labels)
% % --------------------------------------------------------------------
%   fprintf('\t[INFO] Augmenting `%s` data...\n', data_type);
%   fprintf('\t\t[INFO] Initial `%s` data count: %d.\n', data_type, size(data, 4));
%   rotation_angle = 45;
%   degrees = 0:rotation_angle:360 - rotation_angle;
%   fprintf('\t\t\t[INFO] Number of degrees: %d.\n', length(degrees));
%   fprintf('\t\t\t[INFO] Number of flips: %d.\n', 2);
%   new_data = zeros(size(data, 1), size(data, 2), size(data, 3), size(data, 4) * length(degrees) * 2);
%   for i = 1:size(data, 4)
%     for degree = degrees
%       new_index = (i - 1) * length(degrees) * 2 + (degree / rotation_angle) * 2 + 1;
%       new_index_left = new_index + 0;
%       new_index_right = new_index + 1;
%       rotated_image = imrotate(data(:,:,:,i), degree, 'crop');
%       new_data(:,:,:,new_index_left) = rotated_image;
%       new_data(:,:,:,new_index_right) = fliplr(rotated_image);
%     end
%   end

%   % repeat labels length(degrees) number of times...
%   new_labels = labels';
%   n = length(degrees) * 2; % *2 for left right flip
%   r = repmat(labels', 1, n)';
%   new_labels = r(:)';
%   assert(size(new_data, 4) == length(new_labels));

%   % shuffle them so we have intermixed rotations of different images
%   total_new_count = size(new_data, 4);
%   ix = randperm(total_new_count);
%   new_data = new_data(:,:,:,ix);
%   new_labels = new_labels(ix);

%   fprintf('\t\t[INFO] Final `%s` data count: %d.\n', data_type, size(new_data, 4));
%   fprintf('\tdone.\n');





% -------------------------------------------------------------------------
function imdb = constructProstateImdb(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing Prostate imdb...\n');

  useLabels = 'Gleason'; % 'Gleason' | 'PIRAD'
  % percentageTrain = .90;
  numberOfTestPatients = 1;
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

  % randomly select some to be training and others to be test.
  % % make sure the test index contains cancerous tissues!!!!
  % found_test_index_with_cancerous_tissue = false;
  % fprintf('[INFO] Trying to find test patient index with cancerous tissue...\n');
  % while ~found_test_index_with_cancerous_tissue
  %   ix = randperm(totalNumberOfPatients);
  %   test_patient_index = ix(end);
  %   singlePatientDirectory = char(allPatientsList(test_patient_index).name);
  %   suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, singlePatientDirectory, '*_Candidate*'));
  %   for j = 1:length(suspiciousTissuesForPatient)
  %     % test_count = test_count + 1;
  %     suspiciousTissueFile = char(suspiciousTissuesForPatient(j).name);
  %     suspiciousTissue = load(fullfile(opts.dataDir, singlePatientDirectory, suspiciousTissueFile));
  %     gleason = suspiciousTissue.Gleason;
  %     pirad = suspiciousTissue.PIRAD;
  %     switch useLabels
  %       case 'Gleason'
  %         if gleason >= 6
  %           found_test_index_with_cancerous_tissue = true;
  %         end
  %       case 'PIRAD'
  %         if pirad >= 4
  %           found_test_index_with_cancerous_tissue = true;
  %         end
  %     end
  %   end
  %   fprintf('\tNot found. Retrying...\n');
  % end
  % fprintf('[INFO] Found (index %d)!!!\n', test_patient_index);
  ix = 1:1:104;
  ix = cat(2, ix(ix ~= opts.patientNumber), opts.patientNumber);

  data = zeros(32, 32, numberOfModalities, totalSuspiciousTissueCount);
  labelsGleason = zeros(1, totalSuspiciousTissueCount);
  labelsPIRAD = zeros(1, totalSuspiciousTissueCount);
  labels = zeros(1, totalSuspiciousTissueCount);  set = zeros(1, totalSuspiciousTissueCount);


  % ---- ---- ---- ---- TRAIN ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

  train_count = 0;
  patient_count = 0;
  fprintf('\t[INFO] Loading TRAIN Patients...\n');
  for i = ix(1 : totalNumberOfPatients - numberOfTestPatients)
    patient_count = patient_count + 1;
    fprintf('\t\t[INFO] Loading up suspicious tissues from patient #%03d (%03d of %d)... ', i, patient_count, totalNumberOfPatients);
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
    fprintf('\t\t[INFO] Loading up suspicious tissues from patient #%03d (%03d of %d)... ', i, patient_count, totalNumberOfPatients);
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
  [data_train, labels_train] = augmentData('train', data_train, labels_train);
  % [data_test, labels_test] = augmentData('test', data_test, labels_test);

  data = cat(4, data_train, data_test);
  labels = cat(2, labels_train, labels_test);
  set = [1*ones(1,length(labels_train)) 3*ones(1,length(labels_test))];

  totalNumberOfSamples = size(data,4);

  assert(totalNumberOfSamples == length(labels));
  assert(totalNumberOfSamples == length(set));

  fprintf('[INFO] total number of samples: %d\n', totalNumberOfSamples);
  fprintf('[INFO] number of `train` data - cancer: %d\n', size(data_train(:,:,:,labels_train == 2),4));
  fprintf('[INFO] number of `train` data - healthy: %d\n', size(data_train(:,:,:,labels_train == 1),4));
  fprintf('[INFO] number of `test` data - cancer: %d\n', size(data_test(:,:,:,labels_test == 2),4));
  fprintf('[INFO] number of `test` data - healthy: %d\n', size(data_test(:,:,:,labels_test == 1),4));

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

  % if ~benign_count
  %   error('[ERROR]. Was not able to identify any benign tissues\n');
  % end

  % if ~malignant_count
  %   error('[ERROR]. Was not able to identify any malignant tissues\n');
  % end

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



% --------------------------------------------------------------------
function [new_data, new_labels] = augmentData(data_type, data, labels)
% --------------------------------------------------------------------
  fprintf('\t[INFO] Augmenting `%s` data...\n', data_type);
  fprintf('\t\t[INFO] Initial `%s` data count: %d.\n', data_type, size(data, 4));
  rotation_angle = 45;
  degrees = 0:rotation_angle:360 - rotation_angle;
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

  % repeat labels length(degrees) number of times...
  new_labels = labels';
  n = length(degrees) * 2; % *2 for left right flip
  r = repmat(labels', 1, n)';
  new_labels = r(:)';
  assert(size(new_data, 4) == length(new_labels));

  % shuffle them so we have intermixed rotations of different images
  total_new_count = size(new_data, 4);
  ix = randperm(total_new_count);
  new_data = new_data(:,:,:,ix);
  new_labels = new_labels(ix);

  fprintf('\t\t[INFO] Final `%s` data count: %d.\n', data_type, size(new_data, 4));
  fprintf('\tdone.\n');








