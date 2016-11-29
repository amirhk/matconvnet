% -------------------------------------------------------------------------
function imdb = constructProstateImdb(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing Prostate imdb...\n');

  useLabels = 'PIRAD'; % 'Gleason' | 'PIRAD'
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

  % for every patient
  count = 0;
  % randomly select some to be training and others to be test
  ix = randperm(totalNumberOfPatients);



  fprintf('\t[INFO] Loading TRAIN Patients...\n');
  for i = ix(1 : totalNumberOfPatients - numberOfTestPatients)
    fprintf('\t\t[INFO] Loading up suspicious tissues from patient #%02d of %d... ', i, totalNumberOfPatients);
    % for every suspicious tissue
    singlePatientDirectory = char(allPatientsList(i).name);
    suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, singlePatientDirectory, '*_Candidate*'));
    for j = 1:length(suspiciousTissuesForPatient)
      count = count + 1;
      suspiciousTissueFile = char(suspiciousTissuesForPatient(j).name);
      suspiciousTissue = load(fullfile(opts.dataDir, singlePatientDirectory, suspiciousTissueFile));
      tmp = zeros(32, 32, numberOfModalities);
      for k = 1:numberOfModalities
        tmp(:,:,k) = suspiciousTissue.(modalititesInUse{k});
      end
      data(:,:,:,count) = tmp;
      labelsGleason(1, count) = suspiciousTissue.Gleason;
      labelsPIRAD(1, count) = suspiciousTissue.PIRAD;
      set(1, count) = 1; % training data
    end
    fprintf('done.\n');
  end
  fprintf('\tdone.\n');



  fprintf('\t[INFO] Loading TEST Patients...\n');
  for i = ix(totalNumberOfPatients - numberOfTestPatients + 1 : end)
    fprintf('\t\t[INFO] Loading up suspicious tissues from patient #%02d of %d... ', i, totalNumberOfPatients);
    % for every suspicious tissue
    singlePatientDirectory = char(allPatientsList(i).name);
    suspiciousTissuesForPatient = dir(fullfile(opts.dataDir, singlePatientDirectory, '*_Candidate*'));
    for j = 1:length(suspiciousTissuesForPatient)
      count = count + 1;
      suspiciousTissueFile = char(suspiciousTissuesForPatient(j).name);
      suspiciousTissue = load(fullfile(opts.dataDir, singlePatientDirectory, suspiciousTissueFile));
      tmp = zeros(32, 32, numberOfModalities);
      for k = 1:numberOfModalities
        tmp(:,:,k) = suspiciousTissue.(modalititesInUse{k});
      end
      data(:,:,:,count) = tmp;
      labelsGleason(1, count) = suspiciousTissue.Gleason;
      labelsPIRAD(1, count) = suspiciousTissue.PIRAD;
      set(1, count) = 3; % testing data
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

  % % shuffle data and labels the same way
  % ix = randperm(totalSuspiciousTissueCount);
  % data = single(data(:,:,:,ix));
  % labels = labels(ix);

  % % take the first 90% to be training data and last 10% to be testing data
  totalNumberOfSamples = totalSuspiciousTissueCount;
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
