% --------------------------------------------------------------------
function output = compareProstateResults(opts)
  % read the `sensitivity_specificity.txt` file in the subdirectories of a parent experimentd directory
  % from there, extract values for each patients leave-1-pateint-out test
  % average them
% --------------------------------------------------------------------
  experimentDirectory = '/Volumes/Amir/results/2016-12-01-01; Radiomics; Gleason; balanced only train not balanced test, augment only train, epoch 30, leave-1-patient-out/leave 1 patient out';
  experimentDirectory = '/Volumes/Amir/results/2016-12-01-01; Radiomics; Gleason; balanced only train not balanced test, augment only train, epoch 30, leave-1-patient-out/leave 1 sample out';
  % experimentDirectory = '/Volumes/Amir/results/prostate compRand + layerwise from cifar';
  allPatientExperimentDirectories = dir(fullfile(experimentDirectory, 'prostate-prostatenet-*'));
  totalNumberOfPatients = length(allPatientExperimentDirectories);
  allPatientResults = {};
  for i = 1:totalNumberOfPatients
    fprintf('[INFO] Reading results for patient #%d... ', i);
    singlePatientExperimentDirectory = char(allPatientExperimentDirectories(i).name);
    singlePatientResultFile = fullfile(singlePatientExperimentDirectory, 'sensitivity_specificity.txt');
    % time to open the file for each patient
    fid = fopen(fullfile(experimentDirectory, singlePatientResultFile));
    tline = fgets(fid);
    singlePatientResults = {};
    count = 1;
    while ischar(tline)
        % [INFO] Accuracy: 0.00000
        index_1 = strfind(tline, '[INFO]');
        index_2 = strfind(tline, ':');
        parameter = tline(index_1 + 7 : index_2 - 1);
        value = tline(index_2 + 2 : end - 1);
        if ~isempty(parameter)
          if count <= 7
            singlePatientResults.train.(parameter) = value;
          else
            singlePatientResults.test.(parameter) = value;
          end
          count = count + 1;
        end
        tline = fgets(fid);
    end
    fclose(fid);
    allPatientResults{i} = singlePatientResults;
    fprintf('done!\n');
  end

  allTrainAccuracy = [];
  allTrainSensitivity = [];
  allTrainSpecificity = [];
  allTestAccuracy = [];
  allTestSensitivity = [];
  allTestSpecificity = [];
  for i = 1:numel(allPatientResults)
    % train
    trainAccuracy = str2num(allPatientResults{i}.train.Accuracy);
    if ~isnan(trainAccuracy)
      allTrainAccuracy(end+1) = trainAccuracy;
    end
    trainSensitivity = str2num(allPatientResults{i}.train.Sensitivity);
    if ~isnan(trainSensitivity)
      allTrainSensitivity(end+1) = trainSensitivity;
    end
    trainSensitivity = str2num(allPatientResults{i}.train.Specificity);
    if ~isnan(trainSensitivity)
      allTrainSpecificity(end+1) = trainSensitivity;
    end
    % test
    testAccuracy = str2num(allPatientResults{i}.test.Accuracy);
    if ~isnan(testAccuracy)
      allTestAccuracy(end+1) = testAccuracy;
    end
    testSensitivity = str2num(allPatientResults{i}.test.Sensitivity);
    if ~isnan(testSensitivity)
      allTestSensitivity(end+1) = testSensitivity;
    end
    testSensitivity = str2num(allPatientResults{i}.test.Specificity);
    if ~isnan(testSensitivity)
      allTestSpecificity(end+1) = testSensitivity;
    end
  end
  disp(mean(allTrainAccuracy));
  disp(mean(allTrainSensitivity));
  disp(mean(allTrainSensitivity));
  disp(mean(allTestAccuracy));
  disp(mean(allTestSensitivity));
  disp(mean(allTestSpecificity));

  fprintf('Average Train Accuracy: %6.5f\n', mean(allTrainAccuracy));
  fprintf('Average Train Sensitivity: %6.5f\n', mean(allTrainSensitivity));
  fprintf('Average Train Sensitivity: %6.5f\n', mean(allTrainSensitivity));
  fprintf('Average Test Accuracy: %6.5f\n', mean(allTestAccuracy));
  fprintf('Average Test Sensitivity: %6.5f\n', mean(allTestSensitivity));
  fprintf('Average Test Specificity: %6.5f\n', mean(allTestSpecificity));

  output.allTrainAccuracy = allTrainAccuracy;
  output.allTrainSensitivity = allTrainSensitivity;
  output.allTrainSensitivity = allTrainSensitivity;
  output.allTestAccuracy = allTestAccuracy;
  output.allTestSensitivity = allTestSensitivity;
  output.allTestSpecificity = allTestSpecificity;
