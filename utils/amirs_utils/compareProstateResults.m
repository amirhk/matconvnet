% --------------------------------------------------------------------
function output = compareProstateResults(opts)
  % read the `sensitivity_specificity.txt` file in the subdirectories of a parent experimentd directory
  % from there, extract values for each patients leave-1-pateint-out test
  % average them
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

  experimentDirectory = '/Volumes/Amir/results/2016-12-02-05; Radiomics; Gleason; leave-1-patient-out; bpd 04';
  % experimentDirectory = '/Volumes/Amir/results/2016-12-02-05; Radiomics; Gleason; leave-1-patient-out; bpd 13';
  experimentDirectory = '/Volumes/Amir/results/2016-12-02-05; Radiomics; Gleason; leave-1-sample-out; bpd 04';
  experimentDirectory = '/Volumes/Amir/results/2016-12-02-05; Radiomics; Gleason; leave-1-sample-out; bpd 13';
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
  fprintf('Average Train Specificity: %6.5f\n', mean(allTrainSpecificity));
  fprintf('Average Test Accuracy: %6.5f\n', mean(allTestAccuracy));
  fprintf('Average Test Sensitivity: %6.5f\n', mean(allTestSensitivity));
  fprintf('Average Test Specificity: %6.5f\n', mean(allTestSpecificity));

  output.allTrainAccuracy = allTrainAccuracy;
  output.allTrainSensitivity = allTrainSensitivity;
  output.allTrainSpecificity = allTrainSpecificity;
  output.allTestAccuracy = allTestAccuracy;
  output.allTestSensitivity = allTestSensitivity;
  output.allTestSpecificity = allTestSpecificity;
