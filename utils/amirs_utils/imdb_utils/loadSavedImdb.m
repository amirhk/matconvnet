function imdb = loadSavedImdb(input_opts)
  dataset = getValueFromFieldOrDefault(input_opts, 'dataset', 'mnist-two-class-9-4');
  posneg_balance = getValueFromFieldOrDefault(input_opts, 'posneg_balance', 'balanced-low');
  fold_number = getValueFromFieldOrDefault(input_opts, 'fold_number', 1); % currently only implemented for prostate data

  afprintf(sprintf('[INFO] Loading imdb (dataset: %s, posneg_balance: %s)\n', dataset, posneg_balance));
  path_to_imdbs = fullfile(getDevPath(), 'data', 'two_class_imdbs');
  switch dataset
    case 'mnist-two-class-9-4'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-low'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-balanced-train-30-30.mat'));
        case 'unbalanced'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-unbalanced-30-6000.mat'));
        case 'balanced-high'
          tmp = load(fullfile(path_to_imdbs, 'mnist', 'saved-two-class-mnist-pos9-neg4-balanced-train-6000-6000.mat'));
      end
    case 'cifar-two-class-deer-horse'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-low'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg8-balanced-train-25-25.mat'));
        case 'unbalanced'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg8-unbalanced-25-5000.mat'));
        case 'balanced-high'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg8-balanced-train-5000-5000.mat'));
        end
    case 'cifar-two-class-deer-truck'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-low'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg10-balanced-train-25-25.mat'));
        case 'unbalanced'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg10-unbalanced-25-5000.mat'));
        case 'balanced-high'
          tmp = load(fullfile(path_to_imdbs, 'cifar', 'saved-two-class-cifar-pos5-neg10-balanced-train-5000-5000.mat'));
        end
    case 'svhn-two-class-9-4'
      % currently fold number is not implemented.
      switch posneg_balance
        case 'balanced-low'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-two-class-svhn-pos9-neg4-balanced-low-train-23-23.mat'));
        case 'unbalanced'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-two-class-svhn-pos9-neg4-unbalanced-train-23-7458.mat'));
        case 'balanced-high'
          tmp = load(fullfile(path_to_imdbs, 'svhn', 'saved-two-class-svhn-pos9-neg4-balanced-high-train-4659-7458.mat'));
        end
    case 'prostate-v2-20-patients'
      switch posneg_balance
        case 'unbalanced'
          switch fold_number
            case 1
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-51-655.mat'));
            case 2
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-62-597.mat'));
            case 3
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-68-544.mat'));
            case 4
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-68-567.mat'));
            case 5
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-unbalaned-train-71-493.mat'));
          end
        case 'balanced-high'
          switch fold_number
            case 1
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-488-597.mat'));
            case 2
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-504-498.mat'));
            case 3
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-512-574.mat'));
            case 4
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-512-599.mat'));
            case 5
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'saved-two-class-prostate-v2-20-patients-pos2-neg1-balanced-high-train-544-588.mat'));
          end
        case 'leave-one-out-unbalanced'
          switch fold_number
            case 1
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-1-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-659.mat');
            case 2
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-2-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-80-636.mat');
            case 3
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-3-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-80-692.mat');
            case 4
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-4-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-677.mat');
            case 5
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-5-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-71-703.mat');
            case 6
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-6-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-75-678.mat');
            case 7
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-7-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-70-692.mat');
            case 8
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-8-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-79-671.mat');
            case 9
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-9-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-74-688.mat');
            case 10
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-10-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-75-660.mat');
            case 11
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-11-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-677.mat');
            case 12
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-12-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-78-691.mat');
            case 13
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-13-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-76-691.mat');
            case 14
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-14-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-75-669.mat');
            case 15
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-15-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-78-603.mat');
            case 16
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-16-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-69-710.mat');
            case 17
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-17-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-75-701.mat');
            case 18
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-18-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-80-681.mat');
            case 19
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-19-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-696.mat');
            case 20
              tmp = load(fullfile(path_to_imdbs, 'prostate', 'leave-one-out', 'patient-20-unbalaced-saved-two-class-prostate-v2-20-patients-pos2-neg1-train-77-691.mat');
          end
        otherwise
          fprintf('TODO: implement!');
      end
    % case 'prostate'
    %   fprintf('TODO: implement!')
    %   % TODO: fixup and test
    %   % opts.general.network_arch = 'prostatenet';
    %   % num_test_patients = 10;
    %   % tmp_opts.data_dir = fullfile(getDevPath(), 'matconvnet/data_1/_prostate');
    %   % tmp_opts.imdb_balanced_dir = fullfile(getDevPath(), 'matconvnet/data_1/balanced-prostate-prostatenet');
    %   % tmp_opts.imdb_balanced_path = fullfile(getDevPath(), 'matconvnet/data_1/balanced-prostate-prostatenet/imdb.mat');
    %   % tmp_opts.leave_out_type = 'special';
    %   % random_patient_indices = randperm(104);
    %   % tmp_opts.leaveOutIndices = random_patient_indices(1:num_test_patients);
    %   % tmp_opts.contrast_normalization = true;
    %   % tmp_opts.whitenData = true;
  end
  imdb = tmp.imdb;
  afprintf(sprintf('[INFO] done!\n'));

  % print info
  printConsoleOutputSeparator();
  fh_imdb_utils = imdbTwoClassUtils;
  fh_imdb_utils.getImdbInfo(imdb, 1);
  printConsoleOutputSeparator();

