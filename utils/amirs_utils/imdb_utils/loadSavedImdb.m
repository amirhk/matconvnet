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
          switch fold
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
          switch fold
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
        otherwise
          fprintf('TODO: implement!')
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

