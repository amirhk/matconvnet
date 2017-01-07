function fh = imdbMultiClassUtils()
  % assign function handles so we can call these local functions from elsewhere
  fh.getImdbInfo = @getImdbInfo;

% -------------------------------------------------------------------------
function [ ...
  data_train_per_class, ...
  data_train_count_per_class, ...
  data_train_indices_per_class, ...
  data_test_per_class, ...
  data_test_count_per_class, ...
  data_test_indices_per_class, ...
  labels_test] = getImdbInfo(imdb, print_info)
% -------------------------------------------------------------------------
  % enforce row vector before doing bsxfun
  imdb.images.labels = reshape(imdb.images.labels, 1, prod(size(imdb.images.labels)));
  imdb.images.set = reshape(imdb.images.set, 1, prod(size(imdb.images.set)));


  unique_classes = unique(imdb.images.labels);

  for class_number = unique_classes
    % train
    data_train_indices_per_class{class_number} = bsxfun(@and, imdb.images.labels == class_number, imdb.images.set == 1);
    data_train_count_per_class{class_number} = sum(data_train_indices_per_class{class_number});
    data_train_per_class{class_number} = imdb.images.data(:,:,:,data_train_indices_per_class{class_number});
    % test
    data_test_indices_per_class{class_number} = bsxfun(@and, imdb.images.labels == class_number, imdb.images.set == 3);
    data_test_count_per_class{class_number} = sum(data_test_indices_per_class{class_number});
    data_test_per_class{class_number} = imdb.images.data(:,:,:,data_test_indices_per_class{class_number});
  end

  if print_info
    afprintf(sprintf('[INFO] imdb info:\n'));
    afprintf(sprintf('[INFO] TRAINING SET:\n'));
    afprintf(sprintf('[INFO] total: %d\n', sum([data_train_count_per_class{:}], 2)), 1);
    for class_number = unique_classes
      afprintf(sprintf('[INFO] class #%d: %d\n', class_number, data_train_count_per_class{class_number}), 1);
    end
    afprintf(sprintf('[INFO] TESTING SET:\n'));
    afprintf(sprintf('[INFO] total: %d\n', sum([data_test_count_per_class{:}], 2)), 1);
    for class_number = unique_classes
      afprintf(sprintf('[INFO] class #%d: %d\n', class_number, data_test_count_per_class{class_number}), 1);
    end
  end
