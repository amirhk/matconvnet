% -------------------------------------------------------------------------
function [top_predictions, all_predictions] = getPredictionsFromModelOnImdb(model, training_method, imdb, set)
% -------------------------------------------------------------------------
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

  printConsoleOutputSeparator();
  afprintf(sprintf('[INFO] Computing predictions from `%s` model on imdb (set `%d`)...\n', training_method, set));
  fh_imdb_utils = imdbTwoClassUtils;
  fh_imdb_utils.getImdbInfo(imdb, 1);
  imdb = filterImdbForSet(imdb, set);
  switch training_method
    case 'svm'
      [top_predictions, all_predictions] = getPredictionsFromSvmStructOnImdb(model, imdb);
    case 'forest'
      [top_predictions, all_predictions] = getPredictionsFromBoostedForestOnImdb(model, imdb);
    case 'cnn'
      [top_predictions, all_predictions] = getPredictionsFromNetOnImdb(model, imdb);
    case 'committee-cnn'
      [top_predictions, all_predictions] = getPredictionsFromCommitteeOnImdb(model, imdb, 'cnn');
    case 'committee-svm'
      [top_predictions, all_predictions] = getPredictionsFromCommitteeOnImdb(model, imdb, 'svm');
    case 'ensemble-cnn'
      [top_predictions, all_predictions] = getPredictionsFromEnsembleOnImdb(model, imdb, 'cnn');
    case 'ensemble-svm'
      [top_predictions, all_predictions] = getPredictionsFromEnsembleOnImdb(model, imdb, 'svm');
  end
  top_predictions = reshape(top_predictions, 1, prod(size(top_predictions)));

% -------------------------------------------------------------------------
function [top_predictions, all_predictions] = getPredictionsFromSvmStructOnImdb(svm_struct, imdb)
% -------------------------------------------------------------------------
  vectorized_data = getVectorizedDataFromImdb(imdb);
  top_predictions = svmclassify(svm_struct, vectorized_data);
  all_predictions = getAllPredictionsFromTopPredictions(top_predictions, imdb);

% -------------------------------------------------------------------------
function [top_predictions, all_predictions] = getPredictionsFromBoostedForestOnImdb(boosted_forest, imdb)
% -------------------------------------------------------------------------
  vectorized_data = getVectorizedDataFromImdb(imdb);
  top_predictions = predict(boosted_forest, vectorized_data);
  all_predictions = getAllPredictionsFromTopPredictions(top_predictions, imdb);

% -------------------------------------------------------------------------
function [top_predictions, all_predictions] = getPredictionsFromNetOnImdb(net, imdb)
% -------------------------------------------------------------------------
  [net, info] = cnnTrain(net, imdb, getBatch(), ...
    'debug_flag', false, ...
    'continue', false, ...
    'num_epochs', 1, ...
    'val', find(imdb.images.set == 3));
  top_predictions = info.top_predictions;
  all_predictions = info.all_predictions;

% -------------------------------------------------------------------------
function [top_predictions, all_predictions] = getPredictionsFromCommitteeOnImdb(committee, imdb, training_method)
% -------------------------------------------------------------------------
  fh_imdb_utils = imdbTwoClassUtils;
  [ ...
    data, ...
    data_train, ...
    data_train_positive, ...
    data_train_negative, ...
    data_train_indices, ...
    data_train_positive_indices, ...
    data_train_negative_indices, ...
    data_train_count, ...
    data_train_positive_count, ...
    data_train_negative_count, ...
    labels_train, ...
    labels_train_positive, ...
    labels_train_negative, ...
    data_test, ...
    data_test_positive, ...
    data_test_negative, ...
    data_test_indices, ...
    data_test_positive_indices, ...
    data_test_negative_indices, ...
    data_test_count, ...
    data_test_positive_count, ...
    data_test_negative_count, ...
    labels_test, ...
    labels_test_positive, ...
    labels_test_negative, ...
  ] = fh_imdb_utils.getImdbInfo(imdb, 1);
  number_of_classes = 2;

  predictions_for_all_classes_and_all_models = zeros(number_of_classes, data_test_count);
  for i = 1 : numel(committee)
    model = committee{i};
    [~, all_predictions] = getPredictionsFromModelOnImdb(model, training_method, imdb, 3);
    predictions_for_all_classes_and_all_models = ...
      predictions_for_all_classes_and_all_models + ...
      all_predictions;
  end
  [~, sorted_top_predictions] = sort(predictions_for_all_classes_and_all_models, 1, 'descend');
  top_predictions = sorted_top_predictions(1,:);
  all_predictions = predictions_for_all_classes_and_all_models;

% -------------------------------------------------------------------------
function [top_predictions, all_predictions] = getPredictionsFromEnsembleOnImdb(ensemble, imdb, training_method)
% -------------------------------------------------------------------------
  fh_imdb_utils = imdbTwoClassUtils;
  [ ...
    data, ...
    data_train, ...
    data_train_positive, ...
    data_train_negative, ...
    data_train_indices, ...
    data_train_positive_indices, ...
    data_train_negative_indices, ...
    data_train_count, ...
    data_train_positive_count, ...
    data_train_negative_count, ...
    labels_train, ...
    labels_train_positive, ...
    labels_train_negative, ...
    data_test, ...
    data_test_positive, ...
    data_test_negative, ...
    data_test_indices, ...
    data_test_positive_indices, ...
    data_test_negative_indices, ...
    data_test_count, ...
    data_test_positive_count, ...
    data_test_negative_count, ...
    labels_test, ...
    labels_test_positive, ...
    labels_test_negative, ...
  ] = fh_imdb_utils.getImdbInfo(imdb, 1);
  number_of_samples = size(imdb.images.data, 4);
  number_of_classes = numel(unique(imdb.images.labels));
  if ~length(fieldnames(ensemble))
    top_predictions = -1 * ones(1, number_of_samples);
    all_predictions = repmat(top_predictions, [number_of_classes, 1]);
    return
  end

  models = {};
  model_weights = [];
  number_of_models_in_ensemble = length(fieldnames(ensemble));
  for iteration = 1:number_of_models_in_ensemble
    models{iteration} = ensemble.(sprintf('iteration_%d', iteration)).trained_model.model;
    model_weights(iteration) = ensemble.(sprintf('iteration_%d', iteration)).trained_model.weight_normalized;
  end

  test_set_predictions_per_model = {};
  for iteration = 1:number_of_models_in_ensemble % looping through all trained models
    afprintf(sprintf('\n'));
    afprintf(sprintf('[INFO] Getting predictions for model #%d (positive: %d, negative: %d)...\n', ...
      iteration, ...
      data_test_positive_count, ...
      data_test_negative_count));
    % works for both `svm` and `cnn`
    model = models{iteration};
    [top_predictions, ~] = getPredictionsFromModelOnImdb(model, training_method, imdb, 3);
    test_set_predictions_per_model{iteration} = top_predictions;
    afprintf(sprintf('done.\n'));
  end

  weighted_ensemble_prediction = zeros(1, number_of_samples);
  for i = 1:number_of_samples
    % Calculating the total weight of the class labels from all the models
    % produced during boosting
    wt_positive = 0; % class 2
    wt_negative = 0; % class 1
    for iteration = 1:number_of_models_in_ensemble % looping through all trained models
       p = test_set_predictions_per_model{iteration}(i);
       if p == 2 % if is positive
           wt_positive = wt_positive + model_weights(iteration);
       else
           wt_negative = wt_negative + model_weights(iteration);
       end
    end

    if (wt_positive > wt_negative)
        weighted_ensemble_prediction(i) = 2;
    else
        weighted_ensemble_prediction(i) = 1;
    end
  end

  top_predictions = weighted_ensemble_prediction;
  all_predictions = getAllPredictionsFromTopPredictions(top_predictions, imdb);

% -------------------------------------------------------------------------
function imdb = filterImdbForSet(imdb, set)
% -------------------------------------------------------------------------
  % imdb should only contain test set ...
  % TODO: extend this file and cnn_train to support testing train set
  imdb.images.data = imdb.images.data(:,:,:,imdb.images.set == set);
  imdb.images.labels = imdb.images.labels(imdb.images.set == set);
  % cnnTrain method only works if the data is in validation set.... so this must be `3`
  % TODO: but what about other methods, like ensemble????????????
  imdb.images.set = 3 * ones(1, length(imdb.images.labels));

% TODO: copy to central file
% -------------------------------------------------------------------------
function fn = getBatch()
% -------------------------------------------------------------------------
  fn = @(x,y) getSimpleNNBatch(x,y);

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
  images = imdb.images.data(:,:,:,batch);
  labels = imdb.images.labels(1,batch);
  if rand > 0.5, images=fliplr(images); end

% -------------------------------------------------------------------------
function vectorized_data = getVectorizedDataFromImdb(imdb)
% -------------------------------------------------------------------------
  number_of_features = prod(size(imdb.images.data(:,:,:,1))); % 32 x 32 x 3 = 3072
  vectorized_data = reshape(imdb.images.data, number_of_features, [])';

% -------------------------------------------------------------------------
function all_predictions = getAllPredictionsFromTopPredictions(top_predictions, imdb)
% -------------------------------------------------------------------------
  % NOT repmat... find the index of the top predicted class
  % for each sample (in each column) and set that to 1.
  number_of_samples = size(imdb.images.data, 4);
  number_of_classes = numel(unique(imdb.images.labels));
  all_predictions = zeros(number_of_classes, number_of_samples);
  for i = 1:number_of_samples
    top_class_prediction = top_predictions(i);
    all_predictions(top_class_prediction, i) = 1;
  end


