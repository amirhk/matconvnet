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



  afprintf(sprintf('[INFO] Computing predictions from model on imdb (set `%d`)...\n', set));
  imdb = filterImdbForSet(imdb, set);
  switch training_method
    case 'svm'
      [top_predictions, all_predictions] = getPredictionsFromSvmStructOnImdb(model, imdb);
    case 'cnn'
      [top_predictions, all_predictions] = getPredictionsFromNetOnImdb(model, imdb);
  end

% -------------------------------------------------------------------------
function [top_predictions, all_predictions] = getPredictionsFromSvmStructOnImdb(svm_struct, imdb)
% -------------------------------------------------------------------------
  vectorized_images = reshape(imdb.images.data, 3072, [])';
  top_predictions = svmclassify(svm_struct, vectorized_images);
  number_of_classes = numel(unique(imdb.images.labels));
  all_predictions = repmat(top_predictions, [number_of_classes, 1]);

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

%   afprintf(sprintf('[INFO] Computing predictions from model on imdb (set `%d`)...\n', set));
%   imdb = filterImdbForSet(imdb, set);
%   switch prediction_type
%     case 'top-class'
%       switch training_method
%         case 'svm'
%           predictions = getTopPredictionsFromSvmStructOnImdb(model, imdb);
%         case 'cnn'
%           predictions = getTopPredictionsFromNetOnImdb(model, imdb);
%       end
%     case 'all-class'

%       switch training_method
%         case 'svm'
%           fprintf('TODO!!');
%           % number_of_classes = numel(unique(imdb.images.labels));
%           % predictions = getTopPredictionsFromSvmStructOnImdb(model, imdb);
%           % predictions = repmat(predictions, [number_of_classes, 1]);
%         case 'cnn'
%           predictions = getAllPredictionsFromNetOnImdb(model, imdb);
%       end

%   end

% % -------------------------------------------------------------------------
% function top_predictions = getTopPredictionsFromSvmStructOnImdb(svm_struct, imdb)
% % -------------------------------------------------------------------------
%   vectorized_images = reshape(imdb.images.data, 3072, [])';
%   top_predictions = svmclassify(svm_struct, vectorized_images);

% % -------------------------------------------------------------------------
% function top_predictions = getTopPredictionsFromNetOnImdb(net, imdb)
% % -------------------------------------------------------------------------
%   [net, info] = cnnTrain(net, imdb, getBatch(), ...
%     'debug_flag', false, ...
%     'continue', false, ...
%     'num_epochs', 1, ...
%     'val', find(imdb.images.set == 3));
%   top_predictions = info.top_predictions;

% % -------------------------------------------------------------------------
% function all_predictions = getAllPredictionsFromNetOnImdb(net, imdb)
% % -------------------------------------------------------------------------
%   [net, info] = cnnTrain(net, imdb, getBatch(), ...
%     'debug_flag', false, ...
%     'continue', false, ...
%     'num_epochs', 1, ...
%     'val', find(imdb.images.set == 3));
%   all_predictions = info.all_predictions;

% -------------------------------------------------------------------------
function imdb = filterImdbForSet(imdb, set)
% -------------------------------------------------------------------------
  % imdb should only contain test set ...
  % TODO: extend this file and cnn_train to support testing train set
  imdb.images.data = imdb.images.data(:,:,:,imdb.images.set == set);
  imdb.images.labels = imdb.images.labels(imdb.images.set == set);
  % cnn_train method only works if the data is in validation set.... so this must be `3`
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
