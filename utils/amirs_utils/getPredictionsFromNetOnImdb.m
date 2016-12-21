% -------------------------------------------------------------------------
function predictions = getPredictionsFromNetOnImdb(net, imdb, set)
% -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Computing predictions from net on imdb (set `%d`)...\n', set));
  imdb = filterImdbForSet(imdb, set);
  [net, info] = cnn_train(net, imdb, getBatch(), ...
    'debugFlag', false, ...
    'continue', false, ...
    'numEpochs', 1, ...
    'val', find(imdb.images.set == 3));
  predictions = info.predictions;

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
