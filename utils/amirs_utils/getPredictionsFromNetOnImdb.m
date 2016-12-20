% -------------------------------------------------------------------------
function predictions = getPredictionsFromNetOnImdb(net, imdb)
% -------------------------------------------------------------------------
  [net, info] = cnn_train(net, imdb, getBatch(), ...
    'debugFlag', false, ...
    'continue', false, ...
    'numEpochs', 1, ...
    'val', find(imdb.images.set == 3));
  predictions = info.predictions;

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
