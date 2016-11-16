% -------------------------------------------------------------------------
function imdb = constructCOIL100Imdb(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing STL-10 imdb...\n');

  numOfObjects = 100; % numOfClasses
  numOfAngles = 72;
  numOfImages = numOfObjects * numOfAngles;
  imageSizeY = 128;
  imageSizeX = 128;
  imageSizeZ = 3;

  trainFile = fullfile(opts.dataDir, 'train.mat');
  testFile = fullfile(opts.dataDir, 'test.mat');

  train_images_indices = 1:2:numOfImages; % every other angle of the image
  test_images_indices = 2:2:numOfImages; % every other angle of the image

  labels = zeros(1, numOfImages);
  for i = 1:numOfObjects
    labels((i-1) * numOfAngles + 1:i * numOfAngles) = i;
  end

  if ~exist(trainFile) || ~exist(testFile)
    fprintf('\t[INFO] no `images.mat` file found; generating a new one from image files...\n');
    images = zeros(numOfImages, imageSizeY * imageSizeX * imageSizeZ); % [100 * 72, 49152]
    for objectNum = 1:1:100
      for angleNum = 0:5:355
        imageName = sprintf('obj%d__%d.png', objectNum, angleNum);
        image = imread(fullfile(opts.dataDir, imageName));
        image = reshape(image, 1, []); % [1, 49152]
        images((objectNum - 1) * numOfAngles + (angleNum + 5) / 5, :) = image;
      end
      if ~mod(objectNum, 5)
        fprintf('\t\t[INFO] finished processing %d files.\n', objectNum);
      end
    end
    fprintf('\tdone\n');

    meta_train.data = images(train_images_indices,:);
    meta_train.labels = labels(train_images_indices);
    meta_test.data = images(test_images_indices,:);
    meta_test.labels = labels(test_images_indices);

    fprintf('\t[INFO] Saving train meta data (large file ~25MB)...');
    save(fullfile(opts.dataDir, 'train.mat'), 'meta_train');
    fprintf('done\n');
    fprintf('\t[INFO] Saving test meta data (large file ~25MB)...');
    save(fullfile(opts.dataDir, 'test.mat'), 'meta_test');
    fprintf('done\n');
  else
    fprintf('\t[INFO] Found pre-existing train and test meta files. Loading... ');
    meta_train = load(trainFile);
    meta_train = meta_train.meta_train;
    meta_test = load(testFile);
    meta_test = meta_test.meta_test;
    fprintf('done.\n');
  end

  % disp(meta_train);
  data_train = meta_train.data;
  labels_train = meta_train.labels;
  fprintf('\t[INFO] Processing and resizing `train` images... ');
  data_train = imresize(reshape(data_train', imageSizeY,imageSizeX,imageSizeZ,[]), [32,32]);
  fprintf('done.\n');
  labels_train = single(labels_train);
  set_train = 1 * ones(1, numOfImages / 2);

  data_test = meta_test.data;
  labels_test = meta_test.labels;
  fprintf('\t[INFO] Processing and resizing `test` images... ');
  data_test = imresize(reshape(data_test', imageSizeY,imageSizeX,imageSizeZ,[]), [32,32]);
  fprintf('done.\n');
  labels_test = single(labels_test);
  set_test = 3 * ones(1, numOfImages / 2);

  data = single(cat(4, data_train, data_test));
  % data = cat(4, data_train, data_test);
  labels = single(cat(2, labels_train, labels_test));
  set = cat(2, set_train, set_test);

  % remove mean in any case
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  % imdb.meta.classes = ... i just made 100 classes myself, 1 per object (that
  % has 72 angles)... so we don't really have names for these classes. Note, we
  % can most definitely come up with fewer classes, say 'cup', 'car', 'fruit'...
  fprintf('done!\n\n');
