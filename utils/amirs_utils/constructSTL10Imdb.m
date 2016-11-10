% -------------------------------------------------------------------------
function imdb = constructSTL10Imdb(opts)
% -------------------------------------------------------------------------
  fprintf('[INFO] Constructing STL-10 imdb...');
  train_file = load(fullfile(opts.dataDir, 'train.mat'));
  test_file = load(fullfile(opts.dataDir, 'test.mat'));

  data_train = imresize(reshape(train_file.X', 96,96,3,[]), [32,32]);
  % data_train = imresize(reshape(im2double(train_file.X'), 96,96,3,[]), [32,32]);
  labels_train = single(train_file.y');
  set_train = 1 * ones(1, 5000);

  data_test = imresize(reshape(test_file.X', 96,96,3,[]), [32,32]);
  % data_test = imresize(reshape(im2double(test_file.X'), 96,96,3,[]), [32,32]);
  labels_test = single(test_file.y');
  set_test = 3 * ones(1, 8000);

  data = single(cat(4, data_train, data_test));
  labels = single(cat(2, labels_train, labels_test));
  set = cat(2, set_train, set_test);

  % remove mean in any case
  dataMean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, dataMean);

  if opts.contrastNormalization
    fprintf('[INFO] contrast-normalizing data... ');
    z = reshape(data,[],13000);
    z = bsxfun(@minus, z, mean(z,1));
    n = std(z,0,1);
    z = bsxfun(@times, z, mean(n) ./ max(n, 40));
    data = reshape(z, 32, 32, 3, []);
    fprintf('done.\n');
  end

  % if opts.whitenData
  %   fprintf('[INFO] whitening data... ');
  %   z = reshape(data,[],13000);
  %   W = z(:,set == 1)*z(:,set == 1)'/13000;
  %   [V,D] = eig(W);
  %   % the scale is selected to approximately preserve the norm of W
  %   d2 = diag(D);
  %   en = sqrt(mean(d2));
  %   z = V*diag(en./max(sqrt(d2), 10))*V'*z;
  %   data = reshape(z, 32, 32, 3, []);
  %   fprintf('done.\n');
  % end

  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  imdb.meta.classes = train_file.class_names; % = test_file.class_names
  fprintf('done!\n\n');
