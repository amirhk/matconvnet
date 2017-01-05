% --------------------------------------------------------------------
function imdb = constructSvhnImdb(opts)
% --------------------------------------------------------------------
  afprintf(sprintf('[INFO] Constructing SVHN imdb...\n'));
  train_file = load(fullfile(opts.imdb.data_dir, 'train_32x32.mat'));
  test_file = load(fullfile(opts.imdb.data_dir, 'test_32x32.mat'));

  data = single(cat(4, train_file.X, test_file.X));
  labels = single(cat(2, train_file.y', test_file.y'));
  set = cat(2, 1 * ones(1, length(train_file.y)), 3 * ones(1, length(test_file.y)));

  % remove mean in any case
  data_mean = mean(data(:,:,:,set == 1), 4);
  data = bsxfun(@minus, data, data_mean);

  % normalize by image mean and std as suggested in `An Analysis of
  % Single-Layer Networks in Unsupervised Feature Learning` Adam
  % Coates, Honglak Lee, Andrew Y. Ng
  if opts.imdb.contrast_normalization
    afprintf(sprintf('[INFO] Contrast-normalizing data... '));
    z = reshape(data,[],size(data, 4));
    z = bsxfun(@minus, z, mean(z,1));
    n = std(z,0,1);
    z = bsxfun(@times, z, mean(n) ./ max(n, 40));
    data = reshape(z, 32, 32, 3, []);
    afprintf(sprintf('done.\n'));
  end

  imdb.images.data = data;
  imdb.images.labels = labels;
  imdb.images.set = set;
  imdb.meta.sets = {'train', 'val', 'test'};
  afprintf(sprintf('done!\n\n'));
