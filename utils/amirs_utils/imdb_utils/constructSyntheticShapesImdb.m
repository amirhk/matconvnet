% -------------------------------------------------------------------------
function imdb = constructSyntheticShapesImdb(total_number_of_samples, dim_image, dim_shape, dim_shape_variance)
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

  afprintf(sprintf('[INFO] Constructing synthetic shapes imdb...\n'));

  data = zeros(dim_image, dim_image, 1, total_number_of_samples);
  labels = zeros(1, total_number_of_samples);

  for i = 1 : total_number_of_samples
    if ~mod(i, total_number_of_samples / 10)
      afprintf(sprintf('[INFO] generated %d samples\n', i));
    end
    if randn() > 0
      canvas_with_shape = drawRandomSquareOfRadiusOnCanvas(dim_image, dim_shape, dim_shape_variance);
      labels(i) = 1;
    else
      canvas_with_shape = drawRandomCircleOfRadiusOnCanvas(dim_image, dim_shape, dim_shape_variance);
      labels(i) = 2;
    end
    data(:,:,1,i) = canvas_with_shape;
  end

  number_of_training_samples = .7 * total_number_of_samples;
  number_of_testing_samples = .3 * total_number_of_samples;
  set = cat(1, 1 * ones(number_of_training_samples, 1), 3 * ones(number_of_testing_samples, 1));


  % shuffle
  ix = randperm(total_number_of_samples);
  imdb.images.data = data(:,:,:,ix);
  imdb.images.labels = labels(ix);
  imdb.images.set = set(ix);
  imdb.name = sprintf('shapes-dim-image-%d-dim-shape-%d-dim-shape-variance-%d', dim_image, dim_shape, dim_shape_variance); % sprintf('shapes-%dD-%d-train-%d-test-%.1f-var', sample_dim, number_of_training_samples, number_of_testing_samples, sample_variance_multiplier);

  afprintf(sprintf('done!\n\n'));
  fh = imdbMultiClassUtils;
  fh.getImdbInfo(imdb, 1);
  save(sprintf('%s.mat', imdb.name), 'imdb');



% -------------------------------------------------------------------------
function canvas_with_shape = drawRandomSquareOfRadiusOnCanvas(dim_image, dim_shape, dim_shape_variance)
% -------------------------------------------------------------------------
  canvas = ones(dim_image, dim_image);
  % canvas = imread('cameraman.tif')
  dim_shape_variance = round(randn() * dim_shape_variance);

  x = randsample(dim_image, 1);
  y = randsample(dim_image, 1);

  square = int32([y, x, dim_shape + dim_shape_variance, dim_shape + dim_shape_variance]);
  shapeInserter = vision.ShapeInserter('Fill',true);
  canvas_with_shape = step(shapeInserter, canvas, square);
  release(shapeInserter);

  % pause(0.01)
  % imshow(canvas_with_shape)

% -------------------------------------------------------------------------
function canvas_with_shape = drawRandomCircleOfRadiusOnCanvas(dim_image, dim_shape, dim_shape_variance)
% -------------------------------------------------------------------------
  canvas = ones(dim_image, dim_image);
  dim_shape_variance = round(randn() * dim_shape_variance);

  x = randsample(dim_image, 1);
  y = randsample(dim_image, 1);

  circle = int32([y, x, (dim_shape + dim_shape_variance) / 2]);
  shapeInserter = vision.ShapeInserter('Fill',true);
  set(shapeInserter,'Shape','Circles');
  canvas_with_shape = step(shapeInserter, canvas, circle);
  release(shapeInserter);

  % pause(0.01)
  % imshow(canvas_with_shape)




















