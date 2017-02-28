% -------------------------------------------------------------------------
function runTempTests()
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
% % POSSIBILITY OF SUCH DAMAGE.

  input_opts.dataset = 'mnist-multi-class-subsampled';
  input_opts.posneg_balance = 'balanced-38';
  % input_opts.dataset = 'mnist';
  % input_opts.posneg_balance = 'whatever';
  input_opts.network_arch = 'lenet';
  input_opts.projection = 'no-projection';
  input_opts.fold_number = 1;
  imdb = loadSavedImdb(input_opts)


  opts.train.number_of_features = prod(size(imdb.images.data(:,:,:,1))); % 32 x 32 x 3 = 3072
  vectorized_data = reshape(imdb.images.data, opts.train.number_of_features, [])';
  labels = imdb.images.labels;
  is_train = imdb.images.set == 1;
  is_test = imdb.images.set == 3;

  vectorized_data_train = vectorized_data(is_train, :);
  vectorized_data_test = vectorized_data(is_test, :);
  labels_train = labels(is_train)';
  labels_test = labels(is_test)';

  theta = train_svm(vectorized_data_train, labels_train, 1);


  % gpu = 2;
  % runEnsembleLarpTests('cifar',  'whatever', 'no-projection', gpu);
  % runEnsembleLarpTests('mnist',  'whatever', 'no-projection', gpu);
  % runEnsembleLarpTests('svhn',   'whatever', 'no-projection', gpu);
  % runEnsembleLarpTests('stl-10', 'whatever', 'no-projection', gpu);
  % runEnsembleLarpTests('coil-100', 'whatever', 'no-projection', gpu);

  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-38', 1);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-100', 1);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-266', 1);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-707', 1);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-1880', 1);

  % gpu = 1;
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-38',   'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-38',   'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-38',   'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-38',   'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-100',  'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-100',  'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-100',  'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-100',  'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-266',  'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-266',  'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-266',  'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-266',  'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-707',  'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-707',  'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-707',  'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-707',  'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-1880', 'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-1880', 'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-1880', 'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-1880', 'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-5000', 'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-5000', 'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-5000', 'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-5000', 'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('cifar',                        'whatever',      'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar',                        'whatever',      'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar',                        'whatever',      'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar',                        'whatever',      'projected-through-larpV3P3+convV0P0+fcV1', gpu);


  % gpu = 2;
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-38',   'no-projection',                            gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-38',   'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-38',   'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-38',   'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-100',  'no-projection',                            gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-100',  'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-100',  'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-100',  'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-266',  'no-projection',                            gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-266',  'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-266',  'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-266',  'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-707',  'no-projection',                            gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-707',  'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-707',  'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-707',  'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-1880', 'no-projection',                            gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-1880', 'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-1880', 'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-1880', 'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-5000', 'no-projection',                            gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-5000', 'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-5000', 'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-5000', 'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('mnist',                        'whatever',      'no-projection',                            gpu);
  % runEnsembleLarpTests('mnist',                        'whatever',      'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist',                        'whatever',      'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('mnist',                        'whatever',      'projected-through-larpV3P3+convV0P0+fcV1', gpu);




  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-38', 'no-projection', 1);
  % runEnsembleLarpTests('mnist-multi-class-subsampled', 'balanced-266', 'no-projection', 1);




  % gpu = 1;
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-38',   'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-38',   'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-38',   'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-38',   'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-100',  'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-100',  'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-100',  'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-100',  'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-266',  'no-projection',                            gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-266',  'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-266',  'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-266',  'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-707',  'no-projection',                            gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-707',  'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-707',  'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-707',  'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-1880', 'no-projection',                            gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-1880', 'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-1880', 'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-1880', 'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-5000', 'no-projection',                            gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-5000', 'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-5000', 'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % % runEnsembleLarpTests('cifar-multi-class-subsampled', 'balanced-5000', 'projected-through-larpV3P3+convV0P0+fcV1', gpu);

  % runEnsembleLarpTests('cifar',                        'whatever',      'no-projection',                            gpu);
  % runEnsembleLarpTests('cifar',                        'whatever',      'projected-through-larpV1P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar',                        'whatever',      'projected-through-larpV3P1+convV0P0+fcV1', gpu);
  % runEnsembleLarpTests('cifar',                        'whatever',      'projected-through-larpV3P3+convV0P0+fcV1', gpu);




























