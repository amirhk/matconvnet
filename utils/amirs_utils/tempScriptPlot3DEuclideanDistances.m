% -------------------------------------------------------------------------
function tempScriptPlot3DEuclideanDistances(dataset, posneg_balance, save_results)
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

  % -------------------------------------------------------------------------
  %                                                                     Setup
  % -------------------------------------------------------------------------
  afprintf(sprintf('[INFO] Setting up experiment...\n'));
  [~, experiments] = setupExperimentsUsingProjectedImbds(dataset, posneg_balance, 1);
  assert(numel(experiments) == 2);
  afprintf(sprintf('[INFO] done!\n'));
  printConsoleOutputSeparator();

  imdb_1 = getVectorizedImdb(experiments{1}.imdb);
  imdb_2 = getVectorizedImdb(experiments{2}.imdb);

  data_original = imdb_1.images.data;
  data_original_a = data_original(imdb_1.images.labels == 1,:);
  data_original_b = data_original(imdb_1.images.labels == 2,:);
  data_angle_separated = imdb_2.images.data;
  data_angle_separated_a = data_angle_separated(imdb_2.images.labels == 1,:);
  data_angle_separated_b = data_angle_separated(imdb_2.images.labels == 2,:);

  figure;

  a = strfind(dataset,'var-');
  b = strfind(dataset(a:end),'-train');
  variance = str2num(dataset(a+4:a+b-2));

  subplot(1,2,1),
  title('Original Data');
  hold on,
  grid on,
  scatter3(data_original_a(:,1), data_original_a(:,2), data_original_a(:,3), 'bs');
  scatter3(data_original_b(:,1), data_original_b(:,2), data_original_b(:,3), 'ro');
  xlim([-15, 15]);
  ylim([-15, 15]);
  zlim([-15, 15]);
  hold off
  view(15, 25);

  subplot(1,2,2),
  title('Angle Separated Data');
  hold on,
  grid on,
  scatter3(data_angle_separated_a(:,1), data_angle_separated_a(:,2), data_angle_separated_a(:,3), 'bs');
  scatter3(data_angle_separated_b(:,1), data_angle_separated_b(:,2), data_angle_separated_b(:,3), 'ro');
  xlim([-15, 15]);
  ylim([-15, 15]);
  zlim([-15, 15]);
  hold off
  view(15, 25);

  suptitle(sprintf('Gaussian Class Variance: %d', variance));
