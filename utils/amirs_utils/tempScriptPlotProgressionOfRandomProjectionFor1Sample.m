% -------------------------------------------------------------------------
function tempScriptPlotProgressionOfRandomProjectionFor1Sample(dataset, posneg_balance, should_filter_out_test_set)
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

fh_projection_utils = projectionUtils;
experiments = {};

afprintf(sprintf('[INFO] Loading original imdb...\n'));
tmp_opts.dataset = dataset;
tmp_opts.posneg_balance = posneg_balance;
original_imdb = loadSavedImdb(tmp_opts, 1);
if should_filter_out_test_set
  original_imdb = filterImdbForSet(original_imdb, 1, 1);
end
afprintf(sprintf('[INFO] done!\n'));

repeat_count = 100;

tmp = getVectorizedImdb(original_imdb);
tmp_sample_projected_wo_relu = zeros(size(tmp.images.data, 2), repeat_count);
tmp_sample_projected_w_relu = zeros(size(tmp.images.data, 2), repeat_count);




s_1 = size(original_imdb.images.data, 1);
s_2 = size(original_imdb.images.data, 2);
s_3 = size(original_imdb.images.data, 3);
s_4 = size(original_imdb.images.data, 4);

tmp_imdb_wo_relu = original_imdb;
tmp_imdb_w_relu = original_imdb;
for i = 1:repeat_count
  tmp_imdb_wo_relu = getVectorizedImdb(fh_projection_utils.getDenslyProjectedImdb(tmp_imdb_wo_relu, 1, 0));
  tmp_sample_projected_wo_relu(:,i) = tmp_imdb_wo_relu.images.data(1,:);
  tmp_imdb_w_relu = getVectorizedImdb(fh_projection_utils.getDenslyProjectedImdb(tmp_imdb_w_relu, 1, 1));
  tmp_sample_projected_w_relu(:,i) = tmp_imdb_w_relu.images.data(1,:);

  tmp_imdb_wo_relu = get4DImdb(tmp_imdb_wo_relu, s_1, s_2, s_3, s_4);
  tmp_imdb_w_relu = get4DImdb(tmp_imdb_w_relu, s_1, s_2, s_3, s_4);

  % pause();
end

figure
subplot(2,1,1)
imshow(tmp_sample_projected_wo_relu);
title('wo relu');

subplot(2,1,2)
imshow(tmp_sample_projected_w_relu);
title('w relu');

