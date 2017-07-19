% -------------------------------------------------------------------------
function average_c_separation = getAverageClassCSeparation(imdb)
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

  % assert(numel(unique(imdb.images.labels)) == 2);

  vectorized_imdb = getVectorizedImdb(imdb);

  vectorized_data_per_class = {};
  unique_labels = reshape(unique(imdb.images.labels), 1, []);
  for j = unique_labels
    vectorized_data_per_class{j} = vectorized_imdb.images.data(imdb.images.labels == j,:);
  end

  class_index_combos = combnk(unique_labels, 2);
  assert(size(class_index_combos, 1) == length(unique_labels) * (length(unique_labels) - 1) / 2, 'wtf!');

  all_c_separations = [];
  afprintf(sprintf('[INFO] computing c-sep for %d pairs:\t', size(class_index_combos, 1)));
  for j = 1 : size(class_index_combos, 1)
    fprintf('%d,  ', j);
    class_index_1 = class_index_combos(j,1);
    class_index_2 = class_index_combos(j,2);
    vectorized_data_1 = vectorized_data_per_class{class_index_1};
    vectorized_data_2 = vectorized_data_per_class{class_index_2};

    mean_1 = mean(vectorized_data_1);
    mean_2 = mean(vectorized_data_2);

    cov_1 = cov(vectorized_data_1);
    cov_2 = cov(vectorized_data_2);

    c_separation = norm(mean_1 - mean_2) / sqrt(max(trace(cov_1), trace(cov_2)));
    all_c_separations(end+1) = c_separation;
  end
  fprintf('\n');

  average_c_separation = mean(all_c_separations);
