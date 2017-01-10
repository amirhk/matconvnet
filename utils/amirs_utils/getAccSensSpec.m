% -------------------------------------------------------------------------
function [acc, sens, spec] = getAccSensSpec(labels, predictions, debug_flag)
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


  % enforce row vector before doing sum(...)
  labels = reshape(labels, 1, prod(size(labels)));
  predictions = reshape(predictions, 1, prod(size(predictions)));
  assert(length(labels) == length(predictions));
  assert(numel(unique(labels)) == 2); % only to be used for two-class imdbs

  positive_class_num = 2;
  negative_class_num = 1;
  TP = sum((labels == predictions) .* (predictions == positive_class_num)); % TP
  TN = sum((labels == predictions) .* (predictions == negative_class_num)); % TN
  FP = sum((labels ~= predictions) .* (predictions == positive_class_num)); % FP
  FN = sum((labels ~= predictions) .* (predictions == negative_class_num)); % FN
  acc = (TP + TN) / (TP + TN + FP + FN);
  sens = TP / (TP + FN);
  spec = TN / (TN + FP);
  if debug_flag
    afprintf( ...
      sprintf('[INFO] Acc: %3.6f Sens: %3.6f Spec: %3.6f\n', ...
      acc, ...
      sens, ...
      spec));
  end
