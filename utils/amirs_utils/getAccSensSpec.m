% -------------------------------------------------------------------------
function [acc, sens, spec] = getAccSensSpec(labels, predictions, debug_flag)
% -------------------------------------------------------------------------
  % enforce row vector before doing sum(...)
  labels = reshape(labels, 1, prod(size(labels)));
  predictions = reshape(predictions, 1, prod(size(predictions)));
  assert(length(labels) == length(predictions));

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
