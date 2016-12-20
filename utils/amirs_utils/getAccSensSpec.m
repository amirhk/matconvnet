% -------------------------------------------------------------------------
function [acc, sens, spec] = getAccSensSpec(labels, predictions)
% -------------------------------------------------------------------------
  positive_class_num = 2;
  negative_class_num = 1;
  TP = sum((labels == predictions) .* (predictions == positive_class_num)); % TP
  TN = sum((labels == predictions) .* (predictions == negative_class_num)); % TN
  FP = sum((labels ~= predictions) .* (predictions == positive_class_num)); % FP
  FN = sum((labels ~= predictions) .* (predictions == negative_class_num)); % FN
  acc = (TP + TN) / (TP + TN + FP + FN);
  sens = TP / (TP + FN);
  spec = TN / (TN + FP);
