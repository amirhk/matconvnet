function theta = train_svm(trainXC, trainY, c_penalty, max_iters)

  number_of_classes = max(trainY);
  %w0 = zeros(size(trainXC,2)*(number_of_classes-1), 1);
  % w0 = zeros(size(trainXC,2)*number_of_classes, 1);
  strategy = 'one_vs_all';
  switch strategy
    case 'one_vs_all'
      number_of_classifiers = number_of_classes; % 10
    case 'one_vs_one'
      number_of_classifiers = number_of_classes * (number_of_classes - 1) / 2; % 45
  end

  w0 = zeros(size(trainXC,2) * number_of_classifiers, 1);
  w = minFunc(@l2svmloss_one_vs_all, w0, struct('MaxIter', max_iters, 'MaxFunEvals', 1000), ...
              trainXC, trainY, number_of_classes, c_penalty);

  theta = reshape(w, size(trainXC,2), number_of_classes);

% 1-vs-all L2-svm loss function;  similar to LibLinear.
function [loss, g] = l2svmloss_one_vs_all(w, trainXC, trainY, number_of_classes, c_penalty)
  [M, N] = size(trainXC);
  number_of_classifiers = number_of_classes; % 10
  theta = reshape(w, N, number_of_classifiers);
  Y = bsxfun(@(trainY,ypos) 2*(trainY==ypos)-1, trainY, 1:number_of_classes);

  margin = max(0, 1 - Y .* (trainXC*theta));
  loss = (0.5 * sum(theta.^2)) + c_penalty*mean(margin.^2);
  loss = sum(loss);
  g = theta - 2*c_penalty/M * (trainXC' * (margin .* Y));
  g = g(:);

  %[v,i] = max(X*theta,[],2);
  %sum(i ~= y) / length(y)


% % 1-vs-1 L2-svm loss function;  similar to LibLinear.
% function [loss, g] = l2svmloss_one_vs_one(w, trainXC, trainY, number_of_classes, c_penalty)
%   [M, N] = size(trainXC);
%   number_of_classifiers = number_of_classes * (number_of_classes - 1) / 2; % 45
%   theta = reshape(w, N, number_of_classifiers);
%   Y = bsxfun(@(trainY,ypos) 2*(trainY==ypos)-1, trainY, 1:number_of_classes);

%   margin = max(0, 1 - Y .* (trainXC*theta));
%   loss = (0.5 * sum(theta.^2)) + c_penalty*mean(margin.^2);
%   loss = sum(loss);
%   g = theta - 2*c_penalty/M * (trainXC' * (margin .* Y));
%   g = g(:);

%   %[v,i] = max(X*theta,[],2);
%   %sum(i ~= y) / length(y)
