function theta = train_svm(trainXC, trainY, C)

  numClasses = max(trainY);
  %w0 = zeros(size(trainXC,2)*(numClasses-1), 1);
  w0 = zeros(size(trainXC,2)*numClasses, 1);
  w = minFunc(@my_l2svmloss, w0, struct('MaxIter', 1000, 'MaxFunEvals', 1000), ...
              trainXC, trainY, numClasses, C);

  theta = reshape(w, size(trainXC,2), numClasses);

% 1-vs-all L2-svm loss function;  similar to LibLinear.
function [loss, g] = my_l2svmloss(w, trainXC, trainY, K, C)
  [M,N] = size(trainXC);
  theta = reshape(w, N,K);
  Y = bsxfun(@(trainY,ypos) 2*(trainY==ypos)-1, trainY, 1:K);

  margin = max(0, 1 - Y .* (trainXC*theta));
  loss = (0.5 * sum(theta.^2)) + C*mean(margin.^2);
  loss = sum(loss);
  g = theta - 2*C/M * (trainXC' * (margin .* Y));
  g = g(:);

  %[v,i] = max(X*theta,[],2);
  %sum(i ~= y) / length(y)
