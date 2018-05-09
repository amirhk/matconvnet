X = ... % generated from randn(d, n) and then centered.
  [-0.9472   -0.9027   -0.3969    1.1126   -0.3651    0.3187   -0.2514    0.6533    0.1893    0.5891;
    1.1735    0.1985   -0.9874    0.0702   -0.4653   -0.2010   -0.0518   -0.6036    0.6354    0.2315;
    1.1846   -0.7662    0.4228   -1.1734    0.1357    0.7378   -0.7375    0.4346    0.8221   -1.0605;
   -1.3976    2.2530   -0.1869   -1.7590    0.3661   -1.1083    0.7946    1.1974   -0.0475   -0.1117;
    0.1487    0.6655   -0.3634    1.2485    0.1658   -0.5706    0.8592   -0.3042   -1.3953   -0.4541];

k = 2; % Projecting into 2 dimensions


% PCA
[U, S, V] = svd(X * X');
U_pca = U(:,1:k);
X_pca = U_pca' * X;


% RP
U_rp = ... % gerenated from U_rp = 1 / sqrt(k) * randn(size(X, 1), k);
  [ 0.2305    0.0016;
    0.0327   -0.0815;
   -0.4871    0.7921;
    0.9523    0.7349;
    0.3850   -0.2448];
X_rp = U_rp' * X;


pdist_original = L2_distance(X, X) .^ 2;
pdist_pca = L2_distance(X_pca, X_pca) .^ 2;
pdist_rp = L2_distance(X_rp, X_rp) .^ 2;


fprintf('|D_{ORIGINAL} - D_{PCA}|_F = %.4f\n', norm(pdist_original - pdist_pca, 'fro'));
fprintf('|D_{ORIGINAL} - D_{RP}|_F = %.4f\n', norm(pdist_original - pdist_rp, 'fro'));


