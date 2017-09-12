d = 100;

x = randn(d,1);
y = randn(d,1);

a = [];
b = [];

repeat_count = 1;
% repeat_count = 100;

hsic_test_results = [];

t_test_results = [];
t_test_p_values = [];

ks_test_results = [];
ks_test_p_values = [];

number_of_samples = 1000;

for kk = 1 : repeat_count

  for ii = 1:number_of_samples
    r_1 = randn(d,1);
    r_2 = randn(d,1);
    r_3 = randn(d,1);
    r_4 = randn(d,1);

    % a(ii) = max([r_1'*x]); % dependent: 10^4
    % b(ii) = max([r_1'*x]);

    % a(ii) = max([r_1'*x]); % independent: 10^-2
    % b(ii) = max([r_2'*y]);

    % a(ii) = max([r_1'*x]); % dependent: 10^2
    % b(ii) = max([r_1'*y]);

    % a(ii) = max([r_1'*x, r_2'*x]); % ~50
    % b(ii) = max([r_1'*y, r_2'*y]);

    a(ii) = max([r_1'*x, r_2'*x, r_3'*x, r_4'*x]); % ~1
    b(ii) = max([r_1'*y, r_2'*y, r_3'*y, r_4'*y]);
  end

  e=ones(number_of_samples,1);
  H=eye(number_of_samples)-(1/number_of_samples)*e*e';

  hsic_test_results(kk) = trace(a'*a*H*b'*b*H)/(number_of_samples^2);
  % [t_test_results(kk), t_test_p_values(kk)] = ttest2(a, b);
  % [ks_test_results(kk), ks_test_p_values(kk)] = kstest2(a, b);

end


hsic_test_results

% t_test_results
% t_test_p_values
% ks_test_results
% ks_test_p_values


% fprintf('Two-sample t-test: \t\t 0 (%%%.2f) ; 1 (%%%.2f)\n', 100 * (1 - sum(t_test_results) / length(t_test_results)), 100 * sum(t_test_results) / length(t_test_results));
% fprintf('Two-sample Kolmogorov-Smirnov test: 0 (%%%.2f) ; 1 (%%%.2f)\n', 100 * (1 - sum(ks_test_results) / length(ks_test_results)), 100 * sum(ks_test_results) / length(ks_test_results));







% clear all

% d=10;
% x=rand(d,1);
% y=rand(d,1);
% n=1000
% e=ones(n,1);
% H=eye(n)-(1/n)*e*e';

% for ii=1:n
% r1=randn(d,1);
% r2=randn(d,1);
% A(ii)=max(r1'*x);
% %B(ii)=A(ii);
% B(ii)=max(r1'*y);
% end

