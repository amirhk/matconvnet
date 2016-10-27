% function [p,mu,phi,lPxtr] = mdgEM(x,K,its,minphi)
%
% Performs EM for a mixture of K axis-aligned (diagonal covariance
% matrix) Gaussians. its iterations are used and the input variances are
% not allowed to fall below minphi (if minphi is not given, its default
% value is 0). The parameters are randomly initialized using the mean
% and variance of each input.
%
% Input:
%
%   x(:,t) = the N-dimensional training vector for the tth training case
%   K = number of Gaussians to use
%   its = number of iterations of EM to apply
%   minphi = minimum variance of sensor noise (default: 0)
%
% Output:
%
%   p = probabilities of clusters
%   mu(:,c) = mean of the cth cluster
%   phi(:,c) = variances for the cth cluster
%   lPxtr(i) = log-probability of data after i-1 iterations
%
%

function [p,mu,phi,lPxtr] = mdgEM(x,K,its,minphi);

if nargin==3 minphi = 0; end;
N = size(x,1); T = size(x,2);

% Initialize the parameters
p = 10+rand(K,1); p = p/sum(p);
mn = mean(x,2); vr = std(x,[],2).^2;
mu = mn*ones(1,K)+randn(N,K).*(sqrt(vr)/10*ones(1,K));
phi = vr*ones(1,K)*2; phi = (phi>=minphi).*phi + (phi<minphi)*minphi;

% Do its iterations of EM
lPxtr = zeros(its,1);
for i=1:its
  % Do the E step
  r = zeros(K,1); rx = zeros(N,K); rDxm2 = zeros(N,K); lPx = zeros(1,T);
  iphi = 1./phi;
  logNorm = log(p)-0.5*N*log(2*pi)-0.5*sum(log(phi'),2);
  logPcx = zeros(K,T);
  for k=1:K
    logPcx(k,:) = logNorm(k)...
                - 0.5*sum((iphi(:,k)*ones(1,T)).*(x-mu(:,k)*ones(1,T)).^2,1);
  end;
  mx = max(logPcx,[],1); Pcx = exp(logPcx-ones(K,1)*mx); norm = sum(Pcx,1);
  PcGx = Pcx./(ones(K,1)*norm); lPx = log(norm) + mx;
  lPxtr(i) = sum(lPx);
  plot([0:i-1],lPxtr(1:i),'r-');
  title('Log-probability of data versus # iterations of EM');
  xlabel('Iterations of EM');
  ylabel('log P(D)');
  drawnow;
  r = mean(PcGx,2);
  rx = zeros(N,K); rDxm2 = zeros(N,K);
  for k=1:K
    rx(:,k) = mean(x.*(ones(N,1)*PcGx(k,:)),2);
    rDxm2(:,k) = mean((x-mu(:,k)*ones(1,T)).^2.*(ones(N,1)*PcGx(k,:)),2);
  end;

  % Do the M step
  p = r;
  mu = rx./(ones(N,1)*r');
  phi = rDxm2./(ones(N,1)*r');
  phi = (phi>=minphi).*phi + (phi<minphi)*minphi;
end;
