function [Hvalues,varargout] = kmd(dat,kern,alpha,varargin)
%  [HVALUES,INFO] = KMD(X,LABELS) calculates the kernel maximum mean
%  discrepancy for samples from two distributions.
%  [..] = KMD(X,LABELS,ALPHA) conducts a test as to whether the samples are
%  from different distributions with level ALPHA  (default 0.05).
%
%  There are three different input choices:
%
%  1)  Input using RBF kernel with automatic kernel width detection.
%
%      This input choice will be adequate for most purposes.
%      Here the calling syntax is [...] = KMD(X,LABELS,...) with samples in the
%      n x p matrix X and corresponding [-1,1] labels in the nx1 vector
%      LABELS. n is the number of observations and p the number of variables.
%
%      The RBF kernel width sigma is computed according to a rule of thumb:
%
%         "Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
%         in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
%         and y of all distances between points from both data sets X and Y."
%
%
%  2) Input using custom kernel matrices
%
%     The kernel matrices can be given directly as global matrices. The
%     following kernel matrices must then exist as global variables in this
%     input choice:   Kxx, Kyy, Kxy
%
%     NOTE: The user is responsible that these global matrices are correct! If no
%     such global variables exist RBF kernel with rule of thumb (see 1))
%     will be used instead!
%
%
%  3) Input using spider
%
%     This input choice uses spider available from
%     http://www.kyb.tuebingen.mpg.de/bs/people/spider/index. In this case call
%     [...] = KMD(SAMPLESDAT,KERN,...). Where SAMPLESDAT is a spider data
%     object including the [-1,1] labels as y-data. KERN is a spider kernel
%     object.
%
%
%  KMD(...,MMD_METHOD) additionally specifies the method for computing
%  MMD asymptotic statistics (see NIPS2006 paper). MMD_METHOD is one (or a
%  cell array of multiple) of
%
%      'bootstrap'     : bootstraps the significance threshold of the MMD
%                        asymptotic statistics (default)
%      'approxmoments' : as 'moments' but uses an approximation to the
%                        4th moment
%                        NOTE: needs MatLab Statistics Toolbox
%      'moments'       : estimates the threshold by matching the first 4
%                        moments to Pearsons curves.
%                        NOTE: needs MatLab Statistics Toolbox and
%                              U4thmoment.c. Compile it with
%                              "mex U4thmoment.c"
%      ''              : skip the asymptotic MMD
%
%  For moderately small data sets take 'approxmoments', since it is then faster
%  and more accurate then 'bootstrap'. For large data sets 'bootstrap' should
%  be faster then 'approxmoments'. 'moments' should only be used for
%  comparisons on very small data sets. See also NIPS 2006 paper for
%  discussion.
%
%
%  Two variables will be given as output [HVALUES,INFO] = KMD(...).
%
%  HVALUES contains the result of each test based on significance niveau
%  ALPHA. HVALUES is 1 if the null hypothesis is rejected and zero if
%  accepted. The ordering of vector HVALUES is [mmd,mmd_rad] with
%
%      1)   mmd       : asymptotic bound (if calculated)
%      2)   mmd_rad   : Rademacher bound
%
%  Additional information is output in structure INFO. For "mmd" and
%  "mmd_rad" it will have the fields
%
%     .H       : if null hypothesis is rejected using test mmd or mmd_rad
%     .val     : value of the test statistics
%     .bound   : bound of the test statistics for given significance niveau
%
%
%  EXAMPLE1 (automatic RBF kernel):
%     % calculating asymptotic MMD  with RBF kernel (automatic width)
%     % for 5% significance niveau
%     clear all
%     X = rand(100,3072);               % random data
%     labels = 2*(rand(100,1)>0.5)-1;   % random labels in [-1,1];
%     [H,info] = kmd(X,labels)          % computes MMD
%                                       % H should of course be [0 0] here
%
%  EXAMPLE2 (custom kernel):
%     x = rand(100,3);                  %first sample
%     y = rand(100,3)+1;                %second sample different
%     global Kxx Kyy Kxy                %kernel matrices
%     %polynomial kernel
%     Kxx = (x*x' + 1).^2;
%     Kyy = (y*y' + 1).^2;
%     Kxy = (x*y' + 1).^2;
%     labels = [ones(size(x,1),1);-ones(size(y,1),1)];
%     [H,info] = kmd([x;y],labels,0.05,'approxmoments');
%     clear global  Kxx Kyy Kxy
%
%
%  EXAMPLE3 (using spider):
%     use_spider;
%     d = data([rand(100,5);rand(100,5)+1],[ones(100,1);-ones(100,1)]);
%     % compute MMD with polynomial  kernel
%     [H,info] = kmd(d,kernel('poly',2),0.05);
%
%
%  Reference:  Gretton, Borgwardt, Rasch, Schoelkopf, Smola. (NIPS 2006)

%
% Author: Malte J. Rasch, malte@igi.tu-graz.ac.at
%
% Version of 2/10/2007 - public version



  global Kxx Kyy Kxy



  %some options
  global SAVE_ALL
  SAVE_ALL = 0;        % whether to save more fields for info
                       % (e.g. bootstrapped samples, moments)

  WITHREPLACEMENT = 1; % method for drawing bootstrapping samples
  NTIMES = 150;        % how many bootstrapping rounds
  FRAC = 1;            % fraction of samples to use for bootstrapping (1 means
                       % sample size M)



  %parameter parsing

  if ~exist('stattypes','var') || isempty('stattypes')
    stattypes = {'MMD'};
  end

  if ~iscell(stattypes)
    stattypes = {stattypes};
  end

  %significance
  if ~exist('alpha','var')
    alpha = 0.05;
  end

  if nargin>3
    %mmd3_method given
    mmd3_method = varargin{1};
  else
    mmd3_method  = 'bootstrap';
  end

  if ~iscell(mmd3_method)
    mmd3_method = {mmd3_method};
  end

  if length(mmd3_method)>1
    %multiple mmd3_methods,  will have extra fields
    SAVE_ALL = 1;
  end

  if ~exist('kern','var') || isempty(kern)
    kern = 'thumb';
  end

  if ~isobject(dat)
    %INPUT CHOICE 2
    GLOBAL_K = 1;
    CLEAR_GLOBAL = 0;

    X = dat;
    labels = kern;

    poslabels = find(labels==1);
    neglabels = find(labels==-1);


    if isempty(Kxx) || ischar(Kxx)
      kern = 'thumb';
    else
      fprintf('WARNING: Calculate MMD using custom kernel in GLOBAL variables.\n');
      fprintf('         Issue ''clear global Kxx'' if you do not want that or do not know what that means\n');
    end


  else
    %INPUT CHOICE 1
    GLOBAL_K = 0;
    CLEAR_GLOBAL = 1;

    labels = get_y(dat);
    X = get_x(dat);

    poslabels = find(labels==1);
    neglabels = find(labels==-1);
  end






  if isempty(poslabels) || isempty(neglabels)
    error('kmd(): each sample should have at least one observation ');
  end



  % KERNEL
  if ischar(kern) && strcmp(kern,'thumb')
    % RULE OF THUMB

    % calculates the kernel matrix and estimate sigma

    CLEAR_GLOBAL = 1;
    GLOBAL_K = 1;


    x = X(poslabels,:);
    y = X(neglabels,:);

    %% Kxx
    G = x*x';
    L = size(G,1);

    nor = G(1:L+1:L^2);
    Kxx = -2*G + repmat(nor',[1,L]) +  repmat(nor,[L,1]);

    clear G L nor

    %% Kyy
    G = y*y';
    L = size(G,1);

    nor = G(1:L+1:L^2);
    Kyy = -2*G + repmat(nor',[1,L]) +  repmat(nor,[L,1]);
    clear G L nor


    %% Kxy
    G = x*y';
    L = size(G,1);

    norx = sum(x.*x,2);
    nory = sum(y.*y,2);
    Kxy = (-2*G + repmat(norx,[1,length(nory)]) +  repmat(nory',[length(norx),1]))';
    clear G L norx nory


    %now get the median distance
    mdist = median(Kxy(Kxy~=0));

    sigma = sqrt(mdist/2);
    if sigma ==0
      sigma =1;
    end

    %apply RBF
    Kxx = exp(-1/2/sigma^2 * Kxx);
    Kyy = exp(-1/2/sigma^2 * Kyy);
    Kxy = exp(-1/2/sigma^2 * Kxy);

    kernelsize = sigma;

    clear sigma mdist x y

    fprintf('Using RBF kernel with sigma=%1.2f\n',kernelsize);

  end

  %sizes
  m = length(poslabels);
  n = length(neglabels);

  N = max(m,n);
  M = min(m,n);


  Hs = [];

  out = [];


  %% MMD

  if any(cell2mat(strfind(stattypes,'MMD'))==1)
    %this is "standard" MMD as compared to linear time MMD


    if ~GLOBAL_K
      dx = data(X(poslabels,:));
      dy = data(X(neglabels,:));

      Kxx = calc(kern,dx);
      Kyy = calc(kern,dy);
      Kxy = calc(kern,dx,dy);
    end


    if exist('kernelsize','var')
      out.mmd.ksize = kernelsize;
    end

    %calculate the mmd3 statistics
    out = submmd(alpha);


    if isempty(mmd3_method)
      %no mmd3 calculation
      out = rmfield(out,'mmd3');
    else

      if any(strcmp(mmd3_method,'bootstrap'))
        % bootstrapping the mmd3 bound

        out = submmd3bound(out,alpha,FRAC,NTIMES,WITHREPLACEMENT);
      end

      %whether to use the moments method
      momentsif = any(strcmp(mmd3_method,'moments'))  || any(strcmp(mmd3_method,'approxmoments')) ;
      APPROXMOMENTS_4TH = any(strcmp(mmd3_method,'approxmoments')) || M>=200;

      %run both ways if requested and sample size low enough
      if SAVE_ALL && M<200 && (any(strcmp(mmd3_method,'moments'))  && any(strcmp(mmd3_method,'approxmoments')) )
        APPROXMOMENTS_4TH =2;
      end

      if momentsif
        %run moments mmd3
        out = subMmd3Moments(out,alpha,APPROXMOMENTS_4TH);
      end
    end

  end




  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %general output




  Hvalues = [out.mmd.H, out.mmd_rad.H];

  if CLEAR_GLOBAL
    clear global Kxx
    clear global Kyy
    clear global Kxy
  end



  %make nicer output if multiple MMD3_METHODS requested
  mmd =out.mmd;
  if isfield(mmd,'H_approxmoments')
    mmd.approxmoments.val = mmd.val;
    mmd.approxmoments.H = mmd.H_approxmoments;
    mmd.approxmoments.bound = mmd.bound_approxmoments;

    mmd = rmfield(mmd,{'H_approxmoments','bound_approxmoments'});
    mmd.H = []; %ambivalent
    mmd.bound = [];
  end

  if isfield(out.mmd,'H_moments')
    mmd.moments.val = mmd.val;
    mmd.moments.H = mmd.H_moments;
    mmd.moments.bound = mmd.bound_moments;

    mmd = rmfield(mmd,{'H_moments','bound_moments'});
    mmd.H = []; %ambivalent
    mmd.bound = [];
  end

  if isfield(out.mmd,'H_boot')
    mmd.bootstrap.val = mmd.val;
    mmd.bootstrap.H = mmd.H_boot;
    mmd.bootstrap.bound = mmd.bound_boot;
    mmd.bootstrap.info = mmd.boot_info;

    mmd = rmfield(mmd,{'H_boot','bound_boot','boot_info'});
    mmd.H = []; %ambivalent
    mmd.bound = [];
  end


  if isempty(mmd.H)
    mmd = rmfield(mmd,{'H','bound','val'});

    %for multiple methods just take the first as asymptotic HVALUE
    Hvalues(1) = mmd.(mmd3_method{1}).H;

  else
    mmd.method = mmd3_method{1};
  end

  out.mmd = mmd;

  varargout{1}.mmd = out.mmd;
  varargout{1}.mmd_rad = out.mmd_rad;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SUB-FUNCTIONS FOR MMD TESTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function out = subMmd3Moments(out,alpha,APPROXIMATION_4TH);
% calculates MMD once with global Ks

  global Kxx Kyy Kxy

  global SAVE_ALL

  m = length(Kxx);
  n = length(Kyy);

  N = max(m,n);
  M = min(m,n);


  %ustats (not normalized yet)
  U = Kxx(1:M,1:M) + Kyy(1:M,1:M) - Kxy(1:M,1:M) - Kxy(1:M,1:M)';



  %variance
  U(1:M+1:M^2) = 0; %clear diag
  U2 = U.^2;
  U_m2 = 2*sum(U2(:))/(M*(M-1))^2; %divided once to get mean, second  because of formula



  %third moment (only 1/M^3  term )
  intU = zeros(size(U)); %intermediate
  for i1=1:M
    for i2 = i1+1:M
      intU(i1,i2) = sum(U(i1,:).*U(i2,:));
    end
  end
  U_m3 = 16 * sum(sum(intU.*U))/ (M*(M-1))^3; %no  (M-2) since above only
                                              %sum not mean % times 2
                                              %since half side only




  %4th moment

  if ~APPROXIMATION_4TH  || (APPROXIMATION_4TH==2 && SAVE_ALL)

    %calculate with MEX
    U_m4 = U4thmoment(U);


    %second term (is negligible anyway)
    U_m4 = U_m4 + 4*U_m2^2/(M*(M-1))^2;

    kurt = U_m4/U_m2^2;

    %compute bound according to pearsrnd
    U_moments = {0,sqrt(U_m2),U_m3/U_m2^(3/2),kurt};

    %just sample the distribution for now
    [r1,type global_p0] = pearsrnd(U_moments{:},5000,1);

    %for now just compute empirical CDF to get the 95% quantile
    [fi,xi]  = ecdf(r1);

    bound= xi(find(fi<1-alpha,1,'last'));

    if SAVE_ALL
      out.mmd.bound_moments = bound;
      out.mmd.H_moments = out.mmd.val>bound;
    end

  end

  if APPROXIMATION_4TH
    %Approximation to the 4th moment
    %take the kurtosis to be the lower limit 2*(square of skewness +1)

    kurt = 2*((U_m3)^2/U_m2^3 + 1);

    U_m4 = kurt * U_m2^2;

    %compute bound according to pearsrnd
    U_moments = {0,sqrt(U_m2),U_m3/U_m2^(3/2),kurt};

    %just sample the distribution for now
    [r1,type global_p0] = pearsrnd(U_moments{:},5000,1);

    %for now just compute empirical CDF to get the 95% quantile
    if ~isnan(r1(1))
      [fi,xi]  = ecdf(r1);


      bound= xi(find(fi<1-alpha,1,'last'));
      if SAVE_ALL
        out.mmd.bound_approxmoments = bound;
        out.mmd.H_approxmoments = out.mmd.val>bound;
      end

    else
      bound = NaN;
      if SAVE_ALL
        out.mmd.H_approxmoments = NaN;
        out.mmd.bound_approxmoments = NaN;
      end
    end

  end

  %output
  if 0
    out.mmd.U_m2 = U_m2;
    out.mmd.U_m3 = U_m3;
    out.mmd.U_m4 = U_m4;
  end

  %overwrite
  out.mmd.bound = bound;
  out.mmd.H = out.mmd.val>bound;





%------------------------------------------------------------------------------------




function out = submmd(alpha);
% calculates MMD once with global Ks

  global Kxx Kyy Kxy

  global SAVE_ALL

  m = length(Kxx);
  n = length(Kyy);

  N = max(m,n);
  M = min(m,n);


  %Kxx
  sumKxx   = sum(Kxx(:));
  if m~=n
    sumKxx_M   = sum(sum(Kxx(1:M,1:M)));
  else
    sumKxx_M = sumKxx;
  end

  dgxx = Kxx(1:m+1:m^2);

  sumKxxnd =  sumKxx - sum(dgxx); %no diags


  R = max(dgxx); %upper bound on kernel, should be one
  R_M = max(dgxx(1:M));

  h_u = sum(Kxx(1:M,1:M)) - dgxx(1:M); %one sided sum


  %Kyy
  sumKyy   = sum(Kyy(:));
  if m~=n
    sumKyy_M   = sum(sum(Kyy(1:M,1:M)));
  else
    sumKyy_M = sumKyy;
  end

  dgyy = Kyy(1:n+1:n^2);

  sumKyynd =  sumKyy - sum(dgyy); %no diags

  R = max([R,max(dgyy)]); %upper bound on kernel, should be one
  R_M = max([R,max(dgyy(1:M))]);

  h_u = h_u + sum(Kyy(1:M,1:M)) - dgyy(1:M); %one sided sum


  %Kxy
  sumKxy   = sum(Kxy(:));
  if m~=n
    sumKxy_M  = sum(sum(Kxy(1:M,1:M)));
  else
    sumKxy_M  = sumKxy;
  end

  dg = Kxy(1:size(Kxy,1)+1:size(Kxy,1)*M); %up to M only

  h_u = h_u - sum(Kxy(1:M,1:M)) - sum(Kxy(1:M,1:M)') + 2*dg; %one sided sum


  %compute MMDs
  %corrolary 11
  mmd1 = sqrt(sumKxx/(m*m) +  sumKyy/(n * n) - 2/m/n * sumKxy);


  %equation 14
  %only for m==n (=M)
  mmd3 = sum(h_u)/M/(M-1);



  %%
  %compute bound for MMD1
  %n=m: just take first M samples

  %rademacher bound on MMD1
  D1 = 2*sqrt(R_M/M) + sqrt(log(1/alpha)*4*R_M/M);




  %%
  % output

  %MMD1 (RADEMACHER)
  out.mmd_rad.val = mmd1;
  out.mmd_rad.bound = D1;
  out.mmd_rad.H = mmd1>D1; %hypothesis H=1 if null hypothesis
                           %is to be rejected with significance alpha

  %MMD3
  out.mmd.val = mmd3;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [info] = submmd3bound(info,alpha,frac,ntimes,WITHREPLACEMENT);
% implements the bootstrapping approach to the MMD3 bound by shuffling
% the kernel matrix
%  frac   : fraction of data used for bootstrap
%  ntimes : how many times MMD is to be evaluated

  global Kxx Kxy Kyy

  global SAVE_ALL



  m = length(Kxx);
  n = length(Kyy);

  M = min(m,n);
  N = max(m,n);

  poslabels = (1:m)';
  neglabels = (m+1:m+n)';


  %bootstrap
  bootmmd3 = zeros(ntimes,1);
  for i =1:ntimes

    [xinds,yinds] = subbootstrap(poslabels,neglabels,frac,WITHREPLACEMENT);


    newm = length(xinds);
    newn = length(yinds);
    newM = min(newm,newn);

    %get new kernel matrices (without concat to big matrix to save memory)
    xind1 = xinds(xinds<=m);
    xind2 = xinds(xinds>m)-m;
    yind1 = yinds(yinds<=m);
    yind2 = yinds(yinds>m)-m;

    %Kxx
    nKxx = [Kxx(xind1,xind1),Kxy(xind2,xind1)';...
              Kxy(xind2,xind1),Kyy(xind2,xind2)];


    dgxx = nKxx(1:newm+1:newm^2);
    h_u = sum(nKxx(1:newM,1:newM)) - dgxx(1:newM); %one sided sum

    clear nKxx

    %Kyy
    nKyy = [Kxx(yind1,yind1),Kxy(yind2,yind1)';...
              Kxy(yind2,yind1), Kyy(yind2,yind2)];

    dgyy = nKyy(1:newn+1:newn^2);
    h_u = h_u + sum(nKyy(1:newM,1:newM)) - dgyy(1:newM); % one sided sum

    clear nKyy

    %Kxy
    nKxy = [Kxx(yind1,xind1),Kxy(xind2,yind1)';...
              Kxy(yind2,xind1),Kyy(yind2,xind2)];

    dg = nKxy(1:size(nKxy,1)+1:size(nKxy,1)*newM); % up to M only
    h_u = h_u - sum(nKxy(1:newM,1:newM)) - sum(nKxy(1:newM,1:newM)') + 2*dg; % one sided sum

    clear nKxy


    %now calculate mmd3
    bootmmd3(i) = sum(h_u)/newM/(newM-1);
  end



  bootmmd3 = sort(bootmmd3,'descend');
  aind = floor(alpha*ntimes); % better less than too much (-> floor);


  %take threshold in between aind and the next smaller value:
  bound = sum(bootmmd3([aind,aind+1]))/2;

  if SAVE_ALL
    info.mmd.boot_info.bootval = bootmmd3; %just for info
    info.mmd.boot_info.WITHREPLACEMENT = WITHREPLACEMENT;
    info.mmd.boot_info.FRAC = frac;


    info.mmd.bound_boot = bound;
    info.mmd.H_boot = info.mmd.val > bound;
  end


  info.mmd.bound = bound;
  info.mmd.H = info.mmd.val > bound;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xinds,yinds] = subbootstrap(poslabels,neglabels,frac,WITHREPLACEMENT)
%gets bootstrapped indices

  m = length(poslabels);
  n = length(neglabels);

  nsamples = ceil(frac*(min(m,n)));

  if WITHREPLACEMENT
    pinds = ceil(rand(nsamples,1)*(m));
    ninds = ceil(rand(nsamples,1)*(n));

  else
    pinds = randperm(m);
    pinds = pinds(1:nsamples);
    ninds = randperm(n);
    ninds = ninds(1:nsamples);
  end

  newlabels = [poslabels(pinds);neglabels(ninds)];



  %shuffle label index
  newlabels = newlabels(randperm(length(newlabels)));

  %index for shuffled set
  xinds = newlabels(1:nsamples);
  yinds = newlabels(nsamples+1:end);



