%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                  Fit a 1D Gaussian Functions to Data
%
%                   Created by Manuel A. Diaz, Jan 2016.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PURPOSE:  
%   Fit a 1D Noisy Gaussian data by two methods: Analitical and LSQ method.
% 
% INPUT:
%   X: one-dimensional data array of size n. 
%   A0 = [Amp,Rate]: Inital guess parameters.
%   A = [Amp,Rate]: simulated exponential decay parameters.
%
% OUTPUT: 
%   Exponential function parameters.
%
% NOTE:
%   1.This routine uses Matlab's 'lsqcurvefit' function to fit.
%   2.The initial values in x0 must be close to x in order for the fit
%   to converge to the values of x (especially if noise is added).
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; % memory
close all; % windows

%% ---Fitting function---
% Coeficients A convention:
%	A = [Amplitude, Rate]
%
% X-data convention:
%	X is of size(1,n) 
% 
% Define f(t) = a*exp(b*t) : 
f = @(A,X) A(1)*exp(A(2)*X);

% Define g(x) = a*exp(-(x-mu)^2/(2*sigma^2)):
g = @(A,X) A(1)*exp(-(X-A(2)).^2/(2*A(3)^2));

%% ---Problem parameters---
% Load Noisy data
x = [-30.6711,-29.4090,-28.1469,-26.8848,-25.6228,-24.3607,...
    -23.0986,-21.8365,-20.5744,-19.3124,-18.0503,-16.7882,...
    -15.5261,-14.2640,-13.0020,-11.7399,-10.4778,-9.2157,...
    -7.9536,-6.6916,-5.4295,-4.1674,-2.9053,-1.6432,-0.3812,...
    0.8809,2.1430,3.4051,4.6671,5.9292,7.1913,8.4534,9.7155,...
    10.9775,12.2396,13.5017,14.7638,16.0259,17.2879,18.5500,...
    19.8121,21.0742,22.3363,23.5983,24.8604,26.1225,27.3846,...
    28.6466,29.9087,31.1708,32.4329];
y = [2.3032,3.8366,1.7067,-3.3135,-2.3072,0.0886,2.8890,0.8625,...
    -0.2992,3.0865,3.5677,4.2557,2.2367,0.9734,0.4389,-0.5235,...
    -2.6371,0.5427,10.5620,24.6725,48.0479,59.2771,58.4992,...
    43.7527,26.4502,16.2949,6.0480,-0.4139,2.1088,0.9435,...
    -0.2748,-3.5531,-0.9084,-3.4756,-3.0706,3.485,-4.169,...
    -0.9073, 3.7780,1.2292,-3.4927,3.3077,0.5342,1.6956,...
    2.2156,-3.7096,-1.6171,3.2179,2.4154,0.4380,-4.3211];

% Plot data
scatter(x,y,'ro'); hold on;

%% ---Fit Data: Analitical Strategy---
% Cut Gaussian bell data
ymax=max(y); xnew=[]; ynew=[];
for n=1:length(x), if y(n)>0.1*ymax; xnew=[xnew,x(n)]; ynew=[ynew,y(n)]; end, end

% Fitting
ylog=log(ynew); xlog=xnew; B=polyfit(xlog,ylog,2);

% Compute Parameters
sigma=sqrt(-1/(2*B(1))); mu=B(2)*sigma^2; a=exp(B(3)+mu^2/(2*sigma^2)); 

% Plot fitting curve
h=4; xfit=-40:1/h:40; A=[a,mu,sigma]; plot(xfit,g(A,xfit),'-r'); grid on;

%% ---Fit Data: Using LSQ method---
% Initial (Guess) Parameters
A0 = [100,0,10];

% Fit using Matlab's Least Squares function
[A1,resnorm,residual,exitflag,output] = lsqcurvefit(g,A0,x,y);

% Plot fitting curve
xfit=-40:1/h:40; plot(xfit,g(A1,xfit),'-b'); disp(output); hold off;

