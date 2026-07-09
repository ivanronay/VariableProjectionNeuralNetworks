clear all;
close all;

addpath("RGW/");
addpath("varpro/");

rng default;

%% Global variables
global n;
global p;
global alpha;
global sigma;
global bmin;

%% Define RGW hyper parameters --> All global
n = 5; % Number of poles
p = 5; % Number of zeros
sigma = 0.5; % Fixed length of the Gaussian for this experiment
bmin = 0.01; % Minimal value of imaginary parts of poles.

%% Add RGW to wavemangr
wavemngr('add','RatGauss','rgw',4,'','rgwwavf',[-4,4]);

%% Create signal
Fs = 500;              % Sampling frequency
dt = 1/Fs;
t = 0:dt:2;             % 2-second signal
N = length(t);

x = cos(2*pi*3*t);                  % low-freq oscillation
x(t > 0.9 & t < 1.1) = x(t > 0.9 & t < 1.1) + 2*cos(2*pi*80*t(t > 0.9 & t < 1.1));  % burst
signal = x;

% Normalize signal
%signal = (signal - mean(signal))/std(signal);

%% Main
scales = logspace(1, 2.5, 80);

% Define function to minimize
obj = @(beta) ComputeEntropy(beta, signal, scales);
x0 = [rand(2*n, 1); rand(p, 1)]*2;%[-1.2382 3.8641 -0.0010 2.1553 0.5009 0.0000 0.5560 1.3958 0.6947]';
options = optimset('Display', 'iter', 'MaxIter', 10000, 'MaxFunEvals', 10000);

% Save initial mother wavelet
alpha = x0;
[f0, t] = wavefun("rgw");

% Run optimization
[xmin, fmin] = fminsearch(obj, x0, options);

alpha = xmin;

C_m = cwt(x, 'amor');
C_r = cwt(x, 'mexh');
C_h = cwt(x, 'haar');
C_me = cwt(x, 'meyr');
C = cwt(signal, scales, 'rgw');
tt = linspace(t(1), t(end), length(signal));

[X,Y] = meshgrid(tt, scales);

figure;
subplot(211);
plot(tt, signal, 'LineWidth', 2); axis tight; grid on;
title('Signal to be analyzed', 'FontSize', 22);
xlabel('Time (s)', 'FontSize', 22);
ylabel('$f(t)$', 'FontSize', 22, 'Interpreter', 'latex');

subplot(212);
imagesc(abs(C_m)./abs(sum(C_m(:)))), colormap hot;
axis tight; 
grid on;
title('Morlet scalogram', 'FontSize', 22);
xlabel('Time (s)', 'FontSize', 22);
ylabel('Scales', 'FontSize', 22);

figure;
subplot(211);
imagesc(abs(C_r)./abs(sum(C_r(:)))), colormap hot;
axis tight; 
grid on;
title('Ricker scalogram', 'FontSize', 22);
xlabel('Time (s)', 'FontSize', 22);
ylabel('Scales', 'FontSize', 22);

subplot(212);
imagesc(abs(C_h)./abs(sum(C_h(:)))), colormap hot;
axis tight; 
grid on;
title('Haar scalogram', 'FontSize', 22);
xlabel('Time (s)', 'FontSize', 22);
ylabel('Scales', 'FontSize', 22);

figure;
subplot(212);
imagesc(abs(C)./abs(sum(C(:)))), colormap hot;
axis tight;
grid on;
title('RGW scalogram', 'FontSize', 22);
xlabel('Time (s)', 'FontSize', 22);
ylabel('Scales', 'FontSize', 22);

subplot(211);
imagesc(abs(C_me)./abs(sum(C_me(:)))), colormap hot;
axis tight;
grid on;
title('Meyer scalogram', 'FontSize', 22);
xlabel('Time (s)', 'FontSize', 22);
ylabel('Scales', 'FontSize', 22);


% Query final mother wavelet
[fN, ~] = wavefun("rgw");
[mf, mt] = wavefun('morl');
[rf, rt] = wavefun("mexh");
[~, hf, ht] = wavefun("haar");
[~, mmf, mmt] = wavefun("meyr");

figure;
plot(mt, mf, 'LineWidth', 2); axis tight; 
title('Morlet wavelet', 'FontSize', 30);
grid on;


figure;
plot(t, fN, 'LineWidth', 2);
grid on;
title('Optimized RGW wavelet', 'FontSize', 30);

figure;
plot(rt, rf, 'LineWidth', 2);
grid on;
title("Ricker wavelet", "FontSize", 30);

figure;
plot(ht, hf, 'LineWidth', 2);
grid on;
title("Haar wavelet", "FontSize", 30);

figure;
plot(mmt, mmf, 'LineWidth', 2);
grid on;
title("Meyer wavelet", "FontSize", 30);

% Delete RGW wavelets
wavemngr('del','RatGauss');

%% Helper functions
function f = ComputeEntropy(beta, signal, scales)
    global alpha;
    alpha = beta; % Set RGW parameters in global variable -> ugly, but works
    C = cwt(signal, scales, 'rgw');
    CC = C(15:end-5,:);
    f = norm(abs(CC(:)), 2)/norm(abs(C(:)), 2);
end