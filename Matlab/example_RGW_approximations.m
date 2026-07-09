%% Clear all & close all
clear all;
close all;

%% Add needed libraries
addpath('./data');
addpath('./exp_fig/');
addpath('./varpro/');
addpath('./RGW');

%% Seed random generator
rng default;

%% Constants
bmin = 0.01;

%% Load test signal & Set RGW parameters

% Tire sensor signal example
load("data/14_new_asphalt");
f = segmented_data(20, 235:275).';
n = 10;
p = 3;
r = 1;
NORMALIZE = 0;

% VEP example
% load("data/AVG50_34.mat");
% f = data(5,1:end-1).'; % for VEP example
% n = 7;
% p = 10;
% r = 7;
% NORMALIZE = 0;

% Healthy ECG example
% load("data/109m.mat");
% f = val(1,3295:3525).';
% n = 8;
% p = 10;
% r = 3;
% NORMALIZE = 1;

% VEB ECG example
% load("data/221m.mat");
% f = val(2, 5490:5790).';
% n = 8;
% p = 10;
% r = 3;
% NORMALIZE = 1;

w = ones(length(f), 1);
t = linspace(-1, 1, length(f));

%% Normalize input signal
if NORMALIZE
    N = length(f);
    baseline = linspace(f(1), f(end), N)';
    f = f - baseline;
    mf = max(abs(f));
    f = f/mf;
end

%% Do varpro with derivative check
polebeg = p+1; poleend = p+1+2*r-1;
alpha0 = rand(2*n+p+2*r+1, 1);
ada = @(alpha) adaRatGaussWav(alpha, t, n, p, r, bmin);

options = optimset('Display','iter','DerivativeCheck','off', 'MaxFunEvals', 5000, 'MaxIter', 1000);
[alpha,c,wresid,resid_norm,y_est,Regression] = ...
     varpro(f, w,alpha0,n,ada,[],[],options);

% Re-add baseline + de-normalize
if NORMALIZE
    f = f*mf;
    f = f + baseline;
    y_est = y_est*mf;
    y_est = y_est + baseline;
end

Phi = adaRatGaussWav(alpha, t, n, p, r, bmin);

figure;
plot(t+1, f, 'LineWidth', 2);
hold on;
plot(t+1, y_est, 'LineWidth', 2);
grid on;
axis tight;
xlabel('Time (s)', 'FontSize', 50);
ylabel('Voltage (mV)', 'FontSize', 50);
title('Tire sensor signal reconstruction', 'FontSize', 55);
legend('Tire sensor signal', 'Approximation (m=10)', 'FontSize', 45, 'location', 'northeast');
pause;
export_fig('figures/tiresensfit.pdf', '-pdf', '-transparent');

% Finally plot the mother wavelet without translation-dilation
alpha_mother = [alpha(1:p+2*r); 1; 0; alpha(end)];
tt = linspace(-2, 2, 1024); % NOTE: Support of the mother wavelet changes with optimization -> this might need adjustment for different experiments to get a good look at it
Phimom = adaRatGaussWav(alpha_mother, tt, 1, p, r, bmin);
figure;
plot(tt, Phimom/norm(Phimom), LineWidth=2);
grid on;
axis tight;
title('Rational Gaussian Mother wavelet', 'Fontsize', 50);
pause;
export_fig('figures/tiresensmom.pdf', '-pdf', '-transparent');

