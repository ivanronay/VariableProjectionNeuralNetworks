%% Clear & close all
clear all;
close all;

%% Seed random generator for reproducibility
rng default;

warning('off', 'all'); % In order to get rid of varpro messages & be able to track progress

%% Load QRS data
QRS_tr = load('data/ecg_train.mat');
QRS_te = load('data/ecg_test.mat');
data = [QRS_tr.samples; QRS_te.samples];

% Get rid of one-hot-encoding for labels
[~, labels_tr] = max(QRS_tr.labels');
[~, labels_te] = max(QRS_te.labels');
labels = [labels_tr, labels_te]';

%% Constants

% Max coeffs
M = 10;

% General constants
DO_RGW = 1;
DO_EWT = 1;
DO_TQT = 1;

Q = size(data, 1); % Number of ecg signals in total
N = size(data, 2); % Number of sample points in each QRS

% RGW-related constants
p = 10;
r = 3;
bmin = 0.01;

% Load initial paramset
alpha_init = load('alpha_init.mat');

% EWT-related constants
PRUNE_EWT_COEFFS_PERC = 100; % Keep this percent of coefficients

% TQT-related constants
PRUNE_TQT_COEFFS_PERC = 100; % Keep this percent of coefficients
Q_factor = 4;

%% Holders for overall stats
PRD_RGW_N_ALL = []; % PRD value for RGW on normal QRS
PRD_EWT_N_ALL = []; % PRD value for EWT on normal QRS
PRD_TQT_N_ALL = []; % PRD value for TQT on normal QRS

PRD_RGW_V_ALL = []; % PRD value for RGW on VEB QRS
PRD_EWT_V_ALL = []; % PRD value for EWT on VEB QRS
PRD_TQT_V_ALL = []; % PRD value for TQT on VEB QRS

PRD_RGW_ALL = []; % PRD value for RGW on every QRS
PRD_EWT_ALL = []; % PRD value for EWT on every QRS
PRD_TQT_ALL = []; % PRD value for TQT on every QRS

STD_RGW_N_ALL = []; % STD value for RGW on normal QRS
STD_EWT_N_ALL = []; % STD value for EWT on normal QRS
STD_TQT_N_ALL = []; % STD value for TQT on normal QRS

STD_RGW_V_ALL = []; % STD value for RGW on VEB QRS
STD_EWT_V_ALL = []; % STD value for EWT on VEB QRS
STD_TQT_V_ALL = []; % STD value for TQT on VEB QRS

STD_RGW_ALL = []; % STD value for RGW on every QRS
STD_EWT_ALL = []; % STD value for EWT on every QRS
STD_TQT_ALL = []; % STD value for TQT on every QRS

QS_RGW_N_ALL = []; % QS value for RGW on normal QRS
QS_EWT_N_ALL = []; % QS value for EWT on normal QRS
QS_TQT_N_ALL = []; % QS value for TQT on normal QRS

QS_RGW_V_ALL = []; % QS value for RGW on VEB QRS
QS_EWT_V_ALL = []; % QS value for EWT on VEB QRS
QS_TQT_V_ALL = []; % QS value for TQT on VEB QRS

QS_RGW_ALL = []; % QS value for RGW on every QRS
QS_EWT_ALL = []; % QS value for EWT on every QRS
QS_TQT_ALL = []; % QS value for TQT on every QRS

CR_ALL = []; % Compression ratio

for c=1:M-1

n = c+1;
KEEP_ONLY_C_TQT = c+1;
KEEP_ONLY_C_EWT = c+1;

CR_ALL = [CR_ALL; N/(c+1)];

%% Create holders for results for current compression
PRD_RGW_N = []; % PRD value for RGW on normal QRS
PRD_EWT_N = []; % PRD value for EWT on normal QRS
PRD_TQT_N = []; % PRD value for TQT on normal QRS

PRD_RGW_V = []; % PRD value for RGW on VEB QRS
PRD_EWT_V = []; % PRD value for EWT on VEB QRS
PRD_TQT_V = []; % PRD value for TQT on VEB QRS

PRD = @(y, y_est) norm(y - y_est)^2/norm(y - mean(y))^2;

%% Run Compression
for k=1:Q
    
    % Display current iteration
    disp(['Coeff size: ', num2str(c), '; Iteration: ', num2str(k), ' of ', num2str(Q)]);
    
    % Extract and normalize current signal
    signal = data(k,:);
    signal = signal(:);
    signal = (signal - mean(signal))/std(signal);
    baseline = linspace(signal(1), signal(end), N)';
    signal = signal - baseline;

    % Run compression schemes one by one
    if DO_RGW
        alpha0 = alpha_init.alpha_opt;
        w = ones(length(signal), 1);
        t = linspace(-1, 1, length(signal));
        ada = @(alpha) adaRatGaussWav(alpha, t, n, p, r, bmin);
        options = optimset('Display','off','DerivativeCheck','off', 'MaxFunEvals', 5000, 'MaxIter', 1000);
        tic
        [alpha_opt,~,~,~,y_est,~] = ...
             varpro(signal, w,alpha0,n,ada,[],[],options);
        time = toc;
        err = PRD(signal, y_est);
        
        if labels(k) == 1
            PRD_RGW_N = [PRD_RGW_N; err];
        else
            PRD_RGW_V = [PRD_RGW_V; err];
        end
    end

    if DO_EWT
        [~, cfs] = ewt(signal);
        cfs_temp = cfs(:);
        [ord_cfs, ind] = sort(abs(cfs_temp), 'ascend');
        
        if PRUNE_EWT_COEFFS_PERC < 100
            M = length(ord_cfs);
            num_to_keep = ceil(M*(PRUNE_EWT_COEFFS_PERC/100));
            ind = ind(1:end-num_to_keep);
        else
            ind = ind(1:end-KEEP_ONLY_C_EWT);
        end

        % Remove small coefficients
        cfs(ind) = 0;

        % Reconstruct the signal and measure the PRD
        y_est = sum(cfs, 2);
        err = PRD(signal, y_est);

        if labels(k) == 1
            PRD_EWT_N = [PRD_EWT_N; err];
        else
            PRD_EWT_V = [PRD_EWT_V; err];
        end
    end

    if DO_TQT
        cfs = tqwt(signal, "QualityFactor", Q_factor);

        coeff_sizes = [];
        cfs_temp = [];
        for j=1:length(cfs)
            coeff_sizes = [coeff_sizes; length(cfs{j})];
            cfs_temp = [cfs_temp; cfs{j}];
        end

        [ord_cfs, ind] = sort(abs(cfs_temp), 'ascend');
        
        if PRUNE_TQT_COEFFS_PERC < 100
            M = length(ord_cfs);
            num_to_keep = ceil(M*(PRUNE_TQT_COEFFS_PERC/100));
            ind = ind(1:end-num_to_keep);
        else
            ind = ind(1:end-KEEP_ONLY_C_TQT);
        end

        % Remove small coefficients
        cfs_temp(ind) = 0;
        
        % Convert coefficients into cell array of appropriate type
        curr_ind = 1;
        for j=1:length(coeff_sizes)
            cfs{j} = cfs_temp(curr_ind:curr_ind + coeff_sizes(j)-1);
            curr_ind = curr_ind + coeff_sizes(j);
        end

        y_est = itqwt(cfs, length(signal),"QualityFactor", Q_factor);

        err = PRD(signal, y_est);

        if labels(k) == 1
            PRD_TQT_N = [PRD_TQT_N; err];
        else
            PRD_TQT_V = [PRD_TQT_V; err];
        end
    end
end

%% Compute stats and save results
PRD_RGW_N_ALL = [PRD_RGW_N_ALL; mean(PRD_RGW_N)]; 
PRD_EWT_N_ALL = [PRD_EWT_N_ALL; mean(PRD_EWT_N)]; 
PRD_TQT_N_ALL = [PRD_TQT_N_ALL; mean(PRD_TQT_N)]; 

PRD_RGW_V_ALL = [PRD_RGW_V_ALL; mean(PRD_RGW_V)]; 
PRD_EWT_V_ALL = [PRD_EWT_V_ALL; mean(PRD_EWT_V)]; 
PRD_TQT_V_ALL = [PRD_TQT_V_ALL; mean(PRD_TQT_V)]; 

PRD_RGW_ALL = [PRD_RGW_ALL; mean([PRD_RGW_N_ALL(c), PRD_RGW_V_ALL(c)])]; 
PRD_EWT_ALL = [PRD_EWT_ALL; mean([PRD_EWT_N_ALL(c), PRD_EWT_V_ALL(c)])]; 
PRD_TQT_ALL = [PRD_TQT_ALL; mean([PRD_TQT_N_ALL(c), PRD_TQT_V_ALL(c)])]; 

STD_RGW_N_ALL = [STD_RGW_N_ALL; std(PRD_RGW_N)];
STD_EWT_N_ALL = [STD_EWT_N_ALL; std(PRD_EWT_N)]; 
STD_TQT_N_ALL = [STD_TQT_N_ALL; std(PRD_TQT_N)]; 

STD_RGW_V_ALL = [STD_RGW_V_ALL; std(PRD_RGW_V)]; 
STD_EWT_V_ALL = [STD_EWT_V_ALL; std(PRD_EWT_V)]; 
STD_TQT_V_ALL = [STD_TQT_V_ALL; std(PRD_TQT_V)]; 

STD_RGW_ALL = [STD_RGW_ALL; mean([STD_RGW_N_ALL(c); STD_RGW_V_ALL(c)])]; 
STD_EWT_ALL = [STD_EWT_ALL; mean([STD_EWT_N_ALL(c); STD_EWT_V_ALL(c)])]; 
STD_TQT_ALL = [STD_TQT_ALL; mean([STD_TQT_N_ALL(c); STD_TQT_V_ALL(c)])]; 

QS_RGW_N_ALL = [QS_RGW_N_ALL; CR_ALL(c)/PRD_RGW_N_ALL(c)]; 
QS_EWT_N_ALL = [QS_EWT_N_ALL; CR_ALL(c)/PRD_EWT_N_ALL(c)]; 
QS_TQT_N_ALL = [QS_TQT_N_ALL; CR_ALL(c)/PRD_TQT_N_ALL(c)]; 

QS_RGW_V_ALL = [QS_RGW_V_ALL; CR_ALL(c)/PRD_RGW_V_ALL(c)]; 
QS_EWT_V_ALL = [QS_EWT_V_ALL; CR_ALL(c)/PRD_EWT_V_ALL(c)]; 
QS_TQT_V_ALL = [QS_TQT_V_ALL; CR_ALL(c)/PRD_TQT_V_ALL(c)]; 

QS_RGW_ALL = [QS_RGW_ALL; mean([QS_RGW_N_ALL(c); QS_RGW_V_ALL(c)])]; 
QS_EWT_ALL = [QS_EWT_ALL; mean([QS_EWT_N_ALL(c); QS_EWT_V_ALL(c)])]; 
QS_TQT_ALL = [QS_TQT_ALL; mean([QS_TQT_N_ALL(c); QS_TQT_V_ALL(c)])];

%% Save results

result = struct;

result.PRD_RGW_N = PRD_RGW_N_ALL;
result.PRD_EWT_N = PRD_EWT_N_ALL;
result.PRD_TQT_N = PRD_TQT_N_ALL;

result.PRD_RGW_V = PRD_RGW_V_ALL;
result.PRD_EWT_V = PRD_EWT_V_ALL;
result.PRD_TQT_V = PRD_TQT_V_ALL;

result.PRD_RGW = PRD_RGW_ALL;
result.PRD_EWT = PRD_EWT_ALL;
result.PRD_TQT = PRD_TQT_ALL;

result.STD_RGW_N = STD_RGW_N_ALL;
result.STD_EWT_N = STD_EWT_N_ALL;
result.STD_TQT_N = STD_TQT_N_ALL;

result.STD_RGW_V = STD_RGW_V_ALL;
result.STD_EWT_V = STD_EWT_V_ALL;
result.STD_TQT_V = STD_TQT_V_ALL;

result.STD_RGW = STD_RGW_ALL;
result.STD_EWT = STD_EWT_ALL;
result.STD_TQT = STD_TQT_ALL;

result.QS_RGW_N = QS_RGW_N_ALL;
result.QS_EWT_N = QS_EWT_N_ALL;
result.QS_TQT_N = QS_TQT_N_ALL;

result.QS_RGW_V = QS_RGW_V_ALL;
result.QS_EWT_V = QS_EWT_V_ALL;
result.QS_TQT_V = QS_TQT_V_ALL;

result.QS_RGW = QS_RGW_ALL;
result.QS_EWT = QS_EWT_ALL;
result.QS_TQT = QS_TQT_ALL;

result.CR = CR_ALL;

save(['compression_results/compression_Q_',num2str(Q_factor)], 'result');

end


