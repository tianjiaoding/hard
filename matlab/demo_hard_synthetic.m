addpath(genpath('./'));

% parameters for generating data
params = struct();
params.D = 30; % dimension of the ambient space
params.K = 3; % number of clusters or hyperplanes
params.r = 0.3;  % number of outliers over the total number of points, i.e., M / (M + N_1 + ... + N_K)

params.alpha = 1; % ratio of the number of inliers between two consecutive clusters
params.sigma = 0; % standard deviation of the noise

% parameters for running the experiments
n_init = 3;

% generate data
data = generate_data(params);

% variables that collect the results
t_hard_l1 = 0;      obj_best_hard_l1 = inf;     Bs_best_hard_l1 = [];       Pis_best_hard_l1 = [];
t_hard_l1p = 0;     obj_best_hard_l1p = inf;    Bs_best_hard_l1p = [];      Pis_best_hard_l1p = [];
t_hard_huberp = 0;  obj_best_hard_huberp = inf; Bs_best_hard_huberp = [];   Pis_best_hard_huberp = [];

% run the experiments
for i = 1:n_init
    % initialize normal vectors
    Bs_init = normc(randn(params.D, params.K));

    [Bs, ~, t, obj, Pi] = HARD_L1(data.Xtilde, Bs_init, params.K, eps, 1000, 1e-3, inf);
    t_hard_l1 = t_hard_l1 + t;
    if obj < obj_best_hard_l1
        obj_best_hard_l1 = obj;
        Bs_best_hard_l1 = Bs;
        Pis_best_hard_l1 = Pi;
    end

    [Bs, ~, t, obj, Pi] = HARD_L1p(data.Xtilde, Bs_init, params.K, eps, 1000, 1e-3, inf);
    t_hard_l1p = t_hard_l1p + t;
    if obj < obj_best_hard_l1p
        obj_best_hard_l1p = obj;
        Bs_best_hard_l1p = Bs;
        Pis_best_hard_l1p = Pi;
    end

    [Bs, ~, t, obj, Pi] = HARD_Huberp(data.Xtilde, Bs_init, params.K, eps, 1000, 1e-3, inf);
    t_hard_huberp = t_hard_huberp + t;
    if obj < obj_best_hard_huberp
        obj_best_hard_huberp = obj;
        Bs_best_hard_huberp = Bs;
        Pis_best_hard_huberp = Pi;
    end
end

% compute clustering accuracy
% for each method, the estimate with the minimum objective value is used
acc_hard_l1 = compute_accuracy(data.C, Pis_best_hard_l1(1:length(data.C)));
acc_hard_l1p = compute_accuracy(data.C, Pis_best_hard_l1p(1:length(data.C)));
acc_hard_huberp = compute_accuracy(data.C, Pis_best_hard_huberp(1:length(data.C)));

% print the results
fprintf('==== Hyperplane Clustering in R^%d with K=%d Hyperplanes ====\n', params.D, params.K);
fprintf('==== Each method is run with %d initializations ====\n', n_init);
fprintf('HARD-L1: accuracy = %f, time = %f\n', acc_hard_l1, t_hard_l1);
fprintf('HARD-L1p: accuracy = %f, time = %f\n', acc_hard_l1p, t_hard_l1p);
fprintf('HARD-Huberp: accuracy = %f, time = %f\n', acc_hard_huberp, t_hard_huberp);
