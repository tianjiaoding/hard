function [info] = generate_data(params)
% Generate synthetic hyperplane clustering data
% params: struct, parameters for the synthetic data
%       .D: int, dimension of the ambient space
%       .K: int, number of clusters or hyperplanes
%       .r: float, number of outliers over the total number of points, i.e., M / (M + N_1 + ... + N_K)
%       .alpha: float, ratio of the number of inliers between two consecutive clusters
%       .sigma: float, standard deviation of the noise


%% basic info
if isfield(params, 'density')
    info.N       =  params.density*(params.D-1)*params.K;    % total number of inliers
    disp(['generating' num2str(params.density) ' inliers per dimension'])
else
    info.N       =  50*(params.D-1)*params.K;    % total number of inliers
    disp(['generating 50 inliers per dimension'])
end

if params.alpha == 1
    info.N1  =  ceil(info.N / params.K);
else
    info.N1  =  ceil(info.N*(1-params.alpha)/(1-params.alpha^params.K));
end

info.b1      =  normc(randn(params.D,1));     
info.M       =  ceil(params.r * info.N / (1 - params.r));
info.C       =  ones(1,info.N1);

for i = 2 : params.K
    eval(['info.N' num2str(i) '= ceil(params.alpha*info.N' num2str(i-1) ');']);
    eval(['info.b' num2str(i) '= normc(randn(params.D,1));']);
    eval(['info.C = [info.C ' num2str(i) '*ones(1,info.N' num2str(i) ')];']);
end

for i = 1:params.K
    eval(['info.U' num2str(i) ' = null(info.b' num2str(i) ''');']);    % orthonormal basis
    eval(['info.P' num2str(i) ' = info.U' num2str(i) '*info.U' num2str(i) ''';']);   % projection
    eval(['info.X' num2str(i) ' = normc(info.P' num2str(i) '*randn(params.D, info.N' num2str(i) ')' ...
        '+ (eye(params.D)-info.P' num2str(i) ')*params.sigma*randn(params.D, info.N' num2str(i) '));']);  % inlier points
end

%% dataset creation
assign_data  =  'info.Xtilde = [';
for i = 1:params.K
    assign_data = strcat(assign_data, ['info.X' num2str(i) ', ']);
end
assign_data = strcat(assign_data, 'info.O];');
info.O = normc(randn(params.D, info.M));    % outliers
eval(assign_data);

return