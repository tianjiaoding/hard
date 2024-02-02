function [B, iter, t_elapsed, J_new, I, J_list] = HARD_L1p(X_tilde, B_0, k, delta, T, epsilon_J, budget)
    % The HARD-$\ell_1$+ algorithm of the paper "HARD: Hyperplane ARrangement Descent" by Tianjiao Ding, Liangzu Peng, and Rene Vidal, published in First Conference on Parsimony and Learning (CPAL 2024).
    
    % X_tilde: D x N matrix, each column is a data point
    % B_0: D x k matrix, initialization of normal vectors, each column is the normal vector of one hyperplane
    % k: number of hyperplanes
    % delta: a small positive number to avoid division by zero
    % T: maximum number of iterations
    % epsilon_J: stopping criterion
    % budget: maximum time allowed in seconds

    % Copyright: Tianjiao Ding, Liangzu Peng, Rene Vidal, 2024.
    
t_start = tic;
[D, N] = size(X_tilde);
Delta_J = Inf;
iter = 0;

B = B_0;
assert(size(B, 2) == k);
W = abs(B'*X_tilde);

J_old = inf;
J_new = inf;
J_list = [];

while (Delta_J > epsilon_J) && (iter < T) && toc(t_start) <= budget
    for i=1:k
        % for each cluster
        w = prod(W([1:(i-1),i+1:end], :), 1);
        w = w ./ max(W(i, :), delta);
        
        R_X = X_tilde .* w * X_tilde';

        [U,L] = eig(R_X);
        [~, perm] = sort(diag(L), 'descend');
        b = U(:, perm(end));

        B(:, i) = b;
        W(i, :) = abs(b'*X_tilde);
    end
    J_new = sum(prod(W, 1));
    iter = iter + 1;
    Delta_J = 1 - sum(J_new) / (sum(J_old) + eps);

    J_old = J_new; 
    J_list(end+1) = J_new;
end
[M, I] = min(W,[],1);
t_elapsed = toc(t_start);

disp(['HARD-L1p: iter = ', num2str(iter), ', J = ', num2str(J_new), ', t = ', num2str(t_elapsed), 's'])

end
