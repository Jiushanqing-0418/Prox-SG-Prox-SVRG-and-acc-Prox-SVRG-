function [x, its, ek, fk, sk, gk] = func_acc_SAGA(para, iGradF, ObjF)
%

fprintf(sprintf('performing acc-SAGA...\n'));
itsprint(sprintf('      step %09d: norm(ek) = %.3e...', 1,1), 1);

% parameters
n = para.n;
m = para.m;
W = para.W;
gamma0 = para.c_gamma * para.beta_fi;
gamma = gamma0;
tau = para.mu * gamma;

% stop cnd, max iteration
tol = para.tol;
maxits = para.maxits;

% initial point
x0 = zeros(n, 1);

G = zeros(n, m);
for i=1:m
    G(:, i) = iGradF(x0, i);
end

mG = sum(G, 2)/m;

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

%%% obtain the minimizer x^\star
ek = zeros(maxits, 1);
fk = zeros(maxits, 1);
sk = zeros(maxits, 1);
gk = zeros(maxits, 1);

g_flag = 1;

x = x0; % xk

l = 0;

its = 1;
while(its<=maxits)
    
    x_old = x;
    
    % j = mod(its-1, m) + 1;
    j = randperm(m, 1);
    
    gj_old = G(:, j);
    
    gj = iGradF(x, j);
    
    w = x - gamma* (gj - gj_old) - gamma*mG;
    x = wthresh(w, 's', tau);
        
    G(:, j) = gj;
    mG = (mG*m + gj - gj_old)/m;
    
    %%% stop?
    normE = norm(x(:) - x_old(:), 'fro');
    
    ek(its) = normE;
    if mod(its, m)==0
        l = l + 1;
        fk(l) = ObjF(x);
    end
    
    if mod(its,1e3)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e...', its,normE), its); end
    
    sk(its) = sum(abs(x)>0);
    gk(its) = gamma;
    
    if g_flag&&(mod(its, m)==0)&&(var(sk(its-m+1:its))<1e-2)
        PT = diag(double(abs(x)>0));
        WT = W*PT;
        b = zeros(m, 1);
        for i=1:m
            WTi = WT(i,:);
            b(i) = norm(WTi)^2;
        end
        beta_fi_new = 1 /max(b);
        
        E = beta_fi_new /para.beta_fi;
        g_flag = 0;
        
    end
    
    if (~g_flag)&&(mod(its, 2*m)==0)
        
        gamma = min(gamma*2, E*gamma0);
        % gamma = para.c_gamma * beta_fi_new /2 /exp(1 - its/maxits);
        tau = para.mu * gamma;
        
    end
    
    %%% stop?
    if ((normE)<tol)||(normE>1e10)||(isnan(normE)); break; end
    
    its = its + 1;
    
end
fprintf('\n');

ek = ek(1:its-1);
sk = sk(1:its-1);
gk = gk(1:its-1);
fk = fk(1:l);
