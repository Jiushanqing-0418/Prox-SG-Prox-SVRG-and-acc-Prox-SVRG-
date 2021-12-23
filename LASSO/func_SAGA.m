function [x, its, ek, fk, sk, gk] = func_SAGA(para, iGradF, ObjF)
%

fprintf( sprintf('performing SAGA...\n') );
itsprint(sprintf('      step %09d: norm(ek) = %.3e...', 1,1), 1);

% parameters
n = para.n;
m = para.m;
gamma = para.c_gamma * para.beta_fi;
tau = para.mu * gamma;

% stop cnd, max iteration
tol = para.tol;
maxits = 1.0*para.maxits;

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
    
    %%% stop?
    if ((normE)<tol)||(normE>1e10); break; end
    
    its = its + 1;
    
end
fprintf('\n');

ek = ek(1:its-1);
sk = sk(1:its-1);
gk = gk(1:its-1);
fk = fk(1:l);
