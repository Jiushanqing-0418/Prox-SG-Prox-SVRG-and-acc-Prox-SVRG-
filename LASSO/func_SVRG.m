function [x, t, ek, fk, sk, gk] = func_SVRG(para, GradF,iGradF, ObjF)
%

fprintf( sprintf('performing SVRG...\n') );
itsprint(sprintf('      step %09d: norm(ek) = %.3e...', 1,1), 1);


% parameters
P = para.P;
m = para.m;
n = para.n;
gamma = para.c_gamma * para.beta_fi;%gamma is the step size
tau = para.mu * gamma;

% stop cnd, max iteration
tol = para.tol;
maxits = para.maxits;


% initial point
x0 = zeros(n, 1);

%
%%%%%%%%%%%%%%%%%%%%%%%%%%
%

ek = zeros(maxits, 1);
fk = zeros(maxits, 1);
sk = zeros(maxits, 1);
gk = zeros(maxits, 1);

x = x0;
x_tilde = x;

l = 0;

its = 1;
t = 1;
while(t<maxits)
    
    g_tilde = GradF(x_tilde);
    
    x = x_tilde;
    for p=1:P%iterate in one stage, total iteration is P
        
        x_old = x;
        
        j = randsample(1:m,1);%here use the uniformly sampling
        
        Gj_k1 = iGradF(x, j);
        Gj_k2 = iGradF(x_tilde, j);
        
        w = x - gamma* ( Gj_k1 - Gj_k2 + g_tilde );
        x = wthresh(w, 's', tau);
        
        %%% stop?
        normE = norm(x(:)-x_old(:), 'fro');
        
        if mod(t, m)==0
            l = l + 1;
            fk(l) = ObjF(x);
        end
        if mod(t,1e3)==0; itsprint(sprintf('      step %09d: norm(ek) = %.3e...', t,normE), t); end
        
        ek(t) = normE;
        sk(t) = sum(abs(x)>0);
        gk(t) = gamma;
        
        t = t + 1;
        
    end
    
    x_tilde = x;
    
    %%% stop?
    if ((normE)<tol)||(normE>1e10); break; end
    
    its = its + 1;
    
end
fprintf('\n');

ek = ek(1:t-1);
fk = fk(1:l);
sk = sk(1:t-1);
gk = gk(1:t-1);