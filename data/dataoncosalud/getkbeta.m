function [k, beta] = getkbeta(R, S)
    % Step 1: Compute d, t, l, m, and n
    d = 6*R + S;
    t = nthroot(64 - d^2, 3);  % Cube root
    l = sqrt(d^2 - 48 + 12*t);
    m = sqrt(2*d^2 - 96 - 12*t + (2*d^3 - 144*d) / l);
    n = sqrt(2*d^2 - 96 - 12*t - (2*d^3 - 144*d) / l);
    
    % Step 2: Calculate the four solutions for U
    U1 = (d + l + m) / 12;
    U2 = (d + l - m) / 12;
    U3 = (d - l + n) / 12;
    U4 = (d - l - n) / 12;
    U_all = [U1, U2, U3, U4];
    
    % Step 3: Compute k and b for each U, and filter valid solutions
    k = [];
    beta = [];
    for i = 1:length(U_all)
        U = U_all(i);
        if isreal(U) && U > 0  % Ensure U is real and positive
            k_temp = sqrt(2*U*R - 2);
            beta_temp = (1/6) * (U*S + 1/U^2 - 3);
            if isreal(k_temp) && k_temp > 0 && isreal(beta_temp) && beta_temp > 0
                k = [k, k_temp];  % Collect valid k values
                beta = [beta, beta_temp];  % Collect valid b values
            end
        end
    end
    
    % Step 4: Return the first valid solution (if multiple exist)
    if isempty(k) || isempty(beta)
        error('No real positive solutions found for the given R and S.');
    end
    k = k(1);
    beta = beta(1);
end
