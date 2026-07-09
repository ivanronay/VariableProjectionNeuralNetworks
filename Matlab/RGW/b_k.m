function [b_k, db_k] = b_k(beta, bmin)
    b_k = (beta^2 + bmin);
    db_k = 2*beta;
end

