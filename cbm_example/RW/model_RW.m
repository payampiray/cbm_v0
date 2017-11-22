function [loglik] = model_RW(params,dat)
% Rescorla-Wagner model for a simple situation that subjects have chosen 
% between two actions. The model updates q-value for every action and
% computes the probability of actions using a softmax rule. 
% [loglik] = model_RW(params,dat)
% The first input to the model is params, which is a row-vector containing 
% two parameters corresponding to the learning rate and the 
% softmax-temperature. 
% The second input, dat, is a structure containing data of this subject
% (outcome and actions).
% The output of the model is the log-likelihood of all data, which is the
% sum of log-probabilities of actions taken across all trials.
% 



% learning rate parameter: since params(1) can take any value, it should be
% transformed to lie in the unit range (between zero and one)
alpha   = 1./(1+exp(-params(1)));

% temperature parameter: since params(2) can take any value, it should be
% transformed to be positive
beta    = exp(params(2));

% unpack dat
outcome   = dat.outcome;
actions   = dat.actions;

% number of trials
T         = size(outcome,1);

% q-value
q       = zeros(1,2);

% probability of actions taken
f       = nan(T,1);

for t=1:T
    
    % probability of taking action 1 using softmax rule
    p     = 1./(1+exp(-beta*(q(1)-q(2))));

    % action taken on this trial
    a    = actions(t);

    % probability of the taken action on this trial
    if a==1        
        f(t) = p;
    else
        f(t) = 1-p;
    end
    
    % outcome given on this trial
    o    = outcome(t);    
    
    % prediction error (delta)
    delta    = o - q(a);
    
    % update q-value using prediction error and learning rate.
    q(a)     = q(a) + (alpha.*delta);
end

% loglik is the output: sum of the log-probability of all actions
loglik = sum(log(f+eps));
end
