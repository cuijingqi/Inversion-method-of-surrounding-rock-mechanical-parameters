function correlatedSamples = lhsgeneral(pd,correlation,n)

l=length(pd);                                                               % number of variables       
RStar=correlation;                                                          % ideal correlation between variables defined by user

x=lhsdesign(n,l,'smooth','off');                                            % generate latin hypercube samples (MATLAB Function, see MATLAB documentation for more information)
independent_samples=zeros(n,l);                                             % preallocation for the matrix

for i=1:l
    prob=x(:,i);
    independent_samples(:,i) = icdf('normal',prob, 0, 1);                   % map latin hypercube samples to values using inverse cumulative distribution functions
end

R=corr(independent_samples);                                                % correlation between the generated values
P=chol(RStar).';                                                            % Cholesky decomposition of the ideal correlation matrix
Q=chol(R).';                                                                % Cholesky decomposition of the actual correlation matrix, 0,
dependent_samples=independent_samples*(P*inv(Q)).';                         % transformation matrix which adds dependency between normal variables 

uniform_dependent_samples=normcdf(dependent_samples);                       % tranforming normal distibution to uniform distribuiton
                                                                            % this transformation preserves the dependency between the variables

for i=1:l
    transformed_samples(:,i)=icdf(pd{i},uniform_dependent_samples(:,i));    % mapping each unifrom variable to the probability distribution defined by user 
end


correlatedSamples=transformed_samples;

fprintf('\n\n')
predefined_correlation=correlation
fprintf('\n\n')
actual_correlation=corr(correlatedSamples)




