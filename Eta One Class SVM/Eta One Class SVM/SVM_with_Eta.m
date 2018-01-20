Ci = table2cell(xtrain);
x_train = cell2mat(Ci);
D = table2cell(ytrain);
y_train = cell2mat(D);
A=11
B = table2cell(ytest);
y_test = cell2mat(B);
A = table2cell(xtest);
x_test = cell2mat(A);
A=12


model = svmtrain(x_train,y_train);

if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3;
end

if ~exist('max_passes', 'var') || isempty(max_passes)
    max_passes = 5;
end
X = x_train;
Y = y_train;
% Data parameters
m = size(X, 1);
n = size(X, 2);

% Map 0 to -1
Y(Y==0) = -1;

C = 10; 
alphas = zeros(m, 1);
b = 0;
E = zeros(m, 1);
passes = 0;
eta = 0;
L = 0;
H = 0;

X2 = sum(X.^2, 2);
nsq=sum(X.^2,2);
K=bsxfun(@minus,nsq,(2*X)*X.');
K=bsxfun(@plus,nsq.',K);
K=exp(-K);
K = gaussianKernel(1, 0,0.0005) .^ K;




dots = 12;
while passes < max_passes
            
    num_changed_alphas = 0;
    for i = 1:m
        
       
        E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);
        xi = (Y(i)*E(i));
        if ((xi < (-tol) && alphas(i) < C) || (xi > (tol) && alphas(i) > 0))
           
            j = ceil(m * rand());
            while j == i  % Make sure i \neq j
                j = ceil(m * rand());
            end

            E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);

            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);
            if (Y(i) == Y(j))
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end
           
            if (L == H)
                continue;
            end

            eta = 2 * K(i,j) - K(i,i) - K(j,j);
            if (eta >= 0)
                continue;
            end
            
            alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;
            alphas(j) = min (H, alphas(j));
            alphas(j) = max (L, alphas(j));
            if (abs(alphas(j) - alpha_j_old) < tol)
                alphas(j) = alpha_j_old;
                continue;
            end
            alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));
            
            b1 = b - E(i) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
            b2 = b - E(j) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

            if (0 < alphas(i) && alphas(i) < C)
                b = b1;
            elseif (0 < alphas(j) && alphas(j) < C)
                b = b2;
            else
                b = (b1+b2)/2;
            end

            num_changed_alphas = num_changed_alphas + 1;

        end
        
    end
    
    if (num_changed_alphas == 0)
        passes = passes + 1;
    else
        passes = 0;
    end

    fprintf('.');
    dots = dots + 1;
    if dots > 78
        dots = 0;
        fprintf('\n');
    end
    
end
idx = alphas > 0;
model.X= X(idx,:);
model.y= Y(idx);
model.kernelFunction = 'gaussianKernel';
model.b= b;
model.alphas= alphas(idx);
model.w = ((alphas.*Y)'*X)';


