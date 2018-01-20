function pred = svmPredict(model,X)
if (size(X, 2) == 1)
    X = X';
end
m = size(X, 1);
p = zeros(m, 1);
pred = zeros(m, 1);

X1 = sum(X.^2, 2);
X2 = sum(model.X.^2, 2)';
transp = transpose(model.X);
K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * transp ));
K = gaussianKernel(1,0,0.0005) .^ K;
K = bsxfun(@times, model.y', K);
K = bsxfun(@times, model.alphas', K);
p = sum(K, 2);

pred(p >= 0) =  1;
pred(p <  0) =  -1;

end



