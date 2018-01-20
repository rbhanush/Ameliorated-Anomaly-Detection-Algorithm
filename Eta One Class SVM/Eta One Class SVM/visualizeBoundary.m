
x1plot = linspace(min(x_test(:,1)), max(x_test(:,1)),117)';
x2plot = linspace(min(y_test(:,1)), max(y_test(:,1)),117)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(x_test));
for i = 1:size(x_test,1)
   vals(:,i) = svmPredict(model, x_test);
end

printClassMetrics (vals(:,1) , y_test);
