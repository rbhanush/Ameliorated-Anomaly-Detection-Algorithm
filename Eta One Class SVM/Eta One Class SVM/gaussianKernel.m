function sim = gaussianKernel(x1, x2, sigma)
x1 = x1(:); x2 = x2(:);
sim = 0;

magnitude = sum((x1-x2).^2);
sim = 2.718^(-magnitude/(2*sigma^2));
end