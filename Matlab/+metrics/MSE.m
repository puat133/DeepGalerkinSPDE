function val = MSE(x1, x2)

val = sqrt(1 / length(x1) * sum((x1(:) - x2(:)) .^ 2));

end