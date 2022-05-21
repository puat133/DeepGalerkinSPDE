function val = mean_log_sq(x1, x2)

val = mean(log((x1(:) - x2(:)).^2));

end

