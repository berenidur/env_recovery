function [x, z, z_interp, X, Z, X_interp, Z_interp] = xz_interp_grid(rf, feat)
    % Process RF data to generate interpolated grids

    z = (1540 / (2 * feat.sf)) * feat.h;
    x = 0.038;

    z = linspace(0, z, size(rf, 1));
    x = linspace(0, x, size(rf, 2));

    res = x(2) - x(1);
    z_interp = z(1):res:z(end);

    [X, Z] = meshgrid(x, z);
    [X_interp, Z_interp] = meshgrid(x, z_interp);
end
