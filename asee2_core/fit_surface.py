# ================================================================================
# file name: fit_surface.py
# description:
# author: Xihan Ma
# date: Jan-10-2025
# ================================================================================

import numpy as np

class FitSurface():

    def __init__(self, fitter):

        self.fitters = {
            'flat': FitFlatSurface,
            'quadratic': FitQuadraticSurface,
        }

        if fitter not in self.fitters:
            raise ValueError(f'Invalid fitter: {fitter}. Choose from {list(self.fitters.keys())}')

        self.fitter = self.fitters[fitter]()
        self.n_coeffs = self.fitter.n_coeffs

    def fit_surface(self, pcd):
        return self.fitter.fit_surface(pcd)
    
    def calculate_normal(self, coeffs, x, y):
        return self.fitter.calculate_normal(coeffs, x, y)
        

class FitFlatSurface():

    n_coeffs = 3

    def __init__(self):
        pass

    def fit_surface(self, pcd):
        x = pcd[:, 0]
        y = pcd[:, 1]
        z = pcd[:, 2]

        A = np.vstack([x, y, np.ones_like(x)]).T

        # TODO: calc residual error
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)  # a, b, c

        return coeffs
    
    def calculate_normal(self, coeffs, x=None, y=None):
        
        a, b, _ = coeffs
        normal = np.array([a, b, -1])
        
        normal_magnitude = np.linalg.norm(normal)
        if normal_magnitude != 0:
            normal = normal / normal_magnitude

        return normal

class FitQuadraticSurface():
    
    n_coeffs = 6

    def __init__(self):
        pass

    def fit_surface(self, pcd):
        x = pcd[:, 0]
        y = pcd[:, 1]
        z = pcd[:, 2]

        A = np.vstack([x**2, y**2, x*y, x, y, np.ones_like(x)]).T

        # TODO: calc residual error
        coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)  # a, b, c, d, e, f

        return coeffs

    def calculate_normal(self, coeffs, x, y):
        """
        Calculate the normal vector at a point (x, y, z) on the quadratic surface

        Args:
            a, b, c, d, e, f: Coefficients of the quadratic surface equation
            x, y, z: Coordinates of the point on the surface
        """
        a, b, c, d, e, _ = coeffs
        dx = 2 * a * x + c * y + d
        dy = 2 * b * y + c * x + e
        dz = -1  # Partial derivative with respect to z
        
        normal = np.array([dx, dy, dz])
        
        normal_magnitude = np.linalg.norm(normal)
        if normal_magnitude != 0:
            normal = normal / normal_magnitude

        return normal

    def sample_surface(self, coeffs, x_range, y_range, resolution=30):
        a, b, c, d, e, f = coeffs

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        x_grid, y_grid = np.meshgrid(x, y)

        z_grid = a * x_grid**2 + b * y_grid**2 + c * x_grid * y_grid + d * x_grid + e * y_grid + f

        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        z_flat = z_grid.flatten()

        points = np.vstack((x_flat, y_flat, z_flat)).T
        return points

