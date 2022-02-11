
import numpy as np
import spectral
from scipy import sparse

class KdVEquation:

    def __init__(self, domain, u):
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype) # 6u*dudx
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)

        p = self.problem.pencils[0]

        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N,dtype = dtype)
        p.M = I
        if dtype == np.complex128:
            diag = -1j*x_basis.wavenumbers(dtype)**3
            p.L = sparse.diags(diag)
        elif dtype == np.float64:
            diag1 = np.zeros(x_basis.N-1)
            diag2 = np.zeros(x_basis.N-1)
            diag1[::2] = x_basis.wavenumbers(dtype)[::2]
            diag1 = diag1**3
            diag2 = -diag1
            diag = [diag1,diag2]
            off = [1,-1]
            diag = sparse.diags(diag,off).toarray()
            p.L = diag

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        x_basis = self.domain.bases[0]
        u = self.u
        dudx = self.dudx
        RHS = self.RHS

        for i in range(num_steps):
            # need to calculate 6u*ux and put it into RHS
            u.require_coeff_space()
            dudx.require_coeff_space()
            if self.dtype == np.complex128:
                dudx.data = 1j*x_basis.wavenumbers(self.dtype)*u.data
            elif self.dtype == np.float64:
                dudx.data[::2] = -u.data[1::2]
                dudx.data[1::2] = u.data[::2]
                dudx.data = x_basis.wavenumbers(self.dtype)*dudx.data
            u.require_grid_space(scales=3/2)
            dudx.require_grid_space(scales=3/2)
            RHS.require_grid_space(scales=3/2)
            RHS.data = 6*u.data * dudx.data

            # take timestep
            ts.step(dt)

class SHEquation:

    def __init__(self, domain, u):
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype) # -u**3+1.8*u**2
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)

        p = self.problem.pencils[0]

        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N,dtype = dtype)
        r = -0.3
        diag = 1-r-2*x_basis.wavenumbers(dtype)**2 + x_basis.wavenumbers(dtype)**4
        p.L = sparse.diags(diag)

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        RHS = self.RHS

        for i in range(num_steps):
            # need to calculate -u**3+1.8*u**2 and put it into RHS
            u.require_coeff_space()
            u.require_grid_space(scales=3)
            RHS.require_grid_space(scales=3)
            RHS.data = -u.data**3+1.8*u.data**2

            # take timestep
            ts.step(dt)
