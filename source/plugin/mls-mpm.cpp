/******************************************************************************
*
* MantaFlow fluid solver framework
* Copyright 2024 Sebastian Barschkis
*
* This program is free software, distributed under the terms of the
* Apache License, Version 2.0
* http://www.apache.org/licenses/LICENSE-2.0
*
* Moving Least Squares Material Point Method (MLS-MPM) plugin
*
*******************************************************************************
*
* Copyright 2018 Taichi MPM Authors
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
******************************************************************************/

/*
 * Algorithm and equations based on the "MPM course":
 * Jiang, Chenfanfu, et al.
 * "The material point method for simulating continuum materials."
 * Acm siggraph 2016 courses. 2016. 1-52.
 */

#include "particle.h"
#include "grid.h"
#include "commonkernels.h"

#include <chrono>
using namespace std::chrono;

namespace Manta {

//! Polar decomposition A=UP
KERNEL(pts, bnd=0) template<class T>
void knMpmPolarDecomposition(const ParticleDataImpl<T>& srcA, ParticleDataImpl<T>& destU, ParticleDataImpl<T>& destP)
{
	polarDecomposition(srcA[idx], destU[idx], destP[idx]);
}

PYTHON() void polarDecomposition2D(const ParticleDataImpl<Matrix2x2f>& A,
	ParticleDataImpl<Matrix2x2f>& U, ParticleDataImpl<Matrix2x2f>& P)
{
	knMpmPolarDecomposition<Matrix2x2f>(A, U, P);
}

PYTHON() void polarDecomposition3D(const ParticleDataImpl<Matrix3x3f>& A,
	ParticleDataImpl<Matrix3x3f>& U, ParticleDataImpl<Matrix3x3f>& P)
{
	knMpmPolarDecomposition<Matrix3x3f>(A, U, P);
}

KERNEL(pts, single, bnd=0) template<class T> void knMpmMapPartsToGrid(
	BasicParticleSystem& pp, MACGrid& vel, Grid<Real>& mass, FlagGrid& flags, const Grid<int>& blockIdxGrid,
	const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<T>& deformationGrad, const ParticleDataImpl<T>& R, const ParticleDataImpl<T>& affineMomentum,
	const Real hardening, const Real E, const Real nu, const Real pmass, const Real pvol, const int sizeK,
	const bool is3D, const int sIdx, const int eIdx)
{
	// Exit if pts index is not in range [sIdx, eIdx]
	if (eIdx > 0 && (idx < sIdx || idx > eIdx)) return;

	// Initial Lame parameters
	const Real mu_0 = E / (2 * (1 + nu));
	const Real lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

	// Lame parameters, MPM course Equation 86. (hardening typically in range [3, 10])
	const Real e = std::exp(hardening * (1.0f - detDeformationGrad[idx]));
	const Real mu = mu_0 * e;
	const Real lambda = lambda_0 * e;

	// Domain parameters
	const Real dt = pp.getParent()->getDt();
	const Real dx = pp.getParent()->getDx();
	const Real inv_dx = 1.0 / dx;
	const Real Dinv = 4 * inv_dx * inv_dx;

	const Vec3 pos = pp.getPos(idx);
	if (!pos.isValid()) return; // omit nans

	// Delete particles in outflow immediately and return
	if (vel.isInBounds(pos) && flags.isOutflow(pos)) {
		pp.kill(idx);
		return;
	}

	const Vec3i base = toVec3i(pos);
	if (!vel.isInBounds(base)) return;
	const Vec3 fx = (pos - toVec3(base)) - 0.5;

	// Quadratic kernels, MPM course Equation 123.
	Vec3 w[3] = {
		Vec3(0.5)  * square(Vec3(0.5) - fx),
		Vec3(0.75) - square(fx),
		Vec3(0.5)  * square(Vec3(0.5) + fx)
	};
	// In 2D the Z component is unused
	if (!is3D) {
		w[0].z = 1.;
		w[1].z = 1.;
		w[2].z = 1.;
	}

	// Current volume
	const Real J = deformationGrad[idx].determinant();

	const T dR = deformationGrad[idx] - R[idx];
	const T PF = (2 * mu * (dR) * deformationGrad[idx].transposed() + lambda * (J-1) * J);

	const T stress = - (dt * pvol) * (Dinv * PF);
	const T affine = stress + pmass * affineMomentum[idx];
	const int sizeI = 3, sizeJ = 3;

	Vec3 dpos;
	Vec3i targetPos;
	Real weight;
	IndexInt targetIdx;

	const Vec3 vel_x_mass = pvel[idx] * pmass;
	// Factor to zero out the last dim in 2D mode
	const Vec3 dimFactor = (is3D) ? Vec3(1) : Vec3(1,1,0);

	// Check if current and outermost cell is in bounds (knowledge saves time in neighbor loop)
	const bool upperInBounds = vel.isInBounds(base + toVec3i(sizeI-1,sizeJ-1,sizeK-1));

	// Neighbor loop: Iterate over neighboring cells of this particle
	for (int k = 0; k < sizeK; ++k)
	for (int j = 0; j < sizeJ; ++j)
	for (int i = 0; i < sizeI; ++i)
	{
		targetPos = base + toVec3i(i-1,j-1,k-1);
		// Only perform bounds check for current ijk if upper bound cell is not in bounds (saves time)
		if (!upperInBounds && !vel.isInBounds(targetPos)) continue;

		weight = w[i].x * w[j].y * w[k].z;
		dpos = (toVec3(targetPos) - pos + 0.5) * dx;

		targetIdx = blockIdxGrid(targetPos);
		vel(targetIdx) += weight * (vel_x_mass + affine * dpos) * dimFactor;
		mass(targetIdx) += weight * pmass;
	}
	flags(base) = (flags(base) | FlagGrid::TypeFluid) & ~FlagGrid::TypeEmpty;}

KERNEL() template<class T> void KernelHelper(
	const Grid<Real>& kernelGrid, BasicParticleSystem& pp, FlagGrid& flags, const Grid<int>& blockIdxGrid,
	MACGrid& vel, MACGrid* velTmp0, MACGrid* velTmp1, MACGrid* velTmp2,
	Grid<Real>& mass, Grid<Real>* massTmp0, Grid<Real>* massTmp1, Grid<Real>* massTmp2,
	const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<T>& deformationGrad, const ParticleDataImpl<T>& affineMomentum, const ParticleDataImpl<T>& rotation,
	const Real hardening, const Real E, const Real nu, const Real pmass, const Real pvol, bool is3D)
{
	// Clear global grids while while new data is computed and stored in tmp grids
	if (k == 0) { vel.clear(); return; }
	if (k == 1) { mass.clear(); return; }

	// Use multiple tmp grids to compute new vel/mass in parallel
	MACGrid* velSelect = nullptr;
	Grid<Real>* massSelect = nullptr;
	int chunkPos = 0;
	if (k == 2) { velSelect = velTmp0; massSelect = massTmp0; chunkPos = 0; }
	if (k == 3) { velSelect = velTmp1; massSelect = massTmp1; chunkPos = 1; }
	if (k == 4) { velSelect = velTmp2; massSelect = massTmp2; chunkPos = 2; }
	if (k >= 5) { return; }

	const int numKernels = 3; // I.e. number of tmp grid pairs 
	const int sizeK = (is3D) ? 3 : 1;
	const int chunkSize = pvel.size() / numKernels;
	const bool isLastChunk = (chunkPos == numKernels-1);
	const int sIdx = chunkPos * chunkSize;
	const int eIdx = (isLastChunk) ? pvel.size() : sIdx + chunkSize;

	knMpmMapPartsToGrid<T>(pp, *velSelect, *massSelect, flags, blockIdxGrid, pvel, detDeformationGrad,
		deformationGrad, rotation, affineMomentum, hardening, E, nu, pmass, pvol, sizeK, is3D, sIdx, eIdx);
}

PYTHON() void mpmMapPartsToGrid2D(
	MACGrid& vel, Grid<Real>& mass, FlagGrid& flags, const Grid<int>& blockIdxGrid,
	BasicParticleSystem& pp, const ParticleDataImpl<Vec3>& pvel,
	const ParticleDataImpl<Real>& detDeformationGrad, const ParticleDataImpl<Matrix2x2f>& deformationGrad,
	const ParticleDataImpl<Matrix2x2f>& affineMomentum, const ParticleDataImpl<Matrix2x2f>& rotation,
	Grid<Real>* kernelGrid, MACGrid* velTmp0, MACGrid* velTmp1, MACGrid* velTmp2,
	Grid<Real>* massTmp0, Grid<Real>* massTmp1, Grid<Real>* massTmp2,
	const Real hardening, const Real E, const Real nu, const Real pmass, const Real pvol)
{
	const bool is3D = false;

	// Compute particle data in single thread
	if (kernelGrid == nullptr) {
		vel.clear();
		mass.clear();
		const int sizeK = 1;
		knMpmMapPartsToGrid<Matrix2x2f>(pp, vel, mass, flags, blockIdxGrid, pvel, detDeformationGrad,
			deformationGrad, rotation, affineMomentum, hardening, E, nu, pmass, pvol, sizeK, is3D, 0, 0);
		return;
	}

	// Compute particle data with multiple threads (number of threads depends on size of kernelgrid)
	KernelHelper<Matrix2x2f>(*kernelGrid, pp, flags, blockIdxGrid, vel, velTmp0, velTmp1, velTmp2, mass, massTmp0, massTmp1, massTmp2,
		pvel, detDeformationGrad, deformationGrad, affineMomentum, rotation, hardening, E, nu, pmass, pvol, is3D);
}

PYTHON() void mpmMapPartsToGrid3D(
	MACGrid& vel, Grid<Real>& mass, FlagGrid& flags, const Grid<int>& blockIdxGrid,
	BasicParticleSystem& pp, const ParticleDataImpl<Vec3>& pvel,
	const ParticleDataImpl<Real>& detDeformationGrad, const ParticleDataImpl<Matrix3x3f>& deformationGrad,
	const ParticleDataImpl<Matrix3x3f>& affineMomentum, const ParticleDataImpl<Matrix3x3f>& rotation,
	Grid<Real>* kernelGrid, MACGrid* velTmp0, MACGrid* velTmp1, MACGrid* velTmp2,
	Grid<Real>* massTmp0, Grid<Real>* massTmp1, Grid<Real>* massTmp2,
	const Real hardening, const Real E, const Real nu, const Real pmass, const Real pvol)
{
	const bool is3D = true;

	// Compute particle data in single thread
	if (kernelGrid == nullptr) {
		vel.clear();
		mass.clear();
		const int sizeK = 3;
		knMpmMapPartsToGrid<Matrix3x3f>(pp, vel, mass, flags, blockIdxGrid, pvel, detDeformationGrad,
			deformationGrad, rotation, affineMomentum, hardening, E, nu, pmass, pvol, sizeK, is3D, 0, 0);
		return;
	}

	// Compute particle data with multiple threads (number of threads depends on size of kernelgrid)
	KernelHelper<Matrix3x3f>(*kernelGrid, pp, flags, blockIdxGrid, vel, velTmp0, velTmp1, velTmp2, mass, massTmp0, massTmp1, massTmp2,
		pvel, detDeformationGrad, deformationGrad, affineMomentum, rotation, hardening, E, nu, pmass, pvol, is3D);
}

KERNEL(idx, bnd=0, reduce=max) returns(Real maxVel=-std::numeric_limits<Real>::max())
Real KnMpmUpdateGrid(
	const FlagGrid& flags, const Vec3& gravity, MACGrid& vel, Grid<Real>& mass, const Grid<Real>& phiObs,
	const MACGrid& fractions, const Grid<Vec3>& blockIjkGrid, const Grid<int>& blockIdxGrid, const MACGrid* obvel,
	MACGrid* velTmp0, MACGrid* velTmp1, MACGrid* velTmp2, Grid<Real>* massTmp0, Grid<Real>* massTmp1, Grid<Real>* massTmp2, int boundaryCondition)
{
	const bool withKernelHelper = velTmp0 && massTmp0 && velTmp1 && massTmp1 && velTmp2 && massTmp2;

	// No grid updates needed in cells with no mass
	if (withKernelHelper && (*massTmp0)[idx] <= 0 && (*massTmp1)[idx] <= 0 && (*massTmp2)[idx] <= 0) return;
	if (!withKernelHelper && mass[idx] <= 0) return;

	const Real dt = flags.getParent()->getDt();

	// If using kernel helper grids, add them to global vel and mass grids
	if (withKernelHelper) {
		// Normalize velocity by mass, then add velocity update
		vel[idx] = ((*velTmp0)[idx] + (*velTmp1)[idx]  + (*velTmp2)[idx]) /
					((*massTmp0)[idx] + (*massTmp1)[idx] + (*massTmp2)[idx]) + dt * gravity;

		// Vel and mass buffers not needed anymore, clear here, is faster than grid.clear()
		if ((*massTmp0)[idx] > 0) { (*velTmp0)[idx] = Vec3(0.); (*massTmp0)[idx] = 0.; }
		if ((*massTmp1)[idx] > 0) { (*velTmp1)[idx] = Vec3(0.); (*massTmp1)[idx] = 0.; }
		if ((*massTmp2)[idx] > 0) { (*velTmp2)[idx] = Vec3(0.); (*massTmp2)[idx] = 0.; }
	}
	else {
		vel[idx] = (vel[idx] / mass[idx]) + dt * gravity;
	}

	// Keep track of maximum velocity, needed for adaptive time-stepping
	const Real s = normSquare(vel[idx]);
	if (s > maxVel) maxVel = s;

	const int i = blockIjkGrid[idx].x, j = blockIjkGrid[idx].y, k = blockIjkGrid[idx].z;

	// Boundary condition
	if (fractions(i,j,k).x < 1 || fractions(i,j,k).y < 1 || (phiObs.is3D() && fractions(i,j,k).z < 1)) {
		// Normal velocity contribution: Vec3 velN = dot(vel[idx], dphi) * dphi;
		// Tangential velocity contribution: Vec3 velT = vel[idx] - velN;
		if (boundaryCondition == 0) { // separate
			Vec3 dphi = getGradient(phiObs, i, j, k);
			normalize(dphi);
			Real normal = dot(vel[idx], dphi);
			vel[idx] -= dphi * std::min(normal, Real(0.));
		} else if (boundaryCondition == 1) { // no-slip: zero in normal and tangential dir
			vel[idx] = Vec3(0.);
		} else if (boundaryCondition == 2) { // free-slip: tangential flows only
			Vec3 dphi = getGradient(phiObs, i, j, k);
			normalize(dphi);
			vel[idx] -= dot(vel[idx], dphi) * dphi;
		}
	}
}

PYTHON() Real mpmUpdateGrid(
	const FlagGrid& flags, const Vec3& gravity, MACGrid& vel, Grid<Real>& mass, const Grid<Real>& phiObs,
	const MACGrid& fractions, const Grid<Vec3>& blockIjkGrid, const Grid<int>& blockIdxGrid, const MACGrid* obvel,
	MACGrid* velTmp0, MACGrid* velTmp1, MACGrid* velTmp2, Grid<Real>* massTmp0, Grid<Real>* massTmp1, Grid<Real>* massTmp2, int boundaryCondition)
{
	unusedParameter(obvel);
	Real maxVel = KnMpmUpdateGrid(flags, gravity, vel, mass, phiObs, fractions, blockIjkGrid, blockIdxGrid, obvel,
		velTmp0, velTmp1, velTmp2, massTmp0, massTmp1, massTmp2, boundaryCondition);
	return sqrt(maxVel);
}

KERNEL(pts, bnd=0) template<class T> void knMpmMapGridToParts(
	BasicParticleSystem& pp, const MACGrid& vel, FlagGrid& flags, const Grid<Real>& phiObs, const Grid<int>& blockIdxGrid,
	ParticleDataImpl<Vec3>& pvel, ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<T>& deformationGrad,
	ParticleDataImpl<T>& affineMomentum, const bool plastic, const int sizeK, const bool is3D)
{
	// Domain parameters
	const Real dt = pp.getParent()->getDt();
	const Real dx = pp.getParent()->getDx();
	const Real inv_dx = 1.0 / dx;

	const Vec3 pos = pp.getPos(idx);
	if (!pos.isValid()) return; // omit nans

	const Vec3i base = toVec3i(pos);
	if (!vel.isInBounds(base)) return;
	const Vec3 fx = (pos - toVec3(base)) - 0.5;

	const int sizeI = 3, sizeJ = 3;

	// Quadratic kernels, MPM course Equation 123.
	Vec3 w[3] = {
		Vec3(0.5)  * square(Vec3(0.5) - fx),
		Vec3(0.75) - square(fx),
		Vec3(0.5)  * square(Vec3(0.5) + fx)
	};
	// In 2D the Z component is unused
	if (!is3D) {
		w[0].z = 1.;
		w[1].z = 1.;
		w[2].z = 1.;
	}

	Vec3 pvelNew = Vec3(0.); // Stores new pvel from for-loop
	T outerProd(Vec3(0.));
	T affineMomentumNew(Vec3(0.)); // Stores new affine momentum from for-loop

	Vec3 dpos, gridVel;
	Vec3i targetPos;
	Real weight;
	IndexInt targetIdx;

	// Factor to zero out the last dim in 2D mode
	const Vec3 dimFactor = (is3D) ? Vec3(1) : Vec3(1,1,0);

	// Check if current and outermost cell is in bounds (knowledge saves time in neighbor loop)
	const bool upperInBounds = vel.isInBounds(base + toVec3i(sizeI-1,sizeJ-1,sizeK-1));

	// Neighbor loop: Iterate over neighboring cells of this particle
	for (int k = 0; k < sizeK; ++k)
	for (int j = 0; j < sizeJ; ++j)
	for (int i = 0; i < sizeI; ++i)
	{
		targetPos = base + toVec3i(i-1,j-1,k-1);

		// Only perform bounds check for current ijk if upper bound cell is not in bounds (saves time)
		if (!upperInBounds && !vel.isInBounds(targetPos)) continue;

		targetIdx = blockIdxGrid(targetPos);
		gridVel = vel(targetIdx) * dimFactor;
		weight = w[i].x * w[j].y * w[k].z;
		pvelNew += weight * gridVel;

		dpos = toVec3(targetPos) - pos + 0.5;

		outerProduct(outerProd, weight * gridVel, dpos * dimFactor);
		affineMomentumNew += 4 * outerProd;
	}
	// Copy accumulated values into actual particle data-structures (outside of for-loop for better performance)
	pvel[idx] = pvelNew;
	affineMomentum[idx] = affineMomentumNew;

	// Advection
	pp[idx].pos += dt * pvel[idx];

	// Deformation update
	T F = (T(Vec3(1.)) + dt * affineMomentum[idx]) * deformationGrad[idx];

	// SVD
	T svdU(Vec3(0.)), Sig(Vec3(0.)), svdV(Vec3(0.));
	svd(F, svdU, svdV, Sig);

	// Snow plasticity
	const int sizeSig = (is3D) ? 3 : 2;
	for (int i = 0; i < sizeSig * int(plastic); i++) {
		Sig(i,i) = clamp(Sig(i,i), Real(1 - 2.5e-2), Real(1 + 7.5e-3));
	}

	const Real oldJ = F.determinant();
	F = svdU * Sig * svdV.transposed();
	const Real Jp_new = clamp(detDeformationGrad[idx] * oldJ / F.determinant(), 0.6f, 20.0f);

	detDeformationGrad[idx] = Jp_new;
	deformationGrad[idx] = F;

	flags(base) = (flags(base) | FlagGrid::TypeEmpty) & ~FlagGrid::TypeFluid;
}

PYTHON() void mpmMapGridToParts2D(
	BasicParticleSystem& pp, const MACGrid& vel, FlagGrid& flags, const Grid<Real>& phiObs, const Grid<int>& blockIdxGrid,
	ParticleDataImpl<Vec3>& pvel, ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix2x2f>& deformationGrad,
	ParticleDataImpl<Matrix2x2f>& affineMomentum, const bool plastic)
{
	const int sizeK = 1;
	const bool is3D = false;
	knMpmMapGridToParts<Matrix2x2f>(pp, vel, flags, phiObs, blockIdxGrid, pvel, detDeformationGrad, deformationGrad, affineMomentum, plastic, sizeK, is3D);
}

PYTHON() void mpmMapGridToParts3D(
	BasicParticleSystem& pp, const MACGrid& vel, FlagGrid& flags, const Grid<Real>& phiObs, const Grid<int>& blockIdxGrid,
	ParticleDataImpl<Vec3>& pvel, ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix3x3f>& deformationGrad,
	ParticleDataImpl<Matrix3x3f>& affineMomentum, const bool plastic)
{
	const int sizeK = 3;
	const bool is3D = true;
	knMpmMapGridToParts<Matrix3x3f>(pp, vel, flags, phiObs, blockIdxGrid, pvel, detDeformationGrad, deformationGrad, affineMomentum, plastic, sizeK, is3D);
}

} // namespace
