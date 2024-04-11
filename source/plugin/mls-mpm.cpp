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

KERNEL(pts, single, bnd=0) template<class T>
void knMpmMapVec3ToMACGrid(
	const BasicParticleSystem& pp, MACGrid& vel, Grid<Real>& mass, FlagGrid& flags,
	const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<T>& deformationGrad, const ParticleDataImpl<T>& R, const ParticleDataImpl<T>& affineMomentum,
	const Real hardening, const Real E, const Real nu, const Real pmass, const Real pvol, int jStart, int kStart, const bool is3D)
{
	// Initial Lame parameters
	const Real mu_0 = E / (2 * (1 + nu));
	const Real lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu));

	// Lame parameters, MPM course Equation 86.
	const Real e = std::exp(hardening * (1.0f - detDeformationGrad[idx]));
	const Real mu = mu_0 * e;
	const Real lambda = lambda_0 * e;

	// Domain parameters
	const Real dt = pp.getParent()->getDt();
	const Real dx = pp.getParent()->getDx();
	const Real inv_dx = 1.0 / dx;
	const Real Dinv = 4 * inv_dx * inv_dx;

	const Vec3 pos = pp.getPos(idx);
	const Vec3i base = toVec3i(pos - Vec3(0.5));
	const Vec3 fx = (pos - toVec3(base));

	// Quadratic kernels, MPM course Equation 123.
	Vec3 w[3] = {
		Vec3(0.5)  * square(Vec3(1.5) - fx), // x = fx
		Vec3(0.75) - square(fx - Vec3(1.0)), // x = fx - 1
		Vec3(0.5)  * square(fx - Vec3(0.5))  // x = fx - 2
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

	const int sizeQKernel = sizeof(w) / sizeof(w[0]);
	const int sizeI = sizeQKernel;
	int sizeK = kStart + 1;
	int sizeJ = jStart + 1;

	// Loop dimension in full range if no specific k or j given
	if (kStart == -1) { kStart = 0; sizeK = sizeQKernel; }
	if (jStart == -1) { jStart = 0; sizeJ = sizeQKernel; }

	Vec3 dpos, aff;
	IndexInt targetPos;
	Real weight;

	const Vec3 vel_x_mass = pvel[idx] * pmass;
	// Factor to zero out the last dim in 2D mode
	const Vec3 dimFactor = (is3D) ? Vec3(1) : Vec3(1,1,0);

	// Check if current and outermost cell is in bounds (knowledge saves time in neighbor loop)
	const bool upperInBounds = vel.isInBounds(base + toVec3i(sizeI-1,sizeJ-1,sizeK-1));
	const bool thisInBounds = vel.isInBounds(base);

	// Neighbor loop: Iterate over neighboring cells of this particle
	for (int k = kStart; k < sizeK; k++)
	for (int j = jStart; j < sizeJ; j++)
	for (int i = 0; i < sizeI; i++)
	{
		// Only perform bounds check for current ijk if this or upper bound cell are not in bounds (saves time)
		if ((!upperInBounds || !thisInBounds) && !vel.isInBounds(base + toVec3i(i,j,k))) continue;

		targetPos = vel.index(base + toVec3i(i,j,k));
		dpos = (Vec3(i,j,k) - fx) * dx;

		aff = affine * dpos;
		weight = w[i].x * w[j].y * w[k].z;

		// Grid access is not cache optimal here (i.e. most expensive part of this function)
		vel(targetPos) += weight * (vel_x_mass + aff) * dimFactor;
		mass(targetPos) += weight * pmass;
	}

	// Set fluid flag, used in grid update
	if (thisInBounds)
		flags(base) = (flags(base) | FlagGrid::TypeFluid) & ~FlagGrid::TypeEmpty;
}

KERNEL() template<class T>
void KernelHelper(const Grid<Real>& kernelGrid, const BasicParticleSystem& pp, FlagGrid& flags,
	MACGrid* vel, MACGrid* velTmp1, MACGrid* velTmp2,
	Grid<Real>* mass, Grid<Real>* massTmp1, Grid<Real>* massTmp2,
	const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<T>& deformationGrad, const ParticleDataImpl<T>& affineMomentum, const ParticleDataImpl<T>& rotation,
	const Real hardening, const Real E, const Real nu, const Real pmass, const Real pvol, bool is3D)
{
	// assertMsg(i <= 2 && j <= 2 && k <= 2, "Unsupported grid size. Ensure size kernelGrid is at most 3x3x3");
	MACGrid* velSelect = nullptr;
	Grid<Real>* massSelect = nullptr;
	if ((k == 0 && is3D) || (j == 0 && !is3D)) { velSelect = vel; massSelect = mass; }
	if ((k == 1 && is3D) || (j == 1 && !is3D)) { velSelect = velTmp1; massSelect = massTmp1; }
	if ((k == 2 && is3D) || (j == 2 && !is3D)) { velSelect = velTmp2; massSelect = massTmp2; }
	// assertMsg(vel != nullptr, "vel grid pointer must not be be null");
	// assertMsg(mass != nullptr, "mass grid pointer must not be be null");

	// -1: loop full range, 0 or k, j: loop in exactly this location
	const int jStart = (is3D) ? -1 : j;
	const int kStart = (is3D) ? k : 0;
	knMpmMapVec3ToMACGrid<T>(pp, *velSelect, *massSelect, flags, pvel, detDeformationGrad, deformationGrad, rotation, affineMomentum, hardening, E, nu, pmass, pvol, jStart, kStart, is3D);
}

PYTHON() void mpmMapPartsToMACGrid2D(MACGrid& vel, Grid<Real>& mass, FlagGrid& flags,
	const BasicParticleSystem& pp, const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<Matrix2x2f>& deformationGrad, const ParticleDataImpl<Matrix2x2f>& affineMomentum, const ParticleDataImpl<Matrix2x2f>& rotation,
	Grid<Real>* kernelGrid=nullptr, MACGrid* velTmp1=nullptr, MACGrid* velTmp2=nullptr, Grid<Real>* massTmp1=nullptr, Grid<Real>* massTmp2=nullptr,
	const Real hardening=10.0f, const Real E=1e4f, const Real nu=0.2f, const Real pmass=1.0f, const Real pvol=1.0f)
{
	vel.clear();
	mass.clear();

	const bool is3D = false;

	// Compute particle data in single thread
	if (kernelGrid == nullptr) {
		const int kStart = 0, jStart = -1;
		const bool is3D = false;
		knMpmMapVec3ToMACGrid<Matrix2x2f>(pp, vel, mass, flags, pvel, detDeformationGrad, deformationGrad, rotation, affineMomentum, hardening, E, nu, pmass, pvol, jStart, kStart, is3D);
		return;
	}

	// Compute particle data with multiple threads
	KernelHelper<Matrix2x2f>(*kernelGrid, pp, flags,
		&vel, velTmp1, velTmp2, &mass, massTmp1, massTmp2, 
		pvel, detDeformationGrad, deformationGrad, affineMomentum, rotation, hardening, E, nu, pmass, pvol, is3D);
}

PYTHON() void mpmMapPartsToMACGrid3D(MACGrid& vel, Grid<Real>& mass, FlagGrid& flags,
	const BasicParticleSystem& pp, const ParticleDataImpl<Vec3>& pvel, const ParticleDataImpl<Real>& detDeformationGrad,
	const ParticleDataImpl<Matrix3x3f>& deformationGrad, const ParticleDataImpl<Matrix3x3f>& affineMomentum, const ParticleDataImpl<Matrix3x3f>& rotation,
	Grid<Real>* kernelGrid=nullptr, MACGrid* velTmp1=nullptr, MACGrid* velTmp2=nullptr,
	Grid<Real>* massTmp1=nullptr, Grid<Real>* massTmp2=nullptr,
	const Real hardening=10.0f, const Real E=1e4f, const Real nu=0.2f, const Real pmass=1.0f, const Real pvol=1.0f)
{
	// assertMsg(vel != nullptr && mass != nullptr, "Missing one of the base grids, check function args");

	// Only have to clear vel and mass, optional kernel buffer grids are cleared in KnMpmUpdateGrid() (this improves performance)
	vel.clear();
	mass.clear();

	const bool is3D = true;

	// Compute particle data in single thread
	if (kernelGrid == nullptr) {
		const int kStart = -1, jStart = -1;
		knMpmMapVec3ToMACGrid<Matrix3x3f>(pp, vel, mass, flags, pvel, detDeformationGrad, deformationGrad, rotation, affineMomentum, hardening, E, nu, pmass, pvol, jStart, kStart, is3D);
		return;
	}

	// Compute particle data with multiple threads
	KernelHelper<Matrix3x3f>(*kernelGrid, pp, flags,
		&vel, velTmp1, velTmp2, &mass, massTmp1, massTmp2, 
		pvel, detDeformationGrad, deformationGrad, affineMomentum, rotation, hardening, E, nu, pmass, pvol, is3D);
}

KERNEL(bnd=0)
void KnMpmUpdateGrid(FlagGrid& flags, const BasicParticleSystem& pp,  const Vec3& gravity, MACGrid& vel, Grid<Real>& mass,
	MACGrid* velTmp1, MACGrid* velTmp2, Grid<Real>* massTmp1, Grid<Real>* massTmp2, const MACGrid* obvel)
{
	bool withKernelHelper = velTmp1 && massTmp1 && velTmp2 && massTmp2;

	// No grid updates needed in cells with no mass
	if (withKernelHelper && mass(i,j,k) <= 0 && (*massTmp1)(i,j,k) <= 0 && (*massTmp2)(i,j,k) <= 0) return;
	if (!withKernelHelper && mass(i,j,k) <= 0) return;

	// If using kernel helper grids, add them to global vel and mass grids
	if (withKernelHelper) {
		vel(i,j,k) += (*velTmp1)(i,j,k) + (*velTmp2)(i,j,k);
		mass(i,j,k) += (*massTmp1)(i,j,k) + (*massTmp2)(i,j,k);

		// Vel and mass kernel buffers not needed anymore, clear here
		(*velTmp1)(i,j,k) = Vec3(0.);
		(*massTmp1)(i,j,k) = 0.;
		(*velTmp2)(i,j,k) = Vec3(0.);
		(*massTmp2)(i,j,k) = 0.;
	}

	// Normalize by mass
	vel(i,j,k) /= mass(i,j,k);
	mass(i,j,k) /= mass(i,j,k);

	// Velocity update in grid
	vel(i,j,k) += pp.getParent()->getDt() * gravity;

	// Handle behavior at boundaries / obstacles (similar to setWallBcs())
	const bool curObs = flags.isObstacle(i,j,k);
	Vec3 bcsVel(0.,0.,0.);
	if (obvel) {
		bcsVel.x = (*obvel)(i,j,k).x;
		bcsVel.y = (*obvel)(i,j,k).y;
		if((*obvel).is3D()) bcsVel.z = (*obvel)(i,j,k).z;
	}
	if (i>0 && flags.isObstacle(i-1,j,k))					{ vel(i,j,k).x = bcsVel.x; mass(i,j,k) = 0.; }
	if (i>0 && curObs && flags.isFluid(i-1,j,k))			{ vel(i,j,k).x = bcsVel.x; mass(i,j,k) = 0.; }
	if (j>0 && flags.isObstacle(i,j-1,k))					{ vel(i,j,k).y = bcsVel.y; mass(i,j,k) = 0.; }
	if (j>0 && curObs && flags.isFluid(i,j-1,k))			{ vel(i,j,k).y = bcsVel.y; mass(i,j,k) = 0.; }
	if(!vel.is3D()) 										{ vel(i,j,k).z = 0; }
	else {	if (k>0 && flags.isObstacle(i,j,k-1))			{ vel(i,j,k).z = bcsVel.z; mass(i,j,k) = 0.; }
			if (k>0 && curObs && flags.isFluid(i,j,k-1))	{ vel(i,j,k).z = bcsVel.z; mass(i,j,k) = 0.; }
	}
}

PYTHON() void mpmUpdateGrid(FlagGrid& flags, const BasicParticleSystem& pp, const Vec3& gravity, MACGrid& vel, Grid<Real>& mass,
	MACGrid* velTmp1=nullptr, MACGrid* velTmp2=nullptr, Grid<Real>* massTmp1=nullptr, Grid<Real>* massTmp2=nullptr, const MACGrid* obvel=nullptr)
{
	KnMpmUpdateGrid(flags, pp, gravity, vel, mass, velTmp1, velTmp2, massTmp1, massTmp2, obvel);
}

KERNEL(pts, bnd=0) template<class T>
void knMpmMapMACGridToVec3(
	BasicParticleSystem& pp, const MACGrid& vel, const Grid<Real>& mass, FlagGrid& flags,
	ParticleDataImpl<Vec3>& pvel, ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<T>& deformationGrad,
	ParticleDataImpl<T>& affineMomentum, const bool plastic, int kStart, const bool is3D)
{
	// Domain parameters
	const Real dt = pp.getParent()->getDt();
	const Real dx = pp.getParent()->getDx();
	const Real inv_dx = 1.0 / dx;

	const Vec3 pos = pp.getPos(idx);
	const Vec3i base = toVec3i(pos - Vec3(0.5));
	const Vec3 fx = (pos - toVec3(base));

	// Quadratic kernels, MPM course Equation 123.
	Vec3 w[3] = {
		Vec3(0.5)  * square(Vec3(1.5) - fx), // x = fx
		Vec3(0.75) - square(fx - Vec3(1.0)), // x = fx - 1
		Vec3(0.5)  * square(fx - Vec3(0.5))  // x = fx - 2
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

	const int sizeQKernel = sizeof(w) / sizeof(w[0]);
	const int sizeI = sizeQKernel, sizeJ = sizeQKernel;
	int sizeK = kStart + 1;
	// Loop k dimension in full range if no specific k given
	if (kStart == -1) { kStart = 0; sizeK = sizeQKernel; }

	Vec3 dpos, gridVel;
	IndexInt targetPos;
	Real weight;

	// Factor to zero out the last dim in 2D mode
	const Vec3 dimFactor = (is3D) ? Vec3(1) : Vec3(1,1,0);

	// Check if current and outermost cell is in bounds (knowledge saves time in neighbor loop)
	const bool upperInBounds = vel.isInBounds(base + toVec3i(sizeI-1,sizeJ-1,sizeK-1));
	const bool thisInBounds = vel.isInBounds(base);

	// Neighbor loop: Iterate over neighboring cells of this particle
	for (int k = kStart; k < sizeK; k++)
	for (int j = 0; j < sizeJ; j++)
	for (int i = 0; i < sizeI; i++)
	{
		// Only perform bounds check for current ijk if this or upper bound cell are not in bounds
		if ((!upperInBounds || !thisInBounds) && !vel.isInBounds(base + toVec3i(i,j,k))) continue;

		targetPos = vel.index(base + toVec3i(i,j,k));
		dpos = (Vec3(i,j,k) - fx);

		gridVel = vel(targetPos) * dimFactor;
		weight = w[i].x * w[j].y * w[k].z;
		pvelNew += weight * gridVel;

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
		Sig(i,i) = clamp(Sig(i,i), 1.0f - 2.5e-2f, 1.0f + 7.5e-3f);
	}

	Real oldJ = F.determinant();
	F = svdU * Sig * svdV.transposed();
	Real Jp_new = clamp(detDeformationGrad[idx] * oldJ / F.determinant(), 0.6f, 20.0f);

	detDeformationGrad[idx] = Jp_new;
	deformationGrad[idx] = F;

	// Clear fluid flag for next iteration
	if (thisInBounds)
		flags(base) = (flags(base) | FlagGrid::TypeEmpty) & ~FlagGrid::TypeFluid;
}

PYTHON() void mpmMapMACGridToParts2D(
	BasicParticleSystem& pp, const MACGrid& vel, const Grid<Real>& mass, FlagGrid& flags,
	ParticleDataImpl<Vec3>& pvel, ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix2x2f>& deformationGrad,
	ParticleDataImpl<Matrix2x2f>& affineMomentum, const bool plastic=true)
{
	const int kStart = 0; // Loop k dim only once in 2D, using k=0
	const bool is3D = false;
	knMpmMapMACGridToVec3<Matrix2x2f>(pp, vel, mass, flags, pvel, detDeformationGrad, deformationGrad, affineMomentum, plastic, kStart, is3D);
}

PYTHON() void mpmMapMACGridToParts3D(
	BasicParticleSystem& pp, const MACGrid& vel, const Grid<Real>& mass, FlagGrid& flags,
	ParticleDataImpl<Vec3>& pvel, ParticleDataImpl<Real>& detDeformationGrad, ParticleDataImpl<Matrix3x3f>& deformationGrad,
	ParticleDataImpl<Matrix3x3f>& affineMomentum, const bool plastic=true)
{
	const int kStart = -1; // Loop full range in k
	const bool is3D = true;
	knMpmMapMACGridToVec3<Matrix3x3f>(pp, vel, mass, flags, pvel, detDeformationGrad, deformationGrad, affineMomentum, plastic, kStart, is3D);
}

} // namespace
