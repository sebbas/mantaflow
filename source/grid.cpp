/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Grid representation
 *
 ******************************************************************************/

#include "grid.h"
#include "levelset.h"
#include "kernel.h"
#include "mantaio.h"
#include <limits>
#include <sstream>
#include <cstring>

#include "commonkernels.h"


using namespace std;
namespace Manta {

//******************************************************************************
// GridBase members

GridBase::GridBase (FluidSolver* parent) 
	: PbClass(parent), mType(TypeNone)
{
	checkParent();
	m3D = getParent()->is3D();
}

//******************************************************************************
// Grid<T> members

// helpers to set type
template<class T> inline GridBase::GridType typeList() { return GridBase::TypeNone; }
template<> inline GridBase::GridType typeList<Real>()  { return GridBase::TypeReal; }
template<> inline GridBase::GridType typeList<int>()   { return GridBase::TypeInt;  }
template<> inline GridBase::GridType typeList<Vec3>()  { return GridBase::TypeVec3; }

template<class T>
Grid<T>::Grid(FluidSolver* parent, bool show, bool sparse)
        : GridBase(parent), mExternalData(false)
{
	mType = typeList<T>();
	mSize = parent->getGridSize();
	mData = parent->getGridPointer<T>();
	
	mStrideZ = parent->is2D() ? 0 : (mSize.x * mSize.y);
	mDx = 1.0 / mSize.max();
	clear();
	setHidden(!show);

#if OPENVDB==1
	mSaveSparse = sparse;
#else
	if (sparse) debMsg("Cannot enable sparse save option without OpenVDB", 1);
	mSaveSparse = false;
#endif
}

template<class T>
Grid<T>::Grid(FluidSolver* parent, T* data, bool show) : Grid<T>::Grid(parent, show)
{
	mData = data;
}

template<class T>
Grid<T>::Grid(const Grid<T>& a)
        : GridBase(a.getParent()), mExternalData(false) {
	mSize = a.mSize;
	mType = a.mType;
	mStrideZ = a.mStrideZ;
	mDx = a.mDx;
	FluidSolver *gp = a.getParent();
	mData = gp->getGridPointer<T>();
	memcpy(mData, a.mData, sizeof(T) * a.mSize.x * a.mSize.y * a.mSize.z);
}

template<class T>
Grid<T>::~Grid() {
    if(!mExternalData)  {
        mParent->freeGridPointer<T>(mData);
    }
}

template<class T>
void Grid<T>::clear() {
	memset(mData, 0, sizeof(T) * mSize.x * mSize.y * mSize.z);    
}

template<class T>
void Grid<T>::swap(Grid<T>& other) {
	if (other.getSizeX() != getSizeX() || other.getSizeY() != getSizeY() || other.getSizeZ() != getSizeZ())
		errMsg("Grid::swap(): Grid dimensions mismatch.");
	
        if(mExternalData || other.mExternalData)
            errMsg("Grid::swap(): Cannot swap if one grid stores mExternalData.");

	T* dswap = other.mData;
	other.mData = mData;
	mData = dswap;

}

template<class T>
int Grid<T>::load(string name) {
	if (name.find_last_of('.') == string::npos)
		errMsg("file '" + name + "' does not have an extension");
	string ext = name.substr(name.find_last_of('.'));
	if (ext == ".raw")
		return readGridRaw(name, this);
	else if (ext == ".uni")
		return readGridUni(name, this);
	else if (ext == ".vol")
		return readGridVol(name, this);
	else if (ext == ".npz")
		return readGridNumpy(name, this);
	else if (ext == ".vdb") {
		std::vector<PbClass*> grids;
		grids.push_back(this);
		return readObjectsVDB(name, &grids);
	} else
		errMsg("file '" + name +"' filetype not supported");
	return 0;
}

template<class T>
int Grid<T>::save(string name) {
	if (name.find_last_of('.') == string::npos)
		errMsg("file '" + name + "' does not have an extension");
	string ext = name.substr(name.find_last_of('.'));
	if (ext == ".raw")
		return writeGridRaw(name, this);
	else if (ext == ".uni")
		return writeGridUni(name, this);
	else if (ext == ".vol")
		return writeGridVol(name, this);
	else if (ext == ".npz")
		return writeGridNumpy(name, this);
	else if (ext == ".vdb") {
		std::vector<PbClass*> grids;
		grids.push_back(this);
		return writeObjectsVDB(name, &grids);
	} else if (ext == ".txt")
		return writeGridTxt(name, this);
	else
		errMsg("file '" + name +"' filetype not supported");
	return 0;
}

//******************************************************************************
// Grid<T> operators

//! Kernel: Compute min value of Real grid
KERNEL(idx, reduce=min) returns(Real minVal=std::numeric_limits<Real>::max())
Real CompMinReal(const Grid<Real>& val) {
	if (val[idx] < minVal)
		minVal = val[idx];
}

//! Kernel: Compute max value of Real grid
KERNEL(idx, reduce=max) returns(Real maxVal=-std::numeric_limits<Real>::max())
Real CompMaxReal(const Grid<Real>& val) {
	if (val[idx] > maxVal)
		maxVal = val[idx];
}

//! Kernel: Compute min value of int grid
KERNEL(idx, reduce=min) returns(int minVal=std::numeric_limits<int>::max())
int CompMinInt(const Grid<int>& val) {
	if (val[idx] < minVal)
		minVal = val[idx];
}

//! Kernel: Compute max value of int grid
KERNEL(idx, reduce=max) returns(int maxVal=-std::numeric_limits<int>::max())
int CompMaxInt(const Grid<int>& val) {
	if (val[idx] > maxVal)
		maxVal = val[idx];
}

//! Kernel: Compute min norm of vec grid
KERNEL(idx, reduce=min) returns(Real minVal=std::numeric_limits<Real>::max())
Real CompMinVec(const Grid<Vec3>& val) {
	const Real s = normSquare(val[idx]);
	if (s < minVal)
		minVal = s;
}

//! Kernel: Compute max norm of vec grid
KERNEL(idx, reduce=max) returns(Real maxVal=-std::numeric_limits<Real>::max())
Real CompMaxVec(const Grid<Vec3>& val) {
	const Real s = normSquare(val[idx]);
	if (s > maxVal)
		maxVal = s;
}

template<class T> Grid<T>& Grid<T>::copyFrom (const Grid<T>& a, bool copyType ) {
	assertMsg (a.mSize.x == mSize.x && a.mSize.y == mSize.y && a.mSize.z == mSize.z, "different grid resolutions "<<a.mSize<<" vs "<<this->mSize );
	memcpy(mData, a.mData, sizeof(T) * mSize.x * mSize.y * mSize.z);
	if(copyType) mType = a.mType; // copy type marker
	return *this;
}
/*template<class T> Grid<T>& Grid<T>::operator= (const Grid<T>& a) {
	note: do not use , use copyFrom instead
}*/

KERNEL(idx) template<class T> void knGridSetConstReal (Grid<T>& me, T val) { me[idx]  = val; }
KERNEL(idx) template<class T> void knGridAddConstReal (Grid<T>& me, T val) { me[idx] += val; }
KERNEL(idx) template<class T> void knGridMultConst (Grid<T>& me, T val) { me[idx] *= val; }

KERNEL(idx) template<class T> void knGridSafeDiv (Grid<T>& me, const Grid<T>& other) { me[idx] = safeDivide(me[idx], other[idx]); }
//KERNEL(idx) template<class T> void gridSafeDiv (Grid<T>& me, const Grid<T>& other) { me[idx] = safeDivide(me[idx], other[idx]); }

KERNEL(idx) template<class T> void knGridClamp(Grid<T>& me, const T& min, const T& max) { me[idx] = clamp(me[idx], min, max); }

template<typename T> inline void stomp(T &v, const T &th) { if(v<th) v=0; }
template<> inline void stomp<Vec3>(Vec3 &v, const Vec3 &th) { if(v[0]<th[0]) v[0]=0; if(v[1]<th[1]) v[1]=0; if(v[2]<th[2]) v[2]=0; }
KERNEL(idx) template<class T> void knGridStomp(Grid<T>& me, const T& threshold) { stomp(me[idx], threshold); }

KERNEL() template<class T> void knPermuteAxes (Grid<T>& self, Grid<T>& target, int axis0, int axis1, int axis2) {
	int i0 = axis0 == 0 ? i : (axis0 == 1 ? j : k);
	int i1 = axis1 == 0 ? i : (axis1 == 1 ? j : k);
	int i2 = axis2 == 0 ? i : (axis2 == 1 ? j : k);
	target(i0,i1,i2) = self(i,j,k);
}

KERNEL(idx) void knJoinVec(Grid<Vec3>& a, const Grid<Vec3>& b, bool keepMax) {
	Real a1 = normSquare(a[idx]);
	Real b1 = normSquare(b[idx]);
	a[idx] = (keepMax) ? max(a1, b1) : min(a1, b1);
}
KERNEL(idx) void knJoinInt(Grid<int>& a, const Grid<int>& b, bool keepMax) {
	a[idx] = (keepMax) ? max(a[idx], b[idx]) : min(a[idx], b[idx]);
}
KERNEL(idx) void knJoinReal(Grid<Real>& a, const Grid<Real>& b, bool keepMax) {
	a[idx] = (keepMax) ? max(a[idx], b[idx]) : min(a[idx], b[idx]);
}

template<class T> Grid<T>& Grid<T>::safeDivide (const Grid<T>& a) {
	knGridSafeDiv<T> (*this, a);
	return *this;
}

template<class T> int Grid<T>::getGridType() {
	return static_cast<int>(mType);
}

template<class T> void Grid<T>::add(const Grid<T>& a) {
	gridAdd<T,T>(*this, a);
}
template<class T> void Grid<T>::add(std::initializer_list<Grid<T> *> grids) {
	gridAddN<T,T>(*this, grids);
}
template<class T> void Grid<T>::sub(const Grid<T>& a) {
	gridSub<T,T>(*this, a);
}
template<class T> void Grid<T>::sub(std::initializer_list<Grid<T> *> grids) {
	gridSubN<T,T>(*this, grids);
}
template<class T> void Grid<T>::addScaled(const Grid<T>& a, const T& factor) { 
	gridScaledAdd<T,T> (*this, a, factor); 
}
template<class T> void Grid<T>::setConst(T a) {
	knGridSetConstReal<T>( *this, T(a) );
}
template<class T> void Grid<T>::addConst(T a) {
	knGridAddConstReal<T>( *this, T(a) );
}
template<class T> void Grid<T>::multConst(T a) {
	knGridMultConst<T>( *this, a );
}

template<class T> void Grid<T>::mult(const Grid<T>& a) {
	gridMult<T,T> (*this, a);
}

template<class T> void Grid<T>::clamp(Real min, Real max) {
	knGridClamp<T> (*this, T(min), T(max) );
}
template<class T> void Grid<T>::stomp(const T& threshold) {
	knGridStomp<T>(*this, threshold);
}
template<class T> void Grid<T>::permuteAxes(int axis0, int axis1, int axis2) {
	if(axis0 == axis1 || axis0 == axis2 || axis1 == axis2 || axis0  > 2 || axis1 > 2 || axis2 > 2 || axis0 < 0 || axis1 < 0  || axis2 < 0)
		return;
	Vec3i size = mParent->getGridSize();
	assertMsg( mParent->is2D() ? size.x == size.y : size.x == size.y && size.y == size.z, "Grid must be cubic!");
	Grid<T> tmp(mParent);
	knPermuteAxes<T>(*this, tmp, axis0, axis1, axis2);
	this->swap(tmp);
}
template<class T> void Grid<T>::permuteAxesCopyToGrid(int axis0, int axis1, int axis2, Grid<T>& out) {
	if(axis0 == axis1 || axis0 == axis2 || axis1 == axis2 || axis0  > 2 || axis1 > 2 || axis2 > 2 || axis0 < 0 || axis1 < 0  || axis2 < 0)
		return;
	assertMsg( this->getGridType() == out.getGridType(), "Grids must have same data type!");
	Vec3i size = mParent->getGridSize();
	Vec3i sizeTarget = out.getParent()->getGridSize();
	assertMsg( sizeTarget[axis0] == size[0] && sizeTarget[axis1] == size[1] && sizeTarget[axis2] == size[2], "Permuted grids must have the same dimensions!");
	knPermuteAxes<T>(*this, out, axis0, axis1, axis2);
}
template<> void Grid<Vec3>::join(const Grid<Vec3>& a, bool keepMax) {
	knJoinVec(*this, a, keepMax);
}
template<> void Grid<int>::join(const Grid<int>& a, bool keepMax) {
	knJoinInt(*this, a, keepMax);
}
template<> void Grid<Real>::join(const Grid<Real>& a, bool keepMax) {
	knJoinReal(*this, a, keepMax);
}

template<> Real Grid<Real>::getMax() const {
	return CompMaxReal (*this);
}
template<> Real Grid<Real>::getMin() const {
	return CompMinReal (*this);
}
template<> Real Grid<Real>::getMaxAbs() const {
	Real amin = CompMinReal (*this);
	Real amax = CompMaxReal (*this);
	return max( fabs(amin), fabs(amax));
}
template<> Real Grid<Vec3>::getMax() const {
	return sqrt(CompMaxVec (*this));
}
template<> Real Grid<Vec3>::getMin() const {
	return sqrt(CompMinVec (*this));
}
template<> Real Grid<Vec3>::getMaxAbs() const {
	return sqrt(CompMaxVec (*this));
}
template<> Real Grid<int>::getMax() const {
	return (Real) CompMaxInt (*this);
}
template<> Real Grid<int>::getMin() const {
	return (Real) CompMinInt (*this);
}
template<> Real Grid<int>::getMaxAbs() const {
	int amin = CompMinInt (*this);
	int amax = CompMaxInt (*this);
	return max( fabs((Real)amin), fabs((Real)amax));
}
template<class T> T* Grid<T>::getDataPointer() {
	return mData;
}

// L1 / L2 functions

//! calculate L1 norm for whole grid with non-parallelized loop
template<class GRID>
Real loop_calcL1Grid (const GRID &grid, int bnd)
{
	double accu = 0.;
	FOR_IJKT_BND(grid, bnd) { accu += norm(grid(i,j,k,t)); }
	return (Real)accu;
}

//! calculate L2 norm for whole grid with non-parallelized loop
// note - kernels "could" be used here, but they can't be templated at the moment (also, that would
// mean the bnd parameter is fixed)
template<class GRID>
Real loop_calcL2Grid(const GRID &grid, int bnd)
{
	double accu = 0.;
	FOR_IJKT_BND(grid, bnd) {
		accu += normSquare(grid(i,j,k,t)); // supported for real and vec3,4 types
	}
	return (Real)sqrt(accu);
}

//! compute L1 norm of whole grid content (note, not parallel at the moment)
template<class T> Real Grid<T>::getL1(int bnd) {
	return loop_calcL1Grid<Grid<T> >(*this, bnd);
}
//! compute L2 norm of whole grid content (note, not parallel at the moment)
template<class T> Real Grid<T>::getL2(int bnd) {
	return loop_calcL2Grid<Grid<T> >(*this, bnd);
}

KERNEL(reduce=+) returns(int cnt=0)
int knCountCells(const FlagGrid& flags, int flag, int bnd, Grid<Real>* mask) { 
	if(mask) (*mask)(i,j,k) = 0.;
	if( bnd>0 && (!flags.isInBounds(Vec3i(i,j,k), bnd)) ) return;
	if (flags(i,j,k) & flag ) {
		cnt++; 
		if(mask) (*mask)(i,j,k) = 1.;
	}
}

//! count number of cells of a certain type flag (can contain multiple bits, checks if any one of them is set - not all!)
int FlagGrid::countCells(int flag, int bnd, Grid<Real>* mask) {
	return knCountCells(*this, flag, bnd, mask);
}

// compute maximal diference of two cells in the grid
// used for testing system
PYTHON() Real gridMaxDiff(Grid<Real>& g1, Grid<Real>& g2)
{
	double maxVal = 0.;
	FOR_IJK(g1) {
		maxVal = std::max(maxVal, (double)fabs(g1(i, j, k) - g2(i, j, k)));
	}
	return maxVal;
}
PYTHON() Real gridMaxDiffInt(Grid<int>& g1, Grid<int>& g2)
{
	double maxVal = 0.;
	FOR_IJK(g1) {
		maxVal = std::max(maxVal, (double)fabs((double)g1(i, j, k) - g2(i, j, k)));
	}
	return maxVal;
}
PYTHON() Real gridMaxDiffVec3(Grid<Vec3>& g1, Grid<Vec3>& g2)
{
	double maxVal = 0.;
	FOR_IJK(g1) {
		// accumulate differences with double precision
		// note - don't use norm here! should be as precise as possible...
		double d = 0.;
		for (int c = 0; c<3; ++c) {
			d += fabs((double)g1(i, j, k)[c] - (double)g2(i, j, k)[c]);
		}
		maxVal = std::max(maxVal, d);
		//maxVal = std::max(maxVal, (double)fabs( norm(g1(i,j,k)-g2(i,j,k)) ));
	}
	return maxVal;
}

KERNEL() void knCopyMacToVec3(MACGrid &source, Grid<Vec3>& target)
{
	target(i,j,k) = source(i,j,k);
}
// simple helper functions to copy (convert) mac to vec3 , and levelset to real grids
// (are assumed to be the same for running the test cases - in general they're not!)
PYTHON() void copyMacToVec3(MACGrid &source, Grid<Vec3>& target)
{
	knCopyMacToVec3(source, target);
}

PYTHON() void convertMacToVec3 (MACGrid &source , Grid<Vec3> &target) { debMsg("Deprecated - do not use convertMacToVec3... use copyMacToVec3 instead",1); copyMacToVec3(source,target); }

KERNEL(bnd=1)
void knResampleVec3ToMac(Grid<Vec3>& source, MACGrid &target)
{
	target(i,j,k)[0] = 0.5*(source(i-1,j,k)[0]+source(i,j,k))[0];
	target(i,j,k)[1] = 0.5*(source(i,j-1,k)[1]+source(i,j,k))[1];
	if(target.is3D()) {
	target(i,j,k)[2] = 0.5*(source(i,j,k-1)[2]+source(i,j,k))[2]; }
}
//! vec3->mac grid conversion , but with full resampling 
PYTHON() void resampleVec3ToMac(Grid<Vec3>& source, MACGrid &target)
{
	knResampleVec3ToMac(source, target);
}

KERNEL(bnd=1)
void knResampleMacToVec3(MACGrid &source, Grid<Vec3>& target)
{
	target(i,j,k) = source.getCentered(i,j,k);
}
//! mac->vec3 grid conversion , with full resampling 
PYTHON() void resampleMacToVec3 (MACGrid &source, Grid<Vec3>& target)
{
	knResampleMacToVec3(source, target);
}

KERNEL() void knCopyLevelsetToReal(LevelsetGrid &source , Grid<Real> &target)
{
	target(i,j,k) = source(i,j,k);
}
PYTHON() void copyLevelsetToReal(LevelsetGrid &source , Grid<Real> &target)
{
	knCopyLevelsetToReal(source, target);
}

KERNEL() void knCopyVec3ToReal(Grid<Vec3> &source, Grid<Real> &targetX, Grid<Real> &targetY, Grid<Real> &targetZ)
{
	targetX(i,j,k) = source(i,j,k).x;
	targetY(i,j,k) = source(i,j,k).y;
	targetZ(i,j,k) = source(i,j,k).z;
}
PYTHON() void copyVec3ToReal (Grid<Vec3> &source, Grid<Real> &targetX, Grid<Real> &targetY, Grid<Real> &targetZ)
{
	knCopyVec3ToReal(source, targetX, targetY, targetZ);
}

KERNEL() void knCopyRealToVec3(Grid<Real> &sourceX, Grid<Real> &sourceY, Grid<Real> &sourceZ, Grid<Vec3> &target)
{
	target(i,j,k).x = sourceX(i,j,k);
	target(i,j,k).y = sourceY(i,j,k);
	target(i,j,k).z = sourceZ(i,j,k);
}
PYTHON() void copyRealToVec3 (Grid<Real> &sourceX, Grid<Real> &sourceY, Grid<Real> &sourceZ, Grid<Vec3> &target)
{
	knCopyRealToVec3(sourceX, sourceY, sourceZ, target);
}

PYTHON() void convertLevelsetToReal (LevelsetGrid &source , Grid<Real> &target) { debMsg("Deprecated - do not use convertLevelsetToReal... use copyLevelsetToReal instead",1); copyLevelsetToReal(source,target); }

template<class T> void Grid<T>::printGrid(int zSlice, bool printIndex, int bnd) {
	std::ostringstream out;
	out << std::endl;
	FOR_IJK_BND(*this,bnd) {
		IndexInt idx = (*this).index(i,j,k);
		if((zSlice>=0 && k==zSlice) || (zSlice<0)) { 
			out << " ";
			if(printIndex &&  this->is3D()) out << "  "<<i<<","<<j<<","<<k <<":";
			if(printIndex && !this->is3D()) out << "  "<<i<<","<<j<<":";
			out << (*this)[idx]; 
			if(i==(*this).getSizeX()-1 -bnd) out << std::endl; 
		}
	}
	out << endl; debMsg("Printing " << this->getName() << out.str().c_str() , 1);
}

//! helper to swap components of a grid (eg for data import)
PYTHON() void swapComponents(Grid<Vec3>& vel, int c1=0, int c2=1, int c3=2) {
	FOR_IJK(vel) {
		Vec3 v = vel(i,j,k);
		vel(i,j,k)[0] = v[c1];
		vel(i,j,k)[1] = v[c2];
		vel(i,j,k)[2] = v[c3];
	}
}

// helper functions for UV grid data (stored grid coordinates as Vec3 values, and uv weight in entry zero)

// make uv weight accesible in python
PYTHON() Real getUvWeight (Grid<Vec3> &uv) { return uv[0][0]; }

// note - right now the UV grids have 0 values at the border after advection... could be fixed with an extrapolation step...

// compute normalized modulo interval
static inline Real computeUvGridTime(Real t, Real resetTime) {
	return fmod( (t / resetTime), (Real)1. );
}
// create ramp function in 0..1 range with half frequency
static inline Real computeUvRamp(Real t) {
	Real uvWeight = 2. * t; 
	if (uvWeight>1.) uvWeight=2.-uvWeight;
	return uvWeight;
}

KERNEL() void knResetUvGrid (Grid<Vec3>& target, const Vec3* offset) {
	Vec3 coord = Vec3((Real)i,(Real)j,(Real)k);
	if (offset) coord += (*offset);
	target(i,j,k) = coord;
}

PYTHON() void resetUvGrid (Grid<Vec3> &target, const Vec3* offset=nullptr)
{
	knResetUvGrid reset(target, offset); // note, llvm complains about anonymous declaration here... ?
}
PYTHON() void updateUvWeight(Real resetTime, int index, int numUvs, Grid<Vec3> &uv, const Vec3* offset=nullptr)
{
	const Real t   = uv.getParent()->getTime();
	Real  timeOff  = resetTime/(Real)numUvs;

	Real lastt = computeUvGridTime(t +(Real)index*timeOff - uv.getParent()->getDt(), resetTime);
	Real currt = computeUvGridTime(t +(Real)index*timeOff                          , resetTime);
	Real uvWeight = computeUvRamp(currt);

	// normalize the uvw weights , note: this is a bit wasteful...
	Real uvWTotal = 0.;
	for(int i=0; i<numUvs; ++i) {
		uvWTotal += computeUvRamp( computeUvGridTime(t +(Real)i*timeOff , resetTime) );
	}
	if(uvWTotal<=VECTOR_EPSILON) { uvWeight =  uvWTotal = 1.; }
	else                           uvWeight /= uvWTotal;

	// check for reset
	if( currt < lastt )
		knResetUvGrid reset(uv, offset);

	// write new weight value to grid
	uv[0] = Vec3( uvWeight, 0.,0.);

	// print info about uv weights?
	debMsg("Uv grid "<<index<<"/"<<numUvs<< " t="<<currt<<" w="<<uvWeight<<", reset:"<<(int)(currt<lastt) , 2);
}

KERNEL() template<class T> void knSetBoundary (Grid<T>& grid, T value, int w) { 
	bool bnd = (i<=w || i>=grid.getSizeX()-1-w || j<=w || j>=grid.getSizeY()-1-w || (grid.is3D() && (k<=w || k>=grid.getSizeZ()-1-w)));
	if (bnd) 
		grid(i,j,k) = value;
}

template<class T> void Grid<T>::setBound(T value, int boundaryWidth) {
	knSetBoundary<T>( *this, value, boundaryWidth );
}


KERNEL() template<class T> void knSetBoundaryNeumann (Grid<T>& grid, int w) { 
	bool set = false;
	int  si=i, sj=j, sk=k;
	if( i<=w) {
		si = w+1; set=true;
	}
	if( i>=grid.getSizeX()-1-w){
		si = grid.getSizeX()-1-w-1; set=true;
	}
	if( j<=w){
		sj = w+1; set=true;
	}
	if( j>=grid.getSizeY()-1-w){
		sj = grid.getSizeY()-1-w-1; set=true;
	}
	if( grid.is3D() ){
		 if( k<=w ) {
			sk = w+1; set=true;
		 }
		 if( k>=grid.getSizeZ()-1-w ) {
			sk = grid.getSizeZ()-1-w-1; set=true;
		 }
	}
	if(set)
		grid(i,j,k) = grid(si, sj, sk);
}

template<class T> void Grid<T>::setBoundNeumann(int boundaryWidth) {
	knSetBoundaryNeumann<T>( *this, boundaryWidth );
}

//! kernel to set velocity components of mac grid to value for a boundary of w cells
KERNEL() void knSetBoundaryMAC (Grid<Vec3>& grid, Vec3 value, int w) { 
	if (i<=w   || i>=grid.getSizeX()  -w || j<=w-1 || j>=grid.getSizeY()-1-w || (grid.is3D() && (k<=w-1 || k>=grid.getSizeZ()-1-w)))
		grid(i,j,k).x = value.x;
	if (i<=w-1 || i>=grid.getSizeX()-1-w || j<=w   || j>=grid.getSizeY()  -w || (grid.is3D() && (k<=w-1 || k>=grid.getSizeZ()-1-w)))
		grid(i,j,k).y = value.y;
	if (i<=w-1 || i>=grid.getSizeX()-1-w || j<=w-1 || j>=grid.getSizeY()-1-w || (grid.is3D() && (k<=w   || k>=grid.getSizeZ()  -w)))
		grid(i,j,k).z = value.z;
} 

//! only set normal velocity components of mac grid to value for a boundary of w cells
KERNEL() void knSetBoundaryMACNorm (Grid<Vec3>& grid, Vec3 value, int w) { 
	if (i<=w   || i>=grid.getSizeX()  -w ) grid(i,j,k).x = value.x;
	if (j<=w   || j>=grid.getSizeY()  -w ) grid(i,j,k).y = value.y;
	if ( (grid.is3D() && (k<=w   || k>=grid.getSizeZ()  -w))) grid(i,j,k).z = value.z;
} 

//! set velocity components of mac grid to value for a boundary of w cells (optionally only normal values)
void MACGrid::setBoundMAC(Vec3 value, int boundaryWidth, bool normalOnly) { 
	if(!normalOnly) knSetBoundaryMAC    ( *this, value, boundaryWidth ); 
	else            knSetBoundaryMACNorm( *this, value, boundaryWidth ); 
}


//! helper kernels for getGridAvg
KERNEL(idx, reduce=+) returns(double result=0.0)
double knGridTotalSum(const Grid<Real>& a, FlagGrid* flags) {
	if(flags) {	if(flags->isFluid(idx)) result += a[idx]; } 
	else      {	result += a[idx]; } 
}

KERNEL(idx, reduce=+) returns(int numEmpty=0)
int knCountFluidCells(FlagGrid& flags) { if (flags.isFluid(idx) ) numEmpty++; }

//! averaged value for all cells (if flags are given, only for fluid cells)
PYTHON() Real getGridAvg(Grid<Real>& source, FlagGrid* flags=nullptr)
{
	double sum = knGridTotalSum(source, flags);

	double cells;
	if(flags) { cells = knCountFluidCells(*flags); }
	else      { cells = source.getSizeX()*source.getSizeY()*source.getSizeZ(); }

	if(cells>0.) sum *= 1./cells;
	else         sum = -1.;
	return sum;
}

//! transfer data between real and vec3 grids

KERNEL(idx) void knGetComponent(const Grid<Vec3>& source, Grid<Real>& target, int component) { 
	target[idx] = source[idx][component]; 
}
PYTHON() void getComponent(const Grid<Vec3>& source, Grid<Real>& target, int component) { knGetComponent(source, target, component); }

KERNEL(idx) void knSetComponent(const Grid<Real>& source, Grid<Vec3>& target, int component) { 
	target[idx][component] = source[idx]; 
}
PYTHON() void setComponent(const Grid<Real>& source, Grid<Vec3>& target, int component) { knSetComponent(source, target, component); }

//******************************************************************************
// Specialization classes

void FlagGrid::InitMinXWall(const int &boundaryWidth, Grid<Real>& phiWalls) {
	const int w = boundaryWidth;
	FOR_IJK(phiWalls) {
		phiWalls(i,j,k) = std::min(i - w - .5, (double)phiWalls(i,j,k));
	}
}

void FlagGrid::InitMaxXWall(const int &boundaryWidth, Grid<Real>& phiWalls) {
	const int w = boundaryWidth;
	FOR_IJK(phiWalls) {
		phiWalls(i,j,k) = std::min(mSize.x-i-1.5-w, (double)phiWalls(i,j,k));
	}
}

void FlagGrid::InitMinYWall(const int &boundaryWidth, Grid<Real>& phiWalls) {
	const int w = boundaryWidth;
	FOR_IJK(phiWalls) {
		phiWalls(i,j,k) = std::min(j - w - .5, (double)phiWalls(i,j,k));
	}
}

void FlagGrid::InitMaxYWall(const int &boundaryWidth, Grid<Real>& phiWalls) {
	const int w = boundaryWidth;
	FOR_IJK(phiWalls) {
		phiWalls(i,j,k) = std::min(mSize.y-j-1.5-w, (double)phiWalls(i,j,k));
	}
}

void FlagGrid::InitMinZWall(const int &boundaryWidth, Grid<Real>& phiWalls) {
	const int w = boundaryWidth;
	FOR_IJK(phiWalls) {
		phiWalls(i,j,k) = std::min(k - w - .5, (double)phiWalls(i,j,k));
	}
}

void FlagGrid::InitMaxZWall(const int &boundaryWidth, Grid<Real>& phiWalls) {
	const int w = boundaryWidth;
	FOR_IJK(phiWalls) {
		phiWalls(i,j,k) = std::min(mSize.z-k-1.5-w, (double)phiWalls(i,j,k));
	}
}

void FlagGrid::initDomain( const int &boundaryWidth
	                     , const string &wallIn
						 , const string &openIn
						 , const string &inflowIn
						 , const string &outflowIn 
						 , Grid<Real>* phiWalls ) {
	
	int  types[6] = {0};
	bool set  [6] = {false};
	// make sure we have at least 6 entries
	string wall    = wallIn;    wall.append("      ");
	string open    = openIn;    open.append("      ");
	string inflow  = inflowIn;  inflow.append("      ");
	string outflow = outflowIn; outflow.append("      ");

	if(phiWalls) phiWalls->setConst(1000000000);

	for (int i = 0; i<6; ++i) {
		//min x-direction
		if(!set[0]) {
			if(open[i]=='x')         {types[0] = TypeOpen;set[0] = true;}
			else if(inflow[i]=='x')  {types[0] = TypeInflow;set[0] = true;}
			else if(outflow[i]=='x') {types[0] = TypeOutflow;set[0] = true;}
			else if(wall[i]=='x') {
				types[0]    = TypeObstacle;
				if(phiWalls) InitMinXWall(boundaryWidth, *phiWalls);
				set[0] = true;
			}			
		}
		//max x-direction
		if(!set[1]) {
			if(open[i]=='X')         {types[1] = TypeOpen;set[1] = true;}
			else if(inflow[i]=='X')  {types[1] = TypeInflow;set[1] = true;}
			else if(outflow[i]=='X') {types[1] = TypeOutflow;set[1] = true;}
			else if(wall[i]=='X')  {
				types[1]    = TypeObstacle;
				if(phiWalls) InitMaxXWall(boundaryWidth, *phiWalls);
				set[1] = true;
			}			
		}
		//min y-direction
		if(!set[2]) {
			if(open[i]=='y')         {types[2] = TypeOpen;set[2] = true;}
			else if(inflow[i]=='y')  {types[2] = TypeInflow;set[2] = true;}
			else if(outflow[i]=='y') {types[2] = TypeOutflow;set[2] = true;}
			else if(wall[i]=='y') {
				types[2]    = TypeObstacle;
				if(phiWalls) InitMinYWall(boundaryWidth, *phiWalls);
				set[2] = true;
			}			
		}
		//max y-direction
		if(!set[3]) {
			if(open[i]=='Y')         {types[3] = TypeOpen;set[3] = true;}
			else if(inflow[i]=='Y')  {types[3] = TypeInflow;set[3] = true;}
			else if(outflow[i]=='Y') {types[3] = TypeOutflow;set[3] = true;}
			else if(wall[i]=='Y') {
				types[3]    = TypeObstacle;
				if(phiWalls) InitMaxYWall(boundaryWidth, *phiWalls);
				set[3] = true;
			}			
		}
		if(this->is3D()) {
		//min z-direction
			if(!set[4]) {
				if(open[i]=='z')         {types[4] = TypeOpen;set[4] = true;}
				else if(inflow[i]=='z')  {types[4] = TypeInflow;set[4] = true;}
				else if(outflow[i]=='z') {types[4] = TypeOutflow;set[4] = true;}
				else if(wall[i]=='z') {
					types[4]    = TypeObstacle;
					if(phiWalls) InitMinZWall(boundaryWidth, *phiWalls);
					set[4] = true;
				}				
			}
			//max z-direction
			if(!set[5]) {
				if(open[i]=='Z')         {types[5] = TypeOpen;set[5] = true;}
				else if(inflow[i]=='Z')  {types[5] = TypeInflow;set[5] = true;}
				else if(outflow[i]=='Z') {types[5] = TypeOutflow;set[5] = true;}
				else if(wall[i]=='Z') {
					types[5]    = TypeObstacle;
					if(phiWalls) InitMaxZWall(boundaryWidth, *phiWalls);
					set[5] = true;
				}				
			}
		}
	}

	setConst(TypeEmpty); 
	initBoundaries(boundaryWidth, types); 
}

void FlagGrid::initBoundaries(const int &boundaryWidth, const int *types) {
	const int w = boundaryWidth;
	FOR_IJK(*this) {
		bool bnd = (i <= w);
		if (bnd) mData[index(i,j,k)] = types[0];
		bnd = (i >= mSize.x-1-w);
		if (bnd) mData[index(i,j,k)] = types[1];
		bnd = (j <= w);
		if (bnd) mData[index(i,j,k)] = types[2];
		bnd = (j >= mSize.y-1-w);
		if (bnd) mData[index(i,j,k)] = types[3];
		if(is3D()) {
			bnd = (k <= w);
			if (bnd) mData[index(i,j,k)] = types[4];
			bnd = (k >= mSize.z-1-w);
			if (bnd) mData[index(i,j,k)] = types[5];
		}
	}
}

void FlagGrid::updateFromLevelset(LevelsetGrid& levelset) {
	FOR_IDX(*this) {
		if (!isObstacle(idx) && !isOutflow(idx)) {
			const Real phi = levelset[idx];
			if (phi <= levelset.invalidTimeValue()) continue;
			
			mData[idx] &= ~(TypeEmpty | TypeFluid); // clear empty/fluid flags
			mData[idx] |= (phi <= 0) ? TypeFluid : TypeEmpty; // set resepctive flag
		}
	}
}   

void FlagGrid::fillGrid(int type) {
	FOR_IDX(*this) {
		if ((mData[idx] & TypeObstacle)==0 && (mData[idx] & TypeInflow)==0&& (mData[idx] & TypeOutflow)==0&& (mData[idx] & TypeOpen)==0)
			mData[idx] = (mData[idx] & ~(TypeEmpty | TypeFluid)) | type;
	}
}

// flag grid helper

bool isIsolatedFluidCell(const IndexInt idx, const FlagGrid &flags)
{
	if(!flags.isFluid(idx)) return false;
	if(flags.isFluid(idx-flags.getStrideX())) return false;
	if(flags.isFluid(idx+flags.getStrideX())) return false;
	if(flags.isFluid(idx-flags.getStrideY())) return false;
	if(flags.isFluid(idx+flags.getStrideY())) return false;
	if(!flags.is3D()) return true;
	if(flags.isFluid(idx-flags.getStrideZ())) return false;
	if(flags.isFluid(idx+flags.getStrideZ())) return false;
	return true;
}

KERNEL(idx)
void knMarkIsolatedFluidCell(FlagGrid &flags, const int mark)
{
	if(isIsolatedFluidCell(idx, flags)) flags[idx] = mark;
}

PYTHON()
void markIsolatedFluidCell(FlagGrid &flags, const int mark)
{
	knMarkIsolatedFluidCell(flags, mark);
}

PYTHON()
void copyMACData(const MACGrid &source, MACGrid &target, const FlagGrid& flags, const int flag, const int bnd)
{
	assertMsg (source.getSize().x == target.getSize().x && source.getSize().y == target.getSize().y && source.getSize().z == target.getSize().z, "different grid resolutions " << source.getSize() << " vs " << target.getSize() );

	// Grid<Real> divGrid(target.getParent());
	// DivergenceOpMAC(divGrid, target);
	// Real fDivOrig = GridSumSqr(divGrid);

	FOR_IJK_BND(target, bnd)
	{
		if(flags.get(i,j,k) & flag)
		{
			target(i,j,k) = source(i,j,k);
		}
	}

	// DivergenceOpMAC(divGrid, target);
	// Real fDivTransfer = GridSumSqr(divGrid);
	// std::cout << "Divergence: " << fDivOrig << " -> " << fDivTransfer << std::endl;
}


// explicit instantiation
template class Grid<int>;
template class Grid<Real>;
template class Grid<Vec3>;

// IndexGrid specials

template<class T>
IndexGrid<T>::IndexGrid(FluidSolver* parent, int indexType, Vec3i blockSize, bool show, bool sparse)
	: Grid<T>(parent, show, sparse), mBlockSize(blockSize), mIndexType(indexType)
{
	mBlockSize.z = (parent->is2D()) ? 1 : mBlockSize.z;
	const Vec3i gridSize = parent->getGridSize();
	assertMsg((gridSize%mBlockSize)==Vec3i(0), "Block size is not compatible with grid size");

	mBlockStride.x = (gridSize.x / blockSize.x);
	mBlockStride.y = (gridSize.y / blockSize.y);
	mBlockStride.z = mBlockStride.x * mBlockStride.y;
	mBlockCellCnt = mBlockSize.x * mBlockSize.y * mBlockSize.z;

	// Populate grid cells with indices based on index type
	this->initialize();
}

KERNEL() void knSetIndicesInt(IndexGrid<int>& target)
{
	if (target.getIndexType() == IndexGrid<int>::IndexLinear) {
		target(i,j,k) = target.index(i, j, k);
	} else if (target.getIndexType() == IndexGrid<int>::IndexBlock) {
		target(i,j,k) = target.blockIndex(i, j, k);
	} else {
		debMsg("Invalid IndexGrid index type", 1);
	}
}
KERNEL() void knSetIndicesVec(IndexGrid<Vec3>& target, IndexGrid<int>* blockIdxGrid=nullptr)
{
	if (target.getIndexType() == IndexGrid<Vec3>::IndexLinear) {
		target(i,j,k) = Vec3(i,j,k);
	} else if (target.getIndexType() == IndexGrid<Vec3>::IndexBlock) {
		IndexInt bidx = (*blockIdxGrid)(i,j,k);
		target(bidx) = Vec3(i,j,k);
	} else {
		debMsg("Invalid IndexGrid index type", 1);
	}
}

template<> void IndexGrid<int>::initialize()
{
	knSetIndicesInt(*this);
}

template<> void IndexGrid<Vec3>::initialize()
{
	IndexGrid<int>* blockIdxGrid = nullptr;
	if (this->getIndexType() == IndexGrid<Vec3>::IndexBlock) {
	}
	blockIdxGrid = new IndexGrid<int>(this->getParent(), IndexGrid<int>::IndexBlock, this->getBlockSize());
	knSetIndicesVec(*this, blockIdxGrid);
	if (blockIdxGrid) {
		delete blockIdxGrid;
	}
}

// explicit instantiation
template class IndexGrid<int>;
template class IndexGrid<Vec3>;

} //namespace
