/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey 
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Levelset
 *
 ******************************************************************************/

#include "levelset.h"
#include "fastmarch.h"
#include "kernel.h"
#include "mcubes.h"
#include "mesh.h"
#include <stack>

using namespace std;
namespace Manta {

//************************************************************************
// Helper functions and kernels for marching

static const int FlagInited = FastMarch<FmHeapEntryOut, +1>::FlagInited;

// neighbor lookup vectors
static const Vec3i neighbors[6] = { Vec3i(-1,0,0), Vec3i(1,0,0), Vec3i(0,-1,0), Vec3i(0,1,0), Vec3i(0,0,-1), Vec3i(0,0,1) };
	
KERNEL(bnd=1) 
void InitFmIn (const FlagGrid& flags, Grid<int>& fmFlags, Grid<Real>& phi, bool ignoreWalls, int obstacleType) {
	const IndexInt idx = flags.index(i,j,k);
	const Real v = phi[idx];
	if (ignoreWalls) {
		if (v>=0. && ((flags[idx] & obstacleType) == 0) )
			fmFlags[idx] = FlagInited;
		else
			fmFlags[idx] = 0;
	} else {
		if (v>=0) fmFlags[idx] = FlagInited;
		else      fmFlags[idx] = 0;
	}
}

KERNEL(bnd=1) 
void InitFmOut (const FlagGrid& flags, Grid<int>& fmFlags, Grid<Real>& phi, bool ignoreWalls, int obstacleType) {
	const IndexInt idx = flags.index(i,j,k);
	const Real v = phi[idx];
	if (ignoreWalls) {
		fmFlags[idx] = (v<0) ? FlagInited : 0;
		if ((flags[idx] & obstacleType) != 0) {
			fmFlags[idx] = 0;
			phi[idx] = 0;
		}
	} else {
		fmFlags[idx] = (v<0) ? FlagInited : 0;
	}
}

KERNEL(bnd=1) 
void SetUninitialized (const Grid<int>& flags, Grid<int>& fmFlags, Grid<Real>& phi, const Real val, int ignoreWalls, int obstacleType) {
	if(ignoreWalls) {
		if ( (fmFlags(i,j,k) != FlagInited) && ((flags(i,j,k) & obstacleType) == 0) ) {
			phi(i,j,k) = val; }
	} else {
		if ( (fmFlags(i,j,k) != FlagInited) ) phi(i,j,k) = val;
	}
}

template<bool inward>
inline bool isAtInterface(const Grid<int>& fmFlags, Grid<Real>& phi, const Vec3i& p) {
	// check for interface
	int max = phi.is3D() ? 6 : 4;
	for (int nb=0; nb<max; nb++) {
		const Vec3i pn(p + neighbors[nb]);
		if (!fmFlags.isInBounds(pn)) continue;
		
		if (fmFlags(pn) != FlagInited) continue;
		if (( inward && phi(pn) >= 0.) || 
			(!inward && phi(pn) <  0.)) return true;
	}
	return false;
}


//************************************************************************
// Levelset class def

LevelsetGrid::LevelsetGrid(FluidSolver* parent, bool show) 
	: Grid<Real>(parent, show) 
{ 
	mType = (GridType)(TypeLevelset | TypeReal);    
}    

LevelsetGrid::LevelsetGrid(FluidSolver* parent, Real* data, bool show)
        : Grid<Real>(parent, data, show)
{
        mType = (GridType)(TypeLevelset | TypeReal);
}

Real LevelsetGrid::invalidTimeValue() {
	return FastMarch<FmHeapEntryOut, 1>::InvalidTime();
}

//! Kernel: perform levelset union
KERNEL(idx) void KnJoin(Grid<Real>& a, const Grid<Real>& b) {
	a[idx] = min(a[idx], b[idx]);
} 
void LevelsetGrid::join(const LevelsetGrid& o) { KnJoin(*this, o); }

//! subtract b, note does not preserve SDF!
KERNEL(idx) void KnSubtract(Grid<Real>& a, const Grid<Real>& b, const FlagGrid* flags, int subtractType) {
	if(flags && ((*flags)(idx) & subtractType) == 0) return;
	if(b[idx]<0.) a[idx] = b[idx] * -1.;
} 
void LevelsetGrid::subtract(const LevelsetGrid& o, const FlagGrid* flags, const int subtractType) { KnSubtract(*this, o, flags, subtractType); }

//! re-init levelset and extrapolate velocities (in & out)
//  note - uses flags to identify border (could also be done based on ls values)
static void doReinitMarch( Grid<Real>& phi,
		const FlagGrid& flags, Real maxTime, MACGrid* velTransport,
		bool ignoreWalls, bool correctOuterLayer, int obstacleType )
{
	const int dim = (phi.is3D() ? 3 : 2); 
	Grid<int> fmFlags( phi.getParent() );

	FastMarch<FmHeapEntryIn, -1> marchIn (flags, fmFlags, phi, maxTime, nullptr );
	
	// march inside
	InitFmIn (flags, fmFlags, phi, ignoreWalls, obstacleType);
	
	FOR_IJK_BND(flags, 1) {
		if (fmFlags(i,j,k) == FlagInited) continue;
		if (ignoreWalls && ((flags(i,j,k) & obstacleType) != 0)) continue;
		const Vec3i p(i,j,k);
				
		if(isAtInterface<true>(fmFlags, phi, p)) {
			// set value
			fmFlags(p) = FlagInited;
			
			// add neighbors that are not at the interface
			for (int nb=0; nb<2*dim; nb++) {
				const Vec3i pn(p + neighbors[nb]); // index always valid due to bnd=1                
				if (ignoreWalls && ((flags.get(pn) & obstacleType) != 0)) continue;
				
				// check neighbors of neighbor
				if (phi(pn) < 0. && !isAtInterface<true>(fmFlags, phi, pn)) {
					marchIn.addToList(pn, p); 
				}
			}            
		}
	}
	marchIn.performMarching();     
	// done with inwards marching
   
	// now march out...    
	
	// set un initialized regions
	SetUninitialized (flags, fmFlags, phi, -maxTime - 1., ignoreWalls, obstacleType); 

	InitFmOut (flags, fmFlags, phi, ignoreWalls, obstacleType);
	
	FastMarch<FmHeapEntryOut, +1> marchOut(flags, fmFlags, phi, maxTime, velTransport );

	// by default, correctOuterLayer is on
	if (correctOuterLayer) {
		// normal version, inwards march is done, now add all outside values (0..2] to list
		// note, this might move the interface a bit! but keeps a nice signed distance field...        
		FOR_IJK_BND(flags, 1) {
			if (ignoreWalls && ((flags(i,j,k) & obstacleType) != 0)) continue;
			const Vec3i p(i,j,k);
			
			// check nbs
			for (int nb=0; nb<2*dim; nb++) {
				const Vec3i pn(p + neighbors[nb]); // index always valid due to bnd=1                
				
				if (fmFlags(pn) != FlagInited) continue;
				if (ignoreWalls && ((flags.get(pn) & obstacleType)) != 0) continue;
				
				const Real nbPhi = phi(pn);
				
				// only add nodes near interface, not e.g. outer boundary vs. invalid region                
				if (nbPhi < 0 && nbPhi >= -2)
					marchOut.addToList(p, pn); 
			}
		}         
	} else {
		// alternative version, keep interface, do not distort outer cells
		// add all ouside values, but not those at the IF layer
		FOR_IJK_BND(flags, 1) {
			if (ignoreWalls && ((flags(i,j,k) & obstacleType) != 0)) continue;
			
			// only look at ouside values
			const Vec3i p(i,j,k);
			if (phi(p) < 0) continue;
			
			if (isAtInterface<false>(fmFlags, phi, p)) {
				// now add all non, interface neighbors
				fmFlags(p) = FlagInited;
				
				// add neighbors that are not at the interface
				for (int nb=0; nb<2*dim; nb++) {
					const Vec3i pn(p + neighbors[nb]); // index always valid due to bnd=1                
					if (ignoreWalls && ((flags.get(pn) & obstacleType) != 0)) continue;
				
					// check neighbors of neighbor
					if (phi(pn) > 0. && !isAtInterface<false>(fmFlags, phi, pn)) {
						marchOut.addToList(pn, p);
					}
				}            
			}
		}
	}    
	marchOut.performMarching();

	// set un initialized regions
	SetUninitialized (flags, fmFlags, phi, +maxTime + 1., ignoreWalls, obstacleType);    
}

//! call for levelset grids & external real grids

void LevelsetGrid::reinitMarching( const FlagGrid& flags, Real maxTime, MACGrid* velTransport,
		bool ignoreWalls, bool correctOuterLayer, int obstacleType )
{
	doReinitMarch( *this, flags, maxTime, velTransport, ignoreWalls, correctOuterLayer, obstacleType );
}


void LevelsetGrid::initFromFlags(const FlagGrid& flags, bool ignoreWalls) {
	FOR_IDX(*this) {
		if (flags.isFluid(idx) || (ignoreWalls && flags.isObstacle(idx)))
			mData[idx] = -0.5;
		else
			mData[idx] = 0.5;
	}
}

/* Helper variables that are used in flood-fill functions. */
static const int ID_UNKNOWN = 0;
static const int ID_VISITED = 1;

/* Fills all cells in the target grid that have not been marked during a flood-fill. */
KERNEL(bnd=boundaryWidth)
void KnFillApply(Grid<Real>& target, Grid<int>& visited, const Real value, const int boundaryWidth, const bool outside) {

	if (visited(i,j,k) == ID_VISITED) return;
	if (outside && target(i,j,k) < 0) return;
	if (!outside && target(i,j,k) >= 0) return;

	/* Actual flood-fill override. */
	target(i,j,k) = value;
}

/* Basic flood fill implementation used to fill inside / outside areas of levelset.
 * Calling this function will ensure that there are no fluid cells inside obstacles.
 * I.e. starting from walls, cells will be tagged in flood-fill fashion, stopping at 0 borders.
 * All remaining cells will be filled with the fill value. Outside mode inverts search behavior. */
void LevelsetGrid::floodFill(const Real value, const bool outside, const int boundaryWidth) {

	/* Sanity check: Filling mode and filling value need to "match". */
	if (outside) { assertMsg(value < 0, "Cannot fill outside with (positive) value " << value);
	} else       { assertMsg(value >= 0, "Cannot fill inside with (negative) value " << value); }

	Grid<Real> levelsetCopy(this->getParent());
	Grid<int> visited(this->getParent());
	std::stack<Vec3i> todoPos;

	const int maxNeighbors = this->is3D() ? 6 : 4;
	const Vec3i maxSize(this->getSize() - 1);

	Vec3i bnd(2 * boundaryWidth);
	if (!this->is3D()) bnd.z = 0;
	const int cellCntNoBnd = (this->getSizeX()-bnd.x) * (this->getSizeY()-bnd.y) * (this->getSizeZ()-bnd.z);

	/* Initialize temporary helper grids. */
	levelsetCopy.copyFrom(*this);
	visited.setConst(ID_UNKNOWN);

	FOR_IJK_BND(visited, boundaryWidth) {

		/* Skip inside / outside cells depending on search mode. */
		if (outside && levelsetCopy(i,j,k) < 0) continue;
		if (!outside && levelsetCopy(i,j,k) >= 0) continue;
		/* Skip cell if it already has been visited. */
		if (visited(i,j,k) == ID_VISITED) continue;

		Vec3i c(i,j,k);

		bool isWallCell = (c.x-boundaryWidth == 0 || c.x == maxSize.x-boundaryWidth);
		isWallCell |= (c.y-boundaryWidth == 0 || c.y == maxSize.y-boundaryWidth);
		if (this->is3D()) isWallCell |= (c.z-boundaryWidth == 0 || c.z == maxSize.z-boundaryWidth);

		/* Only start searching from borders. */
		if (!isWallCell) continue;

		/* Start flood-fill loop by initializing todo stack with current cell. */
		todoPos.push(c);
		visited(c) = ID_VISITED;

		while(!todoPos.empty()) {
			c = todoPos.top();
			todoPos.pop();

			/* Add all neighbor cells to search stack. */
			for (int nb=0; nb<maxNeighbors; nb++) {
				const Vec3i neigh(c + neighbors[nb]);

				if (!visited.isInBounds(neigh, boundaryWidth)) continue;
				/* Skip inside / outside area depening on what we search for. */
				if (outside && levelsetCopy(neigh) < 0) continue;
				if (!outside && levelsetCopy(neigh) >= 0) continue;
				/* Skip neighbor if it already has been visited. */
				if (visited(neigh) == ID_VISITED) continue;

				assertMsg(visited(neigh) == ID_UNKNOWN, "Cell must be of type 'unknown' at this point in the loop");
				todoPos.push(neigh);
				visited(neigh) = ID_VISITED;
			}
			assertMsg(todoPos.size() <= cellCntNoBnd, "Flood-fill todo stack cannot be greater than domain cell count - " << todoPos.size() << " vs " << cellCntNoBnd);
		}
	}
	KnFillApply(*this, visited, value, boundaryWidth, outside);
}

/* Deprecated: Use floodFill() function instead. */
void LevelsetGrid::fillHoles(int maxDepth, int boundaryWidth) {
	debMsg("Deprecated - do not use fillHoles() ... use floodFill() instead", 1);

	Real curVal, i1, i2, j1, j2, k1, k2;
	Vec3i c, cTmp;
	std::stack<Vec3i> undoPos;
	std::stack<Real> undoVal;
	std::stack<Vec3i> todoPos;

	FOR_IJK_BND(*this, boundaryWidth) {

		curVal = mData[index(i,j,k)];
		i1 = mData[index(i-1,j,k)];
		i2 = mData[index(i+1,j,k)];
		j1 = mData[index(i,j-1,k)];
		j2 = mData[index(i,j+1,k)];
		k1 = mData[index(i,j,k-1)];
		k2 = mData[index(i,j,k+1)];

		/* Skip cells inside and cells outside with no inside neighbours early */
		if (curVal < 0.) continue;
		if (curVal > 0. && i1 > 0. && i2 > 0. && j1 > 0. && j2 > 0. && k1 > 0. && k2 > 0.) continue;

		/* Cell at c is positive (outside) and has at least one negative (inside) neighbour cell */
		c = Vec3i(i,j,k);

		/* Current cell is outside and has inside neighbour(s) */
		undoPos.push(c);
		undoVal.push(curVal);
		todoPos.push(c);

		/* Enforce negative cell - if search depth gets exceeded this will be reverted to the original value */
		mData[index(c.x, c.y, c.z)] = -0.5;

		while(!todoPos.empty()) {
			todoPos.pop();

			/* Add neighbouring positive (inside) cells to stacks and set negavtive cell value */
			if (c.x > 0 && mData[index(c.x-1, c.y, c.z)] > 0.) {
				cTmp = Vec3i(c.x-1, c.y, c.z); undoPos.push(cTmp); undoVal.push(mData[index(cTmp)]); todoPos.push(cTmp); mData[index(cTmp)] = -0.5;
			}
			if (c.y > 0 && mData[index(c.x, c.y-1, c.z)] > 0.) {
				cTmp = Vec3i(c.x, c.y-1, c.z); undoPos.push(cTmp); undoVal.push(mData[index(cTmp)]); todoPos.push(cTmp); mData[index(cTmp)] = -0.5;
			}
			if (c.z > 0 && mData[index(c.x, c.y, c.z-1)] > 0.) {
				cTmp = Vec3i(c.x, c.y, c.z-1); undoPos.push(cTmp); undoVal.push(mData[index(cTmp)]); todoPos.push(cTmp); mData[index(cTmp)] = -0.5;
			}
			if (c.x < (*this).getSizeX()-1 && mData[index(c.x+1, c.y, c.z)] > 0.) {
				cTmp = Vec3i(c.x+1, c.y, c.z); undoPos.push(cTmp); undoVal.push(mData[index(cTmp)]); todoPos.push(cTmp); mData[index(cTmp)] = -0.5;
			}
			if (c.y < (*this).getSizeY()-1 && mData[index(c.x, c.y+1, c.z)] > 0.) {
				cTmp = Vec3i(c.x, c.y+1, c.z); undoPos.push(cTmp); undoVal.push(mData[index(cTmp)]); todoPos.push(cTmp); mData[index(cTmp)] = -0.5;
			}
			if (c.z < (*this).getSizeZ()-1 && mData[index(c.x, c.y, c.z+1)] > 0.) {
				cTmp = Vec3i(c.x, c.y, c.z+1); undoPos.push(cTmp); undoVal.push(mData[index(cTmp)]); todoPos.push(cTmp); mData[index(cTmp)] = -0.5;
			}

			/* Restore original value in cells if undo needed ie once cell undo count exceeds given limit */
			if (undoPos.size() > maxDepth) {
				/* Clear todo stack */
				while (!todoPos.empty()) {
					todoPos.pop();
				}
				/* Clear undo stack and revert value */
				while (!undoPos.empty()) {
					c = undoPos.top();
					curVal = undoVal.top();
					undoPos.pop();
					undoVal.pop();
					mData[index(c.x, c.y, c.z)] = curVal;
				}
				break;
			}

			/* Ensure that undo stack is cleared at the end if no more items in todo stack left */
			if (todoPos.empty()) {
				while (!undoPos.empty()) {
					undoPos.pop();
				}
				while (!undoVal.empty()) {
					undoVal.pop();
				}
			}
			/* Pop value for next while iteration */
			else {
				c = todoPos.top();
			}
		}
	}
}

//! run marching cubes to create a mesh for the 0-levelset
void LevelsetGrid::createMesh(Mesh& mesh) {
	assertMsg(is3D(), "Only 3D grids supported so far");
	
	mesh.clear();
		
	const Real invalidTime = invalidTimeValue();
	const Real isoValue = 1e-4;
	
	// create some temp grids
	Grid<int> edgeVX(mParent);
	Grid<int> edgeVY(mParent);
	Grid<int> edgeVZ(mParent);
	
	for(int k=0; k<mSize.z-1; k++)
	for(int j=0; j<mSize.y-1; j++)
	for(int i=0; i<mSize.x-1; i++) {
		 Real value[8] = { get(i,j,k),   get(i+1,j,k),   get(i+1,j+1,k),   get(i,j+1,k),
						   get(i,j,k+1), get(i+1,j,k+1), get(i+1,j+1,k+1), get(i,j+1,k+1) };
		
		// build lookup index, check for invalid times
		bool skip = false;
		int cubeIdx = 0;
		for (int l=0;l<8;l++) {
			value[l] *= -1;
			if (-value[l] <= invalidTime)
				skip = true;
			if (value[l] < isoValue) 
				cubeIdx |= 1<<l;
		}
		if (skip || (mcEdgeTable[cubeIdx] == 0)) continue;
		
		// where to look up if this point already exists
		int triIndices[12];
		int *eVert[12] = { &edgeVX(i,j,k),   &edgeVY(i+1,j,k),   &edgeVX(i,j+1,k),   &edgeVY(i,j,k), 
						   &edgeVX(i,j,k+1), &edgeVY(i+1,j,k+1), &edgeVX(i,j+1,k+1), &edgeVY(i,j,k+1), 
						   &edgeVZ(i,j,k),   &edgeVZ(i+1,j,k),   &edgeVZ(i+1,j+1,k), &edgeVZ(i,j+1,k) };
		
		const Vec3 pos[9] = { Vec3(i,j,k),   Vec3(i+1,j,k),   Vec3(i+1,j+1,k),   Vec3(i,j+1,k),
						Vec3(i,j,k+1), Vec3(i+1,j,k+1), Vec3(i+1,j+1,k+1), Vec3(i,j+1,k+1) };
		
		for (int e=0; e<12; e++) {
			if (mcEdgeTable[cubeIdx] & (1<<e)) {
				// vertex already calculated ?
				if (*eVert[e] == 0) {
					// interpolate edge
					const int e1 = mcEdges[e*2  ];
					const int e2 = mcEdges[e*2+1];
					const Vec3 p1 = pos[ e1  ];    // scalar field pos 1
					const Vec3 p2 = pos[ e2  ];    // scalar field pos 2
					const float valp1  = value[ e1  ];  // scalar field val 1
					const float valp2  = value[ e2  ];  // scalar field val 2
					const float mu = (isoValue - valp1) / (valp2 - valp1);

					// init isolevel vertex
					Node vertex;
					vertex.pos = p1 + (p2-p1)*mu + Vec3(Real(0.5));
					vertex.normal = getNormalized( 
										getGradient( *this, i+cubieOffsetX[e1], j+cubieOffsetY[e1], k+cubieOffsetZ[e1]) * (1.0-mu) +
										getGradient( *this, i+cubieOffsetX[e2], j+cubieOffsetY[e2], k+cubieOffsetZ[e2]) * (    mu)) ;
					
					triIndices[e] = mesh.addNode(vertex) + 1;
					
					// store vertex 
					*eVert[e] = triIndices[e];
				} else {
					// retrieve  from vert array
					triIndices[e] = *eVert[e];
				}
			}
		}
		
		// Create the triangles... 
		for(int e=0; mcTriTable[cubeIdx][e]!=-1; e+=3) {
			mesh.addTri( Triangle( triIndices[ mcTriTable[cubeIdx][e+0]] - 1,
										triIndices[ mcTriTable[cubeIdx][e+1]] - 1,
										triIndices[ mcTriTable[cubeIdx][e+2]] - 1));
		}
	}
	
	//mesh.rebuildCorners();
	//mesh.rebuildLookup();

	// Update mdata fields
	mesh.updateDataFields();
}


} //namespace
