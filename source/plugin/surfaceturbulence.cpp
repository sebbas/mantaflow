/******************************************************************************
 *
 * MantaFlow fluid solver framework 
 * Copyright 2016 Olivier Mercier, oli.mercier@gmail.com
 *
 * This program is free software, distributed under the terms of the
 * GNU General Public License (GPL) 
 * http://www.gnu.org/licenses
 *
 * Surface Turbulence for Particle-Based Liquid Simulations
 * Mercier et al., SIGGRAPH Asia 2015
 * 
 * Possible speedups :
 * - only initialize surface points around coarse particles near the surface. Use the flags in the fluid grid and only use cells with non-fluid neighbors.
 *
 ******************************************************************************/

#include <iomanip>
#include <chrono>
#include "particle.h"

using namespace std;
namespace Manta {

// own namespace for globals
namespace SurfaceTurbulence {


//
// **** surface turbulence parameters ****
//
struct SurfaceTurbulenceParameters {
    int res;
    float outerRadius;
    int surfaceDensity;
    int nbSurfaceMaintenanceIterations;
    float dt;
    float waveSpeed;
    float waveDamping;
    float waveSeedFrequency;
    float waveMaxAmplitude;
    float waveMaxFrequency;
    float waveMaxSeedingAmplitude; // as ratio of max amp;
    float waveSeedingCurvatureThresholdRegionCenter;
    float waveSeedingCurvatureThresholdRegionRadius;
    float waveSeedStepSizeRatioOfMax;
    float innerRadius;
    float meanFineDistance;
    float constraintA;
    float normalRadius;
    float tangentRadius;
    float bndXm, bndXp, bndYm, bndYp, bndZm, bndZp;
};
SurfaceTurbulenceParameters params;


//
// **** acceleration grid for particle neighbor queries ****
//
struct ParticleAccelGrid{
    int res;
    vector<int>*** indices;

    void init(int inRes) {
        res = inRes;
        indices = new vector<int>**[res];
        for(int i=0; i<res; i++){
            indices[i] = new vector<int>*[res];
            for(int j=0; j<res; j++) {
                indices[i][j] = new vector<int>[res];
            }
        }
    }

    void fillWith(BasicParticleSystem& particles) {
        // clear
        for(int i=0; i<res; i++) {
        for(int j=0; j<res; j++) {
        for(int k=0; k<res; k++) {
            indices[i][j][k].clear();
        }}}

        // fill
        for(int id=0;id<particles.size();id++) {
            Vec3 pos = particles.getPos(id);
            int i = clamp<int>(floor(pos.x/params.res*res), 0, res-1);
            int j = clamp<int>(floor(pos.y/params.res*res), 0, res-1);
            int k = clamp<int>(floor(pos.z/params.res*res), 0, res-1);
            indices[i][j][k].push_back(id);
        }
    }

    void fillWith(ParticleDataImpl<Vec3>& particles){
        // clear
        for(int i=0; i<res; i++) {
        for(int j=0; j<res; j++) {
        for(int k=0; k<res; k++) {
            indices[i][j][k].clear();
        }}}

        // fill
        for(int id=0;id<particles.size();id++) {
            Vec3 pos = particles[id];
            int i = clamp<int>(floor(pos.x/params.res*res), 0, res-1);
            int j = clamp<int>(floor(pos.y/params.res*res), 0, res-1);
            int k = clamp<int>(floor(pos.z/params.res*res), 0, res-1);
            indices[i][j][k].push_back(id);
        }
    }
};

#define LOOP_NEIGHBORS_BEGIN(points, center, radius) \
    int minI = clamp<int>(floor((center.x-radius)/params.res*points.accel->res), 0, points.accel->res-1); \
    int maxI = clamp<int>(floor((center.x+radius)/params.res*points.accel->res), 0, points.accel->res-1); \
    int minJ = clamp<int>(floor((center.y-radius)/params.res*points.accel->res), 0, points.accel->res-1); \
    int maxJ = clamp<int>(floor((center.y+radius)/params.res*points.accel->res), 0, points.accel->res-1); \
    int minK = clamp<int>(floor((center.z-radius)/params.res*points.accel->res), 0, points.accel->res-1); \
    int maxK = clamp<int>(floor((center.z+radius)/params.res*points.accel->res), 0, points.accel->res-1); \
    for(int i=minI; i<=maxI; i++) { \
    for(int j=minJ; j<=maxJ; j++) { \
    for(int k=minK; k<=maxK; k++) { \
        for(int idLOOPNEIGHBORS=0;idLOOPNEIGHBORS<(int)points.accel->indices[i][j][k].size();idLOOPNEIGHBORS++) { \
            int idn = points.accel->indices[i][j][k][idLOOPNEIGHBORS]; \
            if(points.isActive(idn)) {
#define LOOP_NEIGHBORS_END \
            } \
        } \
    }}}

#define LOOP_GHOSTS_POS_BEGIN(pos, radius) \
    int flagLOOPGHOSTS = -1; \
    Vec3 gPos; \
    while(flagLOOPGHOSTS < 6) { \
        if     (flagLOOPGHOSTS < 0 && pos.x - params.bndXm <= radius) { flagLOOPGHOSTS = 0; gPos = Vec3(2.f*params.bndXm - pos.x, pos.y, pos.z); } \
        else if(flagLOOPGHOSTS < 1 && params.bndXp - pos.x <= radius) { flagLOOPGHOSTS = 1; gPos = Vec3(2.f*params.bndXp - pos.x, pos.y, pos.z); } \
        else if(flagLOOPGHOSTS < 2 && pos.y - params.bndYm <= radius) { flagLOOPGHOSTS = 2; gPos = Vec3(pos.x, 2.f*params.bndYm - pos.y, pos.z); } \
        else if(flagLOOPGHOSTS < 3 && params.bndYp - pos.y <= radius) { flagLOOPGHOSTS = 3; gPos = Vec3(pos.x, 2.f*params.bndYp - pos.y, pos.z); } \
        else if(flagLOOPGHOSTS < 4 && pos.z - params.bndZm <= radius) { flagLOOPGHOSTS = 4; gPos = Vec3(pos.x, pos.y, 2.f*params.bndZm - pos.z); } \
        else if(flagLOOPGHOSTS < 5 && params.bndZp - pos.Z <= radius) { flagLOOPGHOSTS = 5; gPos = Vec3(pos.x, pos.y, 2.f*params.bndZp - pos.z); } \
        else                                                          { flagLOOPGHOSTS = 6; gPos = Vec3(pos.x, pos.y, pos.z);                    }
#define LOOP_GHOSTS_POS_NORMAL_BEGIN(pos, normal, radius) \
    int flagLOOPGHOSTS = -1; \
    Vec3 gPos, gNormal; \
    while(flagLOOPGHOSTS < 6) { \
        if     (flagLOOPGHOSTS < 0 && pos.x - params.bndXm <= radius) { flagLOOPGHOSTS = 0; gPos = Vec3(2.f*params.bndXm - pos.x, pos.y, pos.z); gNormal = Vec3(-normal.x,  normal.y,  normal.z); } \
        else if(flagLOOPGHOSTS < 1 && params.bndXp - pos.x <= radius) { flagLOOPGHOSTS = 1; gPos = Vec3(2.f*params.bndXp - pos.x, pos.y, pos.z); gNormal = Vec3(-normal.x,  normal.y,  normal.z); } \
        else if(flagLOOPGHOSTS < 2 && pos.y - params.bndYm <= radius) { flagLOOPGHOSTS = 2; gPos = Vec3(pos.x, 2.f*params.bndYm - pos.y, pos.z); gNormal = Vec3( normal.x, -normal.y,  normal.z); } \
        else if(flagLOOPGHOSTS < 3 && params.bndYp - pos.y <= radius) { flagLOOPGHOSTS = 3; gPos = Vec3(pos.x, 2.f*params.bndYp - pos.y, pos.z); gNormal = Vec3( normal.x, -normal.y,  normal.z); } \
        else if(flagLOOPGHOSTS < 4 && pos.z - params.bndZm <= radius) { flagLOOPGHOSTS = 4; gPos = Vec3(pos.x, pos.y, 2.f*params.bndZm - pos.z); gNormal = Vec3( normal.x,  normal.y, -normal.z); } \
        else if(flagLOOPGHOSTS < 5 && params.bndZp - pos.Z <= radius) { flagLOOPGHOSTS = 5; gPos = Vec3(pos.x, pos.y, 2.f*params.bndZp - pos.z); gNormal = Vec3( normal.x,  normal.y, -normal.z); } \
        else                                                          { flagLOOPGHOSTS = 6; gPos = pos;                                          gNormal = normal;                                }
#define LOOP_GHOSTS_END \
    }


//
// **** Wrappers around point sets to attach it an acceleration grid ****
//
struct PointSetWrapper {
    ParticleAccelGrid* accel;

    PointSetWrapper(ParticleAccelGrid* inAccel) {accel = inAccel;}
    virtual void updateAccel() = 0;
};

struct BasicParticleSystemWrapper : PointSetWrapper {
    BasicParticleSystem* points;

    BasicParticleSystemWrapper(ParticleAccelGrid* inAccel) : PointSetWrapper(inAccel) {}

    Vec3 getPos(int id) {return points->getPos(id);}
    void setPos(int id, Vec3 pos) {points->setPos(id, pos);}
    void updateAccel() {accel->fillWith(*points);}
    void clear() {points->clear();}
    int size() {return points->size();}
    bool isActive(int id) {return points->isActive(id);}
    void addParticle(Vec3 pos) {points->addParticle(pos);}
    int getStatus(int id) {return points->getStatus(id);}
    void addBuffered(Vec3 pos) {points->addBuffered(pos);}
    void doCompress() {points->doCompress();}
    void insertBufferedParticles() {points->insertBufferedParticles();}
    void kill(int id) {points->kill(id);}

    bool hasNeighbor(Vec3 pos, float radius) {
        bool answer = false;
        int minI = clamp<int>(floor((pos.x-radius)/params.res*accel->res), 0, accel->res-1);
        int maxI = clamp<int>(floor((pos.x+radius)/params.res*accel->res), 0, accel->res-1);
        int minJ = clamp<int>(floor((pos.y-radius)/params.res*accel->res), 0, accel->res-1);
        int maxJ = clamp<int>(floor((pos.y+radius)/params.res*accel->res), 0, accel->res-1);
        int minK = clamp<int>(floor((pos.z-radius)/params.res*accel->res), 0, accel->res-1);
        int maxK = clamp<int>(floor((pos.z+radius)/params.res*accel->res), 0, accel->res-1);
        for(int i=minI; i<=maxI; i++) {
        for(int j=minJ; j<=maxJ; j++) {
        for(int k=minK; k<=maxK; k++) {
            for(int id=0;id<(int)accel->indices[i][j][k].size();id++) {
                if(points->isActive(accel->indices[i][j][k][id]) &&
                   norm(points->getPos(accel->indices[i][j][k][id]) - pos) <= radius
                ) {answer = true; break;}
            }
        if(answer)break;}
        if(answer)break;}
        if(answer)break;}
        return answer;
    }

    bool hasNeighborOtherThanItself(int idx, float radius) {
        bool answer = false;
        Vec3 pos = points->getPos(idx);
        int minI = clamp<int>(floor((pos.x-radius)/params.res*accel->res), 0, accel->res-1);
        int maxI = clamp<int>(floor((pos.x+radius)/params.res*accel->res), 0, accel->res-1);
        int minJ = clamp<int>(floor((pos.y-radius)/params.res*accel->res), 0, accel->res-1);
        int maxJ = clamp<int>(floor((pos.y+radius)/params.res*accel->res), 0, accel->res-1);
        int minK = clamp<int>(floor((pos.z-radius)/params.res*accel->res), 0, accel->res-1);
        int maxK = clamp<int>(floor((pos.z+radius)/params.res*accel->res), 0, accel->res-1);
        for(int i=minI; i<=maxI; i++) {
        for(int j=minJ; j<=maxJ; j++) {
        for(int k=minK; k<=maxK; k++) {
            for(int id=0;id<(int)accel->indices[i][j][k].size();id++) {
                if(accel->indices[i][j][k][id] != idx &&
                   points->isActive(accel->indices[i][j][k][id]) &&
                   norm(points->getPos(accel->indices[i][j][k][id]) - pos) <= radius
                ) {answer = true; break;}
            }
        if(answer)break;}
        if(answer)break;}
        if(answer)break;}
        return answer;
    }
    
    void removeInvalidIndices(vector<int>& indices) {
        vector<int> copy;
        copy.resize(indices.size());
        for(int i=0; i<(int)indices.size(); i++) {
            copy[i] = indices[i];
        }
        indices.clear();
        for(int i=0; i<(int)copy.size(); i++) {
            if(points->isActive(copy[i])) {
                indices.push_back(copy[i]);
            }
        }
    }
};

struct ParticleDataImplVec3Wrapper : PointSetWrapper {
    ParticleDataImpl<Vec3>* points;

    ParticleDataImplVec3Wrapper(ParticleAccelGrid* inAccel) : PointSetWrapper(inAccel) {}

    Vec3 getVec3(int id) {return (*points)[id];}
    void setVec3(int id, Vec3 vec) {(*points)[id] = vec;}
    void updateAccel() {accel->fillWith(*points);}
    bool isActive(int i) {return true;}

};


//
// **** globals ****
//
ParticleAccelGrid accelCoarse, accelSurface;
BasicParticleSystemWrapper coarseParticles(&accelCoarse), surfacePoints(&accelSurface);
ParticleDataImplVec3Wrapper coarseParticlesPrevPos(&accelCoarse); // WARNING: reusing the coarse accel grid to save space, don't query coarseParticlesPrevPos and coarseParticles at the same time.
vector<Vec3> tempSurfaceVec3; // to store misc info on surface points
vector<float> tempSurfaceFloat; // to store misc info on surface points
int frameCount = 0;




//
//**** weighting kernels *****
//
float triangularWeight(float distance, float radius) {
    return 1.0f - distance/radius;
}
float exponentialWeight(float distance, float radius, float falloff) {
    if(distance > radius) return 0;
    float tmp = distance/radius;
    return expf(-falloff*tmp*tmp);
}

float weightKernelAdvection(float distance) {
    if(distance > 2.f*params.outerRadius) {
        return 0;
    } else {
        return triangularWeight(distance, 2.f*params.outerRadius);
    }
}

float weightKernelCoarseDensity(float distance) {
    return exponentialWeight(distance, params.outerRadius, 2.0f);
}

float weightSurfaceNormal(float distance) {
    if(distance > params.normalRadius) {
        return 0;
    } else {
        return triangularWeight(distance, params.normalRadius);
    }
}

float weightSurfaceTangent(float distance) {
    if(distance > params.tangentRadius) {
        return 0;
    } else {
        return triangularWeight(distance, params.tangentRadius);
    }
}


//
// **** utility ****
//

bool isInDomain(Vec3 pos)
{
    return params.bndXm <= pos.x && pos.x <= params.bndXp &&
           params.bndYm <= pos.y && pos.y <= params.bndYp &&
           params.bndZm <= pos.z && pos.z <= params.bndZp ;
}

float smoothstep( float edgeLeft, float edgeRight, float val ){
    float x = clamp((val - edgeLeft)/(edgeRight - edgeLeft), 0.f, 1.f);
    return x*x*(3 - 2*x);
}


//
// **** surface initialization ****
//

void initFines(
    BasicParticleSystemWrapper& coarseParticles,
    BasicParticleSystemWrapper& surfacePoints,
    FlagGrid& flags
){
    unsigned int discretization = (unsigned int) M_PI*(params.outerRadius+params.innerRadius)/params.meanFineDistance;
    float dtheta = 2*params.meanFineDistance/(params.outerRadius+params.innerRadius);
    float outerRadius2 = params.outerRadius*params.outerRadius;

    surfacePoints.clear();
    for (int idx=0; idx<(int)coarseParticles.size(); idx++) {
        
        if(idx % 500 == 0) {cout << "Initializing surface points : " << setprecision(4) << 100.f*idx/coarseParticles.size() << "%" << endl;}
        
        if (coarseParticles.isActive(idx)) {
            
            // check flags if we are near surface
            bool nearSurface = false;
            Vec3 pos = coarseParticles.getPos(idx);
            for(int i=-1; i<=1; i++) {
            for(int j=-1; j<=1; j++) {
            for(int k=-1; k<=1; k++) {
                if( !flags.isFluid( ((int)pos.x)+i, ((int)pos.y)+j, ((int)pos.z)+k ) ) {
                    nearSurface = true;
                    break;
                }
            }}}

            if(nearSurface) {
                for(unsigned int i = 0; i <= discretization/2; ++i){
                    float discretization2 = float(floor(2*M_PI*sin(i*dtheta)/dtheta)+1);
                    for(float phi = 0; phi < 2*M_PI; phi += float(2*M_PI/discretization2)){
                        float theta = i*dtheta;
                        Vec3 normal(sin(theta)*cos(phi), cos(theta), sin(theta)*sin(phi));
                        Vec3 position = coarseParticles.getPos(idx) + params.outerRadius*normal;
    
                        bool valid = true;
                        LOOP_NEIGHBORS_BEGIN(coarseParticles, position, 2.f*params.outerRadius)
                            if(idx!=idn && normSquare(position-coarseParticles.getPos(idn)) < outerRadius2) {
                                valid = false;
                                break;
                            }
                        LOOP_NEIGHBORS_END
                        if(valid) {
                            surfacePoints.addParticle(position);
                        }
                    }
                }
            }

        }
    }
}

//
// **** surface advection ****
//

KERNEL(pts)
void advectSurfacePoints(
        BasicParticleSystemWrapper& surfacePoints,
        BasicParticleSystemWrapper& coarseParticles,
        ParticleDataImplVec3Wrapper& coarseParticlesPrevPos
)
{
    if(surfacePoints.isActive(idx)) {
        Vec3 avgDisplacement(0,0,0);
        float totalWeight = 0;
        Vec3 p = surfacePoints.getPos(idx);
        LOOP_NEIGHBORS_BEGIN(coarseParticlesPrevPos, surfacePoints.getPos(idx), 2.0f*params.outerRadius)
            if((coarseParticles.getStatus(idn) & ParticleBase::PNEW)==0 &&
               (coarseParticles.getStatus(idn) & ParticleBase::PDELETE)==0)
            {
                Vec3 disp = coarseParticles.getPos(idn) - coarseParticlesPrevPos.getVec3(idn);
                float distance = norm(coarseParticlesPrevPos.getVec3(idn) - p);
                float w = weightKernelAdvection(distance);
                avgDisplacement += w * disp;
                totalWeight += w;
            }
        LOOP_NEIGHBORS_END
        if(totalWeight != 0) avgDisplacement /= totalWeight;
        surfacePoints.setPos(idx, p + avgDisplacement);
    }
}


//
// **** value and gradient of level-set band constraint ****
//
float computeConstraintLevel(
        BasicParticleSystemWrapper& coarseParticles,
        Vec3 pos
){
    float lvl = 0.0f;
    LOOP_NEIGHBORS_BEGIN(coarseParticles, pos, 1.5f*params.outerRadius)
        lvl += expf(-params.constraintA*normSquare(coarseParticles.getPos(idn) - pos));
    LOOP_NEIGHBORS_END
    if(lvl > 1.0f) lvl = 1.0f;
    lvl = (sqrtf(-logf(lvl)/params.constraintA)-params.innerRadius)/(params.outerRadius-params.innerRadius);
    return lvl;
}

Vec3 computeConstraintGradient(
        BasicParticleSystemWrapper& coarseParticles,
        Vec3 pos
){
    Vec3 gradient(0,0,0);
    LOOP_NEIGHBORS_BEGIN(coarseParticles, pos, 1.5f*params.outerRadius)
        gradient += 2.f*params.constraintA*(float)(expf(-params.constraintA*normSquare(coarseParticles.getPos(idn) - pos)))*(pos - coarseParticles.getPos(idn));
    LOOP_NEIGHBORS_END
    return getNormalized(gradient);
}


//
// **** compute surface normals ****
//

KERNEL(pts)
void computeSurfaceNormals(
        BasicParticleSystemWrapper& surfacePoints,
        BasicParticleSystemWrapper& coarseParticles,
        ParticleDataImpl<Vec3>& surfaceNormals
){
        Vec3 pos = surfacePoints.getPos(idx);

        // approx normal with gradient
        Vec3 gradient = computeConstraintGradient(coarseParticles, pos);

        // get tangent frame
        Vec3 n = getNormalized(gradient);
        Vec3 vx(1,0,0);
        Vec3 vy(0,1,0);
        float dotX = dot(n, vx);
        float dotY = dot(n, vy);
        Vec3 t1 = getNormalized(fabs(dotX)<fabs(dotY) ? cross(n, vx) : cross(n, vy));
        Vec3 t2 = getNormalized( cross(n,t1) ); // initial frame

        // linear fit of neighboring surface points in approximated tangent frame
        float sw = 0, swx = 0, swy = 0, swxy = 0, swx2 = 0, swy2 = 0, swxz = 0, swyz = 0, swz = 0;
        LOOP_NEIGHBORS_BEGIN(surfacePoints, pos, params.normalRadius)
            LOOP_GHOSTS_POS_BEGIN(surfacePoints.getPos(idn), params.normalRadius)
                 float x = dot(gPos - pos, t1);
                 float y = dot(gPos - pos, t2);
                 float z = dot(gPos - pos, n);
                 float w = weightSurfaceNormal(norm(pos - gPos));
                 swx2 += w*x*x;
                 swy2 += w*y*y;
                 swxy += w*x*y;
                 swxz += w*x*z;
                 swyz += w*y*z;
                 swx  += w*x;
                 swy  += w*y;
                 swz  += w*z;
                 sw   += w;                             
            LOOP_GHOSTS_END
        LOOP_NEIGHBORS_END
        float det = -sw*swxy*swxy + 2.f*swx*swxy*swy - swx2*swy*swy - swx*swx*swy2 + sw*swx2*swy2;
        if(det == 0) {surfaceNormals[idx]=Vec3(0,0,0);}
        else {
            Vec3 abc = 1.f/det*Vec3(
                    swxz*(-swy*swy+sw*swy2)  + swyz*(-sw*swxy+swx*swy)  + swz*(swxy*swy-swx*swy2),
                    swxz*(-sw*swxy+swx*swy)  + swyz*(-swx*swx+sw*swx2)  + swz*(swx*swxy-swx2*swy),
                    swxz*(swxy*swy-swx*swy2) + swyz*(swx*swxy-swx2*swy) + swz*(-swxy*swxy + swx2*swy2)
                    );
            Vec3 normal = -getNormalized(t1*abc.x + t2*abc.y - n);
            if(dot(gradient, normal) < 0) {normal = -normal;}
            surfaceNormals[idx] = normal;
        }
}


//
// **** smooth surface normals ****
//

KERNEL(pts)
void computeAveragedNormals(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals
){
    Vec3 pos = surfacePoints.getPos(idx);
    Vec3 newNormal = Vec3(0,0,0);
    LOOP_NEIGHBORS_BEGIN(surfacePoints, pos, params.normalRadius)
        float w = weightSurfaceNormal(norm(pos - surfacePoints.getPos(idn)));
        newNormal += w * surfaceNormals[idn];
    LOOP_NEIGHBORS_END
    tempSurfaceVec3[idx] = getNormalized(newNormal);
}

KERNEL(pts)
void assignNormals(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals
) {
    surfaceNormals[idx] = tempSurfaceVec3[idx];
}

void smoothSurfaceNormals(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals
){
    tempSurfaceVec3.resize(surfacePoints.size());
    
    computeAveragedNormals(surfacePoints, surfaceNormals);
    assignNormals(surfacePoints, surfaceNormals);
}



//
// **** addition/deletion of particles. Not parallel to prevent write/delete conflicts ****
//

void addDeleteSurfacePoints(
        BasicParticleSystemWrapper& surfacePoints
){
    int fixedSize = surfacePoints.size();
    for (int idx=0; idx<fixedSize; idx++) {
        // compute proxy tangent displacement
        Vec3 pos = surfacePoints.getPos(idx);

        Vec3 gradient = computeConstraintGradient(coarseParticles, pos);

        float wt = 0;
        Vec3 tangentDisplacement(0,0,0);
        LOOP_NEIGHBORS_BEGIN(surfacePoints, pos, params.tangentRadius)
            if(idn != idx) {
                Vec3 dir = pos - surfacePoints.getPos(idn);
                float length = norm(dir);
                dir = getNormalized(dir);
    
                // Decompose direction into normal and tangent directions.
                Vec3 dn = dot(dir, gradient)*gradient;
                Vec3 dt = dir - dn;
    
                float w = weightSurfaceTangent(length);
                wt += w;
                tangentDisplacement += w * dt;
            }
        LOOP_NEIGHBORS_END
        if(norm(tangentDisplacement)!=0) {tangentDisplacement = getNormalized(tangentDisplacement);}

        // check density criterion, add surface point if necessary
        Vec3 creationPos = pos + params.meanFineDistance*tangentDisplacement;
        if(
            isInDomain(creationPos) &&
            !surfacePoints.hasNeighbor(creationPos, params.meanFineDistance-(1e-6))
        ) {
            //create point
            surfacePoints.addBuffered(creationPos);
        }

    }
    
    
    surfacePoints.doCompress();
    surfacePoints.insertBufferedParticles();
    
    
    // check density criterion, delete surface points if necessary
    fixedSize = surfacePoints.size();
    for (int idx=0; idx<fixedSize; idx++) {
        if(
            !isInDomain(surfacePoints.getPos(idx)) ||
            surfacePoints.hasNeighborOtherThanItself(idx, 0.67*params.meanFineDistance)
        ) {
            surfacePoints.kill(idx);
        }
    }
    
    // delete surface points if no coarse neighbors in advection radius
    fixedSize = surfacePoints.size();
    for (int idx=0; idx<fixedSize; idx++) {
        Vec3 pos = surfacePoints.getPos(idx);
        if(!coarseParticles.hasNeighbor(pos, 2.f*params.outerRadius)) {
            surfacePoints.kill(idx);
        }
    }
    
    // delete surface point if too far from constraint
    fixedSize = surfacePoints.size();
    for (int idx=0; idx<fixedSize; idx++) {
        float level = computeConstraintLevel(coarseParticles, surfacePoints.getPos(idx));
        if(level < -0.2 || level > 1.2) {
            surfacePoints.kill(idx);
        }
    }
    

    surfacePoints.doCompress();
    surfacePoints.insertBufferedParticles();
}


//
// **** surface maintenance ****
//

KERNEL(pts)
void computeSurfaceDensities(
        BasicParticleSystemWrapper& surfacePoints,
        void* dummy
) {
    Vec3 pos = surfacePoints.getPos(idx);
    float density = 0;
    LOOP_NEIGHBORS_BEGIN(surfacePoints, pos, params.normalRadius)
        LOOP_GHOSTS_POS_BEGIN(surfacePoints.getPos(idn), params.normalRadius)
            density += weightSurfaceNormal(norm(pos-gPos));
        LOOP_GHOSTS_END
    LOOP_NEIGHBORS_END
    tempSurfaceFloat[idx] = density;
}

KERNEL(pts)
void computeSurfaceDisplacements(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals
){
    Vec3 pos = surfacePoints.getPos(idx);
    Vec3 normal = surfaceNormals[idx];
    
    Vec3 displacementNormal(0,0,0);
    Vec3 displacementTangent(0,0,0);
    float wTotal = 0;
    LOOP_NEIGHBORS_BEGIN(surfacePoints, pos, params.normalRadius)
            
        LOOP_GHOSTS_POS_NORMAL_BEGIN(surfacePoints.getPos(idn), surfaceNormals[idn], params.normalRadius)
            Vec3 dir = pos - gPos;
            float length = norm(dir);            
            Vec3 dn = dot(dir,surfaceNormals[idx])*surfaceNormals[idx];
            Vec3 dt = dir - dn;
            if(tempSurfaceFloat[idn]==0) {continue;}
            float w = weightSurfaceNormal( length ) / tempSurfaceFloat[idn];
            
            Vec3 crossVec = getNormalized(cross(normal, -dir));
            Vec3 projectedNormal = getNormalized(gNormal - dot(crossVec,gNormal)*crossVec);
            if(dot(projectedNormal, normal) < 0 || abs(dot(normal,normal+projectedNormal)) < 1e-6) {continue;}
            dn = -dot(normal+projectedNormal,dir)/dot(normal,normal+projectedNormal)*normal;
            
            displacementNormal  += w * dn;
            displacementTangent += w * getNormalized(dt); 
            wTotal += w;
        LOOP_GHOSTS_END
            
    LOOP_NEIGHBORS_END
    if(wTotal != 0) {
        displacementNormal  /= wTotal;
        displacementTangent /= wTotal;
    }
    displacementNormal  *= .75f;
    displacementTangent *= .25f * params.meanFineDistance;
    tempSurfaceVec3[idx] = displacementNormal + displacementTangent;
}

KERNEL(pts)
void applySurfaceDisplacements(
        BasicParticleSystemWrapper& surfacePoints,
        void* dummy
){
    surfacePoints.setPos(idx, surfacePoints.getPos(idx) + tempSurfaceVec3[idx]);
}


void regularizeSurfacePoints(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals
){
    tempSurfaceVec3.resize(surfacePoints.size());
    tempSurfaceFloat.resize(surfacePoints.size()); 
    
    computeSurfaceDensities(surfacePoints, 0);
    computeSurfaceDisplacements(surfacePoints, surfaceNormals);
    applySurfaceDisplacements(surfacePoints, 0);
}


KERNEL(pts)
void constrainSurface(
        BasicParticleSystemWrapper& surfacePoints,
        BasicParticleSystemWrapper& coarseParticles        
) {
        Vec3 pos = surfacePoints.getPos(idx);
        float level = computeConstraintLevel(coarseParticles, surfacePoints.getPos(idx));
        if(level > 1) {
            surfacePoints.setPos(idx, pos - (params.outerRadius-params.innerRadius)*(level-1)*computeConstraintGradient(coarseParticles, surfacePoints.getPos(idx)));
        }else if(level < 0) {
            surfacePoints.setPos(idx, pos - (params.outerRadius-params.innerRadius)*  level  *computeConstraintGradient(coarseParticles, surfacePoints.getPos(idx)));
        }
}


KERNEL(pts)
void interpolateNewWaveData(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<float>& surfaceWaveH,
        ParticleDataImpl<float>& surfaceWaveDtH,
        ParticleDataImpl<float>& surfaceWaveSeed,
        ParticleDataImpl<float>& surfaceWaveSeedAmplitude
){
    if(surfacePoints.getStatus(idx) & ParticleBase::PNEW) {
        Vec3 pos = surfacePoints.getPos(idx);
        surfaceWaveH[idx] = 0;
        surfaceWaveDtH[idx] = 0;
        float wTotal = 0;
        LOOP_NEIGHBORS_BEGIN(surfacePoints, pos, params.tangentRadius)
            if(!(surfacePoints.getStatus(idn) & ParticleBase::PNEW)) {
                float w = weightSurfaceTangent(norm( pos - surfacePoints.getPos(idn) ));
                surfaceWaveH[idx] += w * surfaceWaveH[idn];
                surfaceWaveDtH[idx] += w * surfaceWaveDtH[idn];
                surfaceWaveSeed[idx] += w * surfaceWaveSeed[idn];
                surfaceWaveSeedAmplitude[idx] += w * surfaceWaveSeedAmplitude[idn];
                wTotal += w;
            }
        LOOP_NEIGHBORS_END
        if(wTotal != 0) {
            surfaceWaveH[idx]   /= wTotal;
            surfaceWaveDtH[idx] /= wTotal;
            surfaceWaveSeed[idx]   /= wTotal;
            surfaceWaveSeedAmplitude[idx] /= wTotal;
        }
    }
}


void surfaceMaintenance(
        BasicParticleSystemWrapper& coarseParticles,
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals,
        ParticleDataImpl<float>& surfaceWaveH,
        ParticleDataImpl<float>& surfaceWaveDtH,
        ParticleDataImpl<float>& surfaceWaveSeed,
        ParticleDataImpl<float>& surfaceWaveSeedAmplitude,
        int nbIterations
){
    int countIterations = nbIterations;
    while(countIterations > 0) {
        addDeleteSurfacePoints(surfacePoints);
        surfacePoints.updateAccel();
        computeSurfaceNormals(surfacePoints, coarseParticles, surfaceNormals);
        smoothSurfaceNormals(surfacePoints, surfaceNormals);

        regularizeSurfacePoints(surfacePoints, surfaceNormals);
        surfacePoints.updateAccel();
        constrainSurface(surfacePoints, coarseParticles);
        surfacePoints.updateAccel();

        interpolateNewWaveData(surfacePoints, surfaceWaveH, surfaceWaveDtH, surfaceWaveSeed, surfaceWaveSeedAmplitude);
        
        countIterations--;
    }
}




//
// **** surface wave seeding and evolution ****
//


KERNEL(pts)
void addSeed(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<float>& surfaceWaveH,
        ParticleDataImpl<float>& surfaceWaveSeed
){
    surfaceWaveH[idx] += surfaceWaveSeed[idx];
}


KERNEL(pts)
void computeSurfaceWaveNormal(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals,
        ParticleDataImpl<float>& surfaceWaveH
){
    Vec3 pos = surfacePoints.getPos(idx);

    // get tangent frame
    Vec3 n = getNormalized(surfaceNormals[idx]);
    Vec3 vx(1,0,0);
    Vec3 vy(0,1,0);
    float dotX = dot(n, vx);
    float dotY = dot(n, vy);
    Vec3 t1 = getNormalized(fabs(dotX)<fabs(dotY) ? cross(n, vx) : cross(n, vy));
    Vec3 t2 = getNormalized( cross(n,t1) );

    // linear fit
    float sw = 0, swx = 0, swy = 0, swxy = 0, swx2 = 0, swy2 = 0, swxz = 0, swyz = 0, swz = 0;
    LOOP_NEIGHBORS_BEGIN(surfacePoints, pos, params.tangentRadius)
        LOOP_GHOSTS_POS_BEGIN(surfacePoints.getPos(idn), params.tangentRadius)
            float x = dot(gPos - pos, t1);
            float y = dot(gPos - pos, t2);
            float z = surfaceWaveH[idn];
            float w = weightSurfaceTangent(norm(pos - gPos));
            swx2 += w*x*x;
            swy2 += w*y*y;
            swxy += w*x*y;
            swxz += w*x*z;
            swyz += w*y*z;
            swx  += w*x;
            swy  += w*y;
            swz  += w*z;
            sw   += w;
        LOOP_GHOSTS_END
    LOOP_NEIGHBORS_END
    float det = -sw*swxy*swxy + 2.f*swx*swxy*swy - swx2*swy*swy - swx*swx*swy2 + sw*swx2*swy2;
    if(det == 0) {tempSurfaceVec3[idx]=Vec3(0,0,0);}
    else {
        Vec3 abc = 1.f/det*Vec3(
                    swxz*(-swy*swy+sw*swy2)  + swyz*(-sw*swxy+swx*swy)  + swz*(swxy*swy-swx*swy2),
                    swxz*(-sw*swxy+swx*swy)  + swyz*(-swx*swx+sw*swx2)  + swz*(swx*swxy-swx2*swy),
                    swxz*(swxy*swy-swx*swy2) + swyz*(swx*swxy-swx2*swy) + swz*(-swxy*swxy + swx2*swy2)
                    );
        Vec3 waveNormal = -getNormalized(vx*abc.x + vy*abc.y - Vec3(0,0,1));
        tempSurfaceVec3[idx] = waveNormal;
    }
}

KERNEL(pts)
void computeSurfaceWaveLaplacians(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals,
        ParticleDataImpl<float>& surfaceWaveH
){
    float laplacian = 0;
    float wTotal = 0;
    Vec3 pPos = surfacePoints.getPos(idx);
    Vec3 pNormal = surfaceNormals[idx];

    Vec3 vx(1,0,0);
    Vec3 vy(0,1,0);
    float dotX = dot(pNormal, vx);
    float dotY = dot(pNormal, vy);
    Vec3 t1 = getNormalized(fabs(dotX)<fabs(dotY) ? cross(pNormal, vx) : cross(pNormal, vy));
    Vec3 t2 = getNormalized( cross(pNormal,t1) );

    Vec3 pWaveNormal = tempSurfaceVec3[idx];
    float ph = surfaceWaveH[idx];
    if(pWaveNormal.z == 0) {tempSurfaceFloat[idx]=0;}
    else {

        LOOP_NEIGHBORS_BEGIN(surfacePoints, pPos, params.tangentRadius)
            float nh = surfaceWaveH[idn];
            LOOP_GHOSTS_POS_BEGIN(surfacePoints.getPos(idn), params.tangentRadius)
                Vec3 dir = gPos - pPos;
                float lengthDir = norm(dir);
                if(lengthDir < 1e-5) continue;
                Vec3 tangentDir = lengthDir*getNormalized(dir - dot(dir, pNormal)*pNormal);
                float dirX = dot(tangentDir, t1);
                float dirY = dot(tangentDir, t2);
                float dz = nh - ph - (-pWaveNormal.x/pWaveNormal.z)*dirX - (-pWaveNormal.y/pWaveNormal.z)*dirY;
                float w = weightSurfaceTangent(norm(pPos - gPos));
                wTotal += w;
                laplacian += clamp(w * 4*dz/(lengthDir*lengthDir), -100.f, 100.f);
            LOOP_GHOSTS_END
        LOOP_NEIGHBORS_END
        if(wTotal != 0) {tempSurfaceFloat[idx] = laplacian/wTotal;}
        else {tempSurfaceFloat[idx] = 0;}
    }
}


KERNEL(pts)
void evolveWave(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<float>& surfaceWaveH,
        ParticleDataImpl<float>& surfaceWaveDtH,
        ParticleDataImpl<float>& surfaceWaveSeed
){
    surfaceWaveDtH[idx] += params.waveSpeed*params.waveSpeed * params.dt * tempSurfaceFloat[idx];
    surfaceWaveDtH[idx] /= (1 + params.dt * params.waveDamping);
    surfaceWaveH[idx]   += params.dt * surfaceWaveDtH[idx];
    surfaceWaveH[idx]   /= (1 + params.dt * params.waveDamping);
    surfaceWaveH[idx] -= surfaceWaveSeed[idx];
    
    // clamp H and DtH (to prevent rare extreme behaviors)
    surfaceWaveDtH[idx] = clamp(surfaceWaveDtH[idx], -params.waveMaxFrequency*params.waveMaxAmplitude, params.waveMaxFrequency*params.waveMaxAmplitude);
    surfaceWaveH[idx]   = clamp(surfaceWaveH[idx], -params.waveMaxAmplitude, params.waveMaxAmplitude);
}


KERNEL(pts)
void computeSurfaceCurvature(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals
){
    Vec3 pPos = surfacePoints.getPos(idx);
    float wTotal = 0;
    float curv = 0;
    Vec3 pNormal = surfaceNormals[idx];

    LOOP_NEIGHBORS_BEGIN(surfacePoints, pPos, params.normalRadius)
        LOOP_GHOSTS_POS_NORMAL_BEGIN(surfacePoints.getPos(idn), surfaceNormals[idn], params.normalRadius)    
            Vec3 dir = pPos - gPos;
            if(dot(pNormal, gNormal) < 0) {continue;} // backfacing
            float dist = norm(dir);
            if(dist < params.normalRadius/100.f){ continue; }

            float distn = dot(dir, pNormal);

            float w = weightSurfaceNormal(dist);
            curv += w * distn;
            wTotal += w;
        LOOP_GHOSTS_END
    LOOP_NEIGHBORS_END
    if(wTotal!=0) {curv /= wTotal;}
    tempSurfaceFloat[idx] = fabs(curv);
}


KERNEL(pts)
void smoothCurvature(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<float>& surfaceWaveSource
){
    Vec3 pPos = surfacePoints.getPos(idx);
    float curv = 0;
    float wTotal = 0;
    
    LOOP_NEIGHBORS_BEGIN(surfacePoints, pPos, params.normalRadius)
        float w = weightSurfaceNormal(norm( pPos - surfacePoints.getPos(idn) ));
        curv += w * tempSurfaceFloat[idn];
        wTotal += w;                
    LOOP_NEIGHBORS_END
    if(wTotal!=0) {curv /= wTotal;}
    surfaceWaveSource[idx] = curv;
}


KERNEL(pts)
void seedWaves(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<float>& surfaceWaveSeed,
        ParticleDataImpl<float>& surfaceWaveSeedAmplitude,
        ParticleDataImpl<float>& surfaceWaveSource
){
    float source = smoothstep(params.waveSeedingCurvatureThresholdRegionCenter - params.waveSeedingCurvatureThresholdRegionRadius, params.waveSeedingCurvatureThresholdRegionCenter + params.waveSeedingCurvatureThresholdRegionRadius, (float) surfaceWaveSource[idx]) * 2.f - 1.f;
    float freq = params.waveSeedFrequency;
    float theta = params.dt * frameCount * params.waveSpeed * freq;
    float costheta = cosf(theta);
    float maxSeedAmplitude = params.waveMaxSeedingAmplitude * params.waveMaxAmplitude;

    surfaceWaveSeedAmplitude[idx] = clamp<float>(surfaceWaveSeedAmplitude[idx]+source*params.waveSeedStepSizeRatioOfMax*maxSeedAmplitude, 0.f, maxSeedAmplitude);
    surfaceWaveSeed[idx] = surfaceWaveSeedAmplitude[idx] * costheta;

    // source values for display (not used after this point anyway)
    surfaceWaveSource[idx] = (source>=0) ? 1 : 0;
}



void surfaceWaves(
        BasicParticleSystemWrapper& surfacePoints,
        ParticleDataImpl<Vec3>& surfaceNormals,
        ParticleDataImpl<float>& surfaceWaveH,
        ParticleDataImpl<float>& surfaceWaveDtH,
        ParticleDataImpl<float>& surfaceWaveSource,
        ParticleDataImpl<float>& surfaceWaveSeed,
        ParticleDataImpl<float>& surfaceWaveSeedAmplitude
){
    addSeed(surfacePoints, surfaceWaveH, surfaceWaveSeed);
    computeSurfaceWaveNormal(surfacePoints, surfaceNormals, surfaceWaveH);
    computeSurfaceWaveLaplacians(surfacePoints, surfaceNormals, surfaceWaveH);
    evolveWave(surfacePoints, surfaceWaveH, surfaceWaveDtH, surfaceWaveSeed);
    computeSurfaceCurvature(surfacePoints, surfaceNormals);
    smoothCurvature(surfacePoints, surfaceWaveSource);
    seedWaves(surfacePoints, surfaceWaveSeed, surfaceWaveSeedAmplitude, surfaceWaveSource);
}



//
// **** main function ****
//



PYTHON() void particleSurfaceTurbulence(
    FlagGrid& flags,
    BasicParticleSystem& coarseParts,
    ParticleDataImpl<Vec3>& coarsePartsPrevPos,
    BasicParticleSystem& surfPoints,
    ParticleDataImpl<Vec3>& surfaceNormals,
    ParticleDataImpl<float>& surfaceWaveH,
    ParticleDataImpl<float>& surfaceWaveDtH,
    BasicParticleSystem& surfacePointsDisplaced,
    ParticleDataImpl<float>& surfaceWaveSource,
    ParticleDataImpl<float>& surfaceWaveSeed,
    ParticleDataImpl<float>& surfaceWaveSeedAmplitude,
    // params with default values
    int res,
    float outerRadius = 1.0f,
    int surfaceDensity = 20,
    int nbSurfaceMaintenanceIterations = 4,
    float dt = 0.005f,
    float waveSpeed = 16.0f,
    float waveDamping = 0.0f,
    float waveSeedFrequency = 4,
    float waveMaxAmplitude = 0.25f,
    float waveMaxFrequency = 800,
    float waveMaxSeedingAmplitude = 0.5, // as multiple of max amplitude
    float waveSeedingCurvatureThresholdRegionCenter = 0.025f, // any curvature higher than this value will seed waves
    float waveSeedingCurvatureThresholdRegionRadius = 0.01f,
    float waveSeedStepSizeRatioOfMax = 0.05f // higher values will result in faster and more violent wave seeding
)
{ 
    static std::chrono::high_resolution_clock::time_point begin, end;

    end = std::chrono::high_resolution_clock::now();
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1000000000.f << " : time sim" << endl;
    begin = std::chrono::high_resolution_clock::now();
    
    // wrap data
    coarseParticles.points = &coarseParts;
    coarseParticlesPrevPos.points = &coarsePartsPrevPos;
    surfacePoints.points = &surfPoints;

    // copy parameters
    params.res = res;
    params.outerRadius = outerRadius;
    params.surfaceDensity = surfaceDensity;
    params.nbSurfaceMaintenanceIterations = nbSurfaceMaintenanceIterations;
    params.dt = dt;
    params.waveSpeed = waveSpeed;
    params.waveDamping = waveDamping;
    params.waveSeedFrequency = waveSeedFrequency;
    params.waveMaxAmplitude = waveMaxAmplitude;
    params.waveMaxFrequency = waveMaxFrequency;
    params.waveMaxSeedingAmplitude = waveMaxSeedingAmplitude;
    params.waveSeedingCurvatureThresholdRegionCenter = waveSeedingCurvatureThresholdRegionCenter;
    params.waveSeedingCurvatureThresholdRegionRadius = waveSeedingCurvatureThresholdRegionRadius;
    params.waveSeedStepSizeRatioOfMax = waveSeedStepSizeRatioOfMax;

    // compute other parameters
    params.innerRadius = params.outerRadius/2.0;
    params.meanFineDistance = M_PI*(params.outerRadius+params.innerRadius)/params.surfaceDensity;
    params.constraintA = logf(2.0f/(1.0f + weightKernelCoarseDensity(params.outerRadius+params.innerRadius)))/(powf((params.outerRadius+params.innerRadius)/2,2) - params.innerRadius*params.innerRadius);
    params.normalRadius = 0.5f*(params.outerRadius + params.innerRadius);
    params.tangentRadius = 2.1f*params.meanFineDistance;
    params.bndXm = params.bndYm = params.bndZm = 2;
    params.bndXp = params.bndYp = params.bndZp = params.res-2;

    if(frameCount==0) {

        // initialize accel grids
        accelCoarse.init(2.f*res/params.outerRadius);
        accelSurface.init(1.f*res/(2.f*params.meanFineDistance));

        // update coarse accel structure
        coarseParticles.updateAccel();

        // create surface points
        initFines(coarseParticles, surfacePoints, flags);

        // smooth surface
        surfaceMaintenance(coarseParticles, surfacePoints, surfaceNormals, surfaceWaveH, surfaceWaveDtH, surfaceWaveSeed, surfaceWaveSeedAmplitude, 6*params.nbSurfaceMaintenanceIterations);

        // set wave values to zero
        for (int idx=0; idx<surfacePoints.size(); idx++) {
            surfaceWaveH[idx] = 0;
            surfaceWaveDtH[idx] = 0;
            surfaceWaveSeed[idx] = 0;
            surfaceWaveSeedAmplitude[idx] = 0;
        }

    } else {

        // update coarse accel structure with previous coarse particles positions
        coarseParticlesPrevPos.updateAccel();

        //advect surface points following coarse particles
        advectSurfacePoints(surfacePoints, coarseParticles, coarseParticlesPrevPos);
        surfacePoints.updateAccel();

        // update acceleration structure for surface points
        coarseParticles.updateAccel();

        //surface maintenance
        surfaceMaintenance(coarseParticles, surfacePoints, surfaceNormals, surfaceWaveH, surfaceWaveDtH, surfaceWaveSeed, surfaceWaveSeedAmplitude, params.nbSurfaceMaintenanceIterations);
        
        // surface waves
        surfaceWaves(surfacePoints, surfaceNormals, surfaceWaveH, surfaceWaveDtH, surfaceWaveSource, surfaceWaveSeed, surfaceWaveSeedAmplitude);
    }
    frameCount++;

    // save positions as previous positions for next step
    for(int id=0;id<coarseParticles.size();id++) {
        if((coarseParticles.getStatus(id) & ParticleBase::PNEW) == 0 &&
           (coarseParticles.getStatus(id) & ParticleBase::PDELETE) == 0
        ){
            coarseParticlesPrevPos.setVec3(id, coarseParticles.getPos(id));
        }
    }

    // create displaced points for display
    surfacePointsDisplaced.clear();
    for(int idx=0;idx<surfacePoints.size();idx++) {
        if((surfacePoints.getStatus(idx) & ParticleBase::PDELETE) == 0) {
            surfacePointsDisplaced.addParticle(surfacePoints.getPos(idx) + surfaceNormals[idx]*surfaceWaveH[idx]);
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1000000000.f << " : time upres" << endl;
    
    begin = std::chrono::high_resolution_clock::now();
}




PYTHON() void debugCheckParts(BasicParticleSystem& parts, FlagGrid& flags) {
    for(int idx=0;idx<parts.size();idx++) {
        Vec3i p = toVec3i( parts.getPos(idx) );
        if(! flags.isInBounds(p) ) {
            debMsg("bad position??? "<<idx<<" "<< parts.getPos(idx) ,1 ); exit(1);
        }
    }
}




} // namespace SurfaceTurbulence

} // namespace Manta

