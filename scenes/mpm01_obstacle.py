#
# MLS-MPM scene with obstacle
#
from manta import *
import time

# solver params
dim = 2
particleNumber = 2
frames = 10000
res = 80
withMesh = 1
withObs = 1
withScreenshot = 0
withPseudo3D = 0 # simulate with 3D grids but 2D MPM functions
withParallelParts = 1 # enable parallel "parts-to-grid" mapping (noticeably faster when using lots of particles)
withBlockGrid = 1

gs = vec3(res,res,res)
bs = vec3(4) if (withBlockGrid) else vec3(1)
if (dim==2):
	bs.z=1
	gs.z=1
boundaryCondition = 0 # 0: separate, 1: stick, 2: free-slip

s = Solver(name='main', gridSize=gs, dim=dim)
# Adaptive time stepping
fps = 200
timescale = 1
s.frameLength = 0.1 * (25.0 / fps) * timescale # length of one frame (in "world time")
s.cfl         = 0.05
s.timestep    = s.frameLength
s.timestepMin = 1e-5
s.timestepMax = 5e-4
s.updateGui = False  # using gui.update() to refresh GUI (i.e. only update once per frame), is faster this way

timings = Timings()

# converts world gravity to domain gravity
def accelerationToManta(domainRes, worldAcc, worldSize):
	lenFactor = (domainRes / worldSize)
	frameLengthDefault = 0.1
	fpsDefault = 25.0
	timeFactor = ((1/fpsDefault) * (1/frameLengthDefault)) # dt is 0.1 at 25fps
	# e.g. -9.81 m/s^2 * (200 cell / 4 meter) * ((1 s / 25 frame) * (1 frame / 0.1 step))^2 = -78.48 cell/step^2
	return worldAcc * lenFactor * timeFactor**2

gravity   = vec3(0, accelerationToManta(res, -9.81, 1), 0) # gravity and domain size in world units
hardening = 10.0 # Hardening factor
E         = 1e4 # Young's modulus
nu        = 0.2 # Poisson ratio
pmass     = 2.0 # Particle mass
pvol      = 1.0 # Particle volume
plastic   = bool(1) # plastic=0 behave more like soft body

# prepare grids and particles
flags    = s.create(FlagGrid)
dummyVec = s.create(VecGrid) # just for nicer grid view in GUI
vel      = s.create(MACGrid)
mesh     = s.create(Mesh)
pp       = s.create(BasicParticleSystem)
pp.maxParticles = int(5e3) if (dim == 2) else int(1e5) # 2D still runs in realtime on notebook
# add velocity data to particles
pVel     = pp.create(PdataVec3)
# mpm part
dummy    = s.create(RealGrid) # just for nicer grid view in GUI
mass     = s.create(RealGrid)

# placeholders for acceleration grids
kernelGrid = None
velTmp0  = None
velTmp1  = None
velTmp2  = None
massTmp0 = None
massTmp1 = None
massTmp2 = None

if withParallelParts:
	# In 2D parallelize j neighbor cell search, in 3D parallelize k neighbor cell search
	gsKernel       = vec3(1,1,5)
	sAcc           = Solver(name='kernelHelper', gridSize=gsKernel, dim=3)
	sAcc.updateGui = False
	kernelGrid     = sAcc.create(RealGrid)
	velTmp0  = s.create(MACGrid)
	velTmp1  = s.create(MACGrid)
	velTmp2  = s.create(MACGrid)
	massTmp0 = s.create(RealGrid)
	massTmp1 = s.create(RealGrid)
	massTmp2 = s.create(RealGrid)

# acceleration data for particle nbs
pindex   = s.create(ParticleIndexSystem)
gpi      = s.create(IntGrid)
counter  = s.create(IntGrid)
phiParts = s.create(LevelsetGrid)

# obstacle
phiInit  = s.create(LevelsetGrid)
fractions = s.create(MACGrid)
phiInit.setConst(999.)
phiObs   = s.create(LevelsetGrid)

# determinant of deformation grad
Jp = pp.create(PdataReal)

# matrix data structures per particle
F = pp.create(PdataMat2) if (dim == 2 or withPseudo3D) else pp.create(PdataMat3) # deformation grad
C = pp.create(PdataMat2) if (dim == 2 or withPseudo3D) else pp.create(PdataMat3) # affine momentum
R = pp.create(PdataMat2) if (dim == 2 or withPseudo3D) else pp.create(PdataMat3) # rotation mat from F=RS
S = pp.create(PdataMat2) if (dim == 2 or withPseudo3D) else pp.create(PdataMat3) # scale mat from F=RS

# outflow at domain boundaries?
outflowCondition = 'xX' # define open walls
boundaryWidth = 2 # need space at boundaries for neighbor search
flags.initDomain(boundaryWidth=boundaryWidth-1, phiWalls=phiObs, outflow=outflowCondition)

# Setup fluid and obstacle geometries
scene = 5 # center obstacle
#scene = 6 # stairs obstacle

fluid1 = None
fluid2 = None

if scene == 5:
	fluid1 = Box(parent=s, p0=gs*vec3(0.4,0.75,0.4), p1=gs*vec3(0.6,0.95,0.6))
	#fluid1 = Sphere(parent=s, center=gs*vec3(0.5,0.8,0.5), radius=res*0.12)
	phiInit = fluid1.computeLevelset()

if scene == 6:
	fluid1 = Sphere(parent=s, center=gs*vec3(0.15,0.85,0.5), radius=res*0.1)
	fluid2 = Sphere(parent=s, center=gs*vec3(0.35,0.65,0.5), radius=res*0.1)
	phiInit = fluid1.computeLevelset()
	if fluid2: phiInit.join(fluid2.computeLevelset())

if withObs:
	obs1 = None
	obs2 = None
	obs3 = None
	obs4 = None
	obs5 = None
	obs6 = None
	obs7 = None
	if scene == 5:
		obs1 = Box(parent=s, p0=gs*vec3(0.47,0,0), p1=gs*vec3(0.53,0.4,1))
		#obs1 = Sphere(parent=s, center=gs*vec3(0.5,0.1,0.5), radius=res*0.2)

	if scene == 6:
		obs1 = Box(parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(0.1,0.7,1))
		obs2 = Box(parent=s, p0=gs*vec3(0.1,0,0), p1=gs*vec3(0.2,0.6,1))
		obs3 = Box(parent=s, p0=gs*vec3(0.2,0,0), p1=gs*vec3(0.3,0.5,1))
		obs4 = Box(parent=s, p0=gs*vec3(0.3,0,0), p1=gs*vec3(0.4,0.4,1))
		obs5 = Box(parent=s, p0=gs*vec3(0.4,0,0), p1=gs*vec3(0.5,0.3,1))
		obs6 = Box(parent=s, p0=gs*vec3(0.5,0,0), p1=gs*vec3(0.6,0.2,1))
		obs7 = Box(parent=s, p0=gs*vec3(0.6,0,0), p1=gs*vec3(0.7,0.1,1))

	phiObs.join(obs1.computeLevelset())
	if obs2: phiObs.join(obs2.computeLevelset())
	if obs3: phiObs.join(obs3.computeLevelset())
	if obs4: phiObs.join(obs4.computeLevelset())
	if obs5: phiObs.join(obs5.computeLevelset())
	if obs6: phiObs.join(obs6.computeLevelset())
	if obs7: phiObs.join(obs7.computeLevelset())
	extrapolateLsSimple(phi=phiObs, distance=4, inside=True)
	extrapolateLsSimple(phi=phiObs, distance=2, inside=False)

flags.updateFromLevelset(phiInit)
phiInit.subtract(phiObs)

sampleLevelsetWithParticles(phi=phiInit, flags=flags, parts=pp, discretization=particleNumber,
	randomness=.0, reset=false, refillEmpty=false, particleFlag=-1, inRandomOrder=True)

updateFractions(flags=flags, phiObs=phiObs, fractions=fractions, boundaryWidth=boundaryWidth)
setObstacleFlags(flags=flags, phiObs=phiObs, boundaryWidth=boundaryWidth, fractions=fractions, phiOut=None, phiIn=None)

# Initialize particle data structures (after sampling particles!)
Jp.setConst(1.0) # initialize with 0.0 for more rigidity
if not plastic: Jp.setConst(1.0)
identity = mat2() if (dim == 2 or withPseudo3D) else mat3()
zeros = mat2(0) if (dim == 2 or withPseudo3D) else mat3(0)
F.setConst(identity)
C.setConst(zeros)
R.setConst(zeros)
S.setConst(zeros)

if (GUI):
	gui = Gui()
	gui.show()
	gui.pause()

lastFrame = -1
maxVel = 0

# Index grids that contain IndexInt's or ijk's in a blocked layout
blockIdxGrid = s.create(IndexIntGrid, indexType=IndexBlock, blockSize=bs)
blockIjkGrid = s.create(IndexVecGrid, indexType=IndexBlock, blockSize=bs)

startTime = time.time()

# main loop
while s.frame < frames:
	if (maxVel > 0): s.adaptTimestep(maxVel)
	#mantaMsg('\nFrame %i, simulation time %f, timestep %f' % (s.frame, s.timeTotal, s.timestep))

	if (dim == 2 or withPseudo3D):
		polarDecomposition2D(A=F, U=R, P=S)
		mpmMapPartsToGrid2D(vel=vel, mass=mass, flags=flags, pp=pp, pvel=pVel, blockIdxGrid=blockIdxGrid,
			detDeformationGrad=Jp, deformationGrad=F,
			affineMomentum=C, rotation=R, kernelGrid=kernelGrid, velTmp0=velTmp0, velTmp1=velTmp1, velTmp2=velTmp2,
			massTmp0=massTmp0, massTmp1=massTmp1, massTmp2=massTmp2, hardening=hardening, E=E, nu=nu, pmass=pmass, pvol=pvol)
	else:
		polarDecomposition3D(A=F, U=R, P=S)
		mpmMapPartsToGrid3D(vel=vel, mass=mass, flags=flags, pp=pp, pvel=pVel, blockIdxGrid=blockIdxGrid,
			detDeformationGrad=Jp, deformationGrad=F,
			affineMomentum=C, rotation=R, kernelGrid=kernelGrid, velTmp0=velTmp0, velTmp1=velTmp1, velTmp2=velTmp2,
			massTmp0=massTmp0, massTmp1=massTmp1, massTmp2=massTmp2, hardening=hardening, E=E, nu=nu, pmass=pmass, pvol=pvol)

	maxVel = mpmUpdateGrid(flags=flags, gravity=gravity, vel=vel, mass=mass, phiObs=phiObs, fractions=fractions,
		blockIjkGrid=blockIjkGrid, blockIdxGrid=blockIdxGrid,
		velTmp0=velTmp0, velTmp1=velTmp1, velTmp2=velTmp2,
		massTmp0=massTmp0, massTmp1=massTmp1, massTmp2=massTmp2,
		obvel=None, boundaryCondition=boundaryCondition)

	if (dim == 2 or withPseudo3D):
		mpmMapGridToParts2D(pp=pp, vel=vel, pvel=pVel, flags=flags, phiObs=phiObs, blockIdxGrid=blockIdxGrid, detDeformationGrad=Jp, deformationGrad=F,
			affineMomentum=C, plastic=plastic)
	else:
		mpmMapGridToParts3D(pp=pp, vel=vel, pvel=pVel, flags=flags, phiObs=phiObs, blockIdxGrid=blockIdxGrid, detDeformationGrad=Jp, deformationGrad=F,
			affineMomentum=C, plastic=plastic)
	s.step()

	# Update GUI and mesh only once per frame
	if (lastFrame != s.frame):
		if withMesh and dim==3:
			gridParticleIndex(parts=pp , flags=flags, indexSys=pindex, index=gpi, counter=counter)
			#unionParticleLevelset(pp, pindex, flags, gpi, phiParts) # faster but less accurate
			improvedParticleLevelset(pp, pindex, flags, gpi, phiParts , 1 , 1, 1, 0.4, 3.5, None, 0)
			phiParts.createMesh(mesh)
		if (GUI):
			gui.update(s.frame, s.timeTotal)
			if withScreenshot:
				mantaMsg('\nScreenshot at frame %i, simulation time %f, timestep %f' % (s.frame, s.timeTotal, s.timestep))
				gui.screenshot('imgs_%02d/mpm_%02d_%04d.png' % (int(5), int(5), s.frame))

	timings.display()
	lastFrame = s.frame

print('Time taken : %s seconds ---' % (time.time() - startTime))
