/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2020 Sebastian Barschkis, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0 
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * General functions that make use of functions from other io files.
 *
 ******************************************************************************/

#include "mantaio.h"

namespace Manta {

PYTHON() int load(const std::string& name, std::vector<PbClass*>& objects, float worldSize) {
	if (name.find_last_of('.') == std::string::npos)
		errMsg("file '" + name + "' does not have an extension");
	std::string ext = name.substr(name.find_last_of('.'));

	if (ext == ".raw")
		return readGridsRaw(name, &objects);
	else if (ext == ".uni")
		return readGridsUni(name, &objects);
	else if (ext == ".vol")
		return readGridsVol(name, &objects);
	if (ext == ".vdb")
		return readObjectsVDB(name, &objects, worldSize);
	else if (ext == ".npz")
		return readGridsNumpy(name, &objects);
	else if (ext == ".txt")
		return readGridsTxt(name, &objects);
	else
		errMsg("file '" + name +"' filetype not supported");
	return 0;
}

PYTHON() int save(const std::string& name, std::vector<PbClass*>& objects, float worldSize,
	bool skipDeletedParts, int compression, bool precisionHalf, int precision, float clip, const Grid<Real>* clipGrid) {

	if (!precisionHalf) {
		debMsg("Warning: precisionHalf argument is deprecated. Please use precision level instead", 0);
		precision = PRECISION_HALF; // for backwards compatibility
	}

	if (name.find_last_of('.') == std::string::npos)
		errMsg("file '" + name + "' does not have an extension");
	std::string ext = name.substr(name.find_last_of('.'));

	if (ext == ".raw")
		return writeGridsRaw(name, &objects);
	else if (ext == ".uni")
		return writeGridsUni(name, &objects);
	else if (ext == ".vol")
		return writeGridsVol(name, &objects);
	if (ext == ".vdb")
		return writeObjectsVDB(name, &objects, worldSize, skipDeletedParts, compression, precision, clip, clipGrid);
	else if (ext == ".npz")
		return writeGridsNumpy(name, &objects);
	else if (ext == ".txt")
		return writeGridsTxt(name, &objects);
	else
		errMsg("file '" + name +"' filetype not supported");
	return 0;
}

} //namespace
