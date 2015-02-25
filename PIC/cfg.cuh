#include "dependencies.cuh"
#include "structs.cuh"
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <map>

Config getConfig();
void get_derived_vals(Config *cfg);
Config defaultConfig();