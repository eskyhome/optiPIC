#include <random>
#include "Particle.cuh"

std::random_device Particle::sg;
double Particle::seed = sg();
std::uniform_real_distribution<double> Particle::urd(-1.0, 1.0);
std::mt19937_64 Particle::re(Particle::seed);