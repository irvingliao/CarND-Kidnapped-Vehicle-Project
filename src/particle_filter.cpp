/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::numeric_limits;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if (is_initialized) {
    return;
  }

  num_particles = 100;  // Set the number of particles

  // Creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0 ; i < num_particles ; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  // Creates a normal (Gaussian) distribution for x, y and theta
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0 ; i < num_particles ; i++) {
    Particle p = particles[i];

    // Check if yaw_rate is not equal to zero.
    if(fabs(yaw_rate) > 0.00001) {
      particles[i].x += (velocity/yaw_rate)*(sin(p.theta+yaw_rate*delta_t) - sin(p.theta));
      particles[i].y += (velocity/yaw_rate)*(cos(p.theta) - cos(p.theta+yaw_rate*delta_t));
      particles[i].theta += yaw_rate*delta_t;
    } else {
      particles[i].x += velocity*delta_t*cos(p.theta);
      particles[i].y += velocity*delta_t*sin(p.theta);
    }

    // Add Noice
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int j = 0 ; j < observations.size() ; j++) {
    double minDist = numeric_limits<double>::max();
    int mapId = -1;
    for (int k = 0 ; k < predicted.size() ; k++) {
      double xDist = observations[j].x - predicted[k].x;
      double yDist = observations[j].y - predicted[k].y;
      double distance = xDist*xDist + yDist*yDist;
      if (distance < minDist) {
        minDist = distance;
        mapId = predicted[k].id;
      }
      observations[j].id = mapId;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;

  for (int i = 0 ; i < num_particles ; i++) {
    double particleX = particles[i].x;
    double particleY = particles[i].y;
    double theta = particles[i].theta;

    // Landmarks which map location with the sensor range of the particle
    vector<LandmarkObs> predictions;
    for (int j = 0 ; j < landmarks.size() ; j++) {
      int l_id = landmarks[j].id_i;
      float x_f = landmarks[j].x_f;
      float y_f = landmarks[j].y_f;
      double distance = dist(particleX, particleY, x_f, y_f);
      if (distance <= sensor_range) {
        LandmarkObs lm;
        lm.id = l_id;
        lm.x = x_f;
        lm.y = y_f;
        predictions.push_back(lm);
      }
    }

    particles[i].weight = 1.0;
    for (int j = 0 ; j < observations.size() ; j++) {
      LandmarkObs obs;
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;

      // Transform observations from vehicle coordinates to map coordinates
      obs.x = particleX + (cos(theta) * x_obs) - (sin(theta) * y_obs);
      obs.y = particleY + (sin(theta) * x_obs) + (cos(theta) * y_obs);

      // Data Association
      double minDist = numeric_limits<double>::max();
      int mapId = -1;
      double p_x = 0;
      double p_y = 0;
      for (int k = 0 ; k < predictions.size() ; k++) {
        double xDist = obs.x - predictions[k].x;
        double yDist = obs.y - predictions[k].y;
        double distance = xDist*xDist + yDist*yDist;
        if (distance < minDist) {
          minDist = distance;
          mapId = predictions[k].id;
          p_x = predictions[k].x;
          p_y = predictions[k].y;
        }
        obs.id = mapId;
      }

      double w = multiv_prob(std_landmark[0], std_landmark[1], obs.x, obs.y, p_x, p_y);
      particles[i].weight *= w;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
  *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  // Resampling Wheel
  // Get the max weight
  vector<double> w;
  double maxWeight = numeric_limits<double>::min();
  for (int i = 0 ; i < num_particles ; i++) {
    w.push_back(particles[i].weight);
    if (maxWeight < particles[i].weight) {
      maxWeight = particles[i].weight;
    }
  }

  uniform_int_distribution<int> dist_i(0, num_particles - 1);
  uniform_real_distribution<double> dist_w(0.0, maxWeight);
  int index = dist_i(gen);
  double beta = 0.0;

  vector<Particle> resample_ps;
  for (int i = 0 ; i < num_particles ; i++) {
  beta += 2*dist_w(gen);
    while (beta > w[index]) {
      beta -= w[index];
      index = (index+1) % num_particles;
    }
    resample_ps.push_back(particles[index]);
  }
  particles = resample_ps;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}