#ifndef boxm_opt_appearance_estimator_h_
#define boxm_opt_appearance_estimator_h_

#include <vector>
#include <boxm/boxm_apm_traits.h>
#include <boxm/sample/algo/boxm_simple_grey_processor.h>
#include <boxm/sample/algo/boxm_mog_grey_processor.h>
#if 0
#include <vcl_compiler.h>
#include <iostream>
#include <iostream>
#endif

template <boxm_apm_type APM>
void boxm_opt_compute_appearance(std::vector<typename boxm_apm_traits<APM>::obs_datatype> const& obs,
                                 std::vector<float> const& pre,
                                 std::vector<float> const& vis,
                                 typename boxm_apm_traits<APM>::apm_datatype &model)
{
  const float min_sigma=0.1f;
  // initialize appearance model estimate using visibility probabilities as weights.
  boxm_apm_traits<APM>::apm_processor::compute_appearance(obs,vis,model,min_sigma);
#if 1
  // use EM to refine estimate until convergence.
  unsigned int nobs= obs.size();
  if (nobs == 0) {
    // nothing to do.
    return;
  }
  std::vector<float> obs_weights = vis;

  const float epsilon = 1e-4f;
  const unsigned int max_its = 100;
  const float min_weight_change = 1e-5f;
  for (unsigned int i=0; i<max_its; ++i) {
    float max_weight_change = 0.0f;
    // EXPECTATION
    for (unsigned int n=0; n<nobs; ++n) {
      // for each observation, assign probabilities to each of 3 possibilities
      // P1: observation is generated by appearance model.
      // P2: point is visible, but observation is generated by a point "behind" this one. (i.e. point is not on a surface)
      // P3: point is not visible - observation is generated by a point "ahead" of this one.
      const float P1 = vis[n] * boxm_apm_traits<APM>::apm_processor::prob_density(model,obs[n]);
      const float P2 = 0.0f; //post_prob[n];
      const float P3 = pre[n];
      const float normalizing_factor = P1 + P2 + P3;
      float new_obs_weight = 0;
      if (normalizing_factor > epsilon) {
        new_obs_weight = P1 / normalizing_factor;
      }

      // compute delta weight for convergence check
      float weight_delta = std::fabs(obs_weights[n] - new_obs_weight);
      if (weight_delta > max_weight_change) {
        max_weight_change = weight_delta;
      }
      obs_weights[n] = new_obs_weight;
    }
    // check for convergence
    if (max_weight_change < min_weight_change) {
#if 0
    if (nobs > 1) {
      std::cout << "EM converged in " << i << " iterations." << std::endl;
      float total_prob = 0.0f;
      for (unsigned int n=0; n<nobs; ++n) {
        total_prob += obs_weights[n];
      }
      std::cout << "total prob for " << nobs << " observations = " << total_prob << std::endl
               << "sigma = " << model.sigma() << ",  gauss_weight = " << model.gauss_weight() << std::endl;
    }
#endif
      break;
    }
    // MAXIMIZATION
    boxm_apm_traits<APM>::apm_processor::update_appearance(obs,obs_weights,model,min_sigma);
  }

  // adjust model to account for small sample sizes.
  boxm_apm_traits<APM>::apm_processor::finalize_appearance(obs,obs_weights,model);
#endif
  return;
}

#endif // boxm_opt_appearance_estimator_h_
