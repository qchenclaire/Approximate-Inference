// Stan model

      data {
        int N; // sample size
        int D; // dimension of observed vars
        int K; // number of latent groups
        vector[D] y[N]; // data
      }

      parameters {
        ordered[K] mu; // locations of hidden states
        vector<lower = 0>[K] sigma; // variances of hidden states
        simplex[K] theta[D]; // mixture components
      }
      
      model {
        vector[K] obs[D];
        
        // adjust priors according to observation of ys
        //mu[1] ~ normal(-10, 1);
        //mu[2] ~ normal(5, 1);
        //mu[3] ~ normal(10, 1);
        for(k in 1:K){
          mu[k] ~ normal(0, 10);
          sigma[k] ~ inv_gamma(1,1);
        }

        for(d in 1:D){
          theta[d] ~ dirichlet(rep_vector(2.0, K));
        }
      
      
        // likelihood
        for(d in 1:D){
          for(i in 1:N) {
            for(k in 1:K) {
              obs[d][k] = log(theta[d][k]) + normal_lpdf(y[i][d] | mu[k], sigma[k]);
            }
            target += log_sum_exp(obs[d]);
          } 
        }
      } 
