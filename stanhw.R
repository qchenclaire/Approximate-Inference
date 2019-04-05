library(rstan)

y <- read.table("hw2data.txt")
N <- length(y[,1])

plot(density(y[,1]))
plot(density(y[,2]))
plot(density(y[,3]))
plot(density(y[,4]))

## stan data
stan_data <- list(N = N, y = y, K = 3, D = 4)

write("// Stan model

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
      } ",

"stan_model.stan")

## check
stanc("stan_model.stan")

## save filepath
stan_model <- "stan_model.stan"

## fit
fit <- stan(file = stan_model, data = stan_data, warmup = 500, iter = 1000, chains = 8, cores = 4, thin = 1)

## check it out
fit

## look at posterior
posterior <- extract(fit)
hist(posterior$mu)

## some other diagnostics
traceplot(fit)
stan_dens(fit)
stan_hist(fit)

## try out some variational inference methods in Stan...
m <- stan_model(file = "stan_model.stan")
fit2 <- vb(m, data = stan_data, algorithm = "fullrank")
fit2
## look at posterior
posterior2 <- extract(fit2)
hist(posterior2$mu)

## some other diagnostics
#traceplot(fit2)
stan_dens(fit2)
#stan_hist(fit2)
print(fit)
print(fit2)
#library(shinystan)
#launch_shinystan(fit2)
