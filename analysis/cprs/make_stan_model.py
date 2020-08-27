import pickle, pystan

rho_model_gamma = """
data {
    int N; //data length num days
    int K; //Number of mobility indices
    int j; //Number of states
    matrix[N,j] Reff; //response
    matrix[N,K] Mob[j]; //Mobility indices
    matrix[N,K] Mob_std[j]; ///std of mobility
    matrix[N,j] sigma2; //Variances of R_eff from previous study
    vector[N] policy; //Indicators for post policy or not
    matrix[N,j] local; //local number of cases
    matrix[N,j] imported; //imported number of cases

    
    int N_v; //length of VIC days
    int j_v; //second wave states
    matrix[N_v,j_v] Reff_v; //Reff for VIC in June
    matrix[N_v,K] Mob_v[j_v]; //Mob for VIC June
    matrix[N_v,K] Mob_v_std[j_v];// std of mobility
    matrix[N_v,j_v] sigma2_v;// variance of R_eff from previous study
    vector[N_v] policy_v;// micro distancing compliance
    matrix[N_v,j_v] local_v; //local cases in VIC
    matrix[N_v,j_v] imported_v; //imported cases in VIC
    
    vector[N] count_md[j]; //count of always
    vector[N] respond_md[j]; // num respondants
    
    vector[N_v] count_md_v[j_v]; //count of always
    vector[N_v] respond_md_v[j_v]; // num respondants

}
parameters {
    vector[K] bet; //coefficients
    real<lower=0> R_I; //base level imports,
    real<lower=0> R_L; //base level local
    real<lower=0> theta_md; // md weighting
    matrix<lower=0,upper=1>[N,j] prop_md; // proportion who are md'ing
    matrix<lower=0,upper=1>[N_v,j_v] prop_md_v;
    matrix<lower=0,upper=1>[N,j] brho; //estimate of proportion of imported cases
    matrix[N,K] noise[j];
    
    matrix<lower=0,upper=1>[N_v,j_v] brho_v; //estimate of proportion of imported cases
    matrix[N_v,K] noise_v[j_v];
}
transformed parameters {
    matrix[N,j] mu_hat;
    matrix[N_v,j_v] mu_hat_v;
    matrix[N,j] md; //micro distancing
    matrix[N_v,j_v] md_v; 
    
     
    for (i in 1:j) {

        
        for (n in 1:N){

            
            md[n,i] = pow(1+theta_md , -1*prop_md[n,i]);
            
            mu_hat[n,i] = brho[n,i]*R_I + (1-brho[n,i])*2*R_L*(
            (1-policy[n]) + md[n,i]*policy[n] )*inv_logit(
            noise[j][n,:]*(bet)); //mean estimate
        }
    }
    for (i in 1:j_v){
        for (n in 1:N_v){
            
            md_v[n,i] = pow(1+theta_md ,-1*prop_md_v[n,i]);
            
            mu_hat_v[n,i] = brho_v[n,i]*R_I + (1-brho_v[n,i])*2*R_L*(
                (1-policy_v[n]) + md_v[n,i]*policy_v[n] )*inv_logit(
                noise_v[i][n,:]*(bet)); //mean estimate
        }
    }
    
}
model {
    bet ~ normal(0,1);
    theta_md ~ lognormal(0,1);
    //md ~ beta(7,3);
    
    
    R_L ~ gamma(2.4*2.4/0.2,2.4/0.2);
    R_I ~ gamma(0.5*0.5/.2,0.5/.2);

 
    for (i in 1:j) {
        for (n in 1:N){
            prop_md[n,i] ~ beta(1 + count_md[i][n], 1+ respond_md[i][n] - count_md[i][n]);
            brho[n,i] ~ beta( 1+ imported[n,i], 1+ local[n,i]);
            noise[i][n,:] ~ normal( Mob[i][n,:] , Mob_std[i][n,:]);
            mu_hat[n,i] ~ gamma( Reff[n,i]*Reff[n,i]/(sigma2[n,i]), Reff[n,i]/sigma2[n,i]); //Stan uses shape/inverse scale
        }
    }
    for (i in 1:j_v){
        for (n in 1:N_v){
            prop_md_v[n,i] ~ beta(1 + count_md_v[i][n], 1+ respond_md_v[i][n] - count_md_v[i][n]);
            brho_v[n,i] ~ beta( 1+ imported_v[n,i], 1+ local_v[n,i]);
            noise_v[i][n,:] ~ normal( Mob_v[i][n,:] , Mob_v_std[i][n,:]);
            mu_hat_v[n,i] ~ gamma( Reff_v[n,i]*Reff_v[n,i]/(sigma2_v[n,i]), Reff_v[n,i]/sigma2_v[n,i]);
        }
    }
}
"""


sm_pol_gamma = pystan.StanModel(
    model_code = rho_model_gamma,
    model_name ='gamma_pol_state'
)

# save it to the file 'model.pkl' for later use
with open('/model/sm_pol_gamma.pkl', 'wb') as f:
    pickle.dump(sm_pol_gamma, f)