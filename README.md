# covid19-forecasting-aus
code used to forecast covid19 cases

## Workflow and relevant scripts
Below is a summary of the pipeline from case line list data to producing forecasts using this repository.

1. Cori et al. (2013) method to infer an effective reproduction number $R_{eff}$ from case data. Requires:
    * case data in data folder
    * /shortdate/ i.e. 05Aug
   ```
   python model/EpyReff/run_estimator.py /shortdate/
   ```
2. Inference of parameters to produce an effective reproduction number locally $R_L$ from $R_{eff}$ and case data. Requires:
    * Google mobility indices
    * Micro-distancing surveys
    * Cori et al. (2013) $R_{eff}$ estimates from 1.
    * case data
    * /longdate/ i.e 2020-08-05
   ```
    python analysis/cprs/generate_posterior.py /longdate/
   ```
3. Forecasting Google mobility indices and microdistancing trends. Requires:
   * Google mobility indices
   * Micro-distancing surveys
   * Posterior samples from 2.

    ```
    python analysis/cprs/generate_RL_forecasts.py /longdate/
    ```
4.  Simulate cases from $R_L$. Code base lives in `sim_class.py`, but executed by scripts listed below. /num-of-days/ is the number of days since 01/03/2020. Requires:
    * case data
    * $R_L$ distribution file from 3.
    
  * For all states
    ```
    bash all_states.sh /num-of-simulations/ /num-of-days/ /longdate/
    ```

* For a single state (eg. VIC)
    ```
    python run_state.py /num-of-simulations/ /num-of-days/ /longdate/ /state-initials/
    ```

5.  Examine simulation of cases. Requires:
    * case data
    * simulation files of all states from 4.

    ```
    analysis/record_to_csv.py /longdate/
    ```
