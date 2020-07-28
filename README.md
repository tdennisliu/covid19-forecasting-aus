# covid19-forecasting-aus
code used to forecast covid19 cases

## Workflow and relevant scripts
Below is a summary of the pipeline from case line list data to producing forecasts using this repository.

1. LSHTM method to infer an effective reproduction number $R_{eff}$ from case data using EpiEstim. Requires:
    * case data
    * R packages EpiEstim, EpiNow, EpiSoon (see `model/LSHTM_r/install_lshtm_packages_R.r`)
   ```
   Rscript model/LSHTM_r/run_lshtm.R
   ```
2. Inference of parameters to produce an effective reproduction number locally $R_L$ from $R_{eff}$ and case data. Requires:
    * Google mobility indices
    * LSHTM $R_{eff}$ estimates
    * case data
   ```
    model/Reff_traffic.ipynb
   ```
3. Forecasting Google mobility indices and microdistancing trends. Requires:
   * Google mobility indices
   * Posterior samples from 2.

    ```
    model/forecast.ipynb
    ```
4.  Simulate cases from $R_L$. Code base lives in `sim_class.py`, but executed by scripts listed below. /num-of-days/ is the number of days since 01/03/2020. Requires:
    * case data
    * $R_L$ distribution file from 3.
    
  * For all states
    ```
    bash all_states.sh /num-of-simulations/ /num-of-days/ R_L
    ```

* For a single state (eg. VIC)
    ```
    python run_state.py /num-of-simulations/ /num-of-days/ R_L /state-initials/
    ```

5.  Examine simulation of cases. Requires:
    * case data
    * simulation files of all states from 4.

    ```
    analysis/collate_cases.ipynb
    ```
