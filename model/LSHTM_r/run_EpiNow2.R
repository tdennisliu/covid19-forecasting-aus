setwd("~/Documents/GitHub/covid19-forecasting-aus/")
#library(EpiSoon)
library(EpiNow2)
library(lubridate)
#library(EpiEstim)
run_rt_pipe.aus <- function(data.file, case.limit, results.folder.name, rt.windows=NULL){
  
  # INPUTS
  # data.file: "/home/debian/health_updates/COVID-19 UoM 14Apr2020 0800.xlsx"
  # case.limit: 20
  # results.folder.name: "results_parallel"
  # rt.windows: Defualt is 1:7.  
  
  require(lubridate)
  require(dplyr)
  #require(EpiSoon)
  require(EpiNow2)
  #require(EpiEstim)
  # require(NCoVUtils)
  
  dat <- readxl::read_xlsx(data.file)
  
  # Convert to date
  dat <- dat %>% mutate(TRUE_ONSET_DATE=ymd(TRUE_ONSET_DATE), SPECIMEN_DATE=ymd(SPECIMEN_DATE), 
                        NOTIFICATION_DATE=ymd(NOTIFICATION_DATE), NOTIFICATION_RECEIVE_DATE=ymd(NOTIFICATION_RECEIVE_DATE),
                        Diagnosis_Date=ymd(Diagnosis_Date))
  
  # Remove cases without a state
  dat <- dat[!is.na(dat$STATE),]
  
  dat$SPECIMEN_DATE[!grepl("2020", dat$SPECIMEN_DATE)] <- gsub('^\\d\\d\\d\\d', '2020', dat$SPECIMEN_DATE[!grepl("2020", dat$SPECIMEN_DATE)])
  dat$NOTIFICATION_DATE[!grepl("2020", dat$NOTIFICATION_DATE)] <- gsub('^\\d\\d\\d\\d', '2020', dat$NOTIFICATION_DATE[!grepl("2020", dat$NOTIFICATION_DATE)])
  dat$NOTIFICATION_RECEIVE_DATE[!grepl("2020", dat$NOTIFICATION_RECEIVE_DATE)] <- gsub('^\\d\\d\\d\\d', '2020', dat$NOTIFICATION_RECEIVE_DATE[!grepl("2020", dat$NOTIFICATION_RECEIVE_DATE)])
  
  
  dat$import_status <- "imported"
  dat$import_status[grepl(pattern="^1101",dat$PLACE_OF_ACQUISITION)] <- "local"
  # dat$import_status[grepl(pattern="^00038888",dat$PLACE_OF_ACQUISITION)] <- "No info"
  # dat$import_status[is.na(dat$PLACE_OF_ACQUISITION)] <- "Unconfirmed"
  
  dat$import_status[grepl(pattern="^00038888",dat$PLACE_OF_ACQUISITION)] <- "local"
  dat$import_status[is.na(dat$PLACE_OF_ACQUISITION)] <- "local"
  
  # Generate linelist data
  linelist <- dat %>% 
    select(date_onset=TRUE_ONSET_DATE, date_confirm=NOTIFICATION_RECEIVE_DATE, region=STATE, import_status) %>%
    mutate(date_onset = case_when(
      is.na(date_onset) ~ date_confirm - 5,
      TRUE ~ date_onset)) %>%
    mutate(report_delay = as.numeric(date_confirm-date_onset))
  
  
  # linelist$import_status <- as.factor(linelist$import_status)
  linelist$region <- as.factor(linelist$region)
  
  # Remove those with onset date after confirmation date
  linelist <- linelist %>% filter(date_confirm >= date_onset | is.na(date_confirm) | is.na(date_onset))
  
  # Generate case dataframe
  cases <- linelist %>% select(date=date_confirm, import_status, region) %>%
    group_by(date, region) %>%
    count() %>% rename(confirm = n)
  

  if (is.null(rt.windows)){
    rt.windows <- 1:7
  }
  
  #future::plan("multiprocess")
  
  (start.time <- Sys.time())
  cases <- as.data.frame(cases)
  
  generation_time <- list(mean = EpiNow2::covid_generation_times[1, ]$mean,
                          mean_sd = EpiNow2::covid_generation_times[1, ]$mean_sd,
                          sd = EpiNow2::covid_generation_times[1, ]$sd,
                          sd_sd = EpiNow2::covid_generation_times[1, ]$sd_sd,
                          max = 30)
  
  incubation_period <- list(mean = EpiNow2::covid_incubation_period[1, ]$mean,
                            mean_sd = EpiNow2::covid_incubation_period[1, ]$mean_sd,
                            sd = EpiNow2::covid_incubation_period[1, ]$sd,
                            sd_sd = EpiNow2::covid_incubation_period[1, ]$sd_sd,
                            max = 30)
  
  reporting_delay <- EpiNow2::bootstrapped_dist_fit(rlnorm(100, log(6), 1))
  ## Set max allowed delay to 30 days to truncate computation
  reporting_delay$max <- 30
  # Analysis
  EpiNow2::regional_epinow(reported_cases =cases, target_folder=results.folder.name,
                           generation_time = generation_time,
                           delays = list(incubation_period, reporting_delay),
                           horizon = 7,
                               case_limit=case.limit, samples = 1000)
  
  
  (end.time <- Sys.time()-start.time)
  
}

run_rt_pipe.aus("data/COVID-19 UoM 31Jul2020 0850.xlsx", 10,"data/LSHTM_Reff_estimates/")
