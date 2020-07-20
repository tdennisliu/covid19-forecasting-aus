# Script to read in R_eff RDS files and output into csv
setwd('~/Documents/GitHub/covid19-forecasting-aus/data/LSHTM_Reff_estimates/')
date <- '2020-07-16'
states <- c('WA','VIC','TAS','SA','QLD','NSW')
df <- data.frame()
for (i in 1:length(states)) {
  filename <- paste(states[i],'/',date,'/','bigr_estimates','.RDS',sep='')
  df_temp <- readRDS(filename)
  df_temp$state <- rep(states[i],dim(df_temp)[1])

  df<- rbind(df,df_temp)
  
}
#Remove column that is just a list of other columns and rt_type is same as type
df <- subset(df,select=!(names(df) %in% c('R0_range','rt_type')))

filename <- paste('R_eff_',date,'.csv',sep='')
write.csv(df,file=filename,row.names = F)
