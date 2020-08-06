#Install required R packages on phoenix
.libPaths(c("/fast/users/a1193089/local/RLibs",.libPaths()))


install.packages("drat")
drat:::add("epiforecasts")
install.packages("EpiNow2")
