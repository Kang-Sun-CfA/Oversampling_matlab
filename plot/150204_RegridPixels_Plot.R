# R scrpt to plot RegridPixels.x output
# Lei Zhu, 02/04/15

# Load libraries
library(fields);library(maps)

# Set oversampling output folder
output <- "/n/home01/leizhu6/scripts/RegridPixels/output/"

# Set plot domain
plot_region  <- c(-130,-60,10,60) # Plot domain, US

# Set color bar
zlim_min     <- 0  # Plot z bar min
zlim_max     <- 18 # Plot z bar max

# Data lat and lon ranges, from RegridPixels.x 
region  <- c(-143,-57,8,67)

#==================================================
# Plot results from Res = 0.5 run 
#==================================================

res <- 0.5

NRows <- (region[4]-region[3])/res
NCols <- (region[2]-region[1])/res
Lat_common  <- seq((region[3]+0.5*res),(region[3]+0.5*res)+(NRows-1)*res,by= res)
Lon_common  <- seq((region[1]+0.5*res),(region[1]+0.5*res)+(NCols-1)*res,by= res)

Row_down  <- ceiling((plot_region[3]-Lat_common[1])/res)
Row_up    <- ceiling((plot_region[4]-Lat_common[1])/res)
Col_left  <- ceiling((plot_region[1]-Lon_common[1])/res)
Col_right <- ceiling((plot_region[2]-Lon_common[1])/res)

# Read in data
data_raw <- read.table(paste(output,"regrid_Smedt_OMI_HCHO_2013_Aug_Sep_0.5",sep=""),header=F)
head(data_raw)

Nlines <- dim(data_raw)[1]
Value1 <- array(NA,dim=c(NRows,NCols))

for(line in 1:Nlines){
  Value1[data_raw[line,1],data_raw[line,2]] <- data_raw[line,5]
}

# Plot
plotname=paste("Smedt_OMI_HCHO_2013_Aug_Sep_",as.character(res),".png",sep="")
png(plotname, width=8.2, height=8, units="in", res=300)
image.plot(Lon_common[Col_left:Col_right],Lat_common[Row_down:Row_up],t(Value1[Row_down:Row_up,Col_left:Col_right])/1e+15,
           horizontal=T,legend.shrink=1,axis.args = list(cex.axis =1.75),
           legend.width=1.5,legend.mar=4.5,zlim=c(zlim_min, zlim_max),
           legend.args=list(text=expression(paste("[",10^15, " molecules ",cm^-2,"]")),cex=1.75),           
           xlab='',ylab='',midpoint=T,axes=F,ann=F
)
title(xlab="",cex.lab=1.25,font.lab=2)
axis(1,at=pretty(Lon_common[Col_left:Col_right]),tck=-0.015,lwd=2,cex.axis=1.5,font=1)
title(ylab="",cex.lab=1.25,font.lab=2)
axis(2,at=pretty(Lat_common[Row_down:Row_up]),tck=-0.015,lwd=2,cex.axis=1.5,font=1,las=1)
title(main=paste("Smedt_OMI_HCHO_2013_Aug_Sep, Res=",as.character(res),sep=""),cex.main=1.4,font.main=2)

map('state',add=T,lwd=2,col="black")
map('world',add=T,lwd=2,col="black")

box(lwd=3)
dev.off()

#==================================================
# Plot results from Res = 0.1 run
#==================================================

res <- 0.1

NRows <- (region[4]-region[3])/res
NCols <- (region[2]-region[1])/res
Lat_common  <- seq((region[3]+0.5*res),(region[3]+0.5*res)+(NRows-1)*res,by= res)
Lon_common  <- seq((region[1]+0.5*res),(region[1]+0.5*res)+(NCols-1)*res,by= res)

Row_down  <- ceiling((plot_region[3]-Lat_common[1])/res)
Row_up    <- ceiling((plot_region[4]-Lat_common[1])/res)
Col_left  <- ceiling((plot_region[1]-Lon_common[1])/res)
Col_right <- ceiling((plot_region[2]-Lon_common[1])/res)

# Read in data
data_raw <- read.table(paste(output,"regrid_Smedt_OMI_HCHO_2013_Aug_Sep_0.1",sep=""),header=F)
head(data_raw)

Nlines <- dim(data_raw)[1]
Value2 <- array(NA,dim=c(NRows,NCols))

for(line in 1:Nlines){
  Value2[data_raw[line,1],data_raw[line,2]] <- data_raw[line,5]
}

# Plot
plotname=paste("Smedt_OMI_HCHO_2013_Aug_Sep_",as.character(res),".png",sep="")
png(plotname, width=8.2, height=8, units="in", res=300)
image.plot(Lon_common[Col_left:Col_right],Lat_common[Row_down:Row_up],t(Value2[Row_down:Row_up,Col_left:Col_right])/1e+15,
           horizontal=T,legend.shrink=1,axis.args = list(cex.axis =1.75),
           legend.width=1.5,legend.mar=4.5,zlim=c(zlim_min, zlim_max),
           legend.args=list(text=expression(paste("[",10^15, " molecules ",cm^-2,"]")),cex=1.75),           
           xlab='',ylab='',midpoint=T,axes=F,ann=F
)
title(xlab="",cex.lab=1.25,font.lab=2)
axis(1,at=pretty(Lon_common[Col_left:Col_right]),tck=-0.015,lwd=2,cex.axis=1.5,font=1)
title(ylab="",cex.lab=1.25,font.lab=2)
axis(2,at=pretty(Lat_common[Row_down:Row_up]),tck=-0.015,lwd=2,cex.axis=1.5,font=1,las=1)
title(main=paste("Smedt_OMI_HCHO_2013_Aug_Sep, Res=",as.character(res),sep=""),cex.main=1.4,font.main=2)

map('state',add=T,lwd=2,col="black")
map('world',add=T,lwd=2,col="black")

box(lwd=3)
dev.off()
