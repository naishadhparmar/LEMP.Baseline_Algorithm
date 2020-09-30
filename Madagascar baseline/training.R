library(phcfM)
library(vcd)

data <- read.csv("merged_raster_as_csv.csv")

max(data$Y)
min(data$Y)
max(data$X)
min(data$X)
data = subset(data, X > 92.5 & Y >= 28.25)
train = subset(data, Y <= 28.6)
test = subset(data, Y > 28.6)

model = deforestation('Loss ~ X+Y+Roads+Waterways+Elevation+Slope', data=train)

sink("./outputs/model/gamma_selected_coda_output.txt")
print(model$mcmc)
sink()

pdf("./outputs/model/MCMC_posteriors.pdf")
plot(model$mcmc)
dev.off()

MCMC <- as.matrix(model$mcmc)
gamma.hat <- apply(MCMC,2,mean)
write.table(data.frame(gamma=gamma.hat,Parameter=colnames(MCMC),row.names=NULL),"./outputs/model/gamma_selected.txt",sep="\t",row.names=FALSE,quote=FALSE)

theta.prim.mean <- mean(data$Loss)
mf.fixed <- model.frame(formula='Loss ~ X+Y+Roads+Waterways+Elevation+Slope', data=test)
X <- model.matrix(attr(mf.fixed,"terms"),data=mf.fixed)
theta.hat <- inv.logit(X%*%gamma.hat)
theta.prim <- 1-(1-theta.hat)
pred <- rep(0,nrow(test))
threshold <- quantile(theta.prim, 1-theta.prim.mean)
Which <- which(theta.prim >= threshold)
pred[Which] <- 1
test$pred <- pred

n00 <- sum(test$pred==0 & test$Loss==0)
n11 <- sum(test$pred==1 & test$Loss==1)
n01 <- sum(test$pred==0 & test$Loss==1)
n10 <- sum(test$pred==1 & test$Loss==0)

OA = (n11+n00)/(n11+n10+n00+n01)
FOM = n11/(n11+n10+n01)
Sensitivity = n11/(n11+n01)
Specificity = n00/(n00+n10)
TSS <- Sensitivity+Specificity-1
kappa <- Kappa(matrix(c(n00,n01,n10,n11),nrow=2))$Unweighted[1]

metrics_result <- as.data.frame(matrix(NA,nrow=1,ncol=6))
names(metrics_result) <- c("Overall Accuracy","Figure of Merit","Sensitivity","Specificity","True Skill Statistic","Cohen's Kappa")
metrics_result[, 'Overall Accuracy'] <- OA
metrics_result[, 'Figure of Merit'] <- FOM
metrics_result[, 'Sensitivity'] <- Sensitivity
metrics_result[, 'Specificity'] <- Specificity
metrics_result[, 'True Skill Statistic'] <- TSS
metrics_result[, "Cohen's Kappa"] <- kappa
write.csv(metrics_result, './outputs/model/metrics_result.csv', row.names = FALSE)