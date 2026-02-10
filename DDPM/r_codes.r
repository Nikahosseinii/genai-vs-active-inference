library(dendextend)
library(circlize)
library(stylo)

dam <- read.csv('/Users/nikahosseini/Desktop/SUMMER24/Report1/FB/raw_normalized_all.csv', header = TRUE, sep = ";")
epsilon <- 0.00000001
dataset <- dam[1:10000, -c(1,2,3)]
dataset[dataset == "NULL"] <- 0
dataset <- apply(dataset, 1:2, as.numeric)

dataset <- dataset + matrix(runif(nrow(dataset) * ncol(dataset), -epsilon, epsilon), 
                            nrow = nrow(dataset), ncol = ncol(dataset))



d <- dist.entropy(dataset)


hc <- as.dendrogram(hclust(d))

circlize_dendrogram(hc, label_track_height = NA, dend_track_height = 0.6)
