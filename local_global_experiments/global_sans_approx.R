# author: T. Guyet, Inria
# date: 02/2022

library(tidyverse)
library(tsibble)
library(e1071)
library(lubridate)
library(forecast)
library(stringr)
library(randomForest)

#Configuration des variables d'environnement utiles
rm(list=ls()) 

#classifiers = c('lm','svm','rf')
classifiers = c('lm')
feats = c('p','p_tp','p_tp_e','p_tp_e_lisa')
#feats = c('p')
nb_bss_sample = 0 #number of bss in the training dataset (0 means all)
h=93  # taille de l'horizon de prédiction
k=100  # taille de l'historique
max_NA_number = 300 # max number of NA in a time series to handle it
start_train_date="2015-01-01"

# Chargement des données --------------------------------------------------

args <- commandArgs( trailingOnly = TRUE )
if( length(args) == 0 ) {
  rep_data = "../data_collection/"
  rep_results = "./results/"
  datasetfile = 'dataset_2015_2021.csv'
  prefix = '_2202_'
} else {
  rep_data = dirname(args[1])
  rep_results = rep_data
  datasetfile = basename(args[1])
  prefix = paste0(substring(datasetfile,1,nchar(datasetfile)-4),"_")
}

#chargement des données
data <- read_delim(paste0(rep_data,"/",datasetfile), ",")
code_bss <-data$bss %>% unique()

data <- distinct(data, bss, time, .keep_all= TRUE)

# il y a plusieurs données manquantes pour les données de niveau d'eau, on fait donc une imputation de données manquantes

#D'abord, on se débarrasse des séries pour lesquelles il manque trop de données
# -> moins de max_NA_number (mais au moins 1)
code_bss=(data %>% filter(is.na(p)) %>% group_by( bss) %>% summarize( n=n()) %>% filter(n<max_NA_number))$bss
# -> plus, ceux avec aucun NA
code_bss=union(code_bss, (data%>%group_by( bss) %>% summarize( n=sum(p) ) %>% filter( !is.na(n) ))$bss )
code_bss=code_bss[1:50]
data = data %>% filter( bss %in% code_bss )

# On applique une imputation sur chaque sous-groupe
ret = data %>% group_by( bss ) %>% group_map(
  ~ as.vector(na.interp(.x$p))
)
data$p = unlist(ret)

bdlisa <- read_delim(paste0(rep_data,"stations.csv"),",")
bdlisa = bdlisa %>% select(bss,EtatEH,NatureEH,MilieuEH,ThemeEH) %>% drop_na()
bdlisa$EtatEH = as.factor( bdlisa$EtatEH )
bdlisa$NatureEH = as.factor( bdlisa$NatureEH )
bdlisa$MilieuEH = as.factor( bdlisa$MilieuEH )
bdlisa$ThemeEH = as.factor( bdlisa$ThemeEH )
nb_features = 4 #ajout de 4 features aux stations

colnames(data) <- c('bss','t','tp','e','p')

# Préparation du jeu de données d'apprentissage ------------------------------

# Construction du jeu de données et des TN
cat(paste0("prepare train time series\n"))

train_df <-  data %>%filter( t<"2020-10-15" ) %>% select(p,tp,e)
n <- nrow(train_df)

partial <- list()
partial[[1]] <- (data%>%filter( t<"2020-10-15" ))[-(1:k),]
for (j in 1:(k-1)){
  partial[[j+1]] = train_df[-c(1:(k-j),(n-j+1):n),]
} 
partial[[k+1]] = train_df[-((n-k+1):n),]

train = bind_cols(partial)
rm(partial, train_df)
train = train %>% relocate( starts_with("p"), .after=last_col()) %>% 
  relocate( starts_with("tp"), .after=last_col()) %>% 
  relocate( starts_with("e"), .after=last_col())

colnames(train) <- c("bss","t",paste("p_",0:k, sep=""),paste("tp_",0:k, sep=""),paste("e_",0:k, sep=""))
train=train%>%filter( t>(ymd(start_train_date)+k) )

#calcul des TN
TN = train %>% select(bss, t, p_0)
TN$diff=c(0,(TN[2:nrow(train),]$p_0-TN[1:(nrow(train)-1),]$p_0)^2)
TN = TN %>% filter( t>(ymd(start_train_date)+k+1) )
TN = TN %>% group_by( bss ) %>% summarise( TN=mean(diff) )

train = left_join(train, bdlisa, by = "bss")

# Apprentissage des modèles -----------------------------------------------
cat(paste0("learn models\n"))

attributes <- colnames(train)
if (nb_bss_sample!=0 ) {
  samplebss = code_bss%>%sample(nb_bss_sample)
  train = train%>%filter( bss %in% samplebss )
}

#base du modèle
fff=paste0("p_0 ~ ", paste(attributes[str_detect(attributes, "^p_[1-9]")],collapse="+"))

if ('tp_' %in% feats) {
	fff=paste0(fff,"+", paste(attributes[str_detect(attributes, "tp_")],collapse="+"))
}
if( 'e_' %in% feats) {
	fff=paste0(fff,"+", paste(attributes[str_detect(attributes, "e_")],collapse="+"))
}
if( 'lisa' %in% feats ) {
	fff=paste0(fff,"+", paste(attributes[str_detect(attributes, "EH")],collapse="+"))
}


if ('lm' %in% classifiers) {
  model <- lm(as.formula(fff), data = train)
  if( 'lisa' %in% feats ) {
	  model$xlevels[['EtatEH']] <- levels(bdlisa$EtatEH)
	  model$xlevels[['NatureEH']] <- levels(bdlisa$NatureEH)
	  model$xlevels[['MilieuEH']] <- levels(bdlisa$MilieuEH)
	  model$xlevels[['ThemeEH']] <- levels(bdlisa$ThemeEH)
  }
} 
if('svm' %in% classifiers) {
  model <- svm(as.formula(fff), data = train,type='eps-regression',kernel='radial')
}
if('rf' %in% classifiers) {
  model <- randomForest(as.formula(fff), data = train, ntree=100)
}

if ('p_tp_e_lisa' %in% feats) {
    fff=paste0("p_0 ~ ", paste(attributes[str_detect(attributes, "^p_[1-9]|^tp_|^e_|EH")],collapse="+"))
    if ('lm' %in% classifiers) {
      lm_p_tp_e_lisa <- lm(as.formula(fff), data = train)
      lm_p_tp_e_lisa$xlevels[['EtatEH']] <- levels(bdlisa$EtatEH)
      lm_p_tp_e_lisa$xlevels[['NatureEH']] <- levels(bdlisa$NatureEH)
      lm_p_tp_e_lisa$xlevels[['MilieuEH']] <- levels(bdlisa$MilieuEH)
      lm_p_tp_e_lisa$xlevels[['ThemeEH']] <- levels(bdlisa$ThemeEH)
    } 
    if('svm' %in% classifiers) {
      svm_p_tp_e_lisa <- svm(as.formula(fff), data = train,type='eps-regression',kernel='radial')
    }
    if('rf' %in% classifiers) {
      rf_p_tp_e_lisa <- randomForest(as.formula(fff), data = train, ntree=100)
    }
}

if ('p_tp_e' %in% feats) {
    cat(paste0("\train+eto\n"))
    fff=paste0("p_0 ~ ", paste(attributes[str_detect(attributes, "^p_[1-9]|^tp_|^e_")],collapse="+"))
    if ('lm' %in% classifiers) {
      lm_p_tp_e <- lm(as.formula(fff), data = train)  
    } 
    if('svm' %in% classifiers) {
      svm_p_tp_e <- svm(as.formula(fff), data = train,type='eps-regression',kernel='radial')
    }
    if('rf' %in% classifiers) {
      rf_p_tp_e <- randomForest(as.formula(fff), data = train, ntree=100)
    }
}

if ('p_tp' %in% feats) {
    cat(paste0("\train\n"))
    fff=paste0("p_0 ~ ", paste(attributes[str_detect(attributes, "^p_[1-9]|^tp_")],collapse="+"))
    if ('lm' %in% classifiers) {
      lm_p_tp <- lm(as.formula(fff), data = train)  
    } 
    if('svm' %in% classifiers) {
      svm_p_tp <- svm(as.formula(fff), data = train,type='eps-regression',kernel='radial')
    }
    if('rf' %in% classifiers) {
      rf_p_tp <- randomForest(as.formula(fff), data = train, ntree=100)
    }
}

if ('p' %in% feats) {
    cat(paste0("\tno exogeneous variables\n"))
    fff=paste0("p_0 ~ ", paste(attributes[str_detect(attributes, "^p_[1-9]")],collapse="+"))
    if ('lm' %in% classifiers) {
      lm_p <- lm(as.formula(fff), data = train)
    } 
    if('svm' %in% classifiers) {
      svm_p <- svm(as.formula(fff), data = train,type='eps-regression',kernel='radial')
    }
    if('rf' %in% classifiers) {
      rf_p <- randomForest(as.formula(fff), data = train, ntree=100)
    }
}

rm(train)

# Evaluation des modèles --------------------------------------------------

### On fait ensuite tourner les modèles -----------------------------------
cat(paste0("eval models time series\n"))
all_rmsse <- NULL
for( classifier in classifiers ) {
  for( features in feats ) {
    model=paste0(classifier,"_",features)
    cat(paste0("\ttest model: ",model,"\n"))
    #on fait les prévisions avec model
    
    
    cat(paste0("\tprepare test time series\n"))
    # Verification: tous les bss ont bien 103 lignes
    # data %>%filter( (t>=ymd("2020-10-15")-k) &  (t<="2021-01-15") ) %>% group_by(bss) %>% summarise(n =n())
    test_df <-  data %>%filter( (t>=ymd("2020-10-15")-k) &  (t<="2021-01-15") )
    n <- nrow(test_df)
    
    #On conserve la vérité terrain à part
    gt = test_df$p
    
    cat(paste0("\trun forecasts\n"))
    for( i in (k+1):(h+k)) {
      #On construit le jeu de données
      partial <- list()
      partial[[1]] <- test_df[-(1:k),]
      for (j in 1:(k-1)){
        partial[[j+1]] = (test_df%>%select(p,tp,e))[-c(1:(k-j),(n-j+1):n),]
      } 
      partial[[k+1]] = (test_df%>%select(p,tp,e))[-((n-k+1):n),]
      
      test = bind_cols(partial)
      test = test %>% relocate( starts_with("p"), .after=last_col()) %>% 
        relocate( starts_with("tp"), .after=last_col()) %>% 
        relocate( starts_with("e"), .after=last_col())
      
      colnames(test) <- c("bss","t",paste("p_",0:k, sep=""),paste("tp_",0:k, sep=""),paste("e_",0:k, sep=""))
      test=test%>%filter( t>=ymd("2020-10-15") )
      test = left_join(test, bdlisa, by = "bss")
      
      # On construit une ligne de prédiction à faire par piezo (une ligne toutes les 93 pour le jeu de données)
      seq <- seq(i-k, nrow(test), by=h)
      seq_df <- seq(i, nrow(test_df), by=(h+k))
      # On effectue la prédiction
      test_df$p[seq_df] = predict(model, test[seq,])
    }
    
    test_df$gt=gt
    
    ldf = test_df %>%group_by(bss) %>%summarise(mse = sum( (gt-p)^2)/h )
    ldf = left_join(ldf, TN, by = "bss")
    ldf$rmsse = sqrt(ldf$mse/ldf$TN)
    ldf$id_method_ML=classifier
    ldf$type="Global"
    ldf$r=k
    ldf$use_exo_rain=str_detect(features,"tp")
    ldf$use_exo_eto=str_detect(features,"e")
    ldf$use_exo_bdlisa=str_detect(features,"bdlisa")
    ldf$approx_exo=FALSE
    all_rmsse <- rbind(all_rmsse,ldf)
    
    ### sauvegarde partielle
    write.table(all_rmsse,
                paste0(rep_data,"global_sansapprox.tmp.csv"),
                row.names=FALSE,
                sep="\t",
                dec="." )
  }
}

# Sauvegarde des résultats ------------------------------------------------

#on sauvegarde le tableaux avec les RMSSE de l'ensemble des bss

write.table(all_rmsse,
            paste0(rep_data,"global_sansapprox.csv"),
            row.names=FALSE,
            sep="\t",
            dec="." )
